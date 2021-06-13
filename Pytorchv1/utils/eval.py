import os
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from tqdm import tqdm

import nibabel as nib


def transform_single(pred, name, affine=None, store=False):
    pre_label = label(pred > 0)
    pre_reigon = regionprops(pre_label, pred)
    pre_index = [0]
    pre_prob = [0.0]
    pre_label_code = [0]
    for region in pre_reigon:
        pre_index.append(region.label)
        pre_prob.append(region.mean_intensity)
        pre_label_code.append(1)
    info = pd.DataFrame({
        "public_id": [name] * len(pre_index),
        "label_id": pre_index,
        "confidence": pre_prob,
        "label_code": pre_label_code
    })

    if store:
        img = nib.Nifti1Image(pre_label, affine)
        return info, img
    return info


__all__ = ["froc", "plot_froc", "evaluate"]

# detection key FP values
DEFAULT_KEY_FP = (0.5, 1, 2, 4, 8)

# classification confusion matrix settings
label_code_dict = {
    0: "Background",
    1: "Frac"
}

pd.set_option("display.precision", 6)


def _get_gt_class(x):
    # if GT classification exists, use it
    if not pd.isna(x["gt_class"]):
        return x["gt_class"]
    # if the prediction doesn't hit anything, it's a false positive
    else:
        return "FP"


def _compile_pred_metrics(iou_matrix, gt_info, pred_info):
    """
    Compile prediction metrics into a Pandas DataFrame
    Parameters
    ----------
    iou_matrix : numpy.ndarray
        IoU array with shape of (n_pred, n_gt).
    gt_info : pandas.DataFrame
        DataFrame containing GT information.
    pred_info : pandas.DataFrame
        DataFrame containing prediction information.
    Returns
    -------
    pred_metrics : pandas.DataFrame
        A dataframe of prediction metrics.
    """
    # meanings of each column:
    # pred_label --  The index of prediction
    # max_iou -- The highest IoU this prediction has with any certain GT
    # hit_label -- The GT label with which this prediction has the highest IoU
    # prob -- The confidence prediction of this prediction
    # pred_class -- The classification prediction
    # gt_class -- The ground-truth prediction
    # num_gt -- Total number of GT in this volume
    pred_metrics = pd.DataFrame(np.zeros((iou_matrix.shape[0], 3)),
                                columns=["pred_label", "max_iou", "hit_label"])
    pred_metrics["pred_label"] = np.arange(1, iou_matrix.shape[0] + 1)
    pred_metrics["max_iou"] = iou_matrix.max(axis=1)
    pred_metrics["hit_label"] = iou_matrix.argmax(axis=1) + 1

    # if max_iou == 0, this prediction doesn't hit any GT
    pred_metrics["hit_label"] = pred_metrics.apply(lambda x: x["hit_label"]
    if x["max_iou"] > 0 else 0, axis=1)

    # fill in the classification prediction and detection confidence
    pred_metrics = pred_metrics.merge(
        pred_info[["label_id", "label_code", "confidence"]],
        how="left", left_on="pred_label", right_on="label_id")
    pred_metrics.rename({"confidence": "prob", "label_code": "pred_class"},
                        axis=1, inplace=True)
    pred_metrics.drop("label_id", axis=1, inplace=True)

    # compare the classification prediction against GT
    pred_metrics = pred_metrics.merge(gt_info[["label_id", "label_code"]],
                                      how="left", left_on="hit_label", right_on="label_id")
    pred_metrics.rename({"label_code": "gt_class"}, axis=1, inplace=True)
    pred_metrics.drop("label_id", axis=1, inplace=True)
    pred_metrics["gt_class"] = pred_metrics.apply(_get_gt_class, axis=1)

    pred_metrics["num_gt"] = iou_matrix.shape[1]

    return pred_metrics


def evaluate_single_prediction(gt_label, pred_label, gt_info, pred_info):
    """
    Evaluate a single prediction.
    Parameters
    ----------
    gt_label : numpy.ndarray
        The numpy array of ground-truth labelled from 1 to n.
    pred_label : numpy.ndarray
        The numpy array of prediction labelled from 1 to n.
    gt_info : pandas.DataFrame
        DataFrame containing GT information.
    pred_info : pandas.DataFrame
        DataFrame containing prediction information.
    Returns
    -------
    pred_metrics : pandas.DataFrame
        A dataframe of prediction metrics.
    num_gt : int
        Number of GT in this case.
    clf_conf_mat : pandas.DataFrame
        Classification confusion matrix.
    inter_sum : int
        Number of gt-pred intersection voxels.
    union_sum : int
        Number of gt-pred union voxels.
    y_true_sum : int
        Number of GT voxels.
    y_pred_sum : int
        Number of prediction voxels.
    """
    gt_label = gt_label.astype(np.uint8)
    pred_label = pred_label.astype(np.uint8)

    # # GT and prediction must have the same shape
    # assert gt_label.shape == pred_label.shape, \
    #     "The prediction and ground-truth have different shapes. gt:" \
    #         f" {gt_label.shape} and pred: {pred_label.shape}."

    num_gt = gt_label.max()
    num_pred = pred_label.max()

    # if the prediction is empty, return empty pred_metrics
    # and confusion matrix
    if num_pred == 0:
        pred_metrics = pd.DataFrame()

        return pred_metrics, num_gt

    # if GT is empty
    if num_gt == 0:
        pred_metrics = pd.DataFrame([
            {
                "pred_label": i,
                "max_iou": 0,
                "hit_label": 0,
                "gt_class": "FP",
                "num_gt": 0
            }
            for i in range(1, num_pred + 1)])
        pred_metrics = pred_metrics.merge(
            pred_info[["label_id", "label_code", "confidence"]],
            how="left", left_on="pred_label", right_on="label_id")
        pred_metrics.rename(
            {"label_code": "pred_class", "confidence": "prob"}, axis=1,
            inplace=True)
        pred_metrics.drop(["label_id"], axis=1, inplace=True)

        return pred_metrics, num_gt

    # binarize the GT and prediction
    gt_bin = (gt_label > 0).astype(np.uint8)
    pred_bin = (pred_label > 0).astype(np.uint8)

    num_pred = int(pred_label.max())
    iou_matrix = np.zeros((num_gt, num_pred))

    intersection = np.logical_and(gt_bin, pred_bin)
    union = label(np.logical_or(gt_bin, pred_bin))

    # iterate through all intersection area and evaluate predictions
    for region in regionprops(label(intersection)):
        # get an anchor point within the intersection to locate gt & pred
        anchor = tuple(region.coords[0].tolist())

        # get corresponding GT index, pred index and union index
        gt_idx = gt_label[anchor[0], anchor[1], anchor[2]]
        pred_idx = pred_label[anchor[0], anchor[1], anchor[2]]
        union_idx = union[anchor[0], anchor[1], anchor[2]]

        if gt_idx == 0 or pred_idx == 0 or union_idx == 0:
            continue

        inter_area = region.area
        union_area = (union == union_idx).sum()
        iou = inter_area / (union_area + 1e-8)
        iou_matrix[gt_idx - 1, pred_idx - 1] = iou

    iou_matrix = iou_matrix.T
    pred_metrics = _compile_pred_metrics(iou_matrix,
                                                       gt_info, pred_info)

    return pred_metrics, num_gt


def _froc_single_thresh(df_list, num_gts, p_thresh, iou_thresh):
    """
    Calculate the FROC for a single confidence threshold.
    Parameters
    ----------
    df_list : list of pandas.DataFrame
        List of Pandas DataFrame of prediction metrics.
    num_gts : list of int
        List of number of GT in each volume.
    p_thresh : float
        The probability threshold of positive predictions.
    iou_thresh : float
        The IoU threshold of predictions being considered as "hit".
    Returns
    -------
    fp : float
        False positives per scan for this threshold.
    recall : float
        Recall rate for this threshold.
    """
    EPS = 1e-8

    total_gt = sum(num_gts)
    # collect all predictions above the probability threshold
    df_pos_pred = [df.loc[df["prob"] >= p_thresh] for df in df_list
                   if len(df) > 0]

    # calculate total true positives
    total_tp = sum([len(df.loc[df["max_iou"] > iou_thresh, "hit_label"] \
                        .unique()) for df in df_pos_pred])

    # calculate total false positives
    total_fp = sum([len(df) - len(df.loc[df["max_iou"] > iou_thresh])
                    for df in df_pos_pred])

    fp = (total_fp + EPS) / (len(df_list) + EPS)
    recall = (total_tp + EPS) / (total_gt + EPS)

    return fp, recall


def _interpolate_recall_at_fp(fp_recall, key_fp):
    """
    Calculate recall at key_fp using interpolation.
    Parameters
    ----------
    fp_recall : pandas.DataFrame
        DataFrame of FP and recall.
    key_fp : float
        Key FP threshold at which the recall will be calculated.
    Returns
    -------
    recall_at_fp : float
        Recall at key_fp.
    """
    # get fp/recall interpolation points
    fp_recall_less_fp = fp_recall.loc[fp_recall.fp <= key_fp]
    fp_recall_more_fp = fp_recall.loc[fp_recall.fp >= key_fp]

    # if key_fp < min_fp, recall = 0
    if len(fp_recall_less_fp) == 0:
        return 0

    # if key_fp > max_fp, recall = max_recall
    if len(fp_recall_more_fp) == 0:
        return fp_recall.recall.max()

    fp_0 = fp_recall_less_fp["fp"].values[-1]
    fp_1 = fp_recall_more_fp["fp"].values[0]
    recall_0 = fp_recall_less_fp["recall"].values[-1]
    recall_1 = fp_recall_more_fp["recall"].values[0]
    recall_at_fp = recall_0 + (recall_1 - recall_0) \
                   * ((key_fp - fp_0) / (fp_1 - fp_0 + 1e-8))

    return recall_at_fp


def _get_key_recall(fp, recall, key_fp_list):
    """
    Calculate recall at a series of FP threshold.
    Parameters
    ----------
    fp : list of float
        List of FP at different probability thresholds.
    recall : list of float
        List of recall at different probability thresholds.
    key_fp_list : list of float
        List of key FP values.
    Returns
    -------
    key_recall : list of float
        List of key recall at each key FP.
    """
    fp_recall = pd.DataFrame({"fp": fp, "recall": recall}).sort_values("fp")
    key_recall = [_interpolate_recall_at_fp(fp_recall, key_fp)
                  for key_fp in key_fp_list]

    return key_recall


def froc(df_list, num_gts, iou_thresh=0.2, key_fp=DEFAULT_KEY_FP):
    """
    Calculate the FROC curve.
    Parameters
    df_list : list of pandas.DataFrame
        List of prediction metrics.
    num_gts : list of int
        List of number of GT in each volume.
    iou_thresh : float
        The IoU threshold of predictions being considered as "hit".
    key_fp : tuple of float
        The key false positive per scan used in evaluating the sensitivity
        of the model.
    Returns
    -------
    fp : list of float
        List of false positives per scan at different probability thresholds.
    recall : list of float
        List of recall at different probability thresholds.
    key_recall : list of float
        List of key recall corresponding to key FPs.
    avg_recall : float
        Average recall at key FPs. This is the evaluation metric we use
        in the detection track.
    """
    fp_recall = [_froc_single_thresh(df_list, num_gts, p_thresh, iou_thresh)
                 for p_thresh in np.arange(0, 1, 0.01)]
    fp = [x[0] for x in fp_recall]
    recall = [x[1] for x in fp_recall]
    key_recall = _get_key_recall(fp, recall, key_fp)
    avg_recall = np.mean(key_recall)

    return avg_recall


def evaluate(pred_iter, gt_iter, pred_info, gt_info):
    """
    Evaluate predictions against the ground-truth.
    Parameters
    ----------
    gt_dir : str
        The ground-truth nii directory.
    pred_dir : str
        The prediction nii directory.
    Returns
    -------
    eval_results : dict
        Dictionary containing detection and classification results.
    """
    gt_info["label_code"] = gt_info["label_code"].map(label_code_dict)
    pred_info["label_code"] = pred_info["label_code"].map(label_code_dict)
    # # GT and prediction directory sanity check
    # assert len(pred_iter) == len(gt_iter), \
    #     "Unequal number of predictions and ground-truths."
    # assert gt_iter.pid_list == pred_iter.pid_list, \
    #     "Unmatched file names in ground-truth and prediction directory."
    # perform evaluation
    # print(gt_info)
    # print(pred_info)
    # print(pred_info.confidence)
    det_result, num_gt = evaluate_single_prediction(
        gt_iter, pred_iter, gt_info, pred_info)

    # calculate the detection FROC
    avg_recall = froc([det_result], [num_gt])

    return avg_recall
