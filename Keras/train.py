import os
import argparse

import matplotlib
import numpy as np

# matplotlib.use("TkAgg")
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
from models.unet3d import unet_model_3d

from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from keras.utils import Sequence
import math

from keras import backend as K
K.clear_session()
# import os
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"




class SequenceData(Sequence):
    def __init__(self, root_read, channel, batch_size, padding=512):
        # 初始化所需的参数
        self.path = root_read
        self.batch_size = batch_size
        self.channel = channel
        self.padding = padding
        self.data_path = os.path.join(root_read, 'Part2')
        self.label_path = os.path.join(root_read, 'train-labels-2')

    def __len__(self):
        # 让代码知道这个序列的长度
        num_imgs = len(os.listdir(self.data_path))
        return math.ceil(num_imgs / self.batch_size)

    def __getitem__(self, idx):
        # 迭代器部分
        # batch_x = self.x_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        # x_arrays = np.array([self.read_img(filename) for filename in batch_x])    # 读取一批图片
        # batch_y = np.zeros((self.batch_size, 1000))    # 为演示简洁全部假设y为0
        datas = []
        labels = []
        for data in os.listdir(self.data_path)[idx * self.batch_size: (idx + 1) * self.batch_size]:
            data = data[:-13]
            path1 = os.path.join(self.data_path, data + '-image.nii.gz')
            img = np.array(nib.load(path1).dataobj)
            # 预处理
            img = self.pre_pro(img)
            path2 = os.path.join(self.label_path, data + '-label.nii.gz')
            label = np.array(nib.load(path2).dataobj)
            # 补零
            if self.padding > 0 and img.shape[2] < self.padding:
                diff = int((self.padding - img.shape[2]) / 2)
                z1 = np.zeros((512, 512, diff), dtype=int)
                z2 = np.zeros((512, 512, diff), dtype=int)

                img = np.concatenate((z1, img, z2), axis=2)
                label = np.concatenate((z1, label, z2), axis=2)
                if self.padding > 0 and img.shape[2] < self.padding:
                    z = np.zeros((512, 512, 1), dtype=int)
                    img = np.concatenate((z, img), axis=2)
                    label = np.concatenate((z, label), axis=2)
            # None快速增加两个维度到5D，亦可使用np.newaxis()
            datas.append(img[None, None, :])
            labels.append(self.mask_to_onehot(label, self.channel))

        return np.concatenate(datas, axis=0)[:, :, :512, :512, :512], np.concatenate(labels, axis=0)[:, :, :512,
                                                                   :512, :512]

    def mask_to_onehot(self, mask, channel):
        """
        Converts a segmentation mask (H, W, N) to (K,H, W, N) where the first dim is a one
        hot encoding vector, and K is the number of segmented class.
        eg:
        mask:单通道的标注图像
        channel:num
        """
        semantic_map = []
        for colour in range(channel):
            class_map = np.zeros_like(mask)
            class_map[mask == colour] = 1
            semantic_map.append(class_map[None, :])
        semantic_map = np.concatenate(semantic_map, axis=0).astype(np.float32)
        semantic_map = semantic_map[None, :]
        return semantic_map

    def pre_pro(self, img, WC=250, WL=500, normal=False):
        """

        :param img:
        :param WC: CT window center
        :param WL: CT window length
        :param normal:
        :return:
        """
        # CT窗宽调整
        # 人眼是识别不出原始的CT图像的。只有当图像中的人体组织相差2000/16=125个灰阶时才能识别。
        # 但是人体组织一般是相差20~50之间，所以人眼要识别此类图像，就要将它分段放大。
        # 除此之外，人体里面的每个组织的CT值是不一样的。
        # 所以观察不同人体组织时需要调整不同的窗宽与窗位。
        # 肋骨包含骨（400+）部分和软骨（100左右）部分，需要分别加窗后综合
        # 并将int32调整到int8减少数据量
        minWindow = float(WC) - 0.5 * float(WL)
        newimg = (img - minWindow) / float(WL)
        newimg[newimg < 0] = 0
        newimg[newimg > 1] = 1
        if not normal:
            newimg = (newimg * 255).astype('uint8')
        return newimg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename", required=True,
                        help="JSON configuration file specifying the parameters for model training.")
    parser.add_argument("--model_filename",
                        help="Location to save the model during and after training. If this filename exists "
                             "prior to training, the model will be loaded from the filename.",
                        required=True)
    parser.add_argument("--training_log_filename",
                        help="CSV filename to save the to save the training and validation results for each epoch.",
                        required=True)
    parser.add_argument("--fit_gpu_mem", type=float,
                        help="Specify the amount of gpu memory available on a single gpu and change the image size to "
                             "fit into gpu memory automatically. Will try to find the largest image size that will fit "
                             "onto a single gpu. The batch size is overwritten and set to the number of gpus available."
                             " The new image size will be written to a new config file ending named "
                             "'<original_config>_auto.json'. This option is experimental and only works with the UNet "
                             "model. It has only been tested with gpus that have 12GB and 32GB of memory.")
    parser.add_argument("--group_average_filenames")
    args = parser.parse_args()

    return args


def main():
    # 先不管args，用默认参数做，后期调参的时候再改
    # namespace = parse_args()
    save_path = "./output/res"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # padding用512，这是根据model.png得出的图片长宽高都必须为2^（depth-1）的整数倍，而512相对来说比较整适合计算
    padding = 0
    channel = 1
    batch_size = 1
    epochs = 10
    train_steps = math.ceil(len(os.listdir(os.path.join('./train', 'Part2'))) / batch_size)
    test_steps = math.ceil(len(os.listdir(os.path.join('./test', 'Part2'))) / batch_size)
    # train_generator, train_steps, validation_generator, validation_steps = data_loader(channel, batch_size, padding)
    path = os.path.join(save_path, "trained_model.h5")
    if os.path.exists(path):
        from keras.models import load_model
        model = load_model(path)
    else:
        model = unet_model_3d((1, None, None, None), n_labels=channel)

    H = model.fit_generator(generator=SequenceData("./train", channel, batch_size, padding=padding),
                            steps_per_epoch=train_steps, epochs=epochs,  # final train model
                            validation_data=SequenceData("./test", channel, batch_size, padding=padding),
                            validation_steps=test_steps, shuffle=True,
                            verbose=1, workers=1, use_multiprocessing=True,
                            callbacks=[ModelCheckpoint('./output/res/trained_model.h5', save_best_only=True),
                                       CSVLogger('./output/res/training.log', append=True),
                                       ReduceLROnPlateau(factor=0.5, patience=50, verbose=1),
                                       EarlyStopping(verbose=1, patience=None)])

    # 绘制训练 & 验证的准确率值
    plt.plot(H.history['dice_coefficient'])
    plt.plot(H.history['val_dice_coefficient'])
    plt.title('Dice Coefficient')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(save_path, "dice.png"))
    plt.show()

    # 绘制训练 & 验证的损失值
    plt.plot(H.history['loss'])
    plt.plot(H.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(save_path, "loss.png"))
    plt.show()


if __name__ == "__main__":
    main()
