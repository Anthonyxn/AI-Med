from dataset.dataset_rib_val import Val_Dataset
from dataset.dataset_rib_train import Train_Dataset

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config
import pandas as pd

from models import UNet, ResUNet, KiUNet_min, SegNet

from utils import logger, weights_init, metrics, common, loss, eval
import os
import numpy as np
from collections import OrderedDict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def val(model, val_loader, loss_func, gt_info, n_labels):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)
    frocs = AverageMeter()
    with torch.no_grad():
        for idx, (data, target, name) in tqdm(enumerate(val_loader), total=len(val_loader)):
            name = name[0]
            cut_gt_info = gt_info.loc[gt_info["public_id"] == name] \
            .reset_index(drop=True)
            tmp_target = target
            data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target, n_labels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target)

            val_loss.update(loss.item(), data.size(0))
            val_dice.update(output, target)

            # 计算froc
            if not args.cpu:
                res = output.cpu()
            else:
                res = output
            pred = np.argmax(res.numpy()[0, :], axis=0)
            info = eval.transform_single(pred, name)
            tmp_target = tmp_target.numpy()[0,:]
            froc = eval.evaluate(pred,tmp_target,info,cut_gt_info)
            frocs.update(froc)

    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice': val_dice.avg[1], "Val_Froc": frocs.avg})
    return val_log


def train(model, train_loader, optimizer, loss_func, n_labels, alpha):
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)
    train_froc = []
    for idx, (data, target, name) in tqdm(enumerate(train_loader), total=len(train_loader)):
        tmp_target = target
        data, target = data.float(), target.long()
        target = common.to_one_hot_3d(target, n_labels)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss0 = loss_func(output[0], target)
        loss1 = loss_func(output[1], target)
        loss2 = loss_func(output[2], target)
        loss3 = loss_func(output[3], target)

        loss = loss3 + alpha * (loss0 + loss1 + loss2)
        loss.backward()
        optimizer.step()

        train_loss.update(loss3.item(), data.size(0))
        train_dice.update(output[3], target)

        # train部分不计算froc（GPU训练时值不能在CPU中运行）

    val_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice': train_dice.avg[1]})
    return val_log


if __name__ == '__main__':
    args = config.args
    args.dataset_path = "./RibFrac/fixed_train"
    args.val_data_path = "./RibFrac/fixed_val"
    save_path = os.path.join('./experiments', args.save)
    train_gt_info = pd.read_csv(args.train_gt_csv)
    train_gt_info.loc[train_gt_info['label_code'] != 0, 'label_code'] = 1
    val_gt_info = pd.read_csv(args.val_gt_csv)
    val_gt_info.loc[val_gt_info['label_code'] != 0, 'label_code'] = 1
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cpu'
                          if args.cpu else 'cuda'
                          )
    # data info
    train_loader = DataLoader(dataset=Train_Dataset(args), batch_size=args.batch_size, num_workers=args.n_threads,
                              shuffle=True)
    val_loader = DataLoader(dataset=Val_Dataset(args), batch_size=1, num_workers=args.n_threads, shuffle=False)

    # model info
    model = UNet(in_channel=1, out_channel=args.n_labels, training=True).to(device)

    model.apply(weights_init.init_model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # common.print_network(model)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU

    loss = loss.TverskyLoss()

    log = logger.Train_Logger(save_path, "train_log")

    best = [0, 0]  # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器
    alpha = 0.4  # 深监督衰减系数初始值
    for epoch in range(1, args.epochs + 1):
        common.adjust_learning_rate(optimizer, epoch, args)
        train_log = train(model, train_loader, optimizer, loss, args.n_labels, alpha)
        val_log = val(model, val_loader, loss,val_gt_info, args.n_labels)
        log.update(epoch, train_log, val_log)

        # Save checkpoint.
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['Val_dice'] > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))

        # 深监督系数衰减
        if epoch % 30 == 0: alpha *= 0.8

        # # early stopping
        # if args.early_stop is not None:
        #     if trigger >= args.early_stop:
        #         print("=> early stopping")
        #         break
        torch.cuda.empty_cache()
