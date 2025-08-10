"""
@Author: Du Yunhao
@Filename: utils.py
@Contact: dyh_bupt@163.com
@Time: 2022/8/30 21:37
@Discription: utils
"""
import os
import math
import json
import torch
import logging
import numpy as np
from os.path import join
from torch.nn import init
from torchvision import transforms
from collections import defaultdict
import random

MODALITY = {'RGB/IR': -1, 'RGB': 0, 'IR': 1}
MODALITY_ = {-1:'All', 0: 'RGB', 1: 'IR'}
CAMERA = {'LS3': 0, 'G25': 1, 'CQ1': 2, 'W4': 3, 'TSG1': 4, 'TSG2': 5}


def get_auxiliary_alpha(curr_epoch, max_epoch, phi):
    # return phi
    # return 0.5 * math.exp(-phi * curr_epoch / max_epoch)
    return (math.cos(math.pi * curr_epoch / max_epoch) + phi) / (2 + 2 * phi)


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.T, beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist.cpu().numpy()

class ChannelExchange(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img
        else:
            idx = random.randint(0, 3)

            if idx == 0:
                # random select R Channel
                img[1, :, :] = img[0, :, :]
                img[2, :, :] = img[0, :, :]
            elif idx == 1:
                # random select B Channel
                img[0, :, :] = img[1, :, :]
                img[2, :, :] = img[1, :, :]
            elif idx == 2:
                # random select G Channel
                img[0, :, :] = img[2, :, :]
                img[1, :, :] = img[2, :, :]
            else:
                tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
                img[0, :, :] = tmp_img
                img[1, :, :] = tmp_img
                img[2, :, :] = tmp_img
            return img
def get_transform(opt, mode):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize(opt.img_hw),
            transforms.Pad(opt.padding),
            transforms.RandomCrop(opt.img_hw),
            transforms.ToTensor(),
            transforms.Normalize(opt.norm_mean, opt.norm_std),
            # ChannelExchange(p=0.5),
        ])
    elif mode == 'test':
        return transforms.Compose([
            transforms.Resize(opt.img_hw),
            transforms.ToTensor(),
            transforms.Normalize(opt.norm_mean, opt.norm_std)
        ])
    else:
        raise RuntimeError('Error transformation mode.')


def get_lr(opt, curr_epoch):
    if curr_epoch < opt.warmup_epoch:
        return (
            opt.warmup_start_lr
            + (opt.base_lr - opt.warmup_start_lr)
            * curr_epoch
            / opt.warmup_epoch
        )
    else:
        return (
            opt.cosine_end_lr
            + (opt.base_lr - opt.cosine_end_lr)
            * (
                math.cos(
                    math.pi * (curr_epoch - opt.warmup_epoch) / (opt.max_epoch - opt.warmup_epoch)
                )
                + 1.0
            )
            * 0.5
        )


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weights_init_kaiming(m):
    """Copied from https://github.com/mangye16/DDAG/blob/master/model_main.py"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    """Copied from https://github.com/mangye16/DDAG/blob/master/model_main.py"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


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


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix="", lr=0.):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches, lr)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches, lr):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + '] [lr: {:.2e}]'.format(lr)


def save_configs(opt):
    configs = vars(opt)
    os.makedirs(opt.save_dir, exist_ok=True)
    json.dump(
        configs,
        open(join(opt.save_dir, 'config.json'), 'w'),
        indent=2
    )


def get_logger(save_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    filename = join(save_dir, 'log.txt')
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][%(levelname)s] %(message)s')

    # writting to file
    file_handler = logging.FileHandler(filename, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # display in terminal
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(formatter)
    # logger.addHandler(stream_handler)

    return logger


def load_from_ckpt(model, model_name, ckpt_path):
    print(f'load from {ckpt_path}...')
    ckpt = torch.load(ckpt_path)
    epoch = ckpt['epoch']
    state_dict = ckpt[model_name]
    filtered_state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}

    # 加载去掉分类头的参数
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)

    print(f"Loaded with {len(missing)} missing and {len(unexpected)} unexpected keys.")
    return model, epoch
