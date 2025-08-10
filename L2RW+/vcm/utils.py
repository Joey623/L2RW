import os
import numpy as np
from torch.utils.data.sampler import Sampler
import sys
import os.path as osp
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of color image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label


def GenIdx(label):
    pos = []
    unique_label = np.unique(label)
    for i in range(len(unique_label)):
        tmp_pos = [k for k, v in enumerate(label) if v == unique_label[i]]
        pos.append(tmp_pos)
    return pos


def GenCamIdx(gall_img, gall_label, mode):
    if mode == 'indoor':
        camIdx = [1, 2]
    else:
        camIdx = [1, 2, 4, 5]
    gall_cam = []
    for i in range(len(gall_img)):
        gall_cam.append(int(gall_img[i][-10]))

    sample_pos = []
    unique_label = np.unique(gall_label)
    for i in range(len(unique_label)):
        for j in range(len(camIdx)):
            id_pos = [k for k, v in enumerate(gall_label) if v == unique_label[i] and gall_cam[k] == camIdx[j]]
            if id_pos:
                sample_pos.append(id_pos)
    return sample_pos


def ExtractCam(gall_img):
    gall_cam = []
    for i in range(len(gall_img)):
        cam_id = int(gall_img[i][-10])
        # if cam_id ==3:
        # cam_id = 2
        gall_cam.append(cam_id)

    return np.array(gall_cam)


# class IdentitySampler(Sampler):
#     """Sample person identities evenly in each batch.
#         Args:
#             train_color_label, train_thermal_label: labels of two modalities
#             color_pos, thermal_pos: positions of each identity
#             batchSize: batch size
#     """

#     def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize, epoch):
#         uni_label = np.unique(train_color_label)
#         self.n_classes = len(uni_label)

#         N = np.maximum(len(train_color_label), len(train_thermal_label))
#         for j in range(int(N / (batchSize * num_pos)) + 1):
#             batch_idx = np.random.choice(uni_label, batchSize, replace=False)
#             for i in range(batchSize):
#                 sample_color = np.random.choice(color_pos[batch_idx[i]], int(num_pos))
#                 sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], int(num_pos))

#                 if j == 0 and i == 0:
#                     index1 = sample_color
#                     index2 = sample_thermal
#                 else:
#                     index1 = np.hstack((index1, sample_color))
#                     index2 = np.hstack((index2, sample_thermal))

#         self.index1 = index1
#         self.index2 = index2
#         self.N = N

#     def __iter__(self):
#         return iter(np.arange(len(self.index1)))

#     def __len__(self):
#         return self.N
# class IdentitySampler(Sampler):
#     def __init__(self, train_label, color_pos, num_pos, batchsize):
#         uni_label = np.unique(train_label)
#         self.n_classes = len(uni_label)
#         N = len(train_label)
#         # color_pos = GenIdx(train_color_label)
#         for j in range(int(N / (batchsize * num_pos)) + 1):
#             batch_idx = np.random.choice(uni_label, batchsize, replace=False)
#             for i in range(batchsize):
#                 sample_color = np.random.choice(color_pos[batch_idx[i]], int(num_pos))
#                 if j==0 and i==0:
#                     index = sample_color
#                 else:
#                     index = np.hstack((index, sample_color))
#         self.index = index
#         self.N = N

#     def __iter__(self):
#         return iter(np.arange(len(self.index)))

#     def __len__(self):
#         return self.N
class IdentitySampler(Sampler):
    def __init__(self, train_label, color_pos, num_pos, batchsize):
        # 获取唯一标签
        uni_label = np.unique(train_label)
        self.n_classes = len(uni_label)
        N = len(train_label)

        # 生成索引
        index = []
        for j in range(int(N / (batchsize * num_pos)) + 1):
            # 随机选择一批唯一标签
            batch_idx = np.random.choice(uni_label, batchsize, replace=False)
            for i in range(batchsize):
                # 使用实际的标签值而不是索引
                selected_color_pos = color_pos[np.where(uni_label == batch_idx[i])[0][0]]
                sample_color = np.random.choice(selected_color_pos, int(num_pos))
                if j == 0 and i == 0:
                    index = sample_color
                else:
                    index = np.hstack((index, sample_color))

        self.index = index
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index)))

    def __len__(self):
        return self.N


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


def mkdir_if_missing(directory):
    pass


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def set_seed(seed, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('LayerNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)