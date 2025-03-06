import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable
import pdb
import torch.nn.functional as F


class Prototype_loss(nn.Module):
    def __init__(self, weight1=0.5, weight2=0.5):
        super(Prototype_loss, self).__init__()

        self.weight1 = weight1
        self.weight2 = weight2

    def forward(self, x, label, global_prototype, device):
        B, _ = x.shape
        edu_dist = []
        cos_dist = []
        for i in range(B):
            class_label = label[i].item()
            prototype = torch.tensor(global_prototype[class_label]).to(device)
            feature = x[i]
            prototype_distance = F.mse_loss(feature, prototype)
            prototype_cosine = 1 - F.cosine_similarity(feature, prototype, dim=0)
            edu_dist.append(prototype_distance)
            cos_dist.append(prototype_cosine)
        edu_dist = torch.stack(edu_dist)
        cos_dist = torch.stack(cos_dist)
        loss = 0.5 * cos_dist.mean() + 0.5 * edu_dist.mean()

        return loss


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    dist = 2 - 2 * torch.mm(x, y.t())
    return dist


class supcon_loss(nn.Module):
    def __init__(self, num_pos, temperature=0.1, base_temperature=0.07):
        super(supcon_loss, self).__init__()

        self.temp = temperature
        self.base_temp = base_temperature
        self.num_pos = num_pos

    def forward(self, features, labels=None):
        batch_size = features.shape[0]
        # [b] -> [batch_size], batch_size = b // num_pos
        labels = torch.unique_consecutive(labels)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temp)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        device = features.device
        logits_mask = torch.scatter(
            torch.ones_like(mask, device=device),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temp / self.base_temp) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# def supcon_loss(features, labels=None, mask=None, temperature = 0.1):
#     base_temperature = 0.07
#     """Compute loss for model. If both `labels` and `mask` are None,
#     it degenerates to SimCLR unsupervised loss:
#     https://arxiv.org/pdf/2002.05709.pdf
#     Args:
#         features: hidden vector of shape [bsz, n_views, ...].
#         labels: ground truth of shape [bsz].
#         mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
#             has the same class as sample i. Can be asymmetric.
#     Returns:
#         A loss scalar.
#     """
#     device = (torch.device('cuda')
#               if features.is_cuda
#               else torch.device('cpu'))

#     if len(features.shape) < 3:
#         raise ValueError('`features` needs to be [bsz, n_views, ...],'
#                          'at least 3 dimensions are required')
#     if len(features.shape) > 3:
#         features = features.view(features.shape[0], features.shape[1], -1)

#     batch_size = features.shape[0]
#     if labels is not None and mask is not None:
#         raise ValueError('Cannot define both `labels` and `mask`')
#     elif labels is None and mask is None:
#         mask = torch.eye(batch_size, dtype=torch.float32).to(device)
#     elif labels is not None:
#         labels = labels.contiguous().view(-1, 1)
#         if labels.shape[0] != batch_size:
#             raise ValueError('Num of labels does not match num of features')
#         mask = torch.eq(labels, labels.T).float().to(device)
#     else:
#         mask = mask.float().to(device)

#     contrast_count = features.shape[1]
#     contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
#     anchor_feature = contrast_feature
#     anchor_count = contrast_count

#     # compute logits
#     anchor_dot_contrast = torch.div(
#         torch.matmul(anchor_feature, contrast_feature.T),
#     temperature)
#     # for numerical stability
#     logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#     logits = anchor_dot_contrast - logits_max.detach()

#     # tile mask
#     mask = mask.repeat(anchor_count, contrast_count)
#     # mask-out self-contrast cases
#     logits_mask = torch.scatter(
#         torch.ones_like(mask),
#         1,
#         torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
#         0
#     )
#     mask = mask * logits_mask

#     # compute log_prob
#     exp_logits = torch.exp(logits) * logits_mask
#     log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

#     # compute mean of log-likelihood over positive
#     mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

#     # loss
#     loss = - (temperature / base_temperature) * mean_log_prob_pos
#     loss = loss.view(anchor_count, batch_size).mean()

#     return loss


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class PixelCenterTLoss(nn.Module):
    """
    这段代码是一个PyTorch模块，用于实现像素中心距离损失函数。该函数的输入为输入张量和目标张量，其中输入张量包含n个样本的特征向量，目标张量包含这些样本的目标标签。损失函数的计算过程如下：
    1. 把所有样本的特征向量投影到相同目标标签的中心点上，得到中心向量
    2. 计算每个样本的特征向量到中心向量的欧氏距离
    3. 对所有样本计算距离的平均值
    """

    def __init__(self):
        super(PixelCenterTLoss, self).__init__()

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # Come to centers
        centers = []
        for i in range(n):
            centers.append(inputs[targets == targets[i]].mean(0))
        centers = torch.stack(centers)
        dist_pc = (inputs - centers) ** 2
        dist_pc = dist_pc.sum(1)
        dist_pc = dist_pc.sqrt()
        loss = torch.mean(dist_pc)
        return loss


class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss

    # Adaptive weights


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct


class TripletLoss_ADP(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self, alpha=1, gamma=1, square=0):
        super(TripletLoss_ADP, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()
        self.alpha = alpha
        self.gamma = gamma
        self.square = square

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap * self.alpha, is_pos)
        weights_an = softmax_weights(-dist_an * self.alpha, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        # ranking_loss = nn.SoftMarginLoss(reduction = 'none')
        # loss1 = ranking_loss(closest_negative - furthest_positive, y)

        # squared difference
        if self.square == 0:
            y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
            loss = self.ranking_loss(self.gamma * (closest_negative - furthest_positive), y)
        else:
            diff_pow = torch.pow(furthest_positive - closest_negative, 2) * self.gamma
            diff_pow = torch.clamp_max(diff_pow, max=88)

            # Compute ranking hinge loss
            y1 = (furthest_positive > closest_negative).float()
            y2 = y1 - 1
            y = -(y1 + y2)

            loss = self.ranking_loss(diff_pow, y)

        # loss = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct


class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()

    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T = 3

        predict = F.log_softmax(pred / T, dim=1)
        target_data = F.softmax(label / T, dim=1)
        target_data = target_data + 10 ** (-7)
        target = Variable(target_data.data.cuda(), requires_grad=False)
        loss = T * T * ((target * (target.log() - predict)).sum(1).sum() / target.size()[0])
        return loss


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis=1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis=1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx


class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes,
                                                                             batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        # print(mask)

        dist = []
        for i in range(batch_size):
            # print(mask[i])
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss


class inter_center_margin_loss(nn.Module):
    def __init__(self, num_classes, feat_dim, margin):
        super(inter_center_margin_loss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, x, x_recon, labels):
        batch_size = x.size(0)
        center = self.centers[labels]
        dist = (x_recon - center).pow(2).sum(dim=-1)
        center_loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        # Compute pairwise distance, replace by the official when merged
        dist1 = torch.pow(x_recon, 2).sum(dim=1, keepdim=True).expand(batch_size, batch_size)
        dist2 = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, batch_size)

        dist = dist1 + dist2.t()
        dist.addmm_(1, -2, x_recon, x.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], []
        for i in range(batch_size):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return center_loss, loss, correct


class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''

    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.1):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                              self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


from typing import Tuple

import torch
from torch import nn, Tensor


def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    """
    这段代码是一个将标签转化为相似度矩阵的函数。
    输入参数为标准化的特征和标签张量，返回值为正样本相似度矩阵和负样本相似度矩阵（用布尔类型的张量表示）。
    """
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


# m=0.25, gamma=64
class CircleLoss(nn.Module):
    """
    Sun Y, Cheng C, Zhang Y, et al. Circle loss: A unified perspective of pair similarity optimization.
    """

    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


class CircleLoss_2(nn.Module):
    '''
    a circle loss implement, simple and crude
    '''

    def __init__(self, m: float, gamma: float):
        super(CircleLoss_2, self).__init__()
        self.m = m
        self.gamma = gamma

    def forward(self, features: Tensor, pids: Tensor, dist: str = 'cos'):
        if pids.size(0) == features.size(0):
            m = pids.size(0)
        else:
            raise ValueError('Error in the dim of the pids')

        # compute the mask of pos_mat & neg_mat
        pos = pids.unsqueeze(1).expand(m, m).eq(pids.unsqueeze(0).expand(m, m)).float()
        pos_mat = pos - torch.eye(m, m, device=pos.device)
        neg_mat = pids.unsqueeze(1).expand(m, m).ne(pids.unsqueeze(0).expand(m, m)).float()
        # cos or euclidean
        if dist == 'cos':
            features = F.normalize(features)
            # cosine dist of the vecters
            dist_cos = torch.matmul(features, features.t())  # 32 * 32

            s_p = dist_cos * pos_mat
            s_n = dist_cos * neg_mat
            alpha_p = torch.clamp_min(-s_p.detach() + 1 + self.m, min=0.)
            alpha_n = torch.clamp_min(s_n.detach() + self.m, min=0.)
            delta_p = 1 - self.m
            delta_n = self.m
            # logit_p = - self.gamma * alpha_p * (s_p - delta_p) * pos_mat
            logit_p = - self.gamma * alpha_p * (s_p - delta_p)
            # logit_n = self.gamma * alpha_n * (s_n - delta_n) * neg_mat
            logit_n = self.gamma * alpha_n * (s_n - delta_n)
            exp_p = torch.exp(logit_p) * pos_mat
            exp_n = torch.exp(logit_n) * neg_mat
            # loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
            loss = F.softplus(exp_p.sum(dim=1).log() + exp_n.sum(dim=1).log()).mean()

            return loss

        if dist == 'euclidean':
            dist = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(m, m)
            dist = dist + dist.t()
            dist.addmm_(1, -2, features, features.t())
            dist.clamp(1e-12).sqrt()

            d_p = dist * pos_mat
            d_n = dist * neg_mat

            logit_p = 0.5 * (d_p + self.m) * pos_mat
            logit_n = - 0.5 * d_n

            loss = F.softplus(
                torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
            # embed()
            return loss


# margin:0.35
# gamma:128

class PairCircle(nn.Module):
    '''
    a circle loss implement, simple and crude
    '''

    def __init__(self, margin: float, gamma: float):
        super(PairCircle, self).__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, embedding: Tensor, targets: Tensor):
        embedding = F.normalize(embedding, dim=1)

        dist_mat = torch.matmul(embedding, embedding.t())

        N = dist_mat.size(0)

        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        # Mask scores related to itself
        is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg

        alpha_p = torch.clamp_min(-s_p.detach() + 1 + self.margin, min=0.)
        alpha_n = torch.clamp_min(s_n.detach() + self.margin, min=0.)
        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = - self.gamma * alpha_p * (s_p - delta_p) + (-99999999.) * (1 - is_pos)
        logit_n = self.gamma * alpha_n * (s_n - delta_n) + (-99999999.) * (1 - is_neg)

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return loss


def softmax_weights_1(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def hard_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pair wise distance between samples, shape [N, M]
      is_pos: positive index with shape [N, M]
      is_neg: negative index with shape [N, M]
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N]
    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N]
    dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)

    return dist_ap, dist_an


def weighted_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the weighted positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      is_pos:
      is_neg:
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2

    is_pos = is_pos
    is_neg = is_neg
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights_1(dist_ap, is_pos)
    weights_an = softmax_weights_1(-dist_an, is_neg)

    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)

    return dist_ap, dist_an


class triplet_loss_soft(nn.Module):
    '''
    a circle loss implement, simple and crude
    '''

    def __init__(self, margin=0, norm_feat=False, hard_mining=True):
        super(triplet_loss_soft, self).__init__()
        self.margin = margin
        self.norm_feat = norm_feat
        self.hard_mining = hard_mining

    def forward(self, embedding: Tensor, targets: Tensor):

        if self.norm_feat:
            dist_mat = cosine_dist(embedding, embedding)
        else:
            dist_mat = euclidean_dist(embedding, embedding)

        # For distributed training, gather all features from different process.
        # if comm.get_world_size() > 1:
        #     all_embedding = torch.cat(GatherLayer.apply(embedding), dim=0)
        #     all_targets = concat_all_gather(targets)
        # else:
        #     all_embedding = embedding
        #     all_targets = targets

        N = dist_mat.size(0)
        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        if self.hard_mining:
            dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
        else:
            dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if self.margin > 0:
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=self.margin)
        else:
            loss = F.soft_margin_loss(dist_an - dist_ap, y)
            # fmt: off
            if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
            # fmt: on
        return loss


class DCL(nn.Module):
    def __init__(self, num_pos=4, feat_norm='no'):
        super(DCL, self).__init__()
        self.num_pos = num_pos
        self.feat_norm = feat_norm

    def forward(self, inputs, targets):
        if self.feat_norm == 'yes':
            inputs = F.normalize(inputs, p=2, dim=-1)

        N = inputs.size(0)
        id_num = N // 2 // self.num_pos

        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())
        is_neg_c2i = is_neg[::self.num_pos, :].chunk(2, 0)[0]  # mask [id_num, N]

        centers = []
        for i in range(id_num):
            centers.append(inputs[targets == targets[i * self.num_pos]].mean(0))
        centers = torch.stack(centers)

        dist_mat = pdist_torch(centers, inputs)  # c-i

        an = dist_mat * is_neg_c2i
        an = an[an > 1e-6].view(id_num, -1)

        d_neg = torch.mean(an, dim=1, keepdim=True)
        mask_an = (an - d_neg).expand(id_num, N - 2 * self.num_pos).lt(0)  # mask
        an = an * mask_an

        list_an = []
        for i in range(id_num):
            list_an.append(torch.mean(an[i][an[i] > 1e-6]))
        an_mean = sum(list_an) / len(list_an)

        ap = dist_mat * ~is_neg_c2i
        ap_mean = torch.mean(ap[ap > 1e-6])

        loss = ap_mean / an_mean

        return loss


class MSEL(nn.Module):
    def __init__(self, num_pos, feat_norm='no'):
        super(MSEL, self).__init__()
        self.num_pos = num_pos
        self.feat_norm = feat_norm

    def forward(self, inputs, targets):
        if self.feat_norm == 'yes':
            inputs = F.normalize(inputs, p=2, dim=-1)

        target, _ = targets.chunk(2, 0)
        N = target.size(0)

        dist_mat = pdist_torch(inputs, inputs)

        dist_intra_rgb = dist_mat[0: N, 0: N]
        dist_cross_rgb = dist_mat[0: N, N: 2 * N]
        dist_intra_ir = dist_mat[N: 2 * N, N: 2 * N]
        dist_cross_ir = dist_mat[N: 2 * N, 0: N]

        # shape [N, N]
        is_pos = target.expand(N, N).eq(target.expand(N, N).t())

        dist_intra_rgb = is_pos * dist_intra_rgb
        intra_rgb, _ = dist_intra_rgb.topk(self.num_pos - 1, dim=1, largest=True, sorted=False)  # remove itself
        intra_mean_rgb = torch.mean(intra_rgb, dim=1)

        dist_intra_ir = is_pos * dist_intra_ir
        intra_ir, _ = dist_intra_ir.topk(self.num_pos - 1, dim=1, largest=True, sorted=False)
        intra_mean_ir = torch.mean(intra_ir, dim=1)

        dist_cross_rgb = dist_cross_rgb[is_pos].contiguous().view(N, -1)  # [N, num_pos]
        cross_mean_rgb = torch.mean(dist_cross_rgb, dim=1)

        dist_cross_ir = dist_cross_ir[is_pos].contiguous().view(N, -1)  # [N, num_pos]
        cross_mean_ir = torch.mean(dist_cross_ir, dim=1)

        loss = (torch.mean(torch.pow(cross_mean_rgb - intra_mean_rgb, 2)) +
                torch.mean(torch.pow(cross_mean_ir - intra_mean_ir, 2))) / 2

        return loss


class MSEL_Cos(nn.Module):  # for features after bn
    def __init__(self, num_pos):
        super(MSEL_Cos, self).__init__()
        self.num_pos = num_pos

    def forward(self, inputs, targets):
        inputs = nn.functional.normalize(inputs, p=2, dim=1)

        target, _ = targets.chunk(2, 0)
        N = target.size(0)

        dist_mat = 1 - torch.matmul(inputs, torch.t(inputs))

        dist_intra_rgb = dist_mat[0: N, 0: N]
        dist_cross_rgb = dist_mat[0: N, N: 2 * N]
        dist_intra_ir = dist_mat[N: 2 * N, N: 2 * N]
        dist_cross_ir = dist_mat[N: 2 * N, 0: N]

        # shape [N, N]
        is_pos = target.expand(N, N).eq(target.expand(N, N).t())

        dist_intra_rgb = is_pos * dist_intra_rgb
        intra_rgb, _ = dist_intra_rgb.topk(self.num_pos - 1, dim=1, largest=True, sorted=False)  # remove itself
        intra_mean_rgb = torch.mean(intra_rgb, dim=1)

        dist_intra_ir = is_pos * dist_intra_ir
        intra_ir, _ = dist_intra_ir.topk(self.num_pos - 1, dim=1, largest=True, sorted=False)
        intra_mean_ir = torch.mean(intra_ir, dim=1)

        dist_cross_rgb = dist_cross_rgb[is_pos].contiguous().view(N, -1)  # [N, num_pos]
        cross_mean_rgb = torch.mean(dist_cross_rgb, dim=1)

        dist_cross_ir = dist_cross_ir[is_pos].contiguous().view(N, -1)  # [N, num_pos]
        cross_mean_ir = torch.mean(dist_cross_ir, dim=1)

        loss = (torch.mean(torch.pow(cross_mean_rgb - intra_mean_rgb, 2)) +
                torch.mean(torch.pow(cross_mean_ir - intra_mean_ir, 2))) / 2

        return loss


class MSEL_Feat(nn.Module):  # compute MSEL loss by the distance between sample and center
    def __init__(self, num_pos):
        super(MSEL_Feat, self).__init__()
        self.num_pos = num_pos

    def forward(self, input1, input2):
        N = input1.size(0)
        id_num = N // self.num_pos

        feats_rgb = input1.chunk(id_num, 0)
        feats_ir = input2.chunk(id_num, 0)

        loss_list = []
        for i in range(id_num):
            cross_center_rgb = torch.mean(feats_rgb[i], dim=0)  # cross center
            cross_center_ir = torch.mean(feats_ir[i], dim=0)

            for j in range(self.num_pos):
                feat_rgb = feats_rgb[i][j]
                feat_ir = feats_ir[i][j]

                intra_feats_rgb = torch.cat((feats_rgb[i][0:j], feats_rgb[i][j + 1:]), dim=0)  # intra center
                intra_feats_ir = torch.cat((feats_rgb[i][0:j], feats_rgb[i][j + 1:]), dim=0)

                intra_center_rgb = torch.mean(intra_feats_rgb, dim=0)
                intra_center_ir = torch.mean(intra_feats_ir, dim=0)

                dist_intra_rgb = pdist_torch(feat_rgb.view(1, -1), intra_center_rgb.view(1, -1))
                dist_intra_ir = pdist_torch(feat_ir.view(1, -1), intra_center_ir.view(1, -1))

                dist_cross_rgb = pdist_torch(feat_rgb.view(1, -1), cross_center_ir.view(1, -1))
                dist_cross_ir = pdist_torch(feat_ir.view(1, -1), cross_center_rgb.view(1, -1))

                loss_list.append(
                    torch.pow(dist_cross_rgb - dist_intra_rgb, 2) + torch.pow(dist_cross_ir - dist_intra_ir, 2))

        loss = sum(loss_list) / N / 2

        return loss


class SP(nn.Module):
    def __init__(self):
        super(SP, self).__init__()

    def forward(self, feat_v, feat_t):
        feat_v = feat_v.view(feat_v.size(0), -1)
        G_v = torch.mm(feat_v, feat_v.t())
        norm_G_v = F.normalize(G_v, p=2, dim=1)

        feat_t = feat_t.view(feat_t.size(0), -1)
        G_t = torch.mm(feat_t, feat_t.t())
        norm_G_t = F.normalize(G_t, p=2, dim=1)

        loss = F.mse_loss(norm_G_v, norm_G_t)

        return loss


class CMMD(nn.Module):
    def __init__(self, num_pos=4):
        super(CMMD, self).__init__()
        self.num_pos = num_pos

    def forward(self, feat_v, feat_t):
        feat_v = feat_v.view(feat_v.size(0), -1)
        feat_v = F.normalize(feat_v, dim=-1)
        feat_v_s = torch.split(feat_v, self.num_pos)

        feat_t = feat_t.view(feat_t.size(0), -1)
        feat_t = F.normalize(feat_t, dim=-1)
        feat_t_s = torch.split(feat_t, self.num_pos)

        losses = [self.mmd_loss(f_v, f_t) for f_v, f_t in zip(feat_v_s, feat_t_s)]
        loss = sum(losses) / len(losses)

        return loss

    def mmd_loss(self, f_v, f_t):
        return (self.poly_kernel(f_v, f_v).mean() + self.poly_kernel(f_t, f_t).mean()
                - 2 * self.poly_kernel(f_v, f_t).mean())

    def poly_kernel(self, a, b):
        a = a.unsqueeze(0)
        b = b.unsqueeze(1)
        res = (a * b).sum(-1).pow(2)
        return res


class AdaSPLoss(object):
    """
    Adaptive sparse pairwise (AdaSP) loss
    """

    def __init__(self, temp=0.04, loss_type='adasp'):
        self.temp = temp
        self.loss_type = loss_type

    def __call__(self, feats, targets):

        feats_n = nn.functional.normalize(feats, dim=1)

        bs_size = feats_n.size(0)
        N_id = len(torch.unique(targets))
        N_ins = bs_size // N_id

        scale = 1. / self.temp

        sim_qq = torch.matmul(feats_n, feats_n.T)
        sf_sim_qq = sim_qq * scale

        right_factor = torch.from_numpy(np.kron(np.eye(N_id), np.ones((N_ins, 1)))).cuda()
        pos_mask = torch.from_numpy(np.kron(np.eye(N_id), np.ones((N_ins, 1)))).cuda()
        left_factor = torch.from_numpy(np.kron(np.eye(N_id), np.ones((1, N_ins)))).cuda()

        ## hard-hard mining for pos
        mask_HH = torch.from_numpy(np.kron(np.eye(N_id), -1. * np.ones((N_ins, N_ins)))).cuda()
        mask_HH[mask_HH == 0] = 1.

        ID_sim_HH = torch.exp(sf_sim_qq.mul(mask_HH))
        ID_sim_HH = ID_sim_HH.mm(right_factor)
        ID_sim_HH = left_factor.mm(ID_sim_HH)

        pos_mask_id = torch.eye(N_id).cuda()
        pos_sim_HH = ID_sim_HH.mul(pos_mask_id)
        pos_sim_HH[pos_sim_HH == 0] = 1.
        pos_sim_HH = 1. / pos_sim_HH
        ID_sim_HH = ID_sim_HH.mul(1 - pos_mask_id) + pos_sim_HH.mul(pos_mask_id)

        ID_sim_HH_L1 = nn.functional.normalize(ID_sim_HH, p=1, dim=1)

        ## hard-easy mining for pos
        mask_HE = torch.from_numpy(np.kron(np.eye(N_id), -1. * np.ones((N_ins, N_ins)))).cuda()
        mask_HE[mask_HE == 0] = 1.

        ID_sim_HE = torch.exp(sf_sim_qq.mul(mask_HE))
        ID_sim_HE = ID_sim_HE.mm(right_factor)

        pos_sim_HE = ID_sim_HE.mul(pos_mask)
        pos_sim_HE[pos_sim_HE == 0] = 1.
        pos_sim_HE = 1. / pos_sim_HE
        ID_sim_HE = ID_sim_HE.mul(1 - pos_mask) + pos_sim_HE.mul(pos_mask)

        # hard-hard for neg
        ID_sim_HE = left_factor.mm(ID_sim_HE)

        ID_sim_HE_L1 = nn.functional.normalize(ID_sim_HE, p=1, dim=1)

        l_sim = torch.log(torch.diag(ID_sim_HH))
        s_sim = torch.log(torch.diag(ID_sim_HE))

        weight_sim_HH = torch.log(torch.diag(ID_sim_HH)).detach() / scale
        weight_sim_HE = torch.log(torch.diag(ID_sim_HE)).detach() / scale
        wt_l = 2 * weight_sim_HE.mul(weight_sim_HH) / (weight_sim_HH + weight_sim_HE)
        wt_l[weight_sim_HH < 0] = 0
        both_sim = l_sim.mul(wt_l) + s_sim.mul(1 - wt_l)

        adaptive_pos = torch.diag(torch.exp(both_sim))

        pos_mask_id = torch.eye(N_id).cuda()
        adaptive_sim_mat = adaptive_pos.mul(pos_mask_id) + ID_sim_HE.mul(1 - pos_mask_id)

        adaptive_sim_mat_L1 = nn.functional.normalize(adaptive_sim_mat, p=1, dim=1)

        loss_sph = -1 * torch.log(torch.diag(ID_sim_HH_L1)).mean()
        loss_splh = -1 * torch.log(torch.diag(ID_sim_HE_L1)).mean()
        loss_adasp = -1 * torch.log(torch.diag(adaptive_sim_mat_L1)).mean()

        if self.loss_type == 'sp-h':
            loss = loss_sph.mean()
        elif self.loss_type == 'sp-lh':
            loss = loss_splh.mean()
        elif self.loss_type == 'adasp':
            loss = loss_adasp

        return loss
