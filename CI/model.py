import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import weights_init_classifier, weights_init_kaiming
from torch.cuda.amp import autocast
from resnet import resnet50


class GeMP(nn.Module):
    def __init__(self, p=3.0, eps=1e-12):
        super(GeMP, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        p, eps = self.p, self.eps
        if x.ndim != 2:
            batch_size, fdim = x.shape[:2]
            x = x.view(batch_size, fdim, -1)
        return (torch.mean(x ** p, dim=-1) + eps) ** (1 / p)


class base_module(nn.Module):
    def __init__(self, pretrained=True):
        super(base_module, self).__init__()

        base = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.base = base

    def forward(self, x):
        x = self.base(x)
        return x


class embed_net(nn.Module):
    def __init__(self, class_num, pool_dim=2048, pretrained=True):
        super(embed_net, self).__init__()
        self.base = base_module(pretrained=pretrained)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.relu = nn.ReLU()
        self.pool = GeMP()

    @autocast()
    def forward(self, x):

        feat = self.base(x)
        if self.training:
            b, c, h, w = feat.shape
            feat = self.relu(feat)
            feat = feat.view(b, c, h * w)
            feat = self.pool(feat)
            feat_after_BN = self.bottleneck(feat)

            cls_id = self.classifier(feat_after_BN)

            return {
                'cls_id': cls_id,
                'feat': feat_after_BN,
            }
        else:
            b, c, h, w = feat.shape
            feat = self.relu(feat)
            feat = feat.view(b, c, h * w)
            feat = self.pool(feat)
            feat_after_BN = self.bottleneck(feat)

            return F.normalize(feat, p=2.0, dim=1), F.normalize(feat_after_BN, p=2.0, dim=1)