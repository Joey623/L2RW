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



class temporal_module(nn.Module):
    def __init__(self, feat_dim=2048):
        super(temporal_module, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        """
        :param x: shape [b,t,c]
        :return: shape [b,c]
        """
        b, t, c = x.size()

        x = x.permute(0, 2, 1)
        x = self.gap(x)
        x = x.view(b, -1)
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
        self.temporal_module = temporal_module(feat_dim=pool_dim)
        # self.lstm = nn.LSTM(pool_dim, pool_dim, 2)
        # self.temporal_feat_learning = temporal_feat_learning()

    @autocast()
    def forward(self, x, seq_len=6):
        # [b,tc,h,w]
        b, c, h, w = x.shape
        t = seq_len
        # [bt,c,h,w]
        x = x.view(int(b*t), int(c/t), h, w)
        feat = self.base(x)
        b, c, h, w = feat.shape
        feat = self.relu(feat)
        feat = feat.view(b, c, h * w)
        # [bt,c]
        feat = self.pool(feat)
        # [b,t,c]
        feat = feat.view(feat.size(0)//t, t, -1)
        # [b,c]
        feat = self.temporal_module(feat)
        feat_after_BN = self.bottleneck(feat)

        if self.training:
            cls_id = self.classifier(feat_after_BN)

            return {
                'cls_id':cls_id,
                'feat':feat_after_BN,
            }
        else:
            return F.normalize(feat, p=2.0, dim=1), F.normalize(feat_after_BN, p=2.0, dim=1)
