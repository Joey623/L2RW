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


class visible_module(nn.Module):
    def __init__(self, pretrained=True):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)

        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)

        return x


class thermal_module(nn.Module):
    def __init__(self, pretrained=True):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)

        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)

        return x


class base_module(nn.Module):
    def __init__(self, pretrained=True):
        super(base_module, self).__init__()

        base = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.base = base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        return x


class embed_net(nn.Module):
    def __init__(self, class_num, pool_dim=2048, pretrained=True):
        super(embed_net, self).__init__()

        self.visible = visible_module(pretrained=pretrained)
        self.thermal = thermal_module(pretrained=pretrained)
        self.base = base_module(pretrained=pretrained)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.relu = nn.ReLU()
        self.pool = GeMP()

    @autocast()
    def forward(self, x1, x2, modal=0):

        if modal == 0:
            x1 = self.visible(x1)
            x2 = self.thermal(x2)
            x = torch.cat((x1, x2), dim=0)
            del x1, x2

        elif modal == 1:
            x = self.visible(x1)
            del x1, x2

        elif modal == 2:
            x = self.thermal(x2)
            del x1, x2

        if self.training:

            feat = self.base(x)
            b, c, h, w = feat.shape
            feat = self.relu(feat)
            feat = feat.view(b, c, h * w)
            feat = self.pool(feat)
            feat_after_BN = self.bottleneck(feat)

            cls_id = self.classifier(feat_after_BN)

            return {
                'cls_id': cls_id,
                'feat': feat,
            }
        else:
            feat = self.base(x)
            b, c, h, w = feat.shape
            feat = self.relu(feat)
            feat = feat.view(b, c, h * w)
            feat = self.pool(feat)
            feat_after_BN = self.bottleneck(feat)

            return F.normalize(feat, p=2.0, dim=1), F.normalize(feat_after_BN, p=2.0, dim=1)