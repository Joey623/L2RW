import math
import torch
import torch.nn as nn
from torch.nn import init, Softmax
import torch.nn.functional as F
from loss import MMDLoss
from utils import weights_init_classifier, weights_init_kaiming
from resnet import resnet50
from torch.cuda.amp import autocast



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class Shift(nn.Module):
    def __init__(self, mode='max'):
        super(Shift, self).__init__()

        self.mode = mode
        if self.mode == 'max':
            self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
            self.pool_w = nn.AdaptiveMaxPool2d((1, None))

        elif self.mode == 'avg':
            self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
            self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        else:
            raise ValueError("Invalid mode! mode must be 'max' or 'avg'.")

    def forward(self, x):
        bias_h = self.pool_h(x)
        bias_h = F.softmax(bias_h, dim=2)
        bias_w = self.pool_w(x)
        bias_w = F.softmax(bias_w, dim=3)
        bias = torch.matmul(bias_h, bias_w)

        return bias


# Heterogeneous Space Shifting
class HSS(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(HSS, self).__init__()
        self.shift = Shift(mode='max')
        self.pool_ah = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_aw = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(inp, mip, kernel_size=1, bias=False),
            nn.BatchNorm2d(mip),
            h_swish(),
        )
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()
        bias = self.shift(x)
        x = x + bias
        a_h = self.pool_ah(x)
        a_w = self.pool_aw(x).permute(0, 1, 3, 2)
        x = torch.cat((a_h, a_w), dim=2)
        x = self.channel_attention(x)
        x_h, x_w = torch.split(x, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        x_h = self.conv_h(x_h).sigmoid()
        x_w = self.conv_w(x_w).sigmoid()
        out = identity * x_w * x_h

        return out


# Common Space Shifting (CSS)
# contribution2
class CSS(nn.Module):
    def __init__(self, inp=2048, oup=2048, reduction=4):
        super(CSS, self).__init__()

        self.inp = inp
        self.oup = oup
        self.r = reduction
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.inp),
            nn.Linear(self.inp, self.inp * self.r),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(self.inp * self.r, self.oup),
            nn.Dropout(0.05)
        )

    def forward(self, x):
        bias = F.softmax(x, dim=1)
        x = x + bias
        x = self.mlp(x)

        return x


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
        self.visible.layer3 = None
        self.visible.layer4 = None

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        x = self.visible.layer1(x)
        x = self.visible.layer2(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, pretrained=True):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)

        self.thermal = model_t
        self.thermal.layer3 = None
        self.thermal.layer4 = None

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x = self.thermal.layer1(x)
        x = self.thermal.layer2(x)
        return x


class base_module(nn.Module):
    def __init__(self, pretrained=True):
        super(base_module, self).__init__()
        base = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)

        self.base = base
        self.base.conv1 = None
        self.base.bn1 = None
        self.base.relu = None
        self.base.maxpool = None
        self.base.layer1 = None
        self.base.layer2 = None
        self.HSS3 = HSS(1024, 1024)
        self.HSS4 = HSS(2048, 2048)

    def forward(self, x):
        x = self.base.layer3(x)
        x = self.HSS3(x)
        x = self.base.layer4(x)
        x = self.HSS4(x)

        return x

# Domain Alignment
class DA(nn.Module):
    def __init__(self, ):
        super(DA, self).__init__()
        self.MMD = MMDLoss()

    def forward(self, x, x_hat):
        b, c = x.shape
        inter = self.MMD(F.normalize(x[:b // 2], p=2, dim=1), F.normalize(x[b // 2:], p=2, dim=1)) \
               + self.MMD(F.normalize(x_hat[:b // 2], p=2, dim=1), F.normalize(x_hat[b // 2:], p=2, dim=1))
        intra = self.MMD(F.normalize(x[:b // 2], p=2, dim=1), F.normalize(x_hat[:b // 2], p=2, dim=1)) \
               + self.MMD(F.normalize(x[b // 2:], p=2, dim=1), F.normalize(x_hat[b // 2:], p=2, dim=1))
        # alpha_1=0.45, alpha_2=0.05
        # L_da = alpha_1 * inter + alpha_2 * intra
        da_loss = 0.45 * inter + 0.05 * intra
        return da_loss

# Domain Distillation
class DD(nn.Module):
    def __init__(self, class_num, pool_dim=2048, tau=0.0):
        super(DD, self).__init__()
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        self.ID = nn.CrossEntropyLoss()
        self.tau = tau
        # C
        self.visible_classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.infrared_classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.visible_classifier_ = nn.Linear(pool_dim, class_num, bias=False)
        self.visible_classifier_.weight.requires_grad_(False)
        # initial
        # W^t= C^t
        self.visible_classifier_.weight.data = self.visible_classifier.weight.data
        self.infrared_classifier_ = nn.Linear(pool_dim, class_num, bias=False)
        self.infrared_classifier_.weight.requires_grad_(False)
        # initial
        self.infrared_classifier_.weight.data = self.infrared_classifier.weight.data

    def forward(self, x, x_hat, label1, label2):
        b, c = x.shape
        x_v = self.visible_classifier(x[:b // 2])
        x_t = self.infrared_classifier(x[b // 2:])
        # Z^t
        # to capture shared cues
        logit_x_id = [x_v, x_t]
        logit_x = torch.cat((x_v, x_t), 0).float()

        # to calculate dd2, omit in paper
        x_hat_v = self.visible_classifier(x_hat[:b // 2])
        x_hat_t = self.infrared_classifier(x_hat[b // 2:])
        # Z_hat^t
        logit_x_hat_id = [x_hat_v, x_hat_t]
        logit_x_hat = torch.cat((x_hat_v, x_hat_t), 0).float()

        with torch.no_grad():
            # update the W
            self.infrared_classifier_.weight.data = self.infrared_classifier_.weight.data * (1 - self.tau) \
                                                    + self.infrared_classifier.weight.data * self.tau
            self.visible_classifier_.weight.data = self.visible_classifier_.weight.data * (1 - self.tau) \
                                                   + self.visible_classifier.weight.data * self.tau

            # update A, eq 14
            logit_a_v = self.infrared_classifier_(x[:b // 2])
            logit_a_t = self.visible_classifier_(x[b // 2:])
            # in supplementary material
            logit_a_hat_v = self.infrared_classifier_(x_hat[:b // 2])
            logit_a_hat_t = self.visible_classifier_(x_hat[b // 2:])

            logit_a = torch.cat((logit_a_v, logit_a_t), 0).float()
            logit_a_hat = torch.cat((logit_a_hat_v, logit_a_hat_t), 0).float()
        # softmax
        logit_x = F.softmax(logit_x, 1)
        logit_x_hat = F.softmax(logit_x_hat, 1)
        logit_a = F.softmax(logit_a, 1)
        logit_a_hat = F.softmax(logit_a_hat, 1)

        # mitigate the cross-modality gap and capture the shared patterns
        dd1 = self.KLDivLoss(logit_a.log(), logit_x) + \
              0.25 * (self.ID(logit_x_id[0], label1) + self.ID(logit_x_id[1], label2))
        dd2 = self.KLDivLoss(logit_a_hat.log(), logit_x_hat) + \
              0.25 * (self.ID(logit_x_hat_id[0], label1) + self.ID(logit_x_hat_id[1], label2))
        
        # L_dd = kl_loss + shared_cues_loss
        dd_loss = dd1 + dd2

        return dd_loss





class embed_net(nn.Module):
    def __init__(self, class_num, pool_dim=2048, pretrained=True):
        super(embed_net, self).__init__()

        self.visible = visible_module(pretrained=pretrained)
        self.thermal = thermal_module(pretrained=pretrained)
        self.base = base_module(pretrained=pretrained)
        self.CSS = CSS()
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.relu = nn.ReLU()
        self.pool = GeMP()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.DA = DA()
        self.DD = DD(class_num=class_num, pool_dim=pool_dim)

    @autocast()
    def forward(self, x_v, x_t, label1=None, label2=None, modal=0):
        if modal == 0:
            x_v = self.visible(x_v)
            x_t = self.thermal(x_t)
            x = torch.cat((x_v, x_t), 0)
            del x_v, x_t

        elif modal == 1:
            x = self.visible(x_v)

        elif modal == 2:
            x = self.thermal(x_t)

        if self.training:
            x = self.base(x)
            b, c, h, w = x.shape
            x_hat = self.gap(x)
            x_hat = x_hat.view(b, -1)
            # Z_hat in papers
            x_hat = self.CSS(x_hat)

            # all value must > 0 due to the GeMP
            x = self.relu(x)
            x = x.view(b, c, h * w)
            # Z in papers
            x = self.pool(x)

            # calculate the da loss
            da_loss = self.DA(x, x_hat)

            x_after_BN = self.bottleneck(x)
            # calculate the id loss
            cls_id = self.classifier(x_after_BN)
            dd_loss = self.DD(x_after_BN, x_hat, label1, label2)

            return {
                'cls_id': cls_id,  # calculate id loss
                'feat': x_after_BN,  # calculate circle loss
                'da': da_loss,
                'dd': dd_loss,
            }

        else:
            x = self.base(x)
            b, c, h, w = x.shape

            x = self.relu(x)
            x = x.view(b, c, h * w)
            x = self.pool(x)
            x_after_BN = self.bottleneck(x)

        return F.normalize(x, p=2.0, dim=1), F.normalize(x_after_BN, p=2.0, dim=1)




