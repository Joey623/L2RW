import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable
import pdb
import torch.nn.functional as F




class PairCircle(nn.Module):
    '''
    a circle loss implement, simple and crude
    '''

    def __init__(self, margin: float, gamma: float):
        super(PairCircle, self).__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, embedding, targets):
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
        loss = self.weight1 * edu_dist.mean() + self.weight2 * cos_dist.mean()

        return loss