"""
@Author: Du Yunhao
@Filename: train.py
@Contact: dyh_bupt@163.com
@Time: 2022/8/31 21:42
@Discription: train
"""
import os
import time
from itertools import cycle
import copy
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
from utils import *
from opts import opt
from test import test
from loss import PairCircle, Prototype_loss
# from loss import get_loss
from dataloader import get_dataloader
from model import embed_net
from evaluation import evaluate, print_metrics
import random


os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
device = 'cuda' if torch.cuda.is_available() else 'cpu'
scaler = GradScaler()

save_configs(opt)
logger = get_logger(opt.save_dir)


class PrototypeMemory(nn.Module):
    def __init__(self, class_num, feature_dim=2048, top_k=5):
        super(PrototypeMemory, self).__init__()

        self.memory = {i: [] for i in range(class_num)}
        self.n_class = class_num
        self.feature_dim = feature_dim
        self.top_k = top_k

    def update_memory(self, features, labels):
        for i in range(features.size(0)):
            label = labels[i].item()
            feature = features[i].detach()
            self.memory[label].append(feature)

    def get_representative_feature(self, ):
        representative_prototypes = {i: torch.zeros(self.feature_dim).to(device) for i in range(self.n_class)}
        for label in sorted(self.memory.keys()):
            features = self.memory[label]
            if len(features) == 0:
                continue
            elif len(features) > 0 and len(features) < self.top_k:
                representative_prototypes[label] = torch.stack(features).mean(0)
            else:
                class_mean = torch.stack(features).mean(0)
                distances = torch.stack(
                    [F.pairwise_distance(class_mean.unsqueeze(0), f.unsqueeze(0)) for f in features])
                sorted_indices = torch.argsort(distances, dim=0)[:self.top_k]
                selected_features = torch.stack([features[idx] for idx in sorted_indices])
                representative_prototypes[label] = selected_features.mean(0)
        return representative_prototypes

    def reset_memory(self):
        self.memory = {i: [] for i in range(self.n_class)}



def get_optimizer(net, opt):
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([{'params': base_params, 'lr': 0.1 * opt.lr},
                           {'params': net.bottleneck.parameters(), 'lr': opt.lr},
                           {'params': net.classifier.parameters(), 'lr': opt.lr},
                           ],
                          weight_decay=5e-4, momentum=0.9, nesterov=True)
    return optimizer


dataloader_query, _ = get_dataloader(opt, 'query', client_id=0, show=False)
dataloader_gallery, _ = get_dataloader(opt, 'gallery', client_id=0, show=False)
class_num = 1074
# dataloader_train = get_dataloader(opt, 'train', True)
dataloader_train = []
net = []
local_prototypes = []
client_num = 6

# 6 clients as bupt dataset has 6 cameras
for trial in range(client_num):
    trainloader, _ = get_dataloader(opt, 'train', client_id=trial+1, show=True)
    dataloader_train.append(trainloader)
    sub_net = embed_net(class_num=class_num)
    sub_net.to(device)
    net.append(sub_net)
    local_prototype = PrototypeMemory(class_num=class_num)
    local_prototype.to(device)
    local_prototypes.append(local_prototype)


# the auxiliary is not use in our setting
# dataloader_auxiliary, _ = get_dataloader(opt, 'auxiliary', False)

# class_num=1074
server_net = embed_net(class_num=class_num)
server_net.to(device)
global_weight = server_net.state_dict()
# global_prototype = PrototypeMemory(class_num=class_num)
global_prototype = {i: torch.zeros(2048).to(device) for i in range(class_num)}
print('==> Building success!..')
# model = get_model(opt, class_num=class_num)

# optimizer = eval(f'optim.{opt.optimizer}')(
#     model.parameters(),
#     lr=opt.base_lr,
#     weight_decay=opt.weight_decay
# )

client_optimizer = []
for i in range(client_num):
    sub_optim = get_optimizer(net[i], opt)
    client_optimizer.append(sub_optim)

def adjust_lr(optimizer, batch_idx, lrs):
    for idx, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lrs[batch_idx - 1] * opt.lr_multipliers[idx]

    return optimizer, optimizer.param_groups[-1]['lr'], optimizer.param_groups[0]['lr']

def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]

criterion_id = nn.CrossEntropyLoss()
criterion_tri = PairCircle(margin=0.45, gamma=64)
criterion_pro = Prototype_loss(weight1=0.5, weight2=0.5)
criterion_id.to(device)
criterion_tri.to(device)
criterion_pro.to(device)


def train(epoch, optimizer, net, opt, global_weight, local_prototype, global_prototype, trainloader):
    if opt.auxiliary:
        raise 'Please check the auxiliary, we dont use it in our setting'
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    pro_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    net.to(device)
    net.load_state_dict(global_weight, strict=False)
    net.train()
    end = time.time()

    lr_start = get_cyclic_lr(epoch, opt.lr, opt.max_epoch, opt.decay_step)
    lr_end = get_cyclic_lr(epoch + 1, opt.lr, opt.max_epoch, opt.decay_step)
    iters = len(trainloader)
    lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

    # Initialize local prototype
    local_prototype.reset_memory()

    for batch_idx, (input1, input2, label, cam) in enumerate(trainloader):
        optimizer, current_lr, base_lr = adjust_lr(optimizer, batch_idx, lrs)
        

        optimizer.zero_grad()

        label = torch.tensor(label, dtype=torch.long).to(device)
        labels = torch.cat((label, label), dim=0)
        data_time.update(time.time() - end)
        with autocast():
            res = net(input1.to(device), input2.to(device))
            B, C = res['feat'].shape

            local_prototype.update_memory(res['feat'], labels)
            loss_id = criterion_id(res['cls_id'], labels)
            loss_tri = criterion_tri(res['feat'], labels) / B
            # loss = loss_id + loss_tri
            if epoch == 0:
                loss = loss_id + loss_tri
            else:
                loss_pro = criterion_pro(res['feat'], labels, global_prototype, device)
                loss = loss_id + loss_tri + loss_pro
            _, predicted = res['cls_id'].max(1)
            correct += (predicted.eq(labels).sum().item())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if epoch == 0:
            train_loss.update(loss.item(), 2 * input1.size(0))
            id_loss.update(loss_id.item(), 2 * input1.size(0))
            tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        else:
            train_loss.update(loss.item(), 2 * input1.size(0))
            id_loss.update(loss_id.item(), 2 * input1.size(0))
            tri_loss.update(loss_tri.item(), 2 * input1.size(0))
            pro_loss.update(loss_pro.item(), 2 * input1.size(0))

        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 10 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.6f} '
                  'base_lr:{:.6f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'CLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'PLoss: {pro_loss.val:.4f} ({pro_loss.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr, base_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss, pro_loss=pro_loss))
    representative_local_prototypes = local_prototype.get_representative_feature()
    return net.state_dict(), representative_local_prototypes

def fed_avg(weight_local, identity_num):
    w_avg = copy.deepcopy(weight_local[0])
    sample_num = sum(identity_num)
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * (identity_num[0] / sample_num)
    for k in w_avg.keys():
        for i in range(1, len(weight_local)):
            w_avg[k] = w_avg[k] + weight_local[i][k] * identity_num[i] / sample_num

    return w_avg


def proto_avg(local_prototypes):
    global_prototype = {i: torch.zeros(2048).to(device) for i in range(class_num)}
    # labels = local_prototypes[0].keys()
    labels = global_prototype.keys()
    for label in labels:
        global_proto = torch.zeros(2048).to(device)
        count = 0
        for local_pro in local_prototypes:
            if label in local_pro:
                global_proto += local_pro[label]
                count += 1

        global_prototype[label] = global_proto / count
    return global_prototype


for epoch in range(0, opt.max_epoch):
    weight_local = []
    identity_num = []
    representative_local_prototype = []

    for index in range(client_num):
        trainloader = dataloader_train[index]
        optimizer = client_optimizer[index]
        print("Client{} training".format(index + 1))
        w, rep_local_proto = train(epoch, client_optimizer[index], net=net[index], opt=opt,
                                   global_weight=global_weight, local_prototype=local_prototypes[index],
                                   global_prototype=global_prototype, trainloader=trainloader)
        id_num = len(trainloader)
        identity_num.append(id_num)
        weight_local.append(w)
        representative_local_prototype.append(rep_local_proto)
    torch.cuda.empty_cache()
    if epoch >= 0:
        print('Start Communication...')
        global_weight = fed_avg(weight_local, identity_num)
        global_prototype = proto_avg(representative_local_prototype)
        server_net.load_state_dict(global_weight, strict=False)
        print('Success!')
    if epoch == 0 or (epoch + 1) % opt.eval_freq == 0:
        CMC, MAP, MINP = test(server_net, dataloader_query, dataloader_gallery, show=True, return_all=True)
        MODE = ['RGB->RGB', 'RGB->IR ', 'IR->RGB ', 'IR->IR  ', 'AllModal']
        print('Epoch:[{}/{}]'.format(epoch, opt.max_epoch))
        for i, mode in enumerate(MODE):
            print('\n{}: Rank1:{:.2f}% Rank5:{:.2f}% Rank10:{:.2f}% Rank20:{:.2f}% mAP:{:.2f}% mINP:{:.2f}%' \
                .format(mode, CMC[i][0], CMC[i][4], CMC[i][9], CMC[i][19], MAP[i], MINP[i]))


    if epoch == opt.max_epoch-1:
        state = {
            'net': global_weight,
            'epoch': epoch,
        }
        model_path = './save_models/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(state, model_path + 'bupt_pure.pth')








