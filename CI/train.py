from __future__ import print_function
import argparse
import copy
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData, LLCMData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from utils import *
from model import embed_net
from loss import PairCircle, Prototype_loss
import logging
import math
from torch.cuda.amp import autocast, GradScaler
import os
import copy
from ChannelAug import ChannelAdap

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu')
parser.add_argument('--method', default='fl', help='dataset name: regdb or sysu')
parser.add_argument('--lr', default=0.2, type=float,
                    help='learning rate, 0.00035 for adam, 0.0009for adamw, 0.1 for sgd')
parser.add_argument('--optim', default='sgd', help='optimizer: sgd or adamw or adam')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=1000, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--num_pos', default=8, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--margin', default=0.5, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--gpu', default='0,1,2,3', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--pool_dim', default=2048)
parser.add_argument('--decay_step', default=16)
parser.add_argument('--warm_up_epoch', default=8, type=int)
parser.add_argument('--max_epoch', default=50)
parser.add_argument('--rerank', default='no', type=str)
parser.add_argument('--dim', default=2048, type=int)
parser.add_argument('--tta', default=False, type=bool)
parser.add_argument('--FL', default=True, type=bool)
parser.add_argument('--lr_multipliers', default=[0.1, 1.0, 1.0])
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
set_seed(args.seed)

dataset = args.dataset
model_path = args.model_path + dataset + '/'
if not os.path.isdir(model_path):
    os.makedirs(model_path)
if dataset == 'sysu':
    data_path = "../data/SYSU-MM01/"
    class_num = 395
    log_path = args.log_path + 'sysu_log/'
elif dataset == 'regdb':
    data_path = "../data/RegDB/"
    class_num = 206
    log_path = args.log_path + 'regdb_log/'
elif dataset == 'llcm':
    data_path = "../data/LLCM/"
    class_num = 713
    log_path = args.log_path + 'llcm_log/'

if not os.path.isdir(log_path):
    os.makedirs(log_path)

suffix = args.method
suffix = suffix + '_{}_p{}_n{}_lr{}_{}_seed_{}'.format(args.dataset, args.batch_size, args.num_pos, args.lr, args.optim,
                                                       args.seed)
if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')


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


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    ChannelExchange(p=0.5)])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize])
end = time.time()

if dataset == 'sysu':
    train_set = []
    net = []
    local_prototypes = []
    if args.FL:
        client_num = 6
        # client_num = 2
        for index in range(client_num):
            subtrain_set = SYSUData(data_path, args=args, client_id=index + 1, transform=transform_train)
            train_set.append(subtrain_set)
            n_class = len(np.unique(subtrain_set.train_label))
            print('Dataset {} statistics:'.format(dataset))
            print('  -----------------------------------------')
            print('  subset   | # ids | # images | # client')
            print('  -----------------------------------------')
            print('  images  | {:5d} | {:8d} |  {:5d}'.format(n_class, len(subtrain_set.train_label), index + 1))
            print('  -----------------------------------------')
            print('==> Building client{} model..'.format(index + 1))
            # sub_net = embed_net(class_num=n_class)
            sub_net = embed_net(class_num=395)
            sub_net.to(device)
            local_prototype = PrototypeMemory(class_num=395)
            local_prototype.to(device)
            print('==> Building success!..')
            net.append(sub_net)
            local_prototypes.append(local_prototype)

    else:
        subtrain_set = SYSUData(data_path, args=args, transform=transform_train)
        train_set.append(subtrain_set)
        n_class = len(np.unique(subtrain_set.train_label))
        print('Dataset {} statistics:'.format(dataset))
        print('  -----------------------------------------')
        print('  subset   | # ids | # images | client')
        print('  -----------------------------------------')
        print('  images  | {:5d} | {:8d}'.format(n_class, len(subtrain_set.train_label), 1))
        print('  -----------------------------------------')
        print('==> Building model..')
        sub_net = embed_net(class_num=n_class)
        sub_net.to(device)
        print('==> Building success!..')
        net.append(sub_net)

    query_img, query_label, query_cam = process_query_sysu(data_path, mode='all')
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode='all', trial=0)

elif dataset == 'regdb':
    train_set = []
    net = []
    local_prototypes = []
    if args.FL:
        client_num = 2
        for index in range(client_num):
            subtrain_set = RegDBData(data_path, args=args, client_id=index + 1, transform=transform_train)
            train_set.append(subtrain_set)
            n_class = len(np.unique(subtrain_set.train_label))
            print('Dataset {} statistics:'.format(dataset))
            print('  -----------------------------------------')
            print('  subset   | # ids | # images | # client')
            print('  -----------------------------------------')
            print('  images  | {:5d} | {:8d} |  {:5d}'.format(n_class, len(subtrain_set.train_label), index + 1))
            print('  -----------------------------------------')
            print('==> Building client{} model..'.format(index + 1))
            # sub_net = embed_net(class_num=n_class)
            sub_net = embed_net(class_num=206)
            sub_net.to(device)
            local_prototype = PrototypeMemory(class_num=206)
            local_prototype.to(device)
            print('==> Building success!..')
            net.append(sub_net)
            local_prototypes.append(local_prototype)

    else:
        subtrain_set = RegDBData(data_path, args=args, transform=transform_train)
        train_set.append(subtrain_set)
        n_class = len(np.unique(subtrain_set.train_label))
        print('Dataset {} statistics:'.format(dataset))
        print('  -----------------------------------------')
        print('  subset   | # ids | # images | client')
        print('  -----------------------------------------')
        print('  images  | {:5d} | {:8d}'.format(n_class, len(subtrain_set.train_label), 1))
        print('  -----------------------------------------')
        print('==> Building model..')
        sub_net = embed_net(class_num=n_class)
        sub_net.to(device)
        print('==> Building success!..')
        net.append(sub_net)
    # (thermal to visible)
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
elif dataset == 'llcm':
    train_set = []
    net = []
    local_prototypes = []
    if args.FL:
        # client_num = 3
        client_num = 9
        for index in range(client_num):
            subtrain_set = LLCMData(data_path, args=args, client_id=index + 1, transform=transform_train)
            train_set.append(subtrain_set)
            n_class = len(np.unique(subtrain_set.train_label))
            print('Dataset {} statistics:'.format(dataset))
            print('  ------------------------------------------')
            print('  subset   | # ids | # images | # client')
            print('  ------------------------------------------')
            print('  images  | {:5d} | {:8d} |  {:5d}'.format(n_class, len(subtrain_set.train_label), index + 1))
            print('  ------------------------------------------')
            print('==> Building client{} model..'.format(index + 1))
            # sub_net = embed_net(class_num=n_class)
            sub_net = embed_net(class_num=713)
            sub_net.to(device)
            local_prototype = PrototypeMemory(class_num=713)
            local_prototype.to(device)
            print('==> Building success!..')
            net.append(sub_net)
            local_prototypes.append(local_prototype)

    else:
        subtrain_set = LLCMData(data_path, args=args, transform=transform_train)
        train_set.append(subtrain_set)
        n_class = len(np.unique(subtrain_set.train_label))
        print('Dataset {} statistics:'.format(dataset))
        print('  ------------------------------')
        print('  subset   | # ids | # images | client')
        print('  ------------------------------')
        print('  images  | {:5d} | {:8d}'.format(n_class, len(subtrain_set.train_label), 1))
        print('  ------------------------------')
        print('==> Building model..')
        sub_net = embed_net(class_num=713)
        sub_net.to(device)
        print('==> Building success!..')
        net.append(sub_net)
    query_img, query_label, query_cam = process_query_llcm(data_path, mode=2)
    gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=1, trial=0)

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} validate statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building server model..')
server_net = embed_net(class_num=class_num)
server_net.to(device)
global_weight = server_net.state_dict()
# global_prototype = PrototypeMemory(class_num=class_num)
global_prototype = {i: torch.zeros(2048).to(device) for i in range(class_num)}
print('==> Building success!..')
cudnn.benchmark = True

criterion_id = nn.CrossEntropyLoss()
criterion_tri = PairCircle(margin=0.45, gamma=64)
criterion_pro = Prototype_loss(weight1=0.5, weight2=0.5)
criterion_id.to(device)
criterion_tri.to(device)
criterion_pro.to(device)


def get_optimizer(net, args):
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    if args.optim == 'sgd':
        optimizer = optim.SGD([{'params': base_params, 'lr': 0.1 * args.lr},
                               {'params': net.bottleneck.parameters(), 'lr': args.lr},
                               {'params': net.classifier.parameters(), 'lr': args.lr},
                               ],
                              weight_decay=5e-4, momentum=0.9, nesterov=True)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW([{'params': base_params, 'lr': 0.1 * args.lr},
                                 {'params': net.bottleneck.parameters(), 'lr': args.lr},
                                 {'params': net.classifier.parameters(), 'lr': args.lr},
                                 ],
                                weight_decay=0.01)
    elif args.optim == 'adam':
        optimizer = optim.Adam([{'params': base_params, 'lr': 0.1 * args.lr},
                                {'params': net.bottleneck.parameters(), 'lr': args.lr},
                                {'params': net.classifier.parameters(), 'lr': args.lr},
                                ],
                               weight_decay=5e-4)
    return optimizer


client_optimizer = []
for i in range(len(net)):
    sub_optim = get_optimizer(net[i], args)
    client_optimizer.append(sub_optim)


def adjust_lr(optimizer, batch_idx, lrs):
    for idx, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lrs[batch_idx - 1] * args.lr_multipliers[idx]

    return optimizer, optimizer.param_groups[-1]['lr'], optimizer.param_groups[0]['lr']


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return optimizer, lr, optimizer.param_groups[0]['lr']


# one circle learning rate
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]


def train(epoch, optimizer, net, args, global_weight, local_prototype, global_prototype):
    # optimizer, current_lr, base_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    pro_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0
    # optimizer, current_lr, base_lr = adjust_learning_rate(optimizer, epoch)

    # switch to train mode
    net.to(device)
    net.load_state_dict(global_weight, strict=False)
    net.train()
    end = time.time()
    lr_start = get_cyclic_lr(epoch, args.lr, args.max_epoch, args.decay_step)
    lr_end = get_cyclic_lr(epoch + 1, args.lr, args.max_epoch, args.decay_step)
    iters = len(trainloader)
    lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

    # Initialize local prototype
    local_prototype.reset_memory()

    for batch_idx, (input, label, cam) in enumerate(trainloader):
        optimizer, current_lr, base_lr = adjust_lr(optimizer, batch_idx, lrs)

        optimizer.zero_grad()

        label = torch.tensor(label, dtype=torch.long).to(device)
        data_time.update(time.time() - end)
        with autocast():
            res = net(input.to(device))
            B, C = res['feat'].shape
            local_prototype.update_memory(res['feat'], label)
            loss_id = criterion_id(res['cls_id'], label)
            loss_tri = criterion_tri(res['feat'], label) / B
            # loss = loss_id + loss_tri
            if epoch == 0:
                loss = loss_id + loss_tri
            else:
                loss_pro = criterion_pro(res['feat'], label, global_prototype, device)
                loss = loss_id + loss_tri + loss_pro
            _, predicted = res['cls_id'].max(1)
            correct += (predicted.eq(label).sum().item())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if epoch == 0:
            train_loss.update(loss.item(), input.size(0))
            id_loss.update(loss_id.item(), input.size(0))
            tri_loss.update(loss_tri.item(), input.size(0))
        else:
            train_loss.update(loss.item(), input.size(0))
            id_loss.update(loss_id.item(), input.size(0))
            tri_loss.update(loss_tri.item(), input.size(0))
            pro_loss.update(loss_pro.item(), input.size(0))

        total += label.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 20 == 0:
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


def test(net, epoch):
    # switch to evaluation mode
    with torch.no_grad():
        net.eval()
        # print('Extracting Gallery Feature...')
        start = time.time()
        ptr = 0
        gall_feat = np.zeros((ngall, args.dim))
        gall_feat_att = np.zeros((ngall, args.dim))
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(gall_loader):
                batch_num = input.size(0)
                input = Variable(input.to(device))
                feat, feat_att = net(input)
                if args.tta:
                    feat_tta, feat_att_tta = net(torch.flip(input, dims=[3]))
                    feat += feat_tta
                    feat_att += feat_att_tta
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
                ptr = ptr + batch_num
        # print('Extracting Time:\t {:.3f}'.format(time.time() - start))

        # switch to evaluation
        net.eval()
        # print('Extracting Query Feature...')
        start = time.time()
        ptr = 0
        query_feat = np.zeros((nquery, args.dim))
        query_feat_att = np.zeros((nquery, args.dim))
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(query_loader):
                batch_num = input.size(0)
                input = Variable(input.to(device))
                feat, feat_att = net(input)
                if args.tta:
                    feat_tta, feat_att_tta = net(torch.flip(input, dims=[3]))
                    feat += feat_tta
                    feat_att += feat_att_tta
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
                ptr = ptr + batch_num
        # print('Extracting Time:\t {:.3f}'.format(time.time() - start))

        start = time.time()

        distmat = -np.matmul(query_feat, np.transpose(gall_feat))
        distmat_att = -np.matmul(query_feat_att, np.transpose(gall_feat_att))

        # evaluation
        if dataset == 'regdb':
            cmc, mAP, mINP = eval_regdb(distmat, query_label, gall_label)
            cmc_att, mAP_att, mINP_att = eval_regdb(distmat_att, query_label, gall_label)
        elif dataset == 'sysu':
            cmc, mAP, mINP = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
            cmc_att, mAP_att, mINP_att = eval_sysu(distmat_att, query_label, gall_label, query_cam, gall_cam)
        elif dataset == 'llcm':
            cmc, mAP, mINP = eval_llcm(distmat, query_label, gall_label, query_cam, gall_cam)
            cmc_att, mAP_att, mINP_att = eval_llcm(distmat_att, query_label, gall_label, query_cam, gall_cam)
        # print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att


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


best_epoch = 0
# training
print('==> Start Training...')

scaler = GradScaler()
for epoch in range(start_epoch, args.max_epoch):
    weight_local = []
    identity_num = []
    representative_local_prototype = []
    # identity sampler
    for index in range(len(train_set)):
        color_pos = GenIdx(train_set[index].train_label)
        sampler = IdentitySampler(train_set[index].train_label, color_pos, args.num_pos, args.batch_size)
        train_set[index].Index = sampler.index
        loader_batch = args.batch_size * args.num_pos
        id_num = len(train_set[index])
        id_num = id_num - (id_num % (loader_batch))
        identity_num.append(id_num)
        trainloader = data.DataLoader(train_set[index], batch_size=loader_batch, sampler=sampler, drop_last=True,
                                      num_workers=args.workers)
        optimizer = client_optimizer[index]
        print("Client{} training".format(index + 1))
        w, rep_local_proto = train(epoch, client_optimizer[index], net=net[index], args=args,
                                   global_weight=global_weight, local_prototype=local_prototypes[index],
                                   global_prototype=global_prototype)
        weight_local.append(w)
        representative_local_prototype.append(rep_local_proto)

    if epoch >= 0:
        print('Start Communication...')
        global_weight = fed_avg(weight_local, identity_num)
        global_prototype = proto_avg(representative_local_prototype)
        server_net.load_state_dict(global_weight, strict=False)
        print('Success!')
        print('-' * 30)
    if epoch == args.max_epoch - 1:
        state = {
            'net': global_weight,
            'epoch': epoch,
        }
        torch.save(state, model_path + suffix + '.pth')