import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from data_loader import VideoDataset_train, VideoDataset_test
from data_manager import VCM
from data_manager import *
from eval_metrics import eval_vcm
from utils import *
from model import embed_net
from loss import PairCircle, Prototype_loss
import logging
import math
from torch.cuda.amp import autocast, GradScaler
import os
import copy
import argparse
import time

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='vcm', help='dataset name: regdb or sysu')
# parser.add_argument('--method', default='fl', help='dataset name: regdb or sysu')
parser.add_argument('--lr', default=0.2, type=float,
                    help='learning rate, 0.00035 for adam, 0.0009for adamw, 0.1 for sgd')
parser.add_argument('--seq_lenth', default=6, type=int)
parser.add_argument('--optim', default='sgd', help='optimizer: sgd or adamw or adam')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=1000, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='vcm_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--test-batch', default=32, type=int,
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
parser.add_argument('--max_epoch', default=50, type=int)
parser.add_argument('--rerank', default='no', type=str)
parser.add_argument('--dim', default=2048, type=int)
parser.add_argument('--tta', default=False, type=bool)
parser.add_argument('--FL', default=True, type=bool)
parser.add_argument('--lr_multipliers', default=[0.1, 1.0, 1.0])
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
set_seed(args.seed)

seq_lenth = args.seq_lenth
dataset = args.dataset

model_path = args.model_path + dataset + '/'
if not os.path.isdir(model_path):
    os.makedirs(model_path)
if dataset == 'vcm':
    data_path = "../../../dataset/VCM-HITSZ/"
    class_num = 500
    log_path = args.log_path
if not os.path.isdir(log_path):
    os.makedirs(log_path)
suffix = args.dataset
# vcm_p8_n4_lr0.2_sgd_s6_seed_0.pth
suffix = suffix + '_baseline_p{}_n{}_lr{}_{}_s{}_seed_{}'.format(args.batch_size, args.num_pos, args.lr, args.optim,
                                                        args.seq_lenth, args.seed)
sys.stdout = Logger(log_path + suffix + '_os.txt')
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')



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
    transforms.Resize((args.img_h, args.img_w)),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    ChannelExchange(p=0.5)
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize])

if dataset == 'vcm':
    train_set = []
    net = []
    client_num = 6
    # just for extract the testset, all the client setting has the same query and gallery.
    test_set = VCM(client_id=1)
    for index in range(1, client_num + 1):
        subtrain_set = VCM(client_id=index)
        train_set.append(subtrain_set)
        sub_net = embed_net(class_num=500)
        sub_net.to(device)
        net.append(sub_net)

# only infrared to visible
queryloader = data.DataLoader(
    VideoDataset_test(test_set.query, seq_len=seq_lenth, sample='video_test', transform=transform_test),
    batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
galleryloader = data.DataLoader(
    VideoDataset_test(test_set.gallery, seq_len=seq_lenth, sample='video_test', transform=transform_test),
    batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = test_set.num_train_pids
nquery = test_set.num_query_tracklets
ngall = test_set.num_gallery_tracklets

print('==> Building server model..')
server_net = embed_net(class_num=500)
server_net.to(device)
global_weight = server_net.state_dict()

print('==> Building success!..')
cudnn.benchmark = True

criterion_id = nn.CrossEntropyLoss()
criterion_tri = PairCircle(margin=0.45, gamma=64)
criterion_id.to(device)
criterion_tri.to(device)


def get_optimizer(net, args):
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
    optimizer = optim.SGD([{'params': base_params, 'lr': 0.1 * args.lr},
                           {'params': net.bottleneck.parameters(), 'lr': args.lr},
                           {'params': net.classifier.parameters(), 'lr': args.lr},
                           ],
                          weight_decay=5e-4, momentum=0.9, nesterov=True)
    return optimizer


client_optimizer = []
for i in range(len(net)):
    sub_optim = get_optimizer(net[i], args)
    client_optimizer.append(sub_optim)


def adjust_lr(optimizer, batch_idx, lrs):
    for idx, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lrs[batch_idx - 1] * args.lr_multipliers[idx]

    return optimizer, optimizer.param_groups[-1]['lr'], optimizer.param_groups[0]['lr']


def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]


def train(epoch, optimizer, net, args, global_weight):
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    net.to(device)
    net.load_state_dict(global_weight, strict=False)
    net.train()
    end = time.time()
    lr_start = get_cyclic_lr(epoch, args.lr, args.max_epoch, args.decay_step)
    lr_end = get_cyclic_lr(epoch + 1, args.lr, args.max_epoch, args.decay_step)
    iters = len(trainloader)
    lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])


    for batch_idx, (input, label, cam) in enumerate(trainloader):
        optimizer, current_lr, base_lr = adjust_lr(optimizer, batch_idx, lrs)
        optimizer.zero_grad()
        label = torch.tensor(label, dtype=torch.long).to(device)
        data_time.update(time.time() - end)

        with autocast():
            res = net(input.to(device), seq_len=seq_lenth)
            B, C = res['feat'].shape
            loss_id = criterion_id(res['cls_id'], label)
            loss_tri = criterion_tri(res['feat'], label) / B
            loss = loss_id + loss_tri
            _, predicted = res['cls_id'].max(1)
            correct += (predicted.eq(label).sum().item())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss.update(loss.item(), input.size(0))
        id_loss.update(loss_id.item(), input.size(0))
        tri_loss.update(loss_tri.item(), input.size(0))

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
                  'Accu: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr, base_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss))
    return net.state_dict()


def test(net):
    # infrared to visible
    with torch.no_grad():
        net.eval()
        print('Extracting Gallery Feature...')
        start = time.time()
        ptr = 0
        gall_feat = np.zeros((ngall, args.dim))
        gall_feat_att = np.zeros((ngall, args.dim))
        q_pids, q_camids = [], []
        g_pids, g_camids = [], []
        with torch.no_grad():
            for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
                input = imgs
                label = pids
                batch_num = input.size(0)
                input = Variable(input.to(device))
                feat, feat_att = net(input, seq_len=seq_lenth)
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
                ptr = ptr + batch_num
                g_pids.extend(pids)
                g_camids.extend(camids)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        print("Extracting Time: {}".format(time.time() - start))

        net.eval()
        print('Extracting Query Feature...')
        start = time.time()
        ptr = 0
        query_feat = np.zeros((nquery, args.dim))
        query_feat_att = np.zeros((nquery, args.dim))
        with torch.no_grad():
            for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
                input = imgs
                label = pids
                batch_num = input.size(0)
                input = Variable(input.to(device))
                feat, feat_att = net(input, seq_len=seq_lenth)
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
                ptr = ptr + batch_num
                q_pids.extend(pids)
                q_camids.extend(camids)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        print("Extracting Time: {}".format(time.time() - start))

        start = time.time()
        distmat = -np.matmul(query_feat, gall_feat.T)
        distmat_att = -np.matmul(query_feat_att, np.transpose(gall_feat_att))
        print("Calculated Time: {}".format(time.time() - start))
        cmc, mAP, mINP = eval_vcm(distmat, q_pids, g_pids, q_camids, g_camids)
        cmc_att, mAP_att, mINP_att = eval_vcm(distmat_att, q_pids, g_pids, q_camids, g_camids)
        return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att


# def test_1(net, epoch):
#     with torch.no_grad():
#         net.eval()
#         print('Extracting Gallery Feature...')
#         start = time.time()
#         ptr = 0
#         gall_feat = np.zeros((ngall_1, args.dim))
#         q_pids, q_camids = [], []
#         g_pids, g_camids = [], []
#         with torch.no_grad():
#             for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
#                 input = imgs
#                 label = pids
#                 batch_num = input.size(0)
#                 input = Variable(input.to(device))
#                 feat = net(input, seq_len=seq_lenth)
#                 gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
#                 ptr = ptr + batch_num
#                 g_pids.extend(pids)
#                 g_camids.extend(camids)
#         g_pids = np.asarray(g_pids)
#         g_camids = np.asarray(g_camids)
#         print("Extracting Time: {}".format(time.time() - start))
#
#         net.eval()
#         print('Extracting Query Feature...')
#         start = time.time()
#         ptr = 0
#         query_feat = np.zeros((nquery_1, args.dim))
#         with torch.no_grad():
#             for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
#                 input = imgs
#                 label = pids
#                 batch_num = input.size(0)
#                 input = Variable(input.to(device))
#                 feat = net(input, seq_len=seq_lenth)
#                 query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
#                 ptr = ptr + batch_num
#                 q_pids.extend(pids)
#                 q_camids.extend(camids)
#         q_pids = np.asarray(q_pids)
#         q_camids = np.asarray(q_camids)
#         print("Extracting Time: {}".format(time.time() - start))
#
#         start = time.time()
#         distmat = -np.matmul(query_feat, gall_feat.T)
#         print("Calculated Time: {}".format(time.time() - start))
#         cmc, mAP, mINP = eval_vcm(distmat, q_pids, g_pids, q_camids, g_camids)
#         return cmc, mAP, mINP

def fed_avg(weight_local, identity_num):
    w_avg = copy.deepcopy(weight_local[0])
    sample_num = sum(identity_num)
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * (identity_num[0] / sample_num)
    for k in w_avg.keys():
        for i in range(1, len(weight_local)):
            w_avg[k] = w_avg[k] + weight_local[i][k] * identity_num[i] / sample_num

    return w_avg




best_epoch = 0
# training
print('==> Start Training...')

scaler = GradScaler()
for epoch in range(start_epoch, args.max_epoch):
    weight_local = []
    identity_num = []
    for index in range(len(train_set)):
        color_pos = GenIdx(train_set[index].label)
        sampler = IdentitySampler(train_set[index].label, color_pos, args.num_pos, args.batch_size)
        Index = sampler.index
        loader_batch = args.batch_size * args.num_pos
        id_num = len(train_set[index])
        id_num = id_num - (id_num % (loader_batch))
        identity_num.append(id_num)
        trainloader = data.DataLoader(
            VideoDataset_train(train_set[index].train, seq_len=seq_lenth, sample='video_train',
                               transform=transform_train, Index=Index),
            sampler=sampler, batch_size=loader_batch, num_workers=args.workers, drop_last=True)
        optimizer = client_optimizer[index]
        print("Client{} training".format(index + 1))
        w = train(epoch, client_optimizer[index], net=net[index], args=args,
                                   global_weight=global_weight)
        weight_local.append(w)

    if epoch >= 0:
        print('Start Communication...')
        global_weight = fed_avg(weight_local, identity_num)
        server_net.load_state_dict(global_weight, strict=False)
        print('Success!')
        print('-' * 30)

    if epoch == args.max_epoch - 1:
        print('==> Testing...')
        cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(server_net)
        print('VCM Infrared to Visible')
        print(
            'original feature:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'feature after BN:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        state = {
            'net': global_weight,
            'epoch': epoch,
        }
        torch.save(state, model_path + suffix + '_.pth')














