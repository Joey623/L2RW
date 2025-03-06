from __future__ import print_function
import argparse
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
from data_loader import *
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from utils import *
from model import embed_net
from loss import PairCircle
import logging
import math
from torch.cuda.amp import autocast, GradScaler
import os
parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
# each changes
parser.add_argument('--noticed', default='RegDB', type=str)
parser.add_argument('--protocol', type=str, default="R_S_to_L",
                    help='R_S_to_L, R_L_to_S, L_S_to_R, S_to_S, R_to_R, L_to_L')
# 0.00068
parser.add_argument('--lr', default=0.2, type=float, help='learning rate, 0.00035 for adam, 0.0007for adamw')
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
parser.add_argument('--batch_size', default=6, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--margin', default=0.45, type=float,
                    metavar='margin', help='triplet loss margin')

parser.add_argument('--trial', default=1, type=int, help='trial (only for RegDB dataset)')
parser.add_argument('--method', default='dns', type=str,
                    metavar='m', help='method type: base or agw')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--gpu', default='0,1,2,3', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--pool_dim', default=2048)
parser.add_argument('--decay_step', default=16)
parser.add_argument('--warm_up_epoch', default=8, type=int)
parser.add_argument('--max_epoch', default=30)
parser.add_argument('--rerank', default='no', type=str)
parser.add_argument('--dim', default=2048, type=int)
parser.add_argument('--tta', default=False, type=bool)
parser.add_argument('--lr_multipliers', default=[0.1, 0.2, 0.25, 1.0, 1.0, 1.0])

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
set_seed(args.seed)
protocol = args.protocol

model_path = args.model_path + protocol + '/'
if not os.path.isdir(model_path):
    os.makedirs(model_path)

data_name_list_train, data_name_test = protocol_decoder(args.protocol)
dataset_dir_mapping = {
        "SYSU": '../Dataset/SYSU-MM01/',
        "RegDB": '../Dataset/RegDB/',
        "LLCM": '../Dataset/LLCM/',
        }
if len(data_name_list_train)==1:
    print("trainset: {}".format(data_name_list_train[0]))
    print("testset: {}".format(data_name_test[0]))
elif len(data_name_list_train)==2:
    print("trainset: {}, {}".format(data_name_list_train[0], data_name_list_train[1]))
    print("testset: {}".format(data_name_test[0]))
elif len(data_name_list_train)==3:
    print("trainset: {}, {}, {}".format(data_name_list_train[0], data_name_list_train[1], data_name_list_train[2]))
    print("testset: {}".format(data_name_test[0]))
else:
    raise "Wrong! Check your protocol."

# get testing dataset dir
data_path = dataset_dir_mapping.get(data_name_test[0], '')

log_path = args.log_path + args.protocol + '/'

suffix = protocol
suffix = suffix + '_{}_p{}_n{}_lr_{}_seed_{}'.format(args.method, args.batch_size, args.num_pos, args.lr, args.seed)
if not os.path.isdir(log_path):
    os.makedirs(log_path)

# File_name = log_path + suffix + '.log'
# logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=File_name, filemode='a')
# console = logging.StreamHandler()
# console.setLevel(logging.info)
# formatter = logging.Formatter('%(message)s')
# console.setFormatter(formatter)
# logging.getLogger('').addHandler(console)
sys.stdout = Logger(log_path + suffix + '_os.txt')

print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize])
end = time.time()

test_mode = [1, 2]
trainset = get_datasets(args, MultiVIReIDDataset)
color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
if data_name_test[0] in ["SYSU"]:
    data_path = '../Dataset/SYSU-MM01/'
    print('==> Testing dataset: SYSU-MM01')
    # default mode: all (if you wanna testing indoor mode, just change it on test.py)
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif data_name_test[0] in ["RegDB"]:
    data_path = '../Dataset/RegDB/'
    print('==> Testing dataset: RegDB')
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='visible')

elif data_name_test[0] in ["LLCM"]:
    data_path = '../Dataset/LLCM/'
    print('==> Testing dataset: LLCM')
    query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=0)


gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('data init success!')

print('Protocol {} statistics:'.format(protocol))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
net = embed_net(class_num=n_class)
net.to(device)

print('==> Building success!..')
cudnn.benchmark = True

criterion_id = nn.CrossEntropyLoss()
# margin=0.45, gamma=64
criterion_cir = PairCircle(margin=0.45, gamma=64)


criterion_id.to(device)
criterion_cir.to(device)

ignored_params = list(map(id, net.bottleneck.parameters())) \
                 + list(map(id, net.classifier.parameters())) \
                 + list(map(id, net.DD.parameters())) \
                 + list(map(id, net.base.base.layer3.parameters())) \
                 + list(map(id, net.base.base.layer4.parameters()))

base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

optimizer = optim.SGD([{'params': base_params, 'lr': 0.1 * args.lr},
                       {'params': net.base.base.layer3.parameters(), 'lr': args.lr},
                       {'params': net.base.base.layer4.parameters(), 'lr': args.lr},
                       {'params': net.bottleneck.parameters(), 'lr': args.lr},
                       {'params': net.classifier.parameters(), 'lr': args.lr},
                       {'params': net.DD.parameters(), 'lr': args.lr},
                       ],
                      weight_decay=5e-4, momentum=0.9, nesterov=True)

warm_up_with_cosine_lr = lambda epoch: epoch / args.warm_up_epoch if epoch <= args.warm_up_epoch else \
    0.5 * (math.cos((epoch - args.warm_up_epoch) / (args.max_epoch - args.warm_up_epoch) * math.pi) + 1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

# lr_multipliers = [0.1, 0.2, 0.25, 1.0, 1.0, 1.0]


def adjust_lr(optimizer, batch_idx, lrs):
    for idx, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lrs[batch_idx - 1] * args.lr_multipliers[idx]

    return optimizer, optimizer.param_groups[-1]['lr'], optimizer.param_groups[0]['lr']


# one circle learning rate
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]


def train(epoch, optimizer, net, args):
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    cir_loss = AverageMeter()
    da_loss = AverageMeter()
    dd_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()
    lr_start = get_cyclic_lr(epoch, args.lr, args.max_epoch, args.decay_step)
    lr_end = get_cyclic_lr(epoch + 1, args.lr, args.max_epoch, args.decay_step)

    iters = len(trainloader)
    lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):
        optimizer, current_lr, base_lr = adjust_lr(optimizer, batch_idx, lrs)

        optimizer.zero_grad()
        labels = torch.cat((label1, label2), 0)

        label1 = torch.tensor(label1, dtype=torch.long).to(device)
        label2 = torch.tensor(label2, dtype=torch.long).to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        data_time.update(time.time() - end)
        with autocast():
            res = net(input1.to(device), input2.to(device), label1, label2, modal=0)
            B, C = res['feat'].shape
            # circle loss
            loss_cir = criterion_cir(res['feat'], labels) / B
            # identity loss
            loss_id = criterion_id(res['cls_id'], labels)

            loss_da = res['da']
            loss_dd = res['dd']
            loss = loss_id + loss_cir + 0.2 * loss_da + loss_dd
            _, predicted = res['cls_id'].max(1)
            correct += (predicted.eq(labels).sum().item())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        cir_loss.update(loss_cir.item(), 2 * input1.size(0))
        da_loss.update(loss_da.item(), 2 * input1.size(0))
        dd_loss.update(loss_dd.item(), 2 * input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 100 == 0:
            print('Epoch: [{}][{}/{}] '
                         'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                         'lr:{:.6f} '
                         'base_lr:{:.6f}'
                         'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                         'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                         'CLoss: {cir_loss.val:.4f} ({cir_loss.avg:.4f}) '
                         'da_loss: {da_loss.val:.4f} ({da_loss.avg:.4f}) '
                         'dd_loss: {dd_loss.val:.4f} ({dd_loss.avg:.4f}) '
                         'Accu: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr, base_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, cir_loss=cir_loss,
                da_loss=da_loss, dd_loss=dd_loss))

def test(epoch):
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
                feat, feat_att = net(input, input, None, None, test_mode[0])
                if args.tta:
                    feat_tta, feat_att_tta = net(torch.flip(input, dims=[3]), torch.flip(input, dims=[3]), test_mode[0])
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
                feat, feat_att = net(input, input, None, None, test_mode[1])
                if args.tta:
                    feat_tta, feat_att_tta = net(torch.flip(input, dims=[3]), torch.flip(input, dims=[3]), test_mode[0])
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
        if data_name_test[0] in ["RegDB"]:
            cmc, mAP, mINP = eval_regdb(distmat, query_label, gall_label)
            cmc_att, mAP_att, mINP_att = eval_regdb(distmat_att, query_label, gall_label)
        elif data_name_test[0] in ["SYSU"]:
            cmc, mAP, mINP = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
            cmc_att, mAP_att, mINP_att = eval_sysu(distmat_att, query_label, gall_label, query_cam, gall_cam)
        elif data_name_test[0] in ["LLCM"]:
            cmc, mAP, mINP = eval_llcm(distmat, query_label, gall_label, query_cam, gall_cam)
            cmc_att, mAP_att, mINP_att = eval_llcm(distmat_att, query_label, gall_label, query_cam, gall_cam)
        # print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att


best_epoch = 0
# training
print('==> Start Training...')

scaler = GradScaler()
for epoch in range(start_epoch, args.max_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, trainset.train_thermal_label, color_pos, thermal_pos,
                              args.num_pos, args.batch_size, epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, sampler=sampler, drop_last=True,
                                  num_workers=args.workers)

    #     optimizer.zero_grad()
    if epoch <= args.decay_step:
        scheduler.step()
    train(epoch, optimizer, net=net, args=args)

    if epoch == args.max_epoch-1:
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
        }
        torch.save(state, model_path + suffix + '.pth')