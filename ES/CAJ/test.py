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
import logging
import math
from torch.cuda.amp import autocast, GradScaler
import os

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
# each changes
parser.add_argument('--protocol', type=str, default="R_L_to_S",
                    help='R_S_to_L, R_L_to_S, L_S_to_R, S_to_S, R_to_R, L_to_L')
parser.add_argument('--method', type=str, default="baseline",
                    help='baseline or xxx')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam, 0.0007for adamw')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=1000, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=0, type=int, metavar='N',
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
parser.add_argument('--margin', default=0.5, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--gamma', default=64, type=int,
                    metavar='gamma', help='triplet loss gamma')
parser.add_argument('--trial', default=1, type=int, help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--gpu', default='0,1,2,3', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--pool_dim', default=2048)
parser.add_argument('--max_epoch', default=100)
parser.add_argument('--dim', default=2048, type=int)
parser.add_argument('--resume', '-r', default='R_L_to_S_baseline_CA_p8_n4_lr_0.1_seed_0_best.pth', type=str,
                    help='resume from checkpoint')
parser.add_argument('--tvsearch', default=False, help='True: visible to infrared False: infrared to visible')  # RegDB
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
set_seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def extract_gall_feat(gallery_loader):
    with torch.no_grad():
        model.eval()
        # print('Extracting gallery features...')
        start_time = time.time()
        ptr = 0
        gallery_feats = np.zeros((ngall, args.dim))
        gallery_global_feats = np.zeros((ngall, args.dim))
        with torch.no_grad():
            for idx, (img, _) in enumerate(gallery_loader):
                img = Variable(img.to(device))
                global_feat, feat = model(img, img, modal=test_mode[0])
                batch_num = img.size(0)
                gallery_feats[ptr:ptr + batch_num, :] = feat.cpu().numpy()
                gallery_global_feats[ptr:ptr + batch_num, :] = global_feat.cpu().numpy()
                ptr = ptr + batch_num
        duration = time.time() - start_time
    # print('Extracting time: {}s'.format(int(round(duration))))
    return gallery_global_feats, gallery_feats


def extract_query_feat(query_loader):
    with torch.no_grad():
        model.eval()
        # print('Extracting query features...')
        start_time = time.time()
        ptr = 0
        query_feats = np.zeros((nquery, args.dim))
        query_global_feats = np.zeros((nquery, args.dim))
        with torch.no_grad():
            for idx, (img, _) in enumerate(query_loader):
                img = Variable(img.to(device))
                global_feat, feat = model(img, img, modal=test_mode[1])
                batch_num = img.size(0)
                query_feats[ptr:ptr + batch_num, :] = feat.cpu().numpy()
                query_global_feats[ptr:ptr + batch_num, :] = global_feat.cpu().numpy()
                ptr = ptr + batch_num
        duration = time.time() - start_time
        # print('Extracting time: {}s'.format(int(round(duration))))
    return query_global_feats, query_feats


args, unparsed = parser.parse_known_args()

protocol = args.protocol
data_name_list_train, data_name_test = protocol_decoder(args.protocol)
if data_name_test[0] in ['SYSU']:
    data_path = '../Dataset/SYSU-MM01/'
    if len(data_name_list_train) == 1:
        num_classes = 395
    elif len(data_name_list_train) == 2:
        num_classes = 919
    else:
        raise "Wrong protocol!"
    test_mode = [1, 2]
elif data_name_test[0] in ['RegDB']:
    data_path = '../Dataset/RegDB/'
    if len(data_name_list_train) == 1:
        num_classes = 206
    elif len(data_name_list_train) == 2:
        num_classes = 1108
    else:
        raise "Wrong protocol!"
    test_mode = [1, 2]
elif data_name_test[0] in ['LLCM']:
    data_path = '../Dataset/LLCM/'
    if len(data_name_list_train) == 1:
        num_classes = 713
    elif len(data_name_list_train) == 2:
        num_classes = 601
    else:
        raise "Wrong protocol!"
    test_mode = [1, 2]

cudnn.benchmark = True
# cudnn.deterministic = True

print('==> Building model......')

model = embed_net(num_classes, no_local='on', gm_pool='on', arch=args.arch)
model.to(device)

print('==> Testing......')
# define transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize])

end = time.time()

if data_name_test[0] in ['SYSU']:
    if len(args.resume) > 0:
        model_path = args.model_path + args.protocol + '/' + args.resume
        print('==> Loading weights from checkpoint......')
        checkpoint = torch.load(model_path, map_location='cuda:0')
        print('==> best epoch', checkpoint['epoch'])
        model.load_state_dict(checkpoint['net'])
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    nquery = len(query_label)
    query_feat, query_feat_att = extract_query_feat(query_loader)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode='all', trial=trial)
        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        ngall = len(gall_label)
        gall_feat, gall_feat_att = extract_gall_feat(gall_loader)

        distmat = -np.matmul(query_feat, np.transpose(gall_feat))
        distmat_att = -np.matmul(query_feat_att, np.transpose(gall_feat_att))

        cmc, mAP, mINP = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(distmat_att, query_label, gall_label, query_cam, gall_cam)

        if trial == 0:

            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP
            all_cmc_att = cmc_att
            all_mAP_att = mAP_att
            all_mINP_att = mINP_att

        else:

            all_cmc += cmc
            all_mAP += mAP
            all_mINP += mINP
            all_cmc_att += cmc_att
            all_mAP_att += mAP_att
            all_mINP_att += mINP_att

        print('SYSU Test Trial: {}'.format(trial))
        print(
            'original feature:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'feature after BN:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))





elif data_name_test[0] in ['LLCM']:
    if len(args.resume) > 0:
        model_path = args.model_path + args.protocol + '/' + args.resume
        print('==> Loading weights from checkpoint......')
        checkpoint = torch.load(model_path)
        print('==> best epoch', checkpoint['epoch'])
        model.load_state_dict(checkpoint['net'])
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

    query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    nquery = len(query_label)
    query_feat, query_feat_att = extract_query_feat(query_loader)

    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=1, trial=trial)
        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        # gallset = TestData(gall_img, gall_label, transform=corruption_transform_test, img_size=(args.img_w, args.img_h))
        gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        ngall = len(gall_label)
        gall_feat, gall_feat_att = extract_gall_feat(gall_loader)

        distmat = -np.matmul(query_feat, np.transpose(gall_feat))
        distmat_att = -np.matmul(query_feat_att, np.transpose(gall_feat_att))

        cmc, mAP, mINP = eval_llcm(distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_llcm(distmat_att, query_label, gall_label, query_cam, gall_cam)

        if trial == 0:

            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP
            all_cmc_att = cmc_att
            all_mAP_att = mAP_att
            all_mINP_att = mINP_att

        else:

            all_cmc += cmc
            all_mAP += mAP
            all_mINP += mINP
            all_cmc_att += cmc_att
            all_mAP_att += mAP_att
            all_mINP_att += mINP_att

        print('LLCM Test Trial: {}'.format(trial))
        print(
            'original feature:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'feature after BN:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))

elif data_name_test[0] in ['RegDB']:
    if len(args.resume) > 0:
        model_path = args.model_path + args.protocol + '/' + args.resume
        print('==> Loading weights from checkpoint......')
        checkpoint = torch.load(model_path)
        print('==> best epoch', checkpoint['epoch'])
        model.load_state_dict(checkpoint['net'])
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

    for trial in range(10):
        test_trial = trial + 1
        query_img, query_label = process_test_regdb(data_path, trial=test_trial, modal='thermal')
        gall_img, gall_label = process_test_regdb(data_path, trial=test_trial, modal='visible')

        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

        gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        nquery = len(query_label)
        ngall = len(gall_label)

        query_feat, query_feat_att = extract_query_feat(query_loader)
        gall_feat, gall_feat_att = extract_gall_feat(gall_loader)

        distmat = -np.matmul(query_feat, np.transpose(gall_feat))
        distmat_att = -np.matmul(query_feat_att, np.transpose(gall_feat_att))

        cmc, mAP, mINP = eval_regdb(distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att = eval_regdb(distmat_att, query_label, gall_label)

        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP
            all_cmc_att = cmc_att
            all_mAP_att = mAP_att
            all_mINP_att = mINP_att

        else:
            all_cmc += cmc
            all_mAP += mAP
            all_mINP += mINP
            all_cmc_att += cmc_att
            all_mAP_att += mAP_att
            all_mINP_att += mINP_att

        if args.tvsearch:
            print('RegDB Test Trial: {}, Visible to Thermal'.format(test_trial))
        else:
            print('RegDB Test Trial: {}, Thermal to Visible'.format(test_trial))

        print(
            'original feature:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'feature after BN:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
all_cmc = all_cmc / 10
all_mAP = all_mAP / 10
all_mINP = all_mINP / 10
all_cmc_att = all_cmc_att / 10
all_mAP_att = all_mAP_att / 10
all_mINP_att = all_mINP_att / 10
print('All Average:')
print(
    'original feature:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        all_cmc[0], all_cmc[4], all_cmc[9], all_cmc[19], all_mAP, all_mINP))
print(
    'feature after BN:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        all_cmc_att[0], all_cmc_att[4], all_cmc_att[9], all_cmc_att[19], all_mAP_att, all_mINP_att))