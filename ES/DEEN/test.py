from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from model import embed_net
from utils import *
import pdb
import scipy.io
import torchvision

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: llcm, regdb or sysu]')
parser.add_argument('--protocol', type=str, default="R_S_to_L",
                    help='R_S_to_L, R_L_to_S, L_S_to_R, S_to_S, R_to_R, L_to_L')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline: resnet50')
parser.add_argument('--resume', '-r', default='R_S_to_L_deen_p6_n4_lr_0.1_seed_0_best.pth', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str, help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=6, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=32, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='awg', type=str, metavar='m', help='method type: base or awg')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='0,1,2,3', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu')  # SYSU-MM01
parser.add_argument('--tvsearch', default=True,
                    help='True->Infrared to Visible while False->Visible to Infrared')  # RegDB

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# set_seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pool_dim = 2048


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_gall_feat(gall_loader):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat1 = np.zeros((ngall, pool_dim))
    gall_feat2 = np.zeros((ngall, pool_dim))
    gall_feat3 = np.zeros((ngall, pool_dim))
    gall_feat4 = np.zeros((ngall, pool_dim))
    gall_feat5 = np.zeros((ngall, pool_dim))
    gall_feat6 = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input1 = Variable(input.cuda())
            feat, feat_att = net(input1, input1, test_mode[0])
            gall_feat1[ptr:ptr + batch_num, :] = feat[:batch_num].detach().cpu().numpy()
            gall_feat2[ptr:ptr + batch_num, :] = feat_att[:batch_num].detach().cpu().numpy()
            gall_feat3[ptr:ptr + batch_num, :] = feat[batch_num:batch_num * 2].detach().cpu().numpy()
            gall_feat4[ptr:ptr + batch_num, :] = feat_att[batch_num:batch_num * 2].detach().cpu().numpy()
            gall_feat5[ptr:ptr + batch_num, :] = feat[batch_num * 2:].detach().cpu().numpy()
            gall_feat6[ptr:ptr + batch_num, :] = feat_att[batch_num * 2:].detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return gall_feat1, gall_feat2, gall_feat3, gall_feat4, gall_feat5, gall_feat6


def extract_query_feat(query_loader):
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat1 = np.zeros((nquery, pool_dim))
    query_feat2 = np.zeros((nquery, pool_dim))
    query_feat3 = np.zeros((nquery, pool_dim))
    query_feat4 = np.zeros((nquery, pool_dim))
    query_feat5 = np.zeros((nquery, pool_dim))
    query_feat6 = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input1 = Variable(input.cuda())
            feat, feat_att = net(input1, input1, test_mode[1])
            query_feat1[ptr:ptr + batch_num, :] = feat[:batch_num].detach().cpu().numpy()
            query_feat2[ptr:ptr + batch_num, :] = feat_att[:batch_num].detach().cpu().numpy()
            query_feat3[ptr:ptr + batch_num, :] = feat[batch_num:batch_num * 2].detach().cpu().numpy()
            query_feat4[ptr:ptr + batch_num, :] = feat_att[batch_num:batch_num * 2].detach().cpu().numpy()
            query_feat5[ptr:ptr + batch_num, :] = feat[batch_num * 2:].detach().cpu().numpy()
            query_feat6[ptr:ptr + batch_num, :] = feat_att[batch_num * 2:].detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return query_feat1, query_feat2, query_feat3, query_feat4, query_feat5, query_feat6


data_name_list_train, data_name_test = protocol_decoder(args.protocol)
if data_name_test[0] in ['SYSU']:
    data_path = './Dataset/SYSU-MM01/'
    if len(data_name_list_train) == 1:
        num_classes = 395
    elif len(data_name_list_train) == 2:
        num_classes = 919
    else:
        raise "Wrong protocol!"
    test_mode = [1, 2]
elif data_name_test[0] in ['RegDB']:
    data_path = './Dataset/RegDB/'
    if len(data_name_list_train) == 1:
        num_classes = 206
    elif len(data_name_list_train) == 2:
        num_classes = 1108
    else:
        raise "Wrong protocol!"
    test_mode = [1, 2]
elif data_name_test[0] in ['LLCM']:
    data_path = './Dataset/LLCM/'
    if len(data_name_list_train) == 1:
        num_classes = 713
    elif len(data_name_list_train) == 2:
        num_classes = 601
    else:
        raise "Wrong protocol!"
    test_mode = [1, 2]

test_mode = [1, 2]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0
print('==> Building model..')
net = embed_net(num_classes, arch=args.arch)
# net = nn.DataParallel(net)
net.to(device)
cudnn.benchmark = True

checkpoint_path = args.model_path

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()

print("Protocol:{}".format(args.protocol))
if data_name_test[0] in ['LLCM']:
    if len(args.resume) > 0:
        model_path = args.model_path + args.protocol + '/' + args.resume
        print('==> Loading weights from checkpoint......')
        checkpoint = torch.load(model_path)
        print('==> best epoch', checkpoint['epoch'])
        net.load_state_dict(checkpoint['net'])
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

    query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    nquery = len(query_label)
    query_feat1, query_feat2, query_feat3, query_feat4, query_feat5, query_feat6 = extract_query_feat(query_loader)

    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=1, trial=trial)
        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        ngall = len(gall_label)
        gall_feat1, gall_feat2, gall_feat3, gall_feat4, gall_feat5, gall_feat6 = extract_gall_feat(trial_gall_loader)

        distmat1 = np.matmul(query_feat1, np.transpose(gall_feat1))
        distmat2 = np.matmul(query_feat2, np.transpose(gall_feat2))
        distmat3 = np.matmul(query_feat3, np.transpose(gall_feat3))
        distmat4 = np.matmul(query_feat4, np.transpose(gall_feat4))
        distmat5 = np.matmul(query_feat5, np.transpose(gall_feat5))
        distmat6 = np.matmul(query_feat6, np.transpose(gall_feat6))
        distmat7 = distmat1 + distmat2 + distmat3 + distmat4 + distmat5 + distmat6
        a = 0.1
        # distmat7 = distmat1 + distmat3 + distmat5 + distmat2 + distmat4 + distmat6
        distmat8 = a * (distmat1 + distmat3 + distmat5) + (1 - a) * (distmat2 + distmat4 + distmat6)

        cmc7, mAP7, mINP7 = eval_llcm(-distmat7, query_label, gall_label, query_cam, gall_cam)
        cmc8, mAP8, mINP8 = eval_llcm(-distmat8, query_label, gall_label, query_cam, gall_cam)

        if trial == 0:

            all_cmc7 = cmc7
            all_mAP7 = mAP7
            all_mINP7 = mINP7

            all_cmc8 = cmc8
            all_mAP8 = mAP8
            all_mINP8 = mINP8

        else:

            all_cmc7 += cmc7
            all_mAP7 += mAP7
            all_mINP7 += mINP7

            all_cmc8 += cmc8
            all_mAP8 += mAP8
            all_mINP8 += mINP8

        print('LLCM Test Trial: {}'.format(trial))
        print(
            'original feature:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc7[0], cmc7[4], cmc7[9], cmc7[19], mAP7, mINP7))
        print(
            'feature after BN:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc8[0], cmc8[4], cmc8[9], cmc8[19], mAP8, mINP8))


elif data_name_test[0] in ['SYSU']:

    if len(args.resume) > 0:
        model_path = args.model_path + args.protocol + '/' + args.resume
        print('==> Loading weights from checkpoint......')
        checkpoint = torch.load(model_path, map_location='cuda:0')
        print('==> best epoch', checkpoint['epoch'])
        net.load_state_dict(checkpoint['net'])
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    nquery = len(query_label)
    query_feat1, query_feat2, query_feat3, query_feat4, query_feat5, query_feat6 = extract_query_feat(query_loader)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode='all', trial=trial)
        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        ngall = len(gall_label)
        gall_feat1, gall_feat2, gall_feat3, gall_feat4, gall_feat5, gall_feat6 = extract_gall_feat(trial_gall_loader)
        distmat1 = np.matmul(query_feat1, np.transpose(gall_feat1))
        distmat2 = np.matmul(query_feat2, np.transpose(gall_feat2))
        distmat3 = np.matmul(query_feat3, np.transpose(gall_feat3))
        distmat4 = np.matmul(query_feat4, np.transpose(gall_feat4))
        distmat5 = np.matmul(query_feat5, np.transpose(gall_feat5))
        distmat6 = np.matmul(query_feat6, np.transpose(gall_feat6))
        distmat7 = distmat1 + distmat2 + distmat3 + distmat4 + distmat5 + distmat6
        a = 0.1
        # distmat7 = distmat1 + distmat3 + distmat5 + distmat2 + distmat4 + distmat6
        distmat8 = a * (distmat1 + distmat3 + distmat5) + (1 - a) * (distmat2 + distmat4 + distmat6)

        cmc7, mAP7, mINP7 = eval_sysu(-distmat7, query_label, gall_label, query_cam, gall_cam)
        cmc8, mAP8, mINP8 = eval_sysu(-distmat8, query_label, gall_label, query_cam, gall_cam)

        if trial == 0:

            all_cmc7 = cmc7
            all_mAP7 = mAP7
            all_mINP7 = mINP7

            all_cmc8 = cmc8
            all_mAP8 = mAP8
            all_mINP8 = mINP8

        else:

            all_cmc7 += cmc7
            all_mAP7 += mAP7
            all_mINP7 += mINP7

            all_cmc8 += cmc8
            all_mAP8 += mAP8
            all_mINP8 += mINP8

        print('SYSU Test Trial: {}'.format(trial))
        print(
            'original feature:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc7[0], cmc7[4], cmc7[9], cmc7[19], mAP7, mINP7))
        print(
            'feature after BN:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc8[0], cmc8[4], cmc8[9], cmc8[19], mAP8, mINP8))

elif data_name_test[0] in ['RegDB']:
    if len(args.resume) > 0:
        model_path = args.model_path + args.protocol + '/' + args.resume
        print('==> Loading weights from checkpoint......')
        checkpoint = torch.load(model_path)
        print('==> best epoch', checkpoint['epoch'])
        net.load_state_dict(checkpoint['net'])
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

        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

        query_feat1, query_feat2, query_feat3, query_feat4, query_feat5, query_feat6 = extract_query_feat(query_loader)
        gall_feat1, gall_feat2, gall_feat3, gall_feat4, gall_feat5, gall_feat6 = extract_gall_feat(gall_loader)

        distmat1 = np.matmul(query_feat1, np.transpose(gall_feat1))
        distmat2 = np.matmul(query_feat2, np.transpose(gall_feat2))
        distmat3 = np.matmul(query_feat3, np.transpose(gall_feat3))
        distmat4 = np.matmul(query_feat4, np.transpose(gall_feat4))
        distmat5 = np.matmul(query_feat5, np.transpose(gall_feat5))
        distmat6 = np.matmul(query_feat6, np.transpose(gall_feat6))
        a = 0.1
        distmat7 = distmat1 + distmat3 + distmat5 + distmat2 + distmat4 + distmat6
        distmat8 = a * (distmat1 + distmat3 + distmat5) + (1 - a) * (distmat2 + distmat4 + distmat6)

        cmc7, mAP7, mINP7 = eval_regdb(-distmat7, gall_label, query_label)
        cmc8, mAP8, mINP8 = eval_regdb(-distmat8, gall_label, query_label)

        all_cmc7 += cmc7
        all_mAP7 += mAP7
        all_mINP7 += mINP7

        all_cmc8 += cmc8
        all_mAP8 += mAP8
        all_mINP8 += mINP8

        print('RegDB Test Trial: {}'.format(test_trial))
        print(
            'original feature:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc7[0], cmc7[4], cmc7[9], cmc7[19], mAP7, mINP7))
        print(
            'feature after BN:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc8[0], cmc8[4], cmc8[9], cmc8[19], mAP8, mINP8))

all_cmc7 = all_cmc7 / 10
all_mAP7 = all_mAP7 / 10
all_mINP7 = all_mINP7 / 10
all_cmc8 = all_mAP8 / 10
all_mAP8 = all_mAP8 / 10
all_mINP8 = all_mINP8 / 10
print('All Average:')
print(
    'original feature:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        all_cmc7[0], all_cmc7[4], all_cmc7[9], all_cmc7[19], all_mAP7, all_mINP7))
print(
    'feature after BN:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        all_cmc8[0], all_cmc8[4], all_cmc8[9], all_cmc8[19], all_mAP8, all_mINP8))