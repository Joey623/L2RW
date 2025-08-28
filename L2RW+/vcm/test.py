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
from data_loader import VideoDataset_test
from data_manager import *
from eval_metrics import eval_vcm
from utils import *
from model import embed_net

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
# each changes
parser.add_argument('--noticed', default='VI-ReID-test', type=str)

parser.add_argument('--dataset', default='vcm', help='dataset name: regdb or sysu')
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
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--test-batch', default=32, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--margin', default=0.5, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--resume', '-r', default='l2rw.pth', type=str,
                    help='resume from checkpoint')

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
parser.add_argument('--max_epoch', default=100)
parser.add_argument('--rerank', default='no', type=str)
parser.add_argument('--dim', default=2048, type=int)
parser.add_argument('--tvsearch', default=0, type=int, help='1:visible to infrared, 0:infrared to visible')
parser.add_argument('--tta', default=False, type=bool)
parser.add_argument('--seq_lenth', default=6, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
set_seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args, unparsed = parser.parse_known_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
dataset = args.dataset

np.random.seed(1)
seq_lenth = args.seq_lenth
data_set = VCM(client_id=1)
nquery = data_set.num_query_tracklets
ngall = data_set.num_gallery_tracklets
nquery_1 = data_set.num_query_tracklets_1
ngall_1 = data_set.num_gallery_tracklets_1
args, unparsed = parser.parse_known_args()

num_classes = 500

net = embed_net(num_classes, pool_dim=args.pool_dim)
net = net.to(device)

cudnn.benchmark = True

# model_path = args.model_path + args.dataset + '/' + args.resume
model_path = './save_models/l2rw_vcm.pth'
# model_path = '../bupt/save_models/bupt.pth'
print('==> Loading weights from checkpoint......')
checkpoint = torch.load(model_path)
print('==> best epoch', checkpoint['epoch'])
state_dict = checkpoint['net']
filtered_state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}

# 加载去掉分类头的参数
missing, unexpected = net.load_state_dict(filtered_state_dict, strict=False)

print(f"Loaded with {len(missing)} missing and {len(unexpected)} unexpected keys.")
# net.load_state_dict(checkpoint['net'], strict=False)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])
queryloader = data.DataLoader(VideoDataset_test(data_set.query, seq_len=seq_lenth, sample='video_test', transform=transform_test),batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
galleryloader = data.DataLoader(VideoDataset_test(data_set.gallery, seq_len=seq_lenth, sample='video_test', transform=transform_test),batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

queryloader_1 = data.DataLoader(VideoDataset_test(data_set.query_1, seq_len=seq_lenth, sample='video_test', transform=transform_test),batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
galleryloader_1 = data.DataLoader(VideoDataset_test(data_set.gallery_1, seq_len=seq_lenth, sample='video_test', transform=transform_test),batch_size=args.test_batch, shuffle=False, num_workers=args.workers)



def test(net):
    # infrared to visible
    with torch.no_grad():
        net.eval()
        print('Extracting Gallery Feature...')
        start = time.time()
        ptr = 0
        gall_feat = np.zeros((ngall, args.dim))
        q_pids, q_camids = [], []
        g_pids, g_camids = [], []
        with torch.no_grad():
            for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
                input = imgs
                label = pids
                batch_num = input.size(0)
                input = Variable(input.to(device))
                _, feat = net(input, seq_len=seq_lenth)
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
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
        with torch.no_grad():
            for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
                input = imgs
                label = pids
                batch_num = input.size(0)
                input = Variable(input.to(device))
                _, feat = net(input, seq_len=seq_lenth)
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                ptr = ptr + batch_num
                q_pids.extend(pids)
                q_camids.extend(camids)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        print("Extracting Time: {}".format(time.time() - start))

        start = time.time()
        distmat = -np.matmul(query_feat, gall_feat.T)
        print("Calculated Time: {}".format(time.time() - start))
        cmc, mAP, mINP = eval_vcm(distmat, q_pids, g_pids, q_camids, g_camids)
        return cmc, mAP, mINP

def test_1(net):
    with torch.no_grad():
        net.eval()
        print('Extracting Gallery Feature...')
        start = time.time()
        ptr = 0
        gall_feat = np.zeros((ngall_1, args.dim))
        q_pids, q_camids = [], []
        g_pids, g_camids = [], []
        with torch.no_grad():
            for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
                input = imgs
                label = pids
                batch_num = input.size(0)
                input = Variable(input.to(device))
                _, feat = net(input, seq_len=seq_lenth)
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
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
        query_feat = np.zeros((nquery_1, args.dim))
        with torch.no_grad():
            for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
                input = imgs
                label = pids
                batch_num = input.size(0)
                input = Variable(input.to(device))
                _, feat = net(input, seq_len=seq_lenth)
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                ptr = ptr + batch_num
                q_pids.extend(pids)
                q_camids.extend(camids)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        print("Extracting Time: {}".format(time.time() - start))

        start = time.time()
        distmat = -np.matmul(query_feat, gall_feat.T)
        print("Calculated Time: {}".format(time.time() - start))
        cmc, mAP, mINP = eval_vcm(distmat, q_pids, g_pids, q_camids, g_camids)
        return cmc, mAP, mINP

# infrared to visible
cmc, mAP, mINP = test(net)
print(
    'infrared to visible:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
# cmc_1, mAP_1, mINP_1 = test_1(net)
# print(
#     'visible to infrared:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
#         cmc_1[0], cmc_1[4], cmc_1[9], cmc_1[19], mAP_1, mINP_1))