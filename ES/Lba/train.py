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
import torchvision.transforms as transforms
from data_loader import *
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from model import embed_net
from utils import *
from loss import OriTripletLoss, TripletLoss_WRT
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter
import logging

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--protocol', type=str, default="R_S_to_L",
                    help='R_S_to_L, R_L_to_S, L_S_to_R, S_to_S, R_to_R, L_to_L')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--max_epoch', default=30)
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='lba', type=str,
                    metavar='m', help='method type: base or agw')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--lw_dt', default=0.5, type=float, help='weight for dense triplet loss')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
set_seed(args.seed)
protocol = args.protocol


model_path = args.model_path + protocol + '/'
if not os.path.isdir(model_path):
    os.makedirs(model_path)

# training: only infrared to visible mode.
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

sys.stdout = Logger(log_path + suffix + '_os.txt')

print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
test_mode = [1, 2]
trainset = get_datasets(args, MultiVIReIDDataset, transform=transform_train)
color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)



if data_name_test[0] in ["SYSU"]:
    data_path = '../Dataset/SYSU-MM01/'
    logging.info('==> Testing dataset: SYSU-MM01')
    # default mode: all (if you wanna testing indoor mode, just change it on test.py)
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif data_name_test[0] in ["RegDB"]:
    data_path = '../Dataset/RegDB/'
    logging.info('==> Testing dataset: RegDB')
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='visible')

elif data_name_test[0] in ["LLCM"]:
    data_path = '../Dataset/LLCM/'
    logging.info('==> Testing dataset: LLCM')
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

logging.info('data init success!')

logging.info('Protocol {} statistics:'.format(protocol))
logging.info('  ------------------------------')
logging.info('  subset   | # ids | # images')
logging.info('  ------------------------------')
logging.info('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
logging.info('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
logging.info('  ------------------------------')
logging.info('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
logging.info('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
logging.info('  ------------------------------')
logging.info('Data Loading Time:\t {:.3f}'.format(time.time() - end))

net = embed_net(args, n_class)
net.to(device)
cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = model_path + args.protocol + '/' + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion_id = nn.CrossEntropyLoss()

loader_batch = args.batch_size * args.num_pos
criterion_tri= OriTripletLoss(batch_size=args.batch_size*args.num_pos, margin=args.margin)

criterion_id.to(device)
criterion_tri.to(device)


if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
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

    return lr


def train(epoch):

    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss_meter = AverageMeter()
    loss_id_meter = AverageMeter()
    if args.method == 'lba':
        loss_ic_meter = AverageMeter()
        loss_dt_meter = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):

        labels = torch.cat((label1, label2), 0)

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())

        labels = Variable(labels.cuda())
        data_time.update(time.time() - end)

        with autocast():
            out = net(input1, input2, use_cmalign=True)
            loss_tri, batch_acc = criterion_tri(out['feat4_p'], labels)
            loss_id = criterion_id(out['cls_id'], labels) + loss_tri
            loss_ic = criterion_id(out['cls_ic_layer3'], labels) + criterion_id(out['cls_ic_layer4'], labels)
            loss_dt = out['loss_dt']

            loss = loss_id + loss_ic + args.lw_dt * loss_dt

            correct += (batch_acc / 2)
            _, predicted = out['cls_id'].max(1)
            correct += (predicted.eq(labels).sum().item() / 2)
            total += labels.size(0)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update P
        train_loss_meter.update(loss.item(), 2 * input1.size(0))
        loss_id_meter.update(loss_id.item(), 2 * input1.size(0))
        loss_ic_meter.update(loss_ic.item(), 2 * input1.size(0))
        loss_dt_meter.update(loss_dt.item(), 2 * input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'lr: {:.3f} '
                  'loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'loss_id: {loss_id.val:.4f} ({loss_id.avg:.4f}) '
                  'loss_ic: {loss_ic.val:.4f} ({loss_ic.avg:.4f}) '
                  'loss_dt: {loss_dt.val:.4f} ({loss_dt.avg:.4f}) '
                  'acc: {acc:.2f}'.format(
                epoch, batch_idx, len(trainloader),
                current_lr,
                train_loss=train_loss_meter,
                loss_id=loss_id_meter,
                loss_ic=loss_ic_meter,
                loss_dt=loss_dt_meter,
                acc=100. * correct / total)
            )


def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat= net(input, input, test_mode[0], use_cmalign=False)['feat4_p_norm']
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat= net(input, input, test_mode[1], use_cmalign=False)['feat4_p_norm']
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = -np.matmul(query_feat, np.transpose(gall_feat))

    # evaluation
    if data_name_test[0] in ["RegDB"]:
        cmc, mAP, mINP = eval_regdb(distmat, query_label, gall_label)
    elif data_name_test[0] in ["SYSU"]:
        cmc, mAP, mINP = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
    elif data_name_test[0] in ["LLCM"]:
        cmc, mAP, mINP = eval_llcm(distmat, query_label, gall_label, query_cam, gall_cam)

    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))


    return cmc, mAP, mINP


# training
scaler = GradScaler()
print('==> Start Training...')
for epoch in range(start_epoch, args.max_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    train(epoch)

    if epoch == args.max_epoch-1:
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
        }
        torch.save(state, model_path + suffix + '.pth')