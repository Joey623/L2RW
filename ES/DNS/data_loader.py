import numpy as np
from PIL import Image
import torch.utils.data as data
import os.path as osp
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing
import torchvision.transforms as transforms
import random
import math
import torch
from utils import protocol_decoder

class ChannelExchange(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, gray=2):
        self.gray = gray

    def __call__(self, img):

        idx = random.randint(0, self.gray)

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



class MultiVIReIDDataset(data.Dataset):
    def __init__(self, args, data_dir1, data_dir2, protocol, colorIndex=None, thermalIndex=None):
        data_path1 = data_dir1 + protocol + '/'
        data_path2 = data_dir2 + protocol + '/'
        train_color_image1 = np.load(data_path1 + 'train_rgb.npy')
        train_color_image2 = np.load(data_path2 + 'train_rgb.npy')
        train_color_image = np.concatenate((train_color_image1, train_color_image2), axis=0)
        self.train_color_image = train_color_image

        train_color_label1 = np.load(data_path1 + 'train_rgb_label.npy')
        train_color_label2 = np.load(data_path2 + 'train_rgb_label.npy')
        train_color_label = np.concatenate((train_color_label1, train_color_label2), axis=0)
        self.train_color_label = train_color_label

        train_thermal_image1 = np.load(data_path1 + 'train_ir.npy')
        train_thermal_image2 = np.load(data_path2 + 'train_ir.npy')
        train_thermal_image = np.concatenate((train_thermal_image1, train_thermal_image2), axis=0)
        self.train_thermal_image = train_thermal_image

        train_thermal_label1 = np.load(data_path1 + 'train_ir_label.npy')
        train_thermal_label2 = np.load(data_path2 + 'train_ir_label.npy')
        train_thermal_label = np.concatenate((train_thermal_label1, train_thermal_label2), axis=0)
        self.train_thermal_label = train_thermal_label

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_thermal = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((args.img_h, args.img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.ColorJitter(hue=0.5),
            normalize,
            ChannelRandomErasing(probability=0.5)])

        self.transform_color = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((args.img_h, args.img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5)])

        self.transform_color1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((args.img_h, args.img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelExchange(gray=2)])

        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        # img1 = self.transform(img1)
        # img2 = self.transform(img2)
        coin = random.randint(0, 2)
        img1 = self.transform_color(img1) if coin == 0 else self.transform_color1(img1)
        img2 = self.transform_thermal(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

# def get_single_dataset(args, VIReIDDataset, data_name=""):
#     if data_name in ["SYSU"]:
#         data_dir = '../Dataset/SYSU-MM01/'
#         data_set = VIReIDDataset(args, data_dir, args.protocol)
#
#     elif data_name in ["RegDB"]:
#         data_dir = '../Dataset/RegDB/'
#         data_set = VIReIDDataset(args, data_dir, args.protocol)
#
#     elif data_name in ["LLCM"]:
#         data_dir = '../Dataset/LLCM/'
#         data_set = VIReIDDataset(args, data_dir, args.protocol)
#
#     return data_set


def get_datasets(args, MultiVIReIDDataset):
    data_name_list_train, _ = protocol_decoder(args.protocol)
    dataset_dir_mapping = {
        "SYSU": '../Dataset/SYSU-MM01/',
        "RegDB": '../Dataset/RegDB/',
        "LLCM": '../Dataset/LLCM/',
    }
    data_dir1 = dataset_dir_mapping.get(data_name_list_train[0], '')
    data_dir2 = dataset_dir_mapping.get(data_name_list_train[1], '')
    train_set = MultiVIReIDDataset(args, data_dir1, data_dir2, args.protocol)

    return train_set


class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(144, 288)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


class TestDataOld(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size=(144, 288)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(data_dir + test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label
