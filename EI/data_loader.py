import numpy as np
from PIL import Image
import torch.utils.data as data
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing
import torchvision.transforms as transforms
import random
import math


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


class SYSUData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None):
        train_image = np.load(data_dir + 'train_img.npy')
        train_label = np.load(data_dir + 'train_label.npy')
        self.train_image = train_image
        self.train_label = train_label

        self.transform = transform
        self.Index = colorIndex

    def __getitem__(self, index):
        img, target = self.train_image[self.Index[index]], self.train_label[
            self.Index[index]]
        img = self.transform(img)

        return img, target
        # img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[
        #     self.cIndex[index]]
        # img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[
        #     self.tIndex[index]]
        # img1 = self.transform(img1)
        # img2 = self.transform(img2)
        # coin = random.randint(0, 2)
        # img1 = self.transform_color(img1) if coin == 0 else self.transform_color1(img1)
        # img2 = self.transform_thermal(img2)

        # return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_label)


class RegDBData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None):
        train_image = np.load(data_dir + 'train_img.npy')
        train_label = np.load(data_dir + 'train_label.npy')
        self.train_image = train_image
        self.train_label = train_label

        self.transform = transform
        self.Index = colorIndex

    def __getitem__(self, index):
        img, target = self.train_image[self.Index[index]], self.train_label[
            self.Index[index]]
        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.train_label)


class LLCMData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None):
        train_image = np.load(data_dir + 'train_img.npy')
        train_label = np.load(data_dir + 'train_label.npy')
        self.train_image = train_image
        self.train_label = train_label

        self.transform = transform
        self.Index = colorIndex

    def __getitem__(self, index):
        img, target = self.train_image[self.Index[index]], self.train_label[
            self.Index[index]]
        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.train_label)


class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(144, 288)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
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


