import numpy as np
from PIL import Image
import torch.utils.data as data
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing
import torchvision.transforms as transforms
import random
import math


class SYSUData(data.Dataset):
    def __init__(self, data_dir, args, client_id=None, transform=None, colorIndex=None):
        self.args = args
        if self.args.FL == False:
            train_image = np.load(data_dir + 'train_img.npy')
            train_label = np.load(data_dir + 'train_label.npy')
            train_cam = np.load(data_dir + 'train_cam.npy')
        else:
            train_image = np.load(data_dir + 'train_img_client{}.npy'.format(client_id))
            train_label = np.load(data_dir + 'train_label_client{}.npy'.format(client_id))
            train_cam = np.load(data_dir + 'train_cam_client{}.npy'.format(client_id))
        self.train_image = train_image
        self.train_label = train_label
        self.train_cam = train_cam

        self.transform = transform
        self.Index = colorIndex

    def __getitem__(self, index):
        img, target, cam = self.train_image[self.Index[index]], self.train_label[
            self.Index[index]], self.train_cam[self.Index[index]]
        img = self.transform(img)

        return img, target, cam

    def __len__(self):
        return len(self.train_label)


class RegDBData(data.Dataset):
    def __init__(self, data_dir, args, client_id=None, transform=None, colorIndex=None):
        self.args = args

        if self.args.FL == False:
            train_image = np.load(data_dir + 'train_img_trial{}.npy'.format(self.args.trial))
            train_label = np.load(data_dir + 'train_label_trial{}.npy'.format(self.args.trial))
            train_cam = np.load(data_dir + 'train_cam_trial{}.npy'.format(self.args.trial))
        else:
            train_image = np.load(data_dir + 'train_img_trial{}_client{}.npy'.format(self.args.trial, client_id))
            train_label = np.load(data_dir + 'train_label_trial{}_client{}.npy'.format(self.args.trial, client_id))
            train_cam = np.load(data_dir + 'train_cam_trial{}_client{}.npy'.format(self.args.trial, client_id))
        self.train_image = train_image
        self.train_label = train_label
        self.train_cam = train_cam

        self.transform = transform
        self.Index = colorIndex

    def __getitem__(self, index):
        img, target, cam = self.train_image[self.Index[index]], self.train_label[
            self.Index[index]], self.train_cam[self.Index[index]]
        img = self.transform(img)

        return img, target, cam

    def __len__(self):
        return len(self.train_label)


class LLCMData(data.Dataset):
    def __init__(self, data_dir, args, client_id=None, transform=None, colorIndex=None):
        self.args = args
        if self.args.FL == False:
            train_image = np.load(data_dir + 'train_img.npy')
            train_label = np.load(data_dir + 'train_label.npy')
            train_cam = np.load(data_dir + 'train_cam.npy')
        else:
            train_image = np.load(data_dir + 'train_img_client{}.npy'.format(client_id))
            train_label = np.load(data_dir + 'train_label_client{}.npy'.format(client_id))
            train_cam = np.load(data_dir + 'train_cam_client{}.npy'.format(client_id))
        self.train_image = train_image
        self.train_label = train_label
        self.train_cam = train_cam

        self.transform = transform
        self.Index = colorIndex

    def __getitem__(self, index):
        img, target, cam = self.train_image[self.Index[index]], self.train_label[
            self.Index[index]], self.train_cam[self.Index[index]]
        img = self.transform(img)

        return img, target, cam

    def __len__(self):
        return len(self.train_label)


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


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label