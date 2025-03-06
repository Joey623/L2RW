import numpy as np
from PIL import Image
import os
import random
import re

data_path = '../data/RegDB/'
cameras = ['cam1', 'cam2']

def load_data(input_data_path, modal='visible'):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [data_path + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        if modal == 'visible':
            file_cam = [1] * len(data_file_list)
        elif modal == 'infrared':
            file_cam = [2] * len(data_file_list)
        else:
            raise RuntimeError('Invalid modal value. Please check the modal parameter.')

    return file_image, file_label, file_cam

def preprocess_regdb(data_path, width=144, height=288, trial=1):
    train_color_list = data_path + 'idx/train_visible_{}'.format(trial) + '.txt'
    train_thermal_list = data_path + 'idx/train_thermal_{}'.format(trial) + '.txt'

    rgb_img, rgb_lbl, rgb_cam = load_data(train_color_list, modal='visible')
    ir_img, ir_lbl, ir_cam = load_data(train_thermal_list, modal='infrared')
    rgb_img.extend(ir_img)
    rgb_lbl.extend(ir_lbl)
    rgb_cam.extend(ir_cam)

    # relabel
    pid_container = set()
    for label in rgb_lbl:
        pid = label
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}
    train_img = []
    train_lbl = []
    train_cam = []
    for index, (image, label, camera) in enumerate(zip(rgb_img, rgb_lbl, rgb_cam)):
        img = Image.open(image)
        img = img.resize((width, height), Image.LANCZOS)
        pix_array = np.array(img)
        train_img.append(pix_array)
        pid = label
        pid = pid2label[pid]
        train_lbl.append(pid)
        cam = int(camera)
        train_cam.append(cam)
    return np.array(train_img), np.array(train_lbl), np.array(train_cam)

def split_client(cam_id, train_img, train_label, train_cam):
    # 找到所有符合指定摄像头ID的图像索引
    cam_indices = [i for i, cam in enumerate(train_cam) if cam == cam_id]

    # 提取符合条件的图像、标签和摄像头信息
    cam_img = [train_img[i] for i in cam_indices]
    cam_label = [train_label[i] for i in cam_indices]
    cam_cam = [train_cam[i] for i in cam_indices]

    return np.array(cam_img), np.array(cam_label), np.array(cam_cam)

for trial in range(1, 11):
    train_img, train_label, train_cam = preprocess_regdb(data_path, width=144, height=288, trial=trial)
    # np.save(data_path + 'train_img_trial{}.npy'.format(trial), np.array(train_img))
    # np.save(data_path + 'train_label_trial{}.npy'.format(trial), np.array(train_label))
    # np.save(data_path + 'train_cam_trial{}.npy'.format(trial), np.array(train_cam))
    for i in range(1, 3):
        imgs, labels, cams = split_client(i, train_img, train_label, train_cam)
        np.save(data_path + 'train_img_trial{}_client{}.npy'.format(trial, i), imgs)
        np.save(data_path + 'train_label_trial{}_client{}.npy'.format(trial, i), labels)
        np.save(data_path + 'train_cam_trial{}_client{}.npy'.format(trial, i), cams)

