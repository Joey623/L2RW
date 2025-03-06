import numpy as np
from PIL import Image
import os
import random
import re

data_path = '../data/LLCM/'
cameras = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6', 'cam7', 'cam8', 'cam9']

def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [data_path + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label
file_path_train = data_path + 'idx/train_vis.txt'
file_path_val = data_path + 'idx/train_nir.txt'

files_rgb, id_train = load_data(file_path_train)
files_ir, id_val = load_data(file_path_val)
id_train.extend(id_val)
files_rgb.extend(files_ir)

combined = list(zip(files_rgb, id_train))
combined.sort(key=lambda x: x[1])
files_rgb, id_train = zip(*combined)
files_rgb = list(files_rgb)
id_train = list(id_train)

unique_labels = sorted(set(id_train))
pid2label = {pid: idx for idx, pid in enumerate(unique_labels)}


# pid_container = set()
# for label in id_train:
#     pid = label
#     pid_container.add(pid)

# pid2label = {pid: label for label, pid in enumerate(pid_container)}
fix_image_width = 144
fix_image_height = 288

def read_imgs(train_image, train_label):
    train_img = []
    train_lbl = []
    train_cam = []
    for index, (image, label) in enumerate(zip(train_image, train_label)):
        img = Image.open(image)
        img = img.resize((fix_image_width, fix_image_height), Image.LANCZOS)
        pix_array = np.array(img)
        train_img.append(pix_array)
        pid = label
        pid = pid2label[pid]
        train_lbl.append(pid)
        cam = int(re.search(r'_c(\d+)_', image).group(1))
        train_cam.append(cam)
    return train_img, train_lbl, train_cam

train_img, train_label, train_cam = read_imgs(files_rgb, id_train)
# np.save(data_path + 'train_img.npy', np.array(train_img))
# np.save(data_path + 'train_label.npy', np.array(train_label))
# np.save(data_path + 'train_cam.npy', np.array(train_cam))

def split_client(cam_id, train_img, train_label, train_cam):
    # 找到所有符合指定摄像头ID的图像索引
    # cam_indices = [i for i, cam in enumerate(train_cam) if cam in cam_id]
    cam_indices = [i for i, cam in enumerate(train_cam) if cam == cam_id]

    # 提取符合条件的图像、标签和摄像头信息
    cam_img = [train_img[i] for i in cam_indices]
    cam_label = [train_label[i] for i in cam_indices]
    cam_cam = [train_cam[i] for i in cam_indices]

    return np.array(cam_img), np.array(cam_label), np.array(cam_cam)

for i in range(1, 10):
    imgs, labels, cams = split_client(i, train_img, train_label, train_cam)
    np.save(data_path + 'train_img_client{}.npy'.format(i), imgs)
    np.save(data_path + 'train_label_client{}.npy'.format(i), labels)
    np.save(data_path + 'train_cam_client{}.npy'.format(i), cams)