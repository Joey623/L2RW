import numpy as np
from PIL import Image
import random
import os


data_path = '../data/SYSU-MM01/'

cameras = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6']
file_path_train = os.path.join(data_path, 'exp/train_id.txt')
file_path_val = os.path.join(data_path, 'exp/val_id.txt')


with open(file_path_train, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_train = ["%04d" % x for x in ids]


with open(file_path_val, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_val = ["%04d" % x for x in ids]

# combine images
id_train.extend(id_val)
files = []
for id in sorted(id_train):
    for cam in cameras:
        img_dir = os.path.join(data_path, cam, id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            files.extend(new_files)


pid_container = set()
for img_path in files:
    pid = int(img_path[-13:-9])
    pid_container.add(pid)
pid2label = {pid: label for label, pid in enumerate(pid_container)}
fix_image_width = 144
fix_image_height = 288


def read_imgs(train_image):
    train_img = []
    train_label = []
    train_cam = []
    for img_path in train_image:
        # img
        img = Image.open(img_path)
        img = img.resize((fix_image_width, fix_image_height), Image.LANCZOS)
        pix_array = np.array(img)

        train_img.append(pix_array)

        # label
        pid = int(img_path[-13:-9])
        pid = pid2label[pid]
        camid = int(img_path[-15])
        train_label.append(pid)
        train_cam.append(camid)
    return train_img, train_label, train_cam

# training images
train_img, train_label, train_cam = read_imgs(files)
# global
# np.save(data_path + 'train_img.npy', np.array(train_img))
# np.save(data_path + 'train_label.npy', np.array(train_label))
# np.save(data_path + 'train_cam.npy', np.array(train_cam))

# def split_client(cam_id, train_img, train_label, train_cam):
#     pid_container = set()
#     cam_img = []
#     cam_label = []
#     cam_cam = []
#     for img, label, cam in zip(train_img, train_label, train_cam):
#         if cam == cam_id:
#             # cam_img.append(img)
#             # cam_label.append(label)
#             # cam_cam.append(cam)
#             pid_container.add(label)
#     pid2label = {pid: label for label, pid in enumerate(pid_container)}
#     for img, label, cam in zip(train_img, train_label, train_cam):
#         if cam == cam_id:
#             cam_img.append(img)
#             cam_label.append(pid2label[int(label)])
#             cam_cam.append(cam)
#
#
#     return np.array(cam_img), np.array(cam_label), np.array(cam_cam)
def split_client(cam_id, train_img, train_label, train_cam):
    # 找到所有符合指定摄像头ID的图像索引
    cam_indices = [i for i, cam in enumerate(train_cam) if cam == cam_id]

    # 提取符合条件的图像、标签和摄像头信息
    cam_img = [train_img[i] for i in cam_indices]
    cam_label = [train_label[i] for i in cam_indices]
    cam_cam = [train_cam[i] for i in cam_indices]

    return np.array(cam_img), np.array(cam_label), np.array(cam_cam)

for i in range(1, 7):
    imgs, labels, cams = split_client(i, train_img, train_label, train_cam)
    np.save(data_path + 'train_img_client{}.npy'.format(i), imgs)
    np.save(data_path + 'train_label_client{}.npy'.format(i), labels)
    np.save(data_path + 'train_cam_client{}.npy'.format(i), cams)
