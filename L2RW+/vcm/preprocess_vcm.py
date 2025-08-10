from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np
import os.path as osp
import torch
import torch.utils.data as data
import random
import math

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def decoder_pic_path(fname):
    base = fname[0:4]
    modality = fname[5]
    if modality == '1' :
        modality_str = 'ir'
    else:
        modality_str = 'rgb'
    T_pos = fname.find('T')
    D_pos = fname.find('D')
    F_pos = fname.find('F')
    camera = fname[D_pos:T_pos]
    picture = fname[F_pos+1:]
    path = base + '/' + modality_str + '/' + camera + '/' + picture
    return path

def get_train_names(path):
    names = []
    with open(path, 'r') as f:
        for line in f:
            new_line = line.rstrip()
            names.append(new_line)
    return names

def get_train_tracks(path):
    names = []
    with open(path, 'r') as f:
        for line in f:
            new_line = line.rstrip()
            new_line.split(' ')
            tmp = new_line.split(' ')[0:]
            tmp = list(map(int, tmp))
            names.append(tmp)
    names = np.array(names)
    return names

def read_tracklet_imgs(tracklet):
    tracklet_list = []
    for path in tracklet:
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((144, 288), Image.LANCZOS)
            img = np.array(img)
            tracklet_list.append(img)
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(path))
            pass
    tracklet_list_array = np.stack(tracklet_list, axis=0)
    return tracklet_list_array

def process_train_tracklet(tracklet):
    process_tracklet = []
    for img_paths_tuple, pid, camid in tracklet:
        imgs = []
        for img_path in img_paths_tuple:
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                img = img.resize((144, 288), Image.LANCZOS)
                img = np.array(img)
                imgs.append(img)
            else:
                print("Image path does not exist: {}".format(img_path))
                continue
        if len(imgs) == len(img_paths_tuple):
            process_tracklet.append((imgs, pid, camid))
    return process_tracklet


def process_train_data(names, meta_data, relabel=False, min_seq_len=6):
    num_tracklets = meta_data.shape[0]
    # print('num_tracklets:', num_tracklets)
    pid_list = list(set(meta_data[:, 3].tolist()))
    # num_pids = len(pid_list)
    if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
    tracklets = []
    num_imgs_per_tracklet = []
    # label = []
    # cam = []
    # imgs = []
    for tracklet_idx in range(num_tracklets):
        data = meta_data[tracklet_idx,...]
        m,start_index,end_index,pid,camid = data
        if relabel: pid = pid2label[pid]
        img_names = names[start_index-1:end_index]
        img_paths = [osp.join(root,'Train',decoder_pic_path(img_name)) for img_name in img_names]
        if len(img_paths) >= min_seq_len:
            img_paths = tuple(img_paths)
            # label.append(pid)
            tracklets.append((img_paths,pid,camid))

            # cam.append(camid)
            num_imgs_per_tracklet.append(len(img_paths))
    
    # num_tracklets = len(tracklets)
    process_tracklet = process_train_tracklet(tracklets)
    return process_tracklet

def split_client(cam_id, tracklets, label, cam):
    cam_indices = [i for i, x in enumerate(cam) if x == cam_id]
    client_tracklets = [tracklets[i] for i in cam_indices]
    client_label = [label[i] for i in cam_indices]
    client_cam = [cam[i] for i in cam_indices]

    return client_tracklets, np.array(client_label), np.array(client_cam)


def get_test_names(path):
    names = []
    with open(path, 'r') as f:
        for line in f:
            new_line = line.rstrip()
            names.append(new_line)
    return names

def get_test_tracks(path):
    names = []
    with open(path, 'r') as f:
        for line in f:
            new_line = line.rstrip()
            new_line.split(' ')
            tmp = new_line.split(' ')[0:]
            tmp = list(map(int, tmp))
            names.append(tmp)
    names = np.array(names)
    return names

def get_query_idx(path):
    with open(path, 'r') as f:
        for line in f:
            new_line = line.rstrip()
            new_line.split(' ')
            tmp = new_line.split(' ')[0:]
            tmp = list(map(int, tmp))
            idxs = tmp
    idxs = np.array(idxs)
    return idxs



def process_test_data(names, meta_data, relabel=False, min_seq_len=6):
    num_tracklets = meta_data.shape[0]
    pid_list = list(set(meta_data[:, 3].tolist()))
    num_pids = len(pid_list)
    if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
    tracklets = []
    num_imgs_per_tracklet = []
    label = []
    cam = []
    for tracklet_idx in range(num_tracklets):
        data = meta_data[tracklet_idx,...]
        m,start_index,end_index,pid,camid = data
        if relabel: pid = pid2label[pid]
        img_names = names[start_index-1:end_index]
        img_paths = [osp.join(root,'Test',decoder_pic_path(img_name)) for img_name in img_names]
        if len(img_paths) >= min_seq_len:
            imgs = read_tracklet_imgs(img_paths)
            # img_paths = tuple(img_paths)
            # tracklets.append((img_paths,pid,camid))
            # num_imgs_per_tracklet.append(len(img_paths))
            label.append(pid)
            cam.append(camid)
    num_tracklets = len(tracklets)
    # return np.array(tracklets), np.array(num_tracklets), np.array(num_pids), np.array(num_imgs_per_tracklet)
    return imgs, np.array(label), np.array(cam)

def process_testset(root, save_path):
    test_name_path = osp.join(root,'info/test_name.txt')
    track_test_info_path = osp.join(root,'info/track_test_info.txt')
    query_IDX_path = osp.join(root,'info/query_IDX.txt')
    test_names = get_test_names(test_name_path)
    track_test = get_test_tracks(track_test_info_path)
    query_IDX = get_query_idx(query_IDX_path)
    query_IDX -= 1
    track_query = track_test[query_IDX, :]
    gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
    track_gallery = track_test[gallery_IDX, :]

    # visible to infrared
    gallery_IDX_1 = get_query_idx(query_IDX_path)
    gallery_IDX_1 -= 1
    track_gallery_1 = track_test[gallery_IDX_1, :]
    query_IDX_1 = [j for j in range(track_test.shape[0]) if j not in gallery_IDX_1]
    track_query_1 = track_test[query_IDX_1, :]

    query, query_label, query_cam = process_test_data(test_names, track_query, relabel=False, min_seq_len=6)
    gallery, gallery_label, gallery_cam = process_test_data(test_names, track_gallery, relabel=False, min_seq_len=6)
    query_1, query_label_1, query_cam_1 = process_test_data(test_names, track_query_1, relabel=False, min_seq_len=6)
    gallery_1, gallery_label_1, gallery_cam_1 = process_test_data(test_names, track_gallery_1, relabel=False, min_seq_len=6)

    np.save(save_path + 't2v_query.npy', query)
    np.save(save_path + 't2v_query_label.npy', query_label)
    np.save(save_path + 't2v_query_cam.npy', query_cam)
    np.save(save_path + 't2v_gallery.npy', gallery)
    np.save(save_path + 't2v_gallery_label.npy', gallery_label)
    np.save(save_path + 't2v_gallery_cam.npy', gallery_cam)
    np.save(save_path + 'v2t_query.npy', query_1)
    np.save(save_path + 'v2t_query_label.npy', query_label_1)
    np.save(save_path + 'v2t_query_cam.npy', query_cam_1)
    np.save(save_path + 'v2t_gallery.npy', gallery_1)
    np.save(save_path + 'v2t_gallery_label.npy', gallery_label_1)
    np.save(save_path + 'v2t_gallery_cam.npy', gallery_cam_1)
    print("test data process dome")

    return 0




if __name__ == '__main__':
    root = '/Volumes/Yan/Joey/code/Dataset/VCM-HITSZ/'
    save_path = root + 'process/'
    # process npy files

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_name_path = osp.join(root,'info/train_name.txt')
    track_train_info_path = osp.join(root,'info/track_train_info.txt')
    train_names = get_train_names(train_name_path)
    track_train = get_train_tracks(track_train_info_path)
    print("start to process train data")
    train = process_train_data(train_names, track_train, relabel=True, min_seq_len=6)
    np.save(save_path + 'train.npy', train, allow_pickle=True)
    print(len(train))

    # print(len(train[0]))
    # print(len(label))
    # print(len(cam))
    # print("train data process done")
    # print("="*30)
    # print("start to split train data according to camera")
    # for i in range(1, 7):
    #     client_tracklets, client_label, client_cam = split_client(i, train, label, cam)

    #     np.save(save_path + 'train_tracklet_client{}.npy'.format(i), client_tracklets)
    #     np.save(save_path + 'train_label_client{}.npy'.format(i), client_label)
    #     np.save(save_path + 'train_cam_client{}.npy'.format(i), client_cam)
    # print("split done")

    # print('='*30)
    # print("start to process test data")
    # process_testset(root, save_path)
    # print("test data process done")
    
    





















