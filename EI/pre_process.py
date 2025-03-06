import numpy as np
from PIL import Image
import os


def read_sysu_imgs(train_image, fix_image_width, fix_image_height, pid2label, add):
    train_img = []
    train_label = []
    for img_path in train_image:
        # img
        img = Image.open(img_path)
        img = img.resize((fix_image_width, fix_image_height), Image.LANCZOS)
        pix_array = np.array(img)

        train_img.append(pix_array)

        # label
        pid = int(img_path[-13:-9])
        pid = pid2label[pid] + add
        train_label.append(pid)
    return np.array(train_img), np.array(train_label)


# preprocess dataset
# add: avoid different datasets have the same id
def preprocess_sysu(data_path, protocol, width=144, height=288, add=0):
    print("Strat Preprocessing SYSU Dataset...")
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

    id_train.extend(id_val)

    files = []
    for id in sorted(id_train):
        for cam in cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files.extend(new_files)

    # relabel
    pid_container = set()
    for img_path in files:
        pid = int(img_path[-13:-9])
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}
    fix_image_width = width
    fix_image_height = height
    save_path = data_path + protocol + '/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    train_img, train_label = read_sysu_imgs(files, fix_image_width, fix_image_height, pid2label, add)
    # rgb images
    np.save(save_path + 'train_img.npy', train_img)
    np.save(save_path + 'train_label.npy', train_label)

    print("Preprocess SYSU Done!")

    return 0


def load_llcm(input_data_path, data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [data_path + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label


def read_llcm_imgs(train_image, train_label, fix_image_width, fix_image_height, pid2label, add=0):
    train_img = []
    train_lbl = []
    for index, (image, label) in enumerate(zip(train_image, train_label)):
        img = Image.open(image)
        img = img.resize((fix_image_width, fix_image_height), Image.LANCZOS)
        pix_array = np.array(img)

        train_img.append(pix_array)
        pid = int(label)
        pid = pid2label[pid] + add
        train_lbl.append(pid)
    return np.array(train_img), np.array(train_lbl)


def preprocess_llcm(data_path, protocol, width=144, height=288, add=0):
    print("Strat Preprocessing LLCM Dataset...")
    file_path_train = data_path + 'idx/train_vis.txt'
    file_path_val = data_path + 'idx/train_nir.txt'
    files_rgb, id_train = load_llcm(file_path_train, data_path)
    files_ir, id_val = load_llcm(file_path_val, data_path)
    id_train.extend(id_val)
    files_rgb.extend(files_ir)

    combined = list(zip(files_rgb, id_train))
    combined.sort(key=lambda x: x[1])
    files_rgb, id_train = zip(*combined)
    files_rgb = list(files_rgb)
    id_train = list(id_train)

    unique_labels = sorted(set(id_train))
    pid2label = {pid: idx for idx, pid in enumerate(unique_labels)}
    fix_image_width = width
    fix_image_height = height

    save_path = data_path + protocol + '/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    train_img, train_label = read_llcm_imgs(files_rgb, id_train, fix_image_width, fix_image_height, pid2label, add)
    np.save(save_path + 'train_img.npy', train_img)
    np.save(save_path + 'train_label.npy', train_label)
    print("Preprocess LLCM Done!")


def load_regdb_data(input_data_path, data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [data_path + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label


def preprocess_regdb(data_path, protocol, width=144, height=288, trial=1, add=0):
    print("Strat Preprocessing RegDB Dataset...")
    train_color_list = data_path + 'idx/train_visible_{}'.format(trial) + '.txt'
    train_thermal_list = data_path + 'idx/train_thermal_{}'.format(trial) + '.txt'

    rgb_img, rgb_lbl = load_regdb_data(train_color_list, data_path)
    ir_img, ir_lbl = load_regdb_data(train_thermal_list, data_path)
    rgb_img.extend(ir_img)
    rgb_lbl.extend(ir_lbl)

    combined = list(zip(rgb_img, rgb_lbl))
    combined.sort(key=lambda x: x[1])
    rgb_img, rgb_lbl = zip(*combined)
    rgb_img = list(rgb_img)
    rgb_lbl = list(rgb_lbl)

    save_path = data_path + protocol + '/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    unique_labels = sorted(set(rgb_lbl))
    pid2label = {pid: idx for idx, pid in enumerate(unique_labels)}
    train_img = []
    train_lbl = []
    for index, (image, label) in enumerate(zip(rgb_img, rgb_lbl)):
        img = Image.open(image)
        img = img.resize((width, height), Image.LANCZOS)
        pix_array = np.array(img)
        train_img.append(pix_array)
        pid = label
        pid = pid2label[pid] + add
        train_lbl.append(pid)

    np.save(save_path + 'train_img.npy', np.array(train_img))
    np.save(save_path + 'train_label.npy', np.array(train_lbl))

    print("Preprocess RegDB Done!")

    return 0


if __name__ == '__main__':
    sysu_data_path = '../data/SYSU-MM01/'
    regdb_data_path = '../data/RegDB/'
    llcm_data_path = '../data/LLCM/'
    protocol = "R_S_to_L"
    preprocess_sysu(sysu_data_path, protocol, width=144, height=288, add=206)
    # preprocess_llcm(llcm_data_path, protocol, width=144, height=288, add=0)
    preprocess_regdb(regdb_data_path, protocol, width=144, height=288, trial=1)

    protocol = "R_L_to_S"
    # preprocess_sysu(sysu_data_path, protocol, width=144, height=288, add=0)
    preprocess_llcm(llcm_data_path, protocol, width=144, height=288, add=206)
    preprocess_regdb(regdb_data_path, protocol, width=144, height=288, trial=1)

    protocol = "L_S_to_R"
    preprocess_sysu(sysu_data_path, protocol, width=144, height=288, add=713)
    preprocess_llcm(llcm_data_path, protocol, width=144, height=288, add=0)
    # preprocess_regdb(regdb_data_path, protocol, width=144, height=288, trial=1)
