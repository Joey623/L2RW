from __future__ import print_function, absolute_import
import os.path as osp
import torch.utils.data as data
import numpy as np
import random
import math


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

class VCMTest(object):
    root = '/Volumes/Yan/Joey/code/Dataset/VCM-HITSZ/'
    # training data
    # train_name_path = osp.join(root,'info/train_name.txt')
    # track_train_info_path = osp.join(root,'info/track_train_info.txt')

    # testing data
    test_name_path = osp.join(root,'info/test_name.txt')
    track_test_info_path = osp.join(root,'info/track_test_info.txt')
    query_IDX_path = osp.join(root,'info/query_IDX.txt')

    def __init__(self, min_seq_len=6):
        self._check_before_run()
        # just for test
        test_names = self._get_names(self.test_name_path)
        track_test = self._get_tracks(self.track_test_info_path)
        query_IDX = self._get_query_idx(self.query_IDX_path)
        query_IDX -= 1

        track_query = track_test[query_IDX,:]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX,:]

        # visible to infrared
        gallery_IDX_1 = self._get_query_idx(self.query_IDX_path)
        gallery_IDX_1 -= 1
        track_gallery_1 = track_test[gallery_IDX_1,:]
        query_IDX_1 = [j for j in range(track_test.shape[0]) if j not in gallery_IDX_1]
        track_query_1 = track_test[query_IDX_1,:]

        query, num_query_tracklets, num_query_pids, num_query_imgs = \
            self._process_data_test(test_names, track_query, relabel=False, min_seq_len=min_seq_len)
        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
            self._process_data_test(test_names, track_gallery, relabel=False, min_seq_len=min_seq_len)
        query_1, num_query_tracklets_1, num_query_pids_1, num_query_imgs_1 = \
            self._process_data_test(test_names, track_query_1, relabel=False, min_seq_len=min_seq_len)
        gallery_1, num_gallery_tracklets_1, num_gallery_pids_1, num_gallery_imgs_1 = \
            self._process_data_test(test_names, track_gallery_1, relabel=False, min_seq_len=min_seq_len)
        
        print("=> VCM-HITSZ Test data Information")
        print("---------------------------------")
        print("subset      | # ids | # tracklets")
        print("---------------------------------")
        print("query       | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("gallery     | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))

        # infrared to visible
        self.query = query
        self.gallery = gallery
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_query_tracklets = num_query_tracklets
        self.num_gallery_tracklets = num_gallery_tracklets

        # visible to infrared
        self.query_1 = query_1
        self.gallery_1 = gallery_1
        self.num_query_pids_1 = num_query_pids_1
        self.num_gallery_pids_1 = num_gallery_pids_1
        self.num_query_tracklets_1 = num_query_tracklets_1
        self.num_gallery_tracklets_1 = num_gallery_tracklets_1
    
    def _check_before_run(self):
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
    
    def _get_names(self, path):
        "get image names, return name list"
        names = []
        with open(path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names
    def _get_tracks(self, path):
        """get tracks file"""
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
    # need to check
    def _get_query_idx(self, path):
        with open(path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                new_line.split(' ')
                tmp = new_line.split(' ')[0:]
                tmp = list(map(int, tmp))
                idxs = tmp
        idxs = np.array(idxs)
        return idxs
    
    def _process_data_test(self, names, meta_data, relabel=False, min_seq_len=0):
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 3].tolist()))
        num_pids = len(pid_list)
        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []
        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            m,start_index,end_index,pid,camid = data
            if relabel: pid = pid2label[pid]
            img_names = names[start_index-1:end_index]
            img_paths = [osp.join(self.root,decoder_pic_path(img_name)) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths,pid,camid))
                num_imgs_per_tracklet.append(len(img_paths))
        num_tracklets = len(tracklets)
        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet



class VCMTrain(object):
    root = '/Volumes/Yan/Joey/code/Dataset/VCM-HITSZ/'
    # training data
    train_name_path = osp.join(root,'info/train_name.txt')
    track_train_info_path = osp.join(root,'info/track_train_info.txt')

    def __init__(self, client_id, min_seq_len=6):
        self._check_before_run()
        self.client_id = client_id
        train_names = self._get_names(self.train_name_path)
        track_train = self._get_tracks(self.track_train_info_path)
        train, num_train_tracklets, num_train_pids, label = \
            self._process_data_train(train_names, track_train, relabel=True, min_seq_len=min_seq_len, client_id=self.client_id)
        # print(train)
        self.train = train
        self.label = label
        print("=> VCM-HITSZ Train data Information".format(client_id))
        print("---------------------------------")
        print("client{}      | # ids | # tracklets".format(client_id))
        print("---------------------------------")
        print("train    | {:5d} | {:8d}".format(num_train_pids,num_train_tracklets))
        print("label    | {:5d}".format(len(np.unique(label))))

    def _check_before_run(self):
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))

    def _get_names(self, path):
        "get image names, return name list"
        names = []
        with open(path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names
    def _get_tracks(self, path):
        """get tracks file"""
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
    def _process_data_train(self, names, meta_data, relabel=False, min_seq_len=0, client_id=1):
        assert 1 <= client_id <= 6, "client_id out of range, should be 1-6"
        num_tracklets = meta_data.shape[0]
        # print(num_tracklets)
        pid_list = list(set(meta_data[:, 3].tolist()))
        num_pids = len(pid_list)
        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []
        label = []
        # meta_data的信息分别是：模态，开始帧，结束帧，pid，camid
        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            m,start_index,end_index,pid,camid = data
            if camid == client_id:
                if relabel: pid = pid2label[pid]
                img_names = names[start_index-1:end_index]
                img_paths = [osp.join(self.root,decoder_pic_path(img_name)) for img_name in img_names]
                if len(img_paths) >= min_seq_len:
                    img_paths = tuple(img_paths)
                    label.append(pid)
                    tracklets.append((img_paths,pid,camid))
                    num_imgs_per_tracklet.append(len(img_paths))
            else:
                continue

        num_tracklets = len(tracklets)
        return tracklets, num_tracklets, num_pids, label



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
class VideoDataset_train(data.Dataset):
    sample_methods = ['evenly', 'random', 'all']
    def __init__(self, dataset, client_id, seq_len=6, sample='evenly', transform=None, colorIndex=None):
        self.dataset = dataset
        self.client_id = client_id
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.index = colorIndex
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[self.index[index]]
        num = len(img_paths)
        S = self.seq_len
        sample_clip = []
        frame_indices = list(range(num))
        if num < S:
            strip = list(range(num)) + [frame_indices[-1]] * (S - num)
            for s in range(S):
                pool = strip[s * 1:(s + 1) * 1]
                sample_clip.append(list(pool))
        else:
            inter_val = math.ceil(num / S)
            strip = list(range(num)) + [frame_indices[-1]] * (S * inter_val - num)
            for s in range(S):
                pool = strip[s * inter_val:(s + 1) * inter_val]
                sample_clip.append(list(pool))
        sample_clip = np.array(sample_clip)
        
        if self.sample == 'random':
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))
            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array([indices])
            imgs = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            return imgs, pid, camid
        elif self.sample == 'video_train':
            idx = np.random.randint(sample_clip.shape[1], sample_clip.shape[0])
            number = sample_clip[np.arange(len(sample_clip)), idx]
            imgs = []
            for index in number:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            return imgs, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))


class VideoDataset_test(data.Dataset):
    """
    Batch shape: (batch, seq_len, c, h, w)
    """
    sample_methods = ['evenly', 'random', 'all']
    def __init__(self, dataset, seq_len=6, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num))
        if num < S:  
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (S - num)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num / S)
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)

        if self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index=0
            frame_indices = range(num)
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            last_seq = list(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            imgs_list=[]
            for indices in indices_list:
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
 
                    img = np.array(img)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
              
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list) 
            return imgs_array, pid, camid

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            num_ir = len(img_paths)
            frame_indices = range(num_ir)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
            imgs_ir = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)

                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid, camid

        if self.sample == 'video_test':
            number = sample_clip_ir[:, 0]
            imgs_ir = []
            for index in number:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)

                img = np.array(img)
                if self.transform is not None: 
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))


if __name__ == '__main__':
    dataset = VCMTrain(client_id=2)

