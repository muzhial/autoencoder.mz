import os
import glob

import numpy as np
import cv2
# import scipy.io as sio
# from skimage import io
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self,
                 data_dir,
                 preproc=None):
        self.data_dir = data_dir
        self.preproc = preproc
        self.files = self._load_files()

    def _load_files(self):
        files = []
        for f in glob.glob(self.data_dir + f'/*.png'):
            files.append(f)
        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]
        frame = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        return self.preproc(frame)


class VideoDataset(Dataset):
    """
    N x C x T x H x W
    """

    def __init__(self,
                 idx_root,
                 frame_root,
                 transform=None,
                 idx_suffix='.npy',
                 cfg=None):
        self.cfg = cfg
        self.idx_root = idx_root
        self.frame_root = frame_root
        self.transform = transform

        self.idx_file_list = self.load_idx_file(
            self.idx_root, idx_suffix, self.frame_root)

        self.idx_file_list = self.idx_file_list[1000:4000]

    def load_idx_file(self, idx_root, idx_suffix, frame_root):
        idx_file_list = []
        for idx_label_file in glob.glob(idx_root + f'/*{idx_suffix}'):
            idx_file_name = os.path.basename(idx_label_file)
            idx_array = np.load(idx_label_file)
            frame_dir = os.path.join(
                frame_root, os.path.splitext(idx_file_name)[0])
            frame_file_list = [n
                for n in os.listdir(frame_dir)
                    if os.path.splitext(n)[1] in self.cfg.SUFFIX]
            frame_file_list.sort()
            for row in idx_array:
                clip_file_list = []
                for col in row:
                    clip_file_list.append(
                        os.path.join(frame_dir, frame_file_list[col]))
                idx_file_list.append(clip_file_list)

        return idx_file_list

    def __len__(self):
        return len(self.idx_file_list)

    def __getitem__(self, index):
        clip_files = self.idx_file_list[index]
        frames = torch.stack(
            [self.transform(cv2.imread(frame, cv2.IMREAD_GRAYSCALE))
                for frame in clip_files],
            dim=1)
        return frames

# All video index files are in one dir.
# N x C x T x H x W

class VideoDatasetTest(Dataset):
    """
    N x C x T x H x W
    """

    def __init__(self,
                 idx_root,
                 frame_root,
                 transform=None,
                 idx_suffix='.npy',
                 cfg=None):
        self.cfg = cfg
        self.idx_root = idx_root
        self.frame_root = frame_root
        self.transform = transform

        self.idx_file_list = self.load_idx_file(
            self.idx_root, idx_suffix, self.frame_root)

        frame_start = 3000
        frame_end = frame_start + 16
        self.idx_file_list = self.idx_file_list[frame_start:frame_end]

    def load_idx_file(self, idx_root, idx_suffix, frame_root):
        idx_file_list = []
        for idx_label_file in glob.glob(idx_root + f'/*{idx_suffix}'):
            idx_file_name = os.path.basename(idx_label_file)
            idx_array = np.load(idx_label_file)
            frame_dir = os.path.join(
                frame_root, os.path.splitext(idx_file_name)[0])
            frame_file_list = [n
                for n in os.listdir(frame_dir)
                    if os.path.splitext(n)[1] in self.cfg.SUFFIX]
            frame_file_list.sort()
            for row in idx_array:
                clip_file_list = []
                for col in row:
                    clip_file_list.append(
                        os.path.join(frame_dir, frame_file_list[col]))
                idx_file_list.append(clip_file_list)

        return idx_file_list

    def __len__(self):
        return len(self.idx_file_list)

    def __getitem__(self, index):
        clip_files = self.idx_file_list[index]
        add_noise = True
        noise_frame_list = [3, 6, 8, 14]

        # frames = torch.stack(
        #     [self.transform(cv2.imread(frame, cv2.IMREAD_GRAYSCALE))
        #         for frame in clip_files],
        #     dim=1)

        frames = []
        for i, frame in enumerate(clip_files):
            img = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
            if add_noise and index in noise_frame_list:
                img[306:356, 306:356] = 0
                if True:
                    if index == 3:
                        cv2.imwrite(f'./results/res_out2/noise_{i}.png', img)
            frames.append(self.transform(img))
        frames = torch.stack(frames, dim=1)

        return frames
