import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.flow_util import dequantize_flow
from basicsr.utils.registry import DATASET_REGISTRY
import os.path as osp


@DATASET_REGISTRY.register()
class VideoRecurrentDataset(data.Dataset):
    def __init__(self, opt):
        super(VideoRecurrentDataset, self).__init__()
        self.opt = opt
        self.keys = []
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.num_frame = opt['num_frame'] #训练的帧数
        self.keys = self.get_keys(opt['meta_info_file'])
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

    @staticmethod
    def get_keys(meta_info_file):
        keys = []
        with open(meta_info_file, 'r') as fin:
            for line in fin:
                scene, total_frame_num, patch_num, _ = line.split()
                keys.extend([f"{scene}/{total_frame_num}/{patch_num}/{i}" for i in range(int(total_frame_num))])
        return keys


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, total_frame_num, patch_num, frame_name = self.keys[index].split('/') # key example: 00/100/12/0001.png

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        total_frame_num = int(total_frame_num)
        if start_frame_idx > total_frame_num - self.num_frame:
            start_frame_idx = random.randint(0, total_frame_num - self.num_frame)
        end_frame_idx = start_frame_idx + self.num_frame
        neighbor_list = list(range(start_frame_idx + 1, end_frame_idx + 1))

        # select the patch in the same position
        patch_idx = '_s{:03d}'.format(random.randrange(0, int(patch_num))) if int(patch_num) > 1 else ''

        img_lqs = []
        img_gts = []
        for i in neighbor_list:
            img_lq_path = osp.join(self.lq_root, clip_name, f'{i:08d}' + patch_idx + '.png')
            img_gt_path = osp.join(self.gt_root, clip_name, f'{i:08d}' + patch_idx + '.png')

            # print(img_lq_path)

            # get LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # randomly crop
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)
