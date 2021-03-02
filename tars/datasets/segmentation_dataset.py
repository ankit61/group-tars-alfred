import os
import bisect
import json
import numpy as np
from vocab import Vocab
import vocab
from tars.base.dataset import Dataset
from tars.alfred.gen.constants import OBJECTS
from PIL import Image


class SegmentationDataset(Dataset):
    def __init__(self, type, splits_file=None):
        super().__init__(type, splits_file=splits_file)
        img_lens = [len(os.listdir(os.path.join(t, self.conf.instance_mask_dir))) for t, _ in self.tasks()]
        self.len = sum(img_lens)
        self.cum_img_lens = np.cumsum(img_lens)
        self.seg_objects = Vocab(OBJECTS)

    def __getitem__(self, idx):
        task_idx = bisect.bisect_left(self.cum_img_lens, idx)
        im_idx = idx - self.cum_img_lens[task_idx]
        task_dir = self.get_task(task_idx)[0]

        rgb_im = self.get_img(task_dir, self.conf.high_res_img_dir, im_idx)
        gt_im = self.get_img(task_dir, self.conf.instance_mask_dir, im_idx)

        gt_im = self.process_raw_gt(gt_im, task_dir)

        return rgb_im, gt_im

    def __len__(self):
        return self.len

    def clean_raw_gt(self, gt_im, task_dir):
        with open(os.path.join(task_dir, self.conf.aug_traj_file), 'r') as f:
            color_data = json.load(f)['scene']['color_to_object_type']

        out = np.array(gt_im)
        for k in color_data:
            obj_idx = self.seg_objects.word2index(color_data[k])
            k = tuple(map(int, k.strip('()').split(', ')))
            mask = (out[:, :, 0] == k[0]) & (out[:, :, 1] == k[1]) & (out[:, :, 2] == k[2])
            out[mask] = [obj_idx] * 3

        return Image.fromarray(out[:, :, 0])
