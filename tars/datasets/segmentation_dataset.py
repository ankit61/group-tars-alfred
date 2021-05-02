import functools
import os
import bisect
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tars.base.dataset import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, type, img_transforms, splits_file=None):
        super().__init__(type, splits_file=splits_file)
        self.img_transforms = img_transforms
        img_lens = [len(os.listdir(os.path.join(t, self.conf.instance_mask_dir))) for t in self.unique_tasks]
        self.cum_img_lens = np.cumsum(img_lens)

    # @functools.lru_cache() # unbounded caching is too much
    def __getitem__(self, idx):
        task_idx = bisect.bisect(self.cum_img_lens, idx)
        im_idx = idx - self.cum_img_lens[task_idx - 1] if task_idx > 0 else idx
        task_dir = self.unique_tasks[task_idx]

        rgb_im = self.img_transforms(self.get_img(task_dir, self.conf.high_res_img_dir, im_idx))
        gt_im = self.get_img(task_dir, self.conf.instance_mask_dir, im_idx)

        im_size = tuple(rgb_im.shape[2:])
        gt_im = self.clean_raw_gt(gt_im, task_dir, im_size)

        return rgb_im, gt_im

    def __len__(self):
        return self.cum_img_lens[-1]

    def clean_raw_gt(self, gt_im, task_dir, img_size):
        with open(os.path.join(task_dir, self.conf.aug_traj_file), 'r') as f:
            color_data = json.load(f)['scene']['color_to_object_type']

        out = np.array(gt_im)
        bg_mask = np.ones_like(out[:, :, 0], dtype=bool)
        for k in color_data:
            obj_idx = self.conf.objects_vocab.word2index(color_data[k]['objectType'])
            k = tuple(map(int, k.strip('()').split(', ')))
            mask = (out[:, :, 0] == k[2]) & (out[:, :, 1] == k[1]) & (out[:, :, 2] == k[0])
            bg_mask = (bg_mask & (~mask))
            out[mask] = [obj_idx] * 3

        out[bg_mask] = [self.conf.objects_vocab.word2index(self.conf.object_na)] * 3

        pil_img = transforms.Resize(img_size)(Image.fromarray(out[:, :, 0]))

        return torch.tensor(np.array(pil_img), dtype=int)
