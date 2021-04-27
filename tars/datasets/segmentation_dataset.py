import os
import bisect
import json
import numpy as np
import torch
from tars.base.dataset import Dataset
from PIL import Image
import time


class SegmentationDataset(Dataset):
    def __init__(self, type, splits_file=None, transforms=None):
        super().__init__(type, splits_file=splits_file)
        img_lens = [len(os.listdir(os.path.join(task_dir, self.conf.instance_mask_dir))) for task_dir in self.unique_tasks]
        self.cum_img_lens = np.cumsum(img_lens)
        self.transforms = transforms

    def __getitem__(self, idx):
        task_idx = bisect.bisect(self.cum_img_lens, idx)
        img_idx = idx - self.cum_img_lens[task_idx - 1] if task_idx > 0 else idx
        task_dir = self.unique_tasks[task_idx]
        img = self.get_img(task_dir, self.conf.high_res_img_dir, img_idx)
        target = self.get_tgt(task_dir, self.conf.target_dir, img_idx)
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return self.cum_img_lens[-1]
