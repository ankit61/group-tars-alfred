import functools
import os
import bisect
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tars.base.dataset import Dataset
from tars.config.base.dataset_config import DatasetConfig


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
        gt_im = self.get_img(task_dir, self.conf.instance_mask_dir, im_idx) # already cleaned in augment trajectories

        gt_im = torch.tensor(np.array(gt_im), dtype=int)

        assert rgb_im.shape[-2:] == gt_im.shape

        # im_size = tuple(rgb_im.shape[2:])
        # gt_im = self.clean_raw_gt(gt_im, task_dir, im_size)

        return rgb_im, gt_im

    def __len__(self):
        return self.cum_img_lens[-1]

    #def clean_raw_gt(self, gt_im, task_dir, im_size):
    #    with open(os.path.join(task_dir, self.conf.aug_traj_file), 'r') as f:
    #        color_data = json.load(f)['scene']['color_to_object_type']

    #    clean_gt = SegmentationDataset.clean_gt_color_data(gt_im, color_data)
    #    pil_img = transforms.Resize(img_size)(Image.fromarray(clean_gt))

    #    return torch.tensor(np.array(pil_img), dtype=int)

    @classmethod
    def clean_gt_color_data(cls, gt_im, color_data):
        # class method so can be accessed by augment_trajectories
        out = gt_im.copy()
        bg_mask = np.ones_like(out[:, :, 0], dtype=bool)
        for k in color_data:
            obj_idx = DatasetConfig.objects_vocab.word2index(color_data[k]['objectType'])
            k = tuple(map(int, k.strip('()').split(', ')))
            mask = (out[:, :, 0] == k[0]) & (out[:, :, 1] == k[1]) & (out[:, :, 2] == k[2])
            bg_mask = (bg_mask & (~mask))
            out[mask] = [obj_idx] * 3

        out[bg_mask] = [DatasetConfig.objects_vocab.word2index(DatasetConfig.object_na)] * 3

        return out[:, :, 0]
