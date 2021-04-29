import os
import bisect
import json
import numpy as np
import torch
from tars.base.dataset import Dataset
from PIL import Image
import time


class InstanceSegmentationDataset(Dataset):
    def __init__(self, type, splits_file=None, transforms=None, preprocess=False):
        super().__init__(type, splits_file=splits_file)
        img_lens = [len(os.listdir(os.path.join(task_dir, self.conf.instance_mask_dir))) for task_dir in self.unique_tasks]
        self.cum_img_lens = np.cumsum(img_lens)
        self.transforms = transforms
        self.preprocess = preprocess

    def __getitem__(self, idx):
        task_idx = bisect.bisect(self.cum_img_lens, idx)
        img_idx = idx - self.cum_img_lens[task_idx - 1] if task_idx > 0 else idx
        task_dir = self.unique_tasks[task_idx]
        img = self.get_img(task_dir, self.conf.high_res_img_dir, img_idx)

        if self.preprocess:
            mask_img = self.get_img(task_dir, self.conf.instance_mask_dir, img_idx)
            return img, mask_img

        target = self.get_target(task_dir, self.conf.instance_target_dir, img_idx, idx)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return self.cum_img_lens[-1]

    def get_target(self, task_dir, target_dir, img_idx, idx):
        targets = os.listdir(os.path.join(task_dir, target_dir))
        obj_data = np.load(os.path.join(task_dir, target_dir, sorted(targets)[img_idx]))
        boxes, labels, masks = obj_data['arr_0'], obj_data['arr_1'], obj_data['arr_2']
        # convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(masks),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target
