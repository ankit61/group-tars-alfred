import os
import bisect
import json
import numpy as np
import torch
from tars.base.dataset import Dataset
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, type, splits_file=None):
        super().__init__(type, splits_file=splits_file)
        img_lens = [len(os.listdir(os.path.join(t, self.conf.instance_mask_dir))) for t in self.unique_tasks]
        self.cum_img_lens = np.cumsum(img_lens)

    def __getitem__(self, idx):
        task_idx = bisect.bisect(self.cum_img_lens, idx)
        im_idx = idx - self.cum_img_lens[task_idx - 1] if task_idx > 0 else idx
        task_dir = self.get_task(task_idx)[0]

        rgb_im = self.get_img(task_dir, self.conf.high_res_img_dir, im_idx)
        gt_im = self.get_img(task_dir, self.conf.instance_mask_dir, im_idx)

        out = np.array(gt_im)

        with open(os.path.join(task_dir, self.conf.aug_traj_file), 'r') as f:
            color_data = json.load(f)['scene']['color_to_object_type'] 
        
        bg_mask = np.ones_like(out[:, :, 0], dtype=bool)
        boxes = []
        labels = []
        masks = []
        for k in color_data:
            # get object mask
            obj_idx = self.conf.objects_vocab.word2index(color_data[k]['objectType'])
            k = tuple(map(int, k.strip('()').split(', ')))
            obj_mask = (out[:, :, 0] == k[2]) & (out[:, :, 1] == k[1]) & (out[:, :, 2] == k[0])
            
            # get object bounding box coordinates
            pos = np.where(obj_mask)
            if len(pos[0]) + len(pos[1]) > 0:
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(obj_idx)
                masks.append(obj_mask)

        # convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.Tensor(labels)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.Tensor([idx])
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

        return rgb_im, target

    def __len__(self):
        return self.cum_img_lens[-1]


