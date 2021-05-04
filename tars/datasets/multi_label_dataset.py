import torch
from tars.base.dataset import Dataset
from tars.datasets.segmentation_dataset import SegmentationDataset
from tars.config.base.dataset_config import DatasetConfig


class MultiLabelDataset(Dataset):
    def __init__(self, type, img_transforms, splits_file=None):
        self.seg_dataset = SegmentationDataset(type, img_transforms, splits_file)
        self.num_classes = len(DatasetConfig.objects_list)

    def __getitem__(self, idx):
        img, gt = self.seg_dataset[idx]
        labels = torch.zeros(self.num_classes, dtype=int)
        labels[gt.unique()] = 1

        return img, labels

    def __len__(self):
        return len(self.seg_dataset)
