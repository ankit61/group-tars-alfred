from torch.utils.data.dataset import T
import torch
from tars.datasets.segmentation_dataset import SegmentationDataset


class MultiLabelDataset(Dataset):
    def __init__(self, type, img_transforms, splits_file=None):
        self.seg_dataset = SegmentationDataset(type, img_transforms, splits_file)

    def __getitem__(self, idx):
        img, gt = self.seg_dataset[idx]
        labels = torch.zeros(gt.shape[0], self.num_classes)
        for i in range(gt.shape[0]):
            labels[i][gt[i].unique()] = 1

    def __len__(self):
        return len(self.seg_dataset)
