import os
import json
from enum import Enum
from torch.utils import data
from PIL import Image
from tars.base.configurable import Configurable


class DatasetType(Enum):
    TRAIN = 'train'
    VAL_SEEN = 'valid_seen'
    VAL_UNSEEN = 'valid_unseen'
    TEST_SEEN = 'test_seen'
    TEST_UNSEEN = 'test_unseen'


class Dataset(Configurable, data.Dataset):
    def __init__(self, type: DatasetType, splits_file=None):
        Configurable.__init__(self)
        data.Dataset.__init__(self)
        self.type = type
        self.data_dir = os.path.join(self.conf.data_base_dir, self.type.value)
        self.splits_file = self.conf.splits_file if splits_file is None else splits_file
        with open(self.splits_file, 'r') as f:
            self.tasks_json = json.load(f)[self.type.value]

        if self.conf.fast_epoch:
            self.tasks_json = self.tasks_json[:self.conf.fast_epoch_size]

        self.unique_tasks = list(set(t for t, _ in self.tasks()))

    def tasks(self):
        for task in self.tasks_json:
            yield task

    def get_task(self, idx):
        task = self.tasks_json[idx]
        return task

    def get_img(self, task_dir, img_dir, idx):
        ims = os.list_dir(os.path.join(task_dir, img_dir))
        return Image.open(sorted(ims)[idx])

    def __len__(self):
        return len(self.tasks_json)
