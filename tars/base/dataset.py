import os
import json
import pickle
from enum import Enum
from vocab import Vocab
from torch.utils import data
from PIL import Image
from collections import OrderedDict
from tars.base.configurable import Configurable
from tars.config.envs.alfred_env_config import AlfredEnvConfig


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

        self.unique_tasks = list(OrderedDict.fromkeys([t for t, _ in self.tasks()]).keys())

    def tasks(self, start_idx=None, end_idx=None):
        start_idx = start_idx if start_idx else self.conf.start_idx
        end_idx = end_idx if end_idx else (self.conf.end_idx if self.conf.end_idx else len(self.tasks_json))
        assert(start_idx < end_idx)
        for task in self.tasks_json[start_idx:end_idx]:
            yield os.path.join(self.data_dir, task['task']), task['repeat_idx']

    def get_task(self, idx):
        task = self.tasks_json[idx]
        return os.path.join(self.data_dir, task['task']), task['repeat_idx']

    def get_img(self, task_dir, img_dir, idx):
        ims = os.listdir(os.path.join(task_dir, img_dir))
        return Image.open(os.path.join(task_dir, img_dir, sorted(ims)[idx]))

    def get_tgt(self, task_dir, tgt_dir, idx):
        tgts = os.listdir(os.path.join(task_dir, tgt_dir))
        with open(os.path.join(task_dir, tgt_dir, sorted(tgts)[idx]), "rb") as f:
            return pickle.load(f)

    def get_insts(self, task_dir, lang_idx):
        with open(os.path.join(task_dir, self.conf.aug_traj_file), 'r') as f:
            anns = json.load(f)['turk_annotations']['anns'][lang_idx]
            return anns['task_desc'], anns['high_descs']

    def get_expert_action(self, task_dir, idx):
        with open(os.path.join(task_dir, self.conf.aug_traj_file), 'r') as f:
            data = json.load(f)
            low_action = data['plan']['low_actions'][idx]

            action = low_action['api_action']['action']

            object_type = self.conf.object_na
            if action in AlfredEnvConfig.interact_actions:
                object_type = low_action['api_action']['objectId'].split('|')[0]

            object_type = self.conf.objects_vocab.word2index(object_type)

            action = AlfredEnvConfig.actions.word2index(action)

            # all processing can image one-to-one correspondence between images and actions
            assert data['images'][idx]['low_idx'] == idx

            return action, object_type

    def __len__(self):
        return len(self.tasks_json)
