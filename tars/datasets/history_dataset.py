import torch
import functools
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from tars.base.dataset import DatasetType, Dataset
from collections import defaultdict
from typing import Union
from enum import Enum


class HistoryType(Enum):
    ACTION = 'action'
    OBJECT = 'object'
    
    
class HistoryDataset(Dataset):
    def __init__(self, type: Union[str, DatasetType], history_type: Union[str, HistoryType], splits_file=None):
        super().__init__(type, splits_file=splits_file)
        self.history_type = history_type if isinstance(type, HistoryType) else HistoryType(history_type)

    @functools.lru_cache()
    def __getitem__(self, task_idx):
        task_dir = self.unique_tasks[task_idx]
        expert_actions, expert_int_objects = self.get_all_expert_actions(task_dir, int_obj_only=True)
        if self.history_type == HistoryType.ACTION:
            return torch.tensor(expert_actions)
        elif self.history_type == HistoryType.OBJECT:
            return torch.tensor(expert_int_objects)

    def __len__(self):
        return len(self.unique_tasks)

    def collate(self, batch):
        out = {}
        # sort by descending length
        sorted_idxs = np.argsort([-seq.shape[0] for seq in batch])
        return rnn_utils.pack_sequence(list(map(batch.__getitem__, sorted_idxs)))
        
    @staticmethod
    def mini_batches(collated_batch):
        '''
            Given a batch of tasks, loop over them in sequence to get parallel
            batches in order
        '''
        start = 0
        for bsz in collated_batch.batch_sizes:
            mini_batch = {}
            mini_batch = collated_batch.data[start:start+bsz]
            yield mini_batch, bsz
            start += bsz
