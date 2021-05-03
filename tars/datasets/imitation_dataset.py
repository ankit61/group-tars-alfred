from collections import defaultdict
import torch
import functools
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from tars.base.dataset import Dataset


class ImitationDataset(Dataset):
    def __init__(self, type, img_transforms, text_transforms, text_collate, splits_file=None):
        super().__init__(type, splits_file=splits_file)

        self.img_transforms = img_transforms
        self.text_transforms = text_transforms
        self.text_collate = text_collate

    # @functools.lru_cache() # need a diff cache
    def __getitem__(self, task_idx):
        task_dir, lang_idx = self.get_task(task_idx)

        feat = {}

        goal_inst, low_insts = self.get_transformed_insts(task_dir, lang_idx)

        feat['goal_inst'] = goal_inst
        feat['low_insts'] = low_insts
        feat['expert_actions'], feat['expert_int_objects'] = self.get_all_expert_actions(task_dir)
        feat['expert_actions'] = torch.tensor(feat['expert_actions'])
        feat['expert_int_objects'] = torch.tensor(feat['expert_int_objects'])

        feat['images'] = self.get_all_imgs(task_dir, self.conf.high_res_img_dir)
        assert len(feat['images']) == len(feat['expert_actions']) and \
                len(feat['expert_actions']) == len(feat['expert_int_objects'])

        for i in range(len(feat['images'])):
            feat['images'][i] = self.img_transforms(feat['images'][i])

        return feat

    def get_transformed_insts(self, task_dir, lang_idx):
        goal_inst, low_insts = self.get_insts(task_dir, lang_idx)
        goal_inst = self.text_transforms([goal_inst], is_goal=True)
        low_insts = self.text_transforms(low_insts, is_goal=False)

        return goal_inst, low_insts

    def __len__(self):
        return len(self.tasks_json)

    def collate(self, batch):
        out = {}

        dict_batch = defaultdict(lambda: list())
        for i in range(len(batch)):
            for k in batch[i].keys():
                dict_batch[k].append(batch[i][k])

        # sort by descending length
        sorted_idxs = np.argsort([-ac.shape[0] for ac in dict_batch['expert_actions']])

        out['expert_actions'] = rnn_utils.pack_sequence(
                                    list(map(dict_batch['expert_actions'].__getitem__, sorted_idxs))
                                )
        out['expert_int_objects'] = rnn_utils.pack_sequence(
                                    list(map(dict_batch['expert_int_objects'].__getitem__, sorted_idxs))
                                )

        out['images'] = rnn_utils.pack_sequence(
                            list(map(lambda i: torch.stack(dict_batch['images'][i]), sorted_idxs))
                        )

        out['goal_inst'] = self.text_collate(dict_batch['goal_inst'])
        out['low_insts'] = self.text_collate(dict_batch['low_insts'])

        assert (out['expert_actions'].batch_sizes == out['expert_int_objects'].batch_sizes).all() \
            and (out['expert_actions'].batch_sizes == out['images'].batch_sizes).all()

        out['batch_sizes'] = out['expert_actions'].batch_sizes

        return out

    @staticmethod
    def mini_batches(batch):
        '''
            Given a batch of tasks, loop over them in sequence to get parallel
            batches in order
        '''

        start = 0
        for bsz in batch['batch_sizes']:
            mini_batch = {}

            mini_batch['expert_actions'] = batch['expert_actions'].data[start:start+bsz]
            mini_batch['expert_int_objects'] = batch['expert_int_objects'].data[start:start+bsz]
            mini_batch['images'] = batch['images'].data[start:start+bsz]

            mini_batch['goal_inst'] = batch['goal_inst']
            mini_batch['low_insts'] = batch['low_insts']

            yield mini_batch, bsz

            start += bsz
