import os
import json
import bisect
import functools
import numpy as np
from tars.base.dataset import Dataset


class ImitationDataset(Dataset):
    def __init__(self, type, img_transforms, text_transforms, splits_file=None):
        super().__init__(type, splits_file=splits_file)

        task_lens = []
        for t, _ in self.tasks():
            data = json.load(open(os.path.join(t, self.conf.aug_traj_file), 'r'))
            assert len(data['plan']['low_actions']) > 0
            task_lens.append(len(data['plan']['low_actions']))

        self.img_transforms = img_transforms
        self.text_transforms = text_transforms
        self.cum_task_lens = np.cumsum(task_lens)

    def __getitem__(self, idx):
        task_idx = bisect.bisect(self.cum_task_lens, idx)
        step_idx = idx - self.cum_task_lens[task_idx - 1] if task_idx > 0 else idx
        task_dir, lang_idx = self.get_task(task_idx)

        # get raw data
        rgb_im = self.get_img(task_dir, self.conf.high_res_img_dir, step_idx)
        goal_inst, low_insts = self.get_transformed_insts(task_dir, lang_idx)
        expert_action, expert_int_obj = self.get_expert_action(task_dir, step_idx)

        # apply transformations
        rgb_im = self.img_transforms(rgb_im)

        return rgb_im, goal_inst, low_insts, expert_action, expert_int_obj

    @functools.lru_cache()
    def get_transformed_insts(self, task_dir, lang_idx):
        goal_inst, low_insts = self.get_insts(task_dir, lang_idx)
        goal_inst = self.text_transforms([goal_inst], is_goal=True)
        low_insts = self.text_transforms(low_insts, is_goal=False)

        # FIXME: these should be tensors to make them batch-able
        return goal_inst, low_insts

    def __len__(self):
        return self.cum_task_lens[-1]
