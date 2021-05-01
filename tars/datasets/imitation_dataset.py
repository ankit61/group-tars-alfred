import torch
import functools
import numpy as np
from tars.base.dataset import Dataset


class ImitationDataset(Dataset):
    def __init__(self, type, img_transforms, text_transforms, splits_file=None):
        super().__init__(type, splits_file=splits_file)

        self.img_transforms = img_transforms
        self.text_transforms = text_transforms

    @functools.lru_cache()
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
