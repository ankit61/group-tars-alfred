import torch
from torchvision import transforms
from tars.base.policy import Policy


class RandomPolicy(Policy):
    def forward(self, img, goal_inst, low_insts):
        action = torch.rand(img.shape[0], self.num_actions)
        action = torch.softmax(action, 1)
        int_mask = torch.rand(img.shape[0], 1, *self.int_mask_size)

        return action, int_mask, None

    @staticmethod
    def get_img_transforms():
        return transforms.ToTensor()

    def get_text_transforms(self):
        return lambda x, is_goal: x
