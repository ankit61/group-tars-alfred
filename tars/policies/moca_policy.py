from tars.base.policy import Policy
from tars.moca.models.model.seq2seq_im_mask import Module as MOCA


class MocaPolicy(Policy):
    def __init__(self, model_load_path=None):
        super().__init__()
        model_load_path = self.conf.saved_model_path if model_load_path is None else model_load_path

        self.model, _ = MOCA.load(model_load_path, device=self.conf.main.device)
        self.model.share_memory()
        self.model.eval()
        self.model.test_mode = True


    def reset():
        self.model.reset()

    def forward(self, img, goal_inst, low_insts):
        pass

    @staticmethod
    def get_img_transforms():
        pass

    def get_text_transforms(self):
        pass
