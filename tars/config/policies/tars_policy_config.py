import torch.optim as optim
from tars.config.base.model_config import ModelConfig
from tars.config.policies.moca_policy_config import MocaPolicyConfig


class TarsPolicyConfig(ModelConfig):
    use_mask = False
    batch_size = 1

    # feature sizes
    context_size = 256
    vision_features_size = 128
    raw_vision_features_size = 512

    # history
    past_actions_len = 10
    past_objects_len = 10

    # embeddings
    action_emb_dim = 64
    object_emb_dim = 64
    action_hist_emb_dim = 256
    int_hist_emb_dim = 256
    # word_emb_dim = 128
    vision_object_emb_dim = 128

    # LSTMs
    inst_hidden_size = 256
    goal_hidden_size = 128

    # context module
    action_readout_path = '/data/best_models/action_history.ckpt'
    int_object_readout_path = '/data/best_models/int_object_history.ckpt'

    # vision module
    use_instance_seg = False
    detection_model_path = '/data/best_models/multi_label_classifier.ckpt'
    max_img_objects = 10

    # readout transformer
    transformer_num_heads = 8
    transformer_num_layers = 2

    # action module
    action_attn_heads = 4

    # contextual embedding model
    context_emb_model_name_or_path = "albert-base-v2"

    # mask rcnn
    mask_rcnn_path = MocaPolicyConfig.mask_rcnn_path

    def get_optim(self, parameters):
        return optim.SGD(parameters, lr=1e-3, momentum=0.9)

    def get_lr_scheduler(self, opt):
        return optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)

