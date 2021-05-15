import torch.nn as nn
import torch.optim as optim
from tars.config.base.model_config import ModelConfig
from tars.config.policies.moca_policy_config import MocaPolicyConfig
from tars.config.base.dataset_config import DatasetConfig


class TarsPolicyConfig(ModelConfig):
    use_mask = False
    batch_size = 1
    acc_grad_batches = 1 if 'small' in DatasetConfig().splits_file else 8
    # effective batch size = acc_grad_batches * batch_size

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
    action_readout_dropout = 0.3
    obj_readout_dropout = 0.3

    # vision module
    use_instance_seg = False
    detection_model_path = '/data/best_models/multi_label_classifier.ckpt'
    max_img_objects = 10
    vision_readout_dropout = 0.3

    # readout transformer
    transformer_num_heads = 8
    transformer_num_layers = 2

    # action module
    action_attn_heads = 4
    inst_lstm_dropout = 0.3

    # contextual embedding model
    context_emb_model_name_or_path = "google/bert_uncased_L-2_H-128_A-2"

    # mask rcnn
    mask_rcnn_path = MocaPolicyConfig.mask_rcnn_path

    # ablations
    remove_context = False
    remove_vision_readout = True
    remove_goal_lstm = False

    # training
    activation = 'ReLU'
    teacher_forcing_init = 1
    teacher_forcing_curriculum = 0.9
    teacher_forcing_step = 5000
    use_pretraining = True

    # initialization
    init_func = 'kaiming_normal_'

    def get_optim(self, parameters):
        return optim.SGD(parameters, lr=1e-3, momentum=0.9)

    def get_lr_scheduler(self, opt):
        return optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)

    def initialize_weights(self, layer):
        assert layer.__class__.__name__ in ['LSTMCell', 'Linear', 'MultiheadAttention']
        assert self.activation in ['LeakyReLU', 'ReLU']
        if layer.__class__.__name__ == 'LSTMCell':
            weights = [layer.weight_hh, layer.weight_ih]
        elif layer.__class__.__name__ == 'Linear':
            weights = [layer.weight]
        elif layer.__class__.__name__ == 'MultiheadAttention':
            weights = [p for n, p in layer.named_parameters() if 'weight' in n]

        init_nonlinearity = 'leaky_relu' if self.activation == 'LeakyReLU' else 'relu'

        if 'kaiming' in self.init_func:
            for w in weights:
                getattr(nn.init, self.init_func)(w, nonlinearity=init_nonlinearity)
        elif 'xavier' in self.init_func or 'orthogonal' in self.init_func:
            for w in weights:
                getattr(nn.init, self.init_func)(w, gain=nn.init.calculate_gain(init_nonlinearity))
        else:
            for w in weights:
                getattr(nn.init, self.init_func)(w)
