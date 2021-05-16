import torch.nn as nn
import torch.optim as optim
from tars.config.base.model_config import ModelConfig
from tars.config.policies.moca_policy_config import MocaPolicyConfig
from tars.config.base.dataset_config import DatasetConfig


class TarsPolicyConfig(ModelConfig):
    use_mask = False
    batch_size = 1
    acc_grad_batches = 1 if 'small' in DatasetConfig().splits_file else 32
    # effective batch size = acc_grad_batches * batch_size

    # feature sizes
    context_size = 512
    vision_features_size = 512
    # raw_vision_features_size = 512

    # history
    past_actions_len = 10
    past_objects_len = 10

    # embeddings
    action_emb_dim = 128
    object_emb_dim = 128
    action_hist_emb_dim = 256
    int_hist_emb_dim = 256
    # word_emb_dim = 128
    vision_object_emb_dim = 128

    # LSTMs
    inst_hidden_size = 512
    goal_hidden_size = 256

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
    transformer_num_layers = 4

    # action module
    action_attn_heads = 4
    inst_lstm_dropout = 0.3
    num_inst_lstm_layers = 1
    num_goal_lstm_layers = 1

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
    teacher_forcing_step = 200
    use_pretraining = False

    # initialization
    init_func = 'kaiming_normal_'

    def get_optim(self, parameters):
        return optim.Adam(parameters, lr=1e-4)

    def get_lr_scheduler(self, opt):
        return optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)

    def initialize_weights(self, layer):
        if sum([p.numel() for p in layer.parameters()]) == 0:
            return # no params to init

        assert layer.__class__.__name__ in ['LSTMCell', 'StackedLSTMCell', 'Linear', 'MultiheadAttention', 'TransformerEncoder', 'Embedding', 'Conv2d', 'Sequential']
        assert self.activation in ['LeakyReLU', 'ReLU']

        if layer.__class__.__name__ == 'StackedLSTMCell':
            weights = [[l.weight_hh, l.weight_ih] for l in layer.lstm_cells]
            weights = [w for ws in weights for w in ws]
        elif layer.__class__.__name__ == 'LSTMCell':
            weights = [layer.weight_hh, layer.weight_ih]
        elif layer.__class__.__name__ == 'Linear':
            weights = [layer.weight]
        elif layer.__class__.__name__ == 'MultiheadAttention':
            weights = [p for n, p in layer.named_parameters() if 'weight' in n]
        elif layer.__class__.__name__ == 'TransformerEncoder':
            weights = [p for n, p in layer.named_parameters() if 'weight' in n and 'norm' not in n]
        elif layer.__class__.__name__ == 'Embedding':
            weights = [layer.weight]
        elif layer.__class__.__name__ == 'Conv2d':
            weights = [layer.weight]
        elif layer.__class__.__name__ == 'Sequential':
            for l in layer:
                self.initialize_weights(l)
            return

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
