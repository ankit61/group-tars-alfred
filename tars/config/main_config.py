import os
from pathlib import Path
import inspect
from getpass import getuser
from datetime import datetime
import multiprocessing as mp
import torch
from tars.base.config import Config
from tars.config.base.dataset_config import DatasetConfig


class MainConfig(Config):

    # compute use
    use_gpu = torch.cuda.is_available()
    gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}' if use_gpu else 'cpu')
    num_threads = mp.cpu_count()

    # wandb
    wandb_entity = 'tars-alfred'
    wandb_project = 'group-tars-alfred'

    # general training
    default_trainer_args = {
        'gpus': 1 if use_gpu else 0,
        'check_val_every_n_epoch': 1000 if 'small_split' in DatasetConfig().splits_file else 1,
        'num_sanity_val_steps': 4,
        'accumulate_grad_batches': 1,
        #'auto_lr_find': True,
        'track_grad_norm': 2,
        'log_every_n_steps': 1,
        #'val_check_interval': 1000,
        'max_epochs': 500
    }

    # basic dirs
    save_dir = os.path.join(Path(__file__).parents[1], 'gen/')

    @classmethod
    def get_save_path(cls, obj, file_name, use_user_time=False):
        caller_file = str(Path(inspect.getfile(obj.__class__)).parent.absolute())
        base = str(Path(__file__).parents[1].absolute())
        assert caller_file.startswith(base), f'get_cur_save_dir only works when called inside of {base}'

        save_dir = os.path.join(cls.save_dir, caller_file[len(base):].strip('/'))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if use_user_time:
            base, ext = os.path.splitext(file_name)
            user = getuser()
            now = datetime.now().strftime('%b_%d__%H_%M')
            file_name = '_'.join([base, user, now]) + ext

        path = os.path.join(save_dir, file_name)
        create_path = path if '.' not in path else os.path.dirname(path)

        if not os.path.exists(create_path):
            os.makedirs(create_path)

        return path
