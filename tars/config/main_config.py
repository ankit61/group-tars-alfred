import os
from pathlib import Path
import inspect
from getpass import getuser
from datetime import datetime
import torch
from tars.base.config import Config


class MainConfig(Config):

    # gpu use
    use_gpu = torch.cuda.is_available()
    gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}' if use_gpu else 'cpu')

    # basic dirs
    alfred_dir = os.path.join(Path(__file__).parents[1], 'alfred/')
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
