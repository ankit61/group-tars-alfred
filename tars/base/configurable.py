import os
import importlib
from pathlib import Path
import inspect
from tars.config.main_config import MainConfig
from tars.base.config import Config


class Configurable(object):
    def __init__(self):
        pass

    def __new__(cls, *args, **kwargs):
        # this helps avoid circular dependencies since the config is accessible
        # even before __init__ is called
        inst = super(Configurable, cls).__new__(cls)
        inst.conf = inst.__get_config()
        inst.conf.main = MainConfig()
        return inst

    def __get_config(self):
        # this function should never be externally called
        cur_cls = self.__class__
        done = False
        while (not done) and cur_cls != Configurable:
            class_file = inspect.getfile(cur_cls)
            base_dir = str(Path(__file__).parent.parent.absolute())

            assert class_file.startswith(base_dir), f'A config can be linked only with classes inside {base_dir}'

            conf_class = cur_cls.__name__ + 'Config'
            conf_file = os.path.join(
                            base_dir, 'config',
                            os.path.splitext(class_file[len(base_dir):])[0].strip('/') + '_config.py'
                        )
            done = os.path.exists(conf_file)
            if not done:
                cur_cls = cur_cls.__bases__[0]

        if not done:
            return Config()

        module_name = os.path.splitext(os.path.basename(conf_file))[0]
        spec = importlib.util.spec_from_file_location(module_name, conf_file)
        conf_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conf_module)
        if not hasattr(conf_module, conf_class):
            raise ValueError(f'{conf_module} must have a {conf_class} class')

        return getattr(conf_module, conf_class)()
