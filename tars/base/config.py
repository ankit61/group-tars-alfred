import os
from pathlib import Path
import importlib
import inspect
from argparse import ArgumentParser
from ruamel.yaml import YAML, dump, RoundTripDumper


class Config():
    def __new__(cls):
        inst = super().__new__(cls)
        inst.load_from_cmd()
        return inst

    def get(self, mode='dict', types=None):
        '''
            Args:
                format: which format to get config: yaml or dict
                types: which types of variables to consider. If None,
                        all types are considered in when `mode` is `dict` and
                        int, float, bool, str, dict, list when `mode` is `yaml`
        '''

        assert mode in ['dict', 'yaml']
        if mode == 'yaml' and types is None:
            types = set([int, float, bool, str, dict, list])
        d = {}

        for k in self._get_vars():
            v = getattr(self, k)
            if isinstance(v, Config):
                d[k] = v.get(mode='dict', types=types)
            elif (types is None) or (type(v) in types):
                d[k] = v

        return d if (mode == 'dict') else dump(d, Dumper=RoundTripDumper)

    def _get_primitives(self, primitive_types=set([int, float, bool, str])):
        d = {}
        for k in self._get_vars():
            v = getattr(self, k)
            if type(v) in primitive_types:
                d[k] = v
        return d

    def _get_vars(self):
        statics = list(
            filter(
                lambda x: not x.startswith('_') and not callable(getattr(self, x)),
                dir(self.__class__)
            )
        )

        members = list(filter(lambda x: not x.startswith('_'), self.__dict__.keys()))

        return members + statics

    def load_from_cmd(self):
        parser = ArgumentParser(allow_abbrev=False)
        cls_prefix = self.__class__.__name__.lower()
        assert cls_prefix.endswith('config')
        cls_prefix = cls_prefix[:-6] # remove 'config' from end
        for k, v in self._get_primitives().items():
            v_type = (lambda x: x.lower() == 'true') if isinstance(v, bool) else type(v)
            parser.add_argument(f'--{cls_prefix}-{k}', type=v_type, default=v, help=f'{self.__class__.__name__}.{k}')

        args, _ = parser.parse_known_args()
        for k, v in self._get_primitives().items():
            setattr(self.__class__, k, getattr(args, f'{cls_prefix}_{k}'))

    @classmethod
    def get_all(cls, mode='dict'):
        assert mode in ['dict', 'yaml']

        out = {}
        config_dir = (os.path.join(Path(__file__).absolute().parents[1], 'config'))
        for root, ds, fs  in os.walk(config_dir):
            for py_f in filter(lambda x: x.endswith('.py'), fs):
                conf_class = os.path.splitext(py_f)[0]
                conf_class = ''.join(map(lambda x: x.title(), conf_class.split('_')))

                # load py_f
                module_name = os.path.splitext(os.path.basename(py_f))[0]
                spec = importlib.util.spec_from_file_location(module_name, os.path.join(root, py_f))
                conf_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(conf_module)

                conf_class = ''
                module_l = ''.join(conf_module.__name__.split('.')[-1].split('_'))
                for m in inspect.getmembers(conf_module):
                    if m[0].lower() == module_l:
                        conf_class = m[0]
                        break
                assert hasattr(conf_module, conf_class), f'{conf_class} does not exist in {conf_module}'
                conf = getattr(conf_module, conf_class)()
                out[conf_class] = conf.get(mode='dict', types=set([int, float, bool, str, dict, list]))

        return out if (mode == 'dict') else dump(out, Dumper=RoundTripDumper)
