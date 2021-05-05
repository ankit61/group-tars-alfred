import inspect
from collections import defaultdict
import argparse
import logging
import torch
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.cuda import is_available
from tars.auxilary_models import *
from tars.policies import *
from tars.config.main_config import MainConfig

def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True, help='the class name of the model to train - can be auxilary model or a policy')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no-log', action='store_true')

    args, _ = parser.parse_known_args()
    return args


def get_init_args(model_class):
    base_types = {str, int, float, bool}
    parser = argparse.ArgumentParser(allow_abbrev=False)
    arg_spec = inspect.getfullargspec(model_class.__init__)
    init_args = arg_spec.args[1:] # skip first arg as that is self 
    type_hints = defaultdict(lambda: str, arg_spec.annotations)

    if len(init_args) != len(type_hints):
        logging.warning(f'Some arguments in {model_class.__name__}.__init__ do not have type hints! Assuming str type as default for use in command line')

    for arg in init_args:
        assert type_hints[arg] in base_types
        parser.add_argument(f'--{arg}', type=type_hints[arg])

    args, _ = parser.parse_known_args()
    return {k: v for k, v in vars(args).items() if v is not None} # remove args not specified


def main():
    args = get_args()
    model_class = globals()[args.model]
    init_args = get_init_args(model_class)
    model = model_class(**init_args)

    # Define logger
    logger = False
    if not args.no_log:
        logger = WandbLogger(
                    project=MainConfig.wandb_project,
                    entity=MainConfig.wandb_entity
                )

        hyperparams = model.conf.get_all()
        logger.log_hyperparams(hyperparams)

    trainer = Trainer(
                logger=logger, resume_from_checkpoint=args.resume,
                **model.get_trainer_kwargs()
            )

    trainer.fit(model)

if __name__ == '__main__':
    main()
