from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from tars.base.configurable import Configurable
from tars.base.config import Config
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from tars.base.configurable import Configurable
import os
import datetime


class Model(Configurable, LightningModule):
    def __init__(self, *args, **kwargs):
        Configurable.__init__(self)
        LightningModule.__init__(self)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def configure_callbacks(self):
        # create directory where checkpoints will be saved
        checkpoint_dirpath = os.path.dirname(os.path.realpath(__file__)) + "/" + datetime.datetime.now().strftime(
            "%d/%m/%Y %H:%M:%S")
        if not os.path.exists(checkpoint_dirpath):
            os.makedirs(checkpoint_dirpath)

        return [
            EarlyStopping(monitor='val_loss/dataloader_idx_0', patience=self.conf.patience),
            ModelCheckpoint(dirpath=checkpoint_dirpath,
                            filename='sample-mnist-{epoch:02d}-{val_loss:.2f}',
                            monitor="val_loss",
                            period=1,
                            save_last=True)
        ]

    def get_trainer_kwargs(self):
        return self.conf.main.default_trainer_args
