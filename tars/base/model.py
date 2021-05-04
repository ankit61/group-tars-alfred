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
        return [
            EarlyStopping(monitor='val_loss/dataloader_idx_0', patience=self.conf.patience),
            ModelCheckpoint(monitor="val_loss/dataloader_idx_0",
                            period=1,
                            save_top_k=-1) # saves every model
        ]

    def get_trainer_kwargs(self):
        return self.conf.main.default_trainer_args
