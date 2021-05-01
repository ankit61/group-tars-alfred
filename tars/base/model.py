from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from tars.base.configurable import Configurable
from tars.base.config import Config
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from tars.base.configurable import Configurable


class Model(LightningModule, Configurable):
    def __init__(self, *args, **kwargs):
        LightningModule.__init__(self)
        Configurable.__init__(self)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def configure_callbacks(self):
        return [
            EarlyStopping(monitor='val_seen_loss'),
            ModelCheckpoint()
        ]
