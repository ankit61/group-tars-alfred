from pytorch_lightning.core.lightning import LightningModule


class Model(LightningModule):
    # not a configurable because will be mostly used as an auxilary component of
    # policy

    def forward(self, *args, **kwargs):
        raise NotImplementedError
