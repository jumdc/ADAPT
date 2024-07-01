"""Log sizes of the train, test and validation sets."""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class SizeDatamodule(Callback):
    """Get the sizes of the train, test and validation sets."""

    def __init__(self, log: bool = True):
        """Initialize the callback."""
        self.logging = log

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """At each epoch start, log the size of the train set."""
        nb = len(trainer.datamodule.train_set)
        sizes = {"size-train": nb}
        if self.logging:
            pl_module.logger.log_metrics(sizes, step=trainer.current_epoch)

    def on_test_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """At each test start, log the size of the test set."""
        sizes = {"size-test": len(trainer.datamodule.test_set)}
        if self.logging:
            pl_module.logger.log_metrics(sizes)

    def on_validation_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """At each validation start, log the size of the validation set."""
        sizes = {
            "size-val-positives": len(trainer.datamodule.val_set),
        }
        if self.logging:
            pl_module.logger.log_metrics(sizes)
