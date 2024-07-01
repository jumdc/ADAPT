"""Train multimodal model."""

# pylint:disable=C0303

import os
import logging
from copy import deepcopy
from datetime import datetime

import hydra
import torch
import wandb
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


from src.models.adapt import ADAPT
from src.utils.training.sizedatamodule import SizeDatamodule
from src.datamodules.multimodal_datamodule import MultimodaDataModule


def training(cfg: DictConfig, idx_cv: int, date=""):
    """Train multimodal model for classification."""
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    if cfg.logger.mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = cfg.paths.logs

    # callbacks
    callbacks = []
    name = f"{date}_{cfg.logger.name}_cv_{idx_cv}"
    model = ADAPT(cfg=cfg, name=name, stage="anchoring")
    if cfg.log:
        logger = hydra.utils.instantiate(cfg.logger, id=name)
        callbacks.extend(
            [LearningRateMonitor(logging_interval="epoch"), SizeDatamodule(cfg.log)]
        )
    else:
        logger = None
    if cfg.checkpoints:
        callbacks.append(
            ModelCheckpoint(
                save_last=False,
                dirpath=f"{cfg.paths.misc}/checkpoints/adapt",
                filename=name,
            )
        )
    if cfg.anchoring:
        logging.info("Anchoring training")
        cfg.multimodal.keep_missing = False
        datamodule = MultimodaDataModule(cfg=cfg, cv=idx_cv)
        trainer = hydra.utils.instantiate(
            cfg.trainer,
            logger=logger,
            max_epochs=cfg.model.anchoring_loss.max_epochs,
            callbacks=callbacks,
        )
        trainer.fit(model, datamodule)

    if cfg.contrastive:
        logging.info("Contrastive training")
        cfg.multimodal.keep_missing = True
        datamodule = MultimodaDataModule(cfg=cfg, cv=idx_cv)
        datamodule.stage = "contrastive"
        model.stage = "contrastive"
        trainer = hydra.utils.instantiate(
            cfg.trainer,
            logger=logger,
            max_epochs=cfg.model.contrastive_loss.max_epochs,
            callbacks=callbacks,
        )
        trainer.fit(model, datamodule)

    ### classification training ###
    if cfg.clf:
        cfg.multimodal.keep_missing = True
        datamodule_clf = MultimodaDataModule(cfg=cfg, cv=idx_cv)
        model.stage = "classification"
        trainer = hydra.utils.instantiate(
            cfg.trainer,
            logger=logger,
            max_epochs=cfg.model.supervised_loss.max_epochs,
            callbacks=callbacks,
        )
        trainer.fit(model, datamodule_clf)
        ######### test #########
        if cfg.test:
            datamodule_clf.cfg.multimodal.path_features_video = (
                cfg.multimodal.path_features_video_test
            )
            datamodule_clf.cfg.multimodal.hyperparams.batch_size = (
                1  # for test batch = 1 -- different size of samples.
            )
            datamodule_clf.cfg.multimodal.hyperparams.sliding_window = (
                False  # remove the sliding window to test on the whole TS.
            )
            trainer.test(model, datamodule=datamodule_clf)

    ### END ###
    if cfg.log:
        wandb.finish()


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """Train function for multimodal model."""
    ### set-up ###
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    torch.set_float32_matmul_precision("high")
    logger_console = logging.getLogger(__name__)
    logger_console.info("start")
    date_time = datetime.now().strftime("%Y-%m-%d_%Hh%M")

    ### cv ###
    for idx_cv in range(cfg.cv):
        cfg_xp = deepcopy(cfg)
        training(cfg=cfg_xp, idx_cv=idx_cv, date=date_time)
    logger_console.info("finished")


if __name__ == "__main__":
    main()
