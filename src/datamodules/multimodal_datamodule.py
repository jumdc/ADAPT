"""Multimodal datamodule."""

import os
import pandas as pd
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from src.datamodules.datasets.stressid import StressID
from src.utils.training.splits import create_splits


class MultimodaDataModule(LightningDataModule):
    """Multimodal datamodule class"""

    def __init__(self, cfg: dict, stage: str = "anchoring", cv: int = 0) -> None:
        """Initialization

        Parameters
        ----------
        cfg : dict
            cfg dict.
        """
        super().__init__()
        self.cfg = cfg
        self.cv = cv
        self.stage = stage
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage=None):
        """Setup

        Parameterss
        ----------
        stage : str, optional
            Stage, by default None
        """

        all_launches = pd.read_csv(
            os.path.join(
                self.cfg["paths"]["data"], self.cfg["multimodal"]["path_labels"]
            ),
            index_col=0,
            sep=",",
        )
        folds = create_splits(
            ids=all_launches.index, cv=self.cfg.cv, seed=self.cfg.seed
        )
        if stage == "fit" or stage is None:
            self.train_set = StressID(
                cfg=self.cfg, stage=self.stage, ids=folds[0][self.cv]
            )
            self.val_set = StressID(
                cfg=self.cfg, stage=self.stage, ids=folds[1][self.cv]
            )
        if stage == "test":
            self.test_set = StressID(
                cfg=self.cfg, stage=self.stage, ids=folds[2][self.cv]
            )

    def train_dataloader(self) -> DataLoader:
        collate = self.train_set.collate if hasattr(self.train_set, "collate") else None
        return DataLoader(
            shuffle=True,
            dataset=self.train_set,
            batch_size=self.cfg["multimodal"]["hyperparams"]["batch_size"],
            num_workers=self.cfg["multimodal"]["hyperparams"]["num_workers"],
            collate_fn=collate,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        collate = self.train_set.collate if hasattr(self.train_set, "collate") else None
        return DataLoader(
            shuffle=False,
            dataset=self.val_set,
            batch_size=self.cfg["multimodal"]["hyperparams"]["batch_size"],
            num_workers=self.cfg["multimodal"]["hyperparams"]["num_workers"],
            collate_fn=collate,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        collate = self.train_set.collate if hasattr(self.train_set, "collate") else None
        return DataLoader(
            shuffle=False,
            dataset=self.test_set,
            collate_fn=collate,
            batch_size=self.cfg["multimodal"]["hyperparams"]["batch_size"],
            num_workers=self.cfg["multimodal"]["hyperparams"]["num_workers"],
        )
