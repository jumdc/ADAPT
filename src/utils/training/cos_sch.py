"""Cos Scheduler."""

import torch
import numpy as np


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine WarmUp Scheduler."""

    def __init__(self, optimizer, epoch_warmup, max_epoch, min_lr=1e-8):
        """
        Cosine learning rate scheduler with warmup.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer.
        epoch_warmup : int
            Number of epochs for warmup.
        max_epoch : int
            Maximum number of epochs the model is trained for.
        min_lr : float, optional
            Minimum learning rate. The default is 1e-9.
        """
        self.warmup = epoch_warmup
        self.max_num_iters = max_epoch
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        """get the learning rate."""
        if self.last_epoch == 0:
            return [self.min_lr]
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        """get the learning rate factor."""
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
