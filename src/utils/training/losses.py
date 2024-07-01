"""Losses"""

import torch
import numpy as np
from torch import nn
from torch import einsum
import torch.nn.functional as F


class InfoNCEAnchor(nn.Module):
    """symmetric infoNCE loss, aka same scheme than in ImageBind"""

    def __init__(self, cfg):
        """init"""
        super().__init__()
        self.cfg = cfg
        logit_scale = torch.ones([]) * self.cfg.model.contrastive_loss.temperature
        self.gamma = cfg.model.contrastive_loss.gamma
        self.modalities = cfg["multimodal"]["modalities"]
        self.anchor = cfg["model"]["anchor"]
        if (
            cfg.model.contrastive_loss.learnable_scale
            and not cfg.model.contrastive_loss.cos
        ):
            self.temperature = nn.Parameter(logit_scale)
        elif not cfg.model.contrastive_loss.cos:
            self.temperature = logit_scale
        else:
            self.temperature_max = self.cfg.model.contrastive_loss.temperature_max
            self.temperature_min = self.cfg.model.contrastive_loss.temperature_min
            self.period = self.cfg.model.contrastive_loss.period

    def compute_loss(self, x_1, x_2, current_epoch=0):
        """Compute loss"""
        if self.gamma != 0:
            x_1 = F.normalize(x_1, dim=-1, p=2)
            x_2 = F.normalize(x_2, dim=-1, p=2)
            noise_to_add = torch.normal(0, 1, size=x_1.shape, device=x_1.device)
            x_1 = x_1 + noise_to_add * self.gamma
            noise_to_add_2 = torch.normal(0, 1, size=x_2.shape, device=x_2.device)
            x_2 = x_2 + noise_to_add_2 * self.gamma
        if self.cfg.model.contrastive_loss.cos:
            self.temperature = torch.ones([]) * (
                (self.temperature_max - self.temperature_min)
                * (1 + np.cos(2 * np.pi * current_epoch / self.period))
                / 2
                + self.temperature_min
            )
        x_1 = F.normalize(x_1, dim=-1, p=2)
        x_2 = F.normalize(x_2, dim=-1, p=2)
        labels = torch.arange(
            x_1.shape[0], device=x_1.device
        )  # entries on the diagonal
        similarity = einsum("i d, j d -> i j", x_1, x_2) / self.temperature
        loss_1 = F.cross_entropy(similarity, labels)
        loss_2 = F.cross_entropy(similarity.T, labels)
        return (loss_1 + loss_2) / 2.0

    def forward(self, outputs, current_epoch=0):
        """forward."""
        loss = torch.tensor(0.0).to(outputs[self.anchor].device)
        for modality in self.modalities:
            if modality != self.anchor:
                loss += self.compute_loss(
                    outputs[self.anchor], outputs[modality], current_epoch
                )
        return loss


class InfoNCE(nn.Module):
    """symmetric infoNCE loss, aka same scheme than in ImageBind."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        logit_scale = torch.ones([]) * self.cfg.model.contrastive_loss.temperature
        self.gamma = cfg.model.contrastive_loss.gamma
        self.modalities = cfg["multimodal"]["modalities"]
        self.anchor = cfg["model"]["anchor"]
        if (
            cfg.model.contrastive_loss.learnable_scale
            and not cfg.model.contrastive_loss.cos
        ):
            self.temperature = nn.Parameter(logit_scale)
        elif not cfg.model.contrastive_loss.cos:
            self.temperature = logit_scale
        else:
            self.temperature_max = self.cfg.model.contrastive_loss.temperature_max
            self.temperature_min = self.cfg.model.contrastive_loss.temperature_min
            self.period = self.cfg.model.contrastive_loss.period

    def forward(self, x_1, x_2, current_epoch=0):
        """forward."""
        if self.gamma != 0:
            x_1 = F.normalize(x_1, dim=-1, p=2)
            x_2 = F.normalize(x_2, dim=-1, p=2)
            noise_to_add = torch.normal(0, 1, size=x_1.shape, device=x_1.device)
            x_1 = x_1 + noise_to_add * self.gamma
            noise_to_add_2 = torch.normal(0, 1, size=x_2.shape, device=x_2.device)
            x_2 = x_2 + noise_to_add_2 * self.gamma
        if self.cfg.model.contrastive_loss.cos:
            self.temperature = torch.ones([]) * (
                (self.temperature_max - self.temperature_min)
                * (1 + np.cos(2 * np.pi * current_epoch / self.period))
                / 2
                + self.temperature_min
            )
        x_1 = F.normalize(x_1, dim=-1, p=2)
        x_2 = F.normalize(x_2, dim=-1, p=2)
        labels = torch.arange(
            x_1.shape[0], device=x_1.device
        )  # entries on the diagonal
        similarity = einsum("i d, j d -> i j", x_1, x_2) / self.temperature
        loss_1 = F.cross_entropy(similarity, labels)
        loss_2 = F.cross_entropy(similarity.T, labels)
        return (loss_1 + loss_2) / 2.0
