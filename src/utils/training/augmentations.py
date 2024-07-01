"""Data augmentation and transforms"""

import copy
from typing import Any

import torch
import numpy as np
from torchvision.transforms import Compose

EPS = torch.finfo(torch.float).eps


class GaussianNoise:
    """Add Gaussian noise to the different modalities."""

    def __init__(self, sigma=0.05, modalities=None) -> None:
        self.sigma = sigma
        self.modalities = modalities

    def __call__(self, x):
        for modality in self.modalities:
            if modality != "TEXT":
                sigma = np.random.choice([0, self.sigma])  # choose to add noise or not
                if sigma != 0:
                    noise = torch.tensor(
                        np.random.normal(0, sigma, size=x[modality].shape),
                        device=x[modality].device,
                        dtype=x[modality].dtype,
                    )
                    x[modality] = x[modality] + noise
        return x


class MoldalityDropout:
    """Modality Dropout for the different modalities."""

    def __init__(self, weight=0.5, modalities=None) -> None:
        self.weight = weight
        self.modalities = list(modalities)

    def __call__(self, x) -> Any:
        # start at 1 because use 0 idx when missing
        available = np.arange(1, len(self.modalities) + 1)
        for _, modality in enumerate(self.modalities):
            indices = np.char.find(np.array(x[f"id_{modality}"]), "missing") != -1
            available[indices] = 0
        non_zeros = np.count_nonzero(available, axis=-1, keepdims=True)[0]
        if non_zeros > 1 and len(available.tolist()) > 1:
            choose_from = available.tolist()
            # we can hide from 1 to nb_avail - 1 channels
            nb_to_hide = np.random.choice(np.arange(1, len(choose_from)))
            for _ in range(nb_to_hide):
                choice = np.random.choice(choose_from)
                choose_from.remove(choice)
                if choice != 0:
                    x[f"id_{self.modalities[choice-1]}"] = "missing"
        return x


class MultiModalAugmentations:
    """Contrastive Transforms for Time Series

    Reference:
    ----------
    [1] Contrastive Pre-Trainingfor Multimodal Medical Time Series
    """

    def __init__(
        self, modalities: list = None, modality_dropout: bool = True, noise_sigma=0.005
    ) -> None:
        """Add masking channels arbitrarily."""
        prime, second = [], []
        prime.extend([GaussianNoise(sigma=noise_sigma, modalities=modalities)])
        second.extend([GaussianNoise(sigma=noise_sigma, modalities=modalities)])
        if modality_dropout:
            second.append(MoldalityDropout(modalities=modalities))
        self.transform_prime = Compose(prime)
        self.transform_second = Compose(second)

    def __call__(self, sample, idx=None):
        x1, x2 = copy.deepcopy(sample), copy.deepcopy(sample)
        x1 = self.transform_prime(x1)
        x2 = self.transform_second(x2)
        return x1, x2
