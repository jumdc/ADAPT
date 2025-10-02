"""Modality-specific encoders."""

# pylint:disable=C0303

import logging
from collections import OrderedDict


import os
import copy
import torch
from torch import nn

from assets.byol_a.models import AudioNTT2020
from src.models.modules.make_ts import ResnetTS, Inception
from src.models.modules.make_vision import make_hiera, make_hiera_image


def clone(module, number_of_copies):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(number_of_copies)])


def default(val, default):
    """Get default."""
    return default if val is None else val


class Encoders(nn.Module):
    """Projection module."""

    def __init__(self, cfg) -> None:
        """Intialization of the projection module."""
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleDict()
        self.modalities = self.cfg["multimodal"]["modalities"]
        for modality in self.modalities:
            self.encoders[modality] = self.make_encoder(modality)

    def forward(self, x) -> dict:
        """Forward pass of the projection."""
        features = {}
        for modality in self.modalities:
            features[modality] = self.encoders[modality](x[modality])
        return features

    def make_encoder(self, modality) -> nn.Module:
        """Make encoder for a given modality

        Parameters
        ----------
        modality : str
            Modality of the encoder.

        Returns
        -------
        nn.Module
            Encoder module.
        """
        out_features = self.cfg["model"]["transformer"]["d_model"]
        encoder = []
        # -- VIDEO
        if modality == "VIDEO":
            if "hiera" in self.cfg["model"]["encoders"]["type"]:
                logging.info("Using hiera")
                model, in_features = make_hiera(
                    freeze=self.cfg["model"]["encoders"]["freeze"], directory=None
                )
                encoder.append(("encoder", model))
            else:
                model = nn.Identity()  # featurization is already done
                in_features = 768

        # -- AUDIO
        elif modality == "AUDIO":
            if "byola" in self.cfg["model"]["encoders"]["type"]:
                logging.info("Using byola")
                model = AudioNTT2020(d=self.cfg["model"]["encoders"]["n_dims_audio"])
                device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
                if self.cfg["model"]["encoders"]["pretrained"]:
                    path = os.path.join(
                        self.cfg["paths"]["misc"],
                        "byol_a/pretrained_weights",
                        self.cfg["model"]["weight_path_byola"],
                    )
                    # state_dict = torch.load(path, map_location=device, weights_only=False)
                    # print(state_dict.keys())
                    model.load_weight(
                        path,
                        device=device
                    )
                in_features = model.fc[0].in_features
                model.fc = nn.Identity()
                encoder.append(("encoder", model))
            else:
                raise ValueError(f"Unknown encoder for modality {modality}")
        # -- IMAGE
        elif modality == "IMAGE":
            if "hiera-image" in self.cfg["model"]["encoders"]["type"]:
                logging.info("Using hiera image")
                model, in_features = make_hiera_image(freeze=False, directory=None)
                encoder.append(("encoder", model))
            else:
                raise ValueError(f"Unknown encoder for modality {modality}")
        # -- Time Series:EDA, ECG, RR
        elif modality in ["EDA", "ECG", "RR"]:
            if "resnet-ts" in self.cfg["model"]["encoders"]["type"]:
                logging.info("Using resnet-ts")
                model = ResnetTS(
                    hidden_channels=self.cfg["model"]["encoders"]["ts_setting"][
                        "hidden"
                    ],
                    kernel_size=self.cfg["model"]["encoders"]["ts_setting"]["kernel"],
                )
                model.classifier = nn.Identity()
                in_features = model.output_features
                encoder.append(("encoder", model))
            elif "inception-ts" in self.cfg["model"]["encoders"]["type"]:
                model = Inception(
                    hidden_channels=self.cfg["model"]["encoders"]["ts_setting"][
                        "hidden"
                    ],
                    kernel_size=self.cfg["model"]["encoders"]["ts_setting"]["kernel"],
                    bottleneck=self.cfg["model"]["encoders"]["ts_setting"][
                        "bottleneck"
                    ],
                    depth=self.cfg["model"]["encoders"]["ts_setting"]["depth"],
                    rezero=self.cfg["model"]["encoders"]["ts_setting"]["rezero"],
                )
                model.classifier = nn.Identity()
                in_features = model.output_features
                encoder.append(("encoder", model))
            else:
                raise ValueError(f"Unknown encoder for modality {modality}")
        else:
            raise ValueError(f"Unknown modality: {modality}")
        if self.cfg["model"]["encoders"]["projection"]:
            encoder.append(
                (
                    "projection",
                    ProjectionHead(
                        in_features=in_features,
                        out_features=out_features,
                    ),
                )
            )
        encoder = nn.Sequential(OrderedDict(encoder))
        return encoder


class ProjectionHead(nn.Module):
    """Projection module."""

    def __init__(self, in_features, out_features, *args, **kwargs) -> None:
        """Initialize the projection module."""
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )

    def forward(self, x) -> torch.Tensor:
        """forward."""
        x = self.projection(x)
        return x
