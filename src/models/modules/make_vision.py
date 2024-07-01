"""Vision models"""

import torch
import hiera
from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights


def make_hiera(freeze: bool = True, directory: str = None):
    """
    Parameters:
    ----------
    freeze : bool
        Freeze the weights of the model
    directory : str
        path to the pretrained model

    NB :
        N,C,D,H,W
    """
    if directory is not None:
        features_extractor = torch.hub.load(
            repo_or_dir=directory,
            model="hiera_base_16x224",
            pretrained=False,
            source="local",
        )
        features_extractor.load_state_dict(
            torch.load(directory + "/hiera_base_16x224.pth")["model_state"], strict=True
        )
    else:
        features_extractor = hiera.hiera_base_16x224(
            pretrained=True, checkpoint="mae_k400_ft_k400"
        )
    if freeze:
        features_extractor.eval()
        for param in features_extractor.parameters():
            param.requires_grad = False
    dim = features_extractor.head.projection.in_features
    features_extractor.head = nn.Identity()
    return features_extractor, dim


def make_resnet3d(freeze: bool = True, directory: int = None):
    """
    Maked 3d resnet.

    Parameters:
    ----------
    freeze : bool
        Freeze the weights of the model
    NB :
        N,C,D,H,W

    [1] : https://github.com/pytorch/vision/tree/main/references/video_classification
    """
    if directory is None:
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    else:
        model = r3d_18(weights=None)
        model.load_state_dict(torch.load(directory + "/r3d_18-b3b3357e.pth"))
    if freeze:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, dim


def make_hiera_image(freeze: bool = True, directory: str = None):
    """
    Make hiera image.

    Parameters:
    ----------
    freeze : bool
        Freeze the weights of the model
    directory : str
        path to the pretrained model

    Returns:
    -------
    features_extractor : nn.Module
        Model
    dim : int
        Dimension of output
    """
    if directory is not None:
        features_extractor = torch.hub.load(
            repo_or_dir=directory,
            model="hiera_tiny_224",
            pretrained=False,
            source="local",
        )
        features_extractor.load_state_dict(
            torch.load(directory + "/hiera_tiny_224.pth")["model_state"], strict=True
        )
    else:
        features_extractor = hiera.hiera_tiny_224(
            pretrained=True, checkpoint="mae_in1k_ft_in1k"
        )
    if freeze:
        features_extractor.eval()
        for param in features_extractor.parameters():
            param.requires_grad = False
    dim = features_extractor.head.projection.in_features
    features_extractor.head = nn.Identity()
    return features_extractor, dim
