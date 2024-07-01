"""
Model for time series features extraction.
------------------------------------------
Resnet implementation follows :
Deep learning for time series classification: a review, Fawaz et al.

    
Inception implementation follows:
---------------------------------
From https://arxiv.org/abs/1909.04939 
and adapted from https://github.com/timeseriesAI/tsai/blob/main/tsai/models/InceptionTime.py

"""

import torch
import pyrootutils
from torch import nn

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


class ResnetTS(nn.Module):
    """ResNet model for time series classification."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        num_classes: int = 2,
        kernel_size: int = 8,
    ) -> None:
        """
        Initialization

        Parameters
        ----------
        in_channels : int
            Number of input channels
        hidden_channels : int
            Number of hidden channels
        num_classes : int, default=2
            Number of classes
        kernel_size : int, default=8
            Kernel size
        """
        super().__init__()
        self.features = nn.Sequential(
            *[
                ResidualBlock(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                ),
                ResidualBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels * 2,
                    kernel_size=kernel_size,
                ),
                ResidualBlock(
                    in_channels=hidden_channels * 2,
                    out_channels=hidden_channels * 2,
                    kernel_size=kernel_size,
                ),
            ]
        )
        self.output_features = hidden_channels * 2
        self.classifier = nn.Linear(hidden_channels * 2, num_classes)

    @staticmethod
    def weight_initilization(layer):
        """Weight initialization for the ResNet model."""
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(
                layer.weight, mode="fan_in", nonlinearity="relu"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResNet model."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x.mean(dim=-1))


class InceptionModule(nn.Module):
    """Inception module implementation."""

    def __init__(self, in_channels, hidden_channels, kernel_size=40, bottleneck=True):
        """
        Initialize the inception module.

        Parameters
        ----------
        in_channels : int
            the input dimension of the network i.e number of channels.
        hidden_dimension : int
            Number of output channels.
            The dimensions of the output of the bottleneck layer.
        kernel_size : int, optional
            The kernel size of the biggest convolutional layer, by default 40
        bottleneck : bool, optional
            Whether to use a bottleneck layer or not, by default True
        """
        super().__init__()
        kernels_size = [kernel_size // (2**i) for i in range(3)]
        bottleneck = bottleneck if in_channels > 1 else False

        self.bottleneck = (
            nn.Conv1d(in_channels, hidden_channels, 1, bias=False)
            if bottleneck
            else nn.Sequential()
        )

        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden_channels if bottleneck else in_channels,
                    hidden_channels,
                    size,
                    padding="same",
                    bias=False,
                )
                for size in kernels_size
            ]
        )
        self.bottleneck_convolution = nn.Sequential(
            *[
                nn.MaxPool1d(3, stride=1, padding=1),
                nn.Conv1d(in_channels, hidden_channels, 1, padding=0, bias=False),
            ]
        )
        self.concat = Concat()
        self.batch_norm = nn.BatchNorm1d(num_features=hidden_channels * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass of the inception module."""
        input_ = x
        if self.bottleneck:
            x = self.bottleneck(input_)
        x = torch.cat(
            [l(x) for l in self.convolutions] + [self.bottleneck_convolution(input_)],
            dim=1,
        )
        return self.relu(self.batch_norm(x))


class Inception(nn.Module):
    """Inception block.

    One inception block is composed of :
        - 1 bottleneck layers (1x1 conv)
        - 3 convolutional layers in parallel with different kernel sizes
        - 1 residual connection
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        residual: bool = True,
        depth: int = 6,
        bottleneck: bool = True,
        rezero: bool = True,
        num_classes: int = 2,
    ) -> None:
        """
        Initialize the inception block.

        Parameters
        ----------
        input_dimension : int
            the input dimension of the network i.e number of channels.
        hidden_dimension : int
            Number of output channels.
        kernel_size : int
            The kernel size of the biggest convolutional layer.
        residual : bool, optional
            Whether to use a residual connection or not, by default True
        depth : int, optional
            The number of inception modules in the block, by default 6
        bottleneck : bool, optional
            Whether to use a bottleneck layer or not, by default True
        """
        super().__init__()
        self.residual, self.depth = residual, depth
        self.inception = nn.ModuleList()
        self.shortcut = nn.ModuleList()
        for idx_depth in range(depth):
            self.inception.append(
                InceptionModule(
                    in_channels if idx_depth == 0 else hidden_channels * 4,
                    hidden_channels,
                    kernel_size,
                    bottleneck,
                )
            )
            if residual and (idx_depth + 1) % 2 == 0:
                input_dim_shortcut = (
                    in_channels if (idx_depth + 1) == 2 else hidden_channels * 4
                )
                out_dim_shortcut = hidden_channels * 4
                self.shortcut.append(
                    nn.BatchNorm1d(input_dim_shortcut)
                    if input_dim_shortcut == out_dim_shortcut
                    else ConvolutionBlock(in_channels, out_dim_shortcut, kernel_size=1)
                )
        if rezero:
            self.rezero = nn.ParameterList(
                [nn.Parameter(torch.zeros(1)) for _ in range(len(self.shortcut))]
            )
        else:
            self.rezero = None
        self.output_features = hidden_channels * 4  # 4 = 3 conv + 1 bottleneck conv
        self.relu = nn.ReLU()
        self.gap = GAP1d()
        self.classifier = nn.Linear(hidden_channels * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the inception block."""
        residual = x
        residual_count = 0
        for idx_depth in range(self.depth):
            x = self.inception[idx_depth](x)
            if (
                self.residual and (idx_depth + 1) % 2 == 0
            ):  # residual connection every 2 inception modules.
                if self.rezero is not None:
                    x = x * self.rezero[residual_count]
                residual = x = self.relu(x + self.shortcut[residual_count](residual))
                residual_count += 1
        representation = self.gap(x)
        x = self.classifier(representation)
        return x, representation


###################### UTILS ############################


class ConvolutionBlock(nn.Module):
    """Convolutional block for the time series classification models."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding: int = None,
        bias: bool = True,
    ) -> None:
        """
        This implementation follows :
        Deep learning for time series classification: a review, Fawaz et al.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Size of the kernel
        stride : int, default=1
            Stride of the convolution
        dilation : int, default=1
            Dilation of the convolution
        bias : bool, default=True
            Whether to use a bias or not
        """
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.layer = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                padding=padding,
                padding_mode="zeros",
                dilation=dilation,
            ),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass of the convolutional block."""
        return self.layer(x)


class ResidualBlock(nn.Module):
    """Residual block for the time series classification models."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 8
    ) -> None:
        """Initialization of the residual block.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        """
        super().__init__()
        channels = [in_channels, out_channels, out_channels, out_channels]
        kernels = [
            kernel_size,
            kernel_size // 2 + 1,
            kernel_size // 4 + 1,
        ]  # kernel of 8 ??
        self.convolutions = nn.Sequential(
            *[
                ConvolutionBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernels[i],
                    padding="same",
                    stride=1,
                )
                for i in range(len(kernels))
            ]
        )
        # expand channels for the sum if necessary
        self.shortcut = (
            nn.BatchNorm1d(num_features=out_channels)
            if in_channels == out_channels
            else ConvolutionBlock(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding="same",
                bias=False,
            )
        )
        self.residual = nn.Sequential(
            *[
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding_mode="zeros",
                ),
                nn.BatchNorm1d(num_features=out_channels),
            ]
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the residual block."""
        x = self.convolutions(x) + self.residual(x)
        return self.relu(x)


class Add(nn.Module):
    """Addition module."""

    def forward(self, x, y):
        """forward pass."""
        super().__init__()
        return x.add(y)

    def __repr__(self):
        """representation"""
        return f"{self.__class__.__name__}"


class Concat(nn.Module):
    """ "Concatenation module."""

    def __init__(self, dim=1):
        """forward pass."""
        super().__init__()
        self.dim = dim

    def forward(self, *x):
        """forward pass."""
        return torch.cat(*x, dim=self.dim)


class Reshape(nn.Module):
    """Reshape module."""

    def __init__(self, *shape):
        """Initialization."""
        super().__init__()
        self.shape = shape

    def forward(self, x):
        """forward pass."""
        return (
            x.reshape(x.shape[0], -1)
            if not self.shape
            else (
                x.reshape(-1)
                if self.shape == (-1,)
                else x.reshape(x.shape[0], *self.shape)
            )
        )


class GAP1d(nn.Module):
    "Global Adaptive Pooling + Flatten"

    def __init__(self, output_size=1):
        """Initialization."""
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = Reshape()

    def forward(self, x):
        """forward pass."""
        return self.flatten(self.gap(x))
