"""
Baseline CNN regressor for particle-level sizing.

This model supports both single-wavelength and multi-wavelength inputs.

Input shape:
    (batch_size, in_channels, 64, 64)

Output shape:
    (batch_size,)
"""

import torch
import torch.nn as nn


class BaselineCNNRegressor(nn.Module):
    """
    Simple CNN regressor for particle sizing.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
        1 for single wavelength.
        3 for early-fusion three-wavelength input.

    base_channels : int
        Number of channels in the first convolution layer.

    dropout_rate : float
        Dropout probability before the final regression head.
    """

    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        dropout_rate=0.2
    ):
        super().__init__()

        self.features = nn.Sequential(
            # 64 x 64
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 32 x 32
            nn.Conv2d(
                base_channels,
                base_channels * 2,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 16 x 16
            nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 8 x 8
            nn.Conv2d(
                base_channels * 4,
                base_channels * 8,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels * 4, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        x = x.squeeze(1)

        return x
    

"""
Configurable CNN regressor for particle-level sizing.

This model supports:
1. Single-wavelength input, e.g. 440-only, 555-only, 740-only.
2. Early-fusion input, e.g. 440 + 555 + 740.

Input shape:
    (batch_size, in_channels, 64, 64)

Output shape:
    (batch_size,)
"""

class ConfigurableCNNRegressor(nn.Module):
    """
    Configurable CNN regressor.

    Parameters
    ----------
    in_channels : int
        Number of input channels.

    conv_channels : list of int
        Number of output channels for each convolutional layer.
        The length of this list determines the number of convolutional layers.

    kernel_sizes : int or list of int
        Kernel size for each convolutional layer.
        If an integer is given, the same kernel size is used for all conv layers.

    dense_units : list of int
        Number of neurons in each dense layer.
        The length of this list determines the number of dense layers.

    dropout_rate : float
        Dropout probability applied after each dense hidden layer.

    use_batch_norm : bool
        Whether to use BatchNorm2d after each convolution.

    use_pooling : bool
        Whether to use MaxPool2d after each convolutional block.
    """

    def __init__(
        self,
        in_channels=1,
        conv_channels=None,
        kernel_sizes=3,
        dense_units=None,
        dropout_rate=0.2,
        use_batch_norm=True,
        use_pooling=True
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = [16, 32, 64, 128]

        if dense_units is None:
            dense_units = [128, 64]

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(conv_channels)

        assert len(kernel_sizes) == len(conv_channels), (
            "kernel_sizes must be an integer or a list with the same length "
            "as conv_channels."
        )

        assert len(conv_channels) > 0, (
            "conv_channels must contain at least one convolutional layer."
        )

        assert len(dense_units) > 0, (
            "dense_units must contain at least one dense layer."
        )

        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_pooling = use_pooling

        # ====================================================
        # Convolutional feature extractor
        # ====================================================

        conv_layers = []

        current_channels = in_channels

        for out_channels, kernel_size in zip(
            conv_channels,
            kernel_sizes
        ):

            padding = kernel_size // 2

            conv_layers.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding
                )
            )

            if use_batch_norm:
                conv_layers.append(
                    nn.BatchNorm2d(out_channels)
                )

            conv_layers.append(
                nn.ReLU(inplace=True)
            )

            if use_pooling:
                conv_layers.append(
                    nn.MaxPool2d(kernel_size=2)
                )

            current_channels = out_channels

        conv_layers.append(
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.features = nn.Sequential(
            *conv_layers
        )

        # ====================================================
        # Dense regression head
        # ====================================================

        dense_layers = []

        current_features = conv_channels[-1]

        for units in dense_units:

            dense_layers.append(
                nn.Linear(
                    current_features,
                    units
                )
            )

            dense_layers.append(
                nn.ReLU(inplace=True)
            )

            if dropout_rate > 0:
                dense_layers.append(
                    nn.Dropout(dropout_rate)
                )

            current_features = units

        dense_layers.append(
            nn.Linear(
                current_features,
                1
            )
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            *dense_layers
        )

    def forward(self, x):

        x = self.features(x)

        x = self.regressor(x)

        x = x.squeeze(1)

        return x