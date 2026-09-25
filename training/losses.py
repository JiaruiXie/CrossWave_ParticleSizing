"""
Loss functions for particle sizing.

Supports both weighted and unweighted regression loss.
"""

import torch
import torch.nn as nn


class RegressionLoss(nn.Module):
    """
    Regression loss for particle sizing.

    Parameters
    ----------
    loss_type : str
        Type of regression loss.
        Options:
            "mse"  : mean squared error
            "mae"  : mean absolute error
            "huber": smooth L1 / Huber loss

    use_weights : bool
        Whether to apply sample weights.

    huber_delta : float
        Delta parameter for Huber loss.
    """

    def __init__(
        self,
        loss_type="mse",
        use_weights=False,
        huber_delta=1.0
    ):
        super().__init__()

        self.loss_type = loss_type
        self.use_weights = use_weights
        self.huber_delta = huber_delta

        assert loss_type in ["mse", "mae", "huber"], (
            "loss_type must be one of: 'mse', 'mae', 'huber'"
        )

    def forward(
        self,
        prediction,
        target,
        weight=None
    ):
        """
        Compute regression loss.

        Parameters
        ----------
        prediction : torch.Tensor
            Shape: (batch,)

        target : torch.Tensor
            Shape: (batch,)

        weight : torch.Tensor or None
            Shape: (batch,)

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """

        if self.loss_type == "mse":

            loss = (prediction - target) ** 2

        elif self.loss_type == "mae":

            loss = torch.abs(
                prediction - target
            )

        elif self.loss_type == "huber":

            loss = nn.functional.huber_loss(
                prediction,
                target,
                delta=self.huber_delta,
                reduction="none"
            )

        if self.use_weights:

            if weight is None:

                raise ValueError(
                    "use_weights=True, but weight is None."
                )

            loss = loss * weight

        return loss.mean()