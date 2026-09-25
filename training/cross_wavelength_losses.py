"""
Loss functions for cross-wavelength knowledge transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossWavelengthAlignmentLoss(nn.Module):
    """
    Regression loss plus cross-wavelength latent alignment.

    The alignment loss encourages latent features from different wavelengths
    of the same particle to be close to their shared mean representation.

    Total loss:
        L = L_reg + lambda_align * L_align
    """

    def __init__(
        self,
        lambda_align=0.01,
        alignment_type="mean",
        use_weights=False
    ):
        super().__init__()

        self.lambda_align = lambda_align
        self.alignment_type = alignment_type
        self.use_weights = use_weights

    def regression_loss(
        self,
        prediction,
        target,
        weight=None
    ):

        loss = (
            prediction - target
        ) ** 2

        if self.use_weights and weight is not None:
            loss = loss * weight

        return loss.mean()

    def mean_alignment_loss(
        self,
        z440,
        z555,
        z740,
        z_mean
    ):

        loss = (
            F.mse_loss(z440, z_mean)
            + F.mse_loss(z555, z_mean)
            + F.mse_loss(z740, z_mean)
        ) / 3.0

        return loss

    def pairwise_alignment_loss(
        self,
        z440,
        z555,
        z740
    ):

        loss = (
            F.mse_loss(z440, z555)
            + F.mse_loss(z440, z740)
            + F.mse_loss(z555, z740)
        ) / 3.0

        return loss

    def forward(
        self,
        outputs,
        target,
        weight=None
    ):

        prediction = outputs["prediction"]

        loss_reg = self.regression_loss(
            prediction=prediction,
            target=target,
            weight=weight
        )

        if self.alignment_type == "mean":

            loss_align = self.mean_alignment_loss(
                z440=outputs["z440"],
                z555=outputs["z555"],
                z740=outputs["z740"],
                z_mean=outputs["z_mean"]
            )

        elif self.alignment_type == "pairwise":

            loss_align = self.pairwise_alignment_loss(
                z440=outputs["z440"],
                z555=outputs["z555"],
                z740=outputs["z740"]
            )

        else:
            raise ValueError(
                f"Unknown alignment_type: {self.alignment_type}"
            )

        total_loss = (
            loss_reg
            + self.lambda_align * loss_align
        )

        return {
            "loss": total_loss,
            "loss_reg": loss_reg.detach(),
            "loss_align": loss_align.detach()
        }