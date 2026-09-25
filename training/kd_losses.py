"""
Knowledge distillation losses for multi-wavelength fusion teacher
to single-wavelength student.

Teacher:
    input = [440, 555, 740]
    output = teacher prediction + teacher embedding

Student:
    input = one selected wavelength
    output = student prediction + student embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionTeacherStudentKDLoss(nn.Module):

    def __init__(
        self,
        alpha_target=1.0,
        alpha_pred=0.5,
        alpha_feature=0.1
    ):
        super().__init__()

        self.alpha_target = alpha_target
        self.alpha_pred = alpha_pred
        self.alpha_feature = alpha_feature

    def forward(
        self,
        student_pred,
        student_embedding,
        teacher_pred,
        teacher_embedding,
        target
    ):

        loss_target = F.mse_loss(
            student_pred,
            target
        )

        loss_pred_distill = F.mse_loss(
            student_pred,
            teacher_pred.detach()
        )

        loss_feature_distill = F.mse_loss(
            student_embedding,
            teacher_embedding.detach()
        )

        total_loss = (
            self.alpha_target * loss_target
            + self.alpha_pred * loss_pred_distill
            + self.alpha_feature * loss_feature_distill
        )

        return {
            "loss": total_loss,
            "loss_target": loss_target.detach(),
            "loss_pred_distill": loss_pred_distill.detach(),
            "loss_feature_distill": loss_feature_distill.detach()
        }