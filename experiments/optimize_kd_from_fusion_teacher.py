"""
Optuna hyperparameter optimization for single-wavelength student
knowledge distillation from a trained 3-channel fusion teacher.

Teacher:
    input = [440, 555, 740]
    frozen fusion model

Student:
    input = selected wavelength only
    optimized by Optuna

Loss:
    L = alpha_target * MSE(student_pred, target)
      + alpha_pred * MSE(student_pred, teacher_pred)
      + alpha_feature * MSE(projected_student_embedding, teacher_embedding)

Model selection:
    maximize validation R2 of student prediction.
"""

import sys
from pathlib import Path
import argparse
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.kd_teacher_dataset import KDTeacherOutputDataset
from models.baseline_cnn import ConfigurableCNNRegressor
from training.kd_losses import FusionTeacherStudentKDLoss
from training.samplers import create_soft_intensity_balanced_sampler
from utils.seed import set_seed


def parse_args():

    parser = argparse.ArgumentParser(
        description="Optimize KD student from fusion teacher."
    )

    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        default=None,
        help="Optional teacher checkpoint path for record keeping only."
    )

    parser.add_argument(
        "--student_channel",
        type=str,
        required=True,
        choices=["440", "555", "740"]
    )

    parser.add_argument(
        "--teacher_output_dir",
        type=str,
        required=True,
        help="Folder containing precomputed teacher predictions and embeddings."
    )

    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument("--alpha_target", type=float, default=1.0)
    parser.add_argument("--alpha_pred", type=float, default=0.5)
    parser.add_argument("--alpha_feature", type=float, default=0.1)

    parser.add_argument("--use_soft_balanced_sampler", action="store_true")
    parser.add_argument("--sampler_alpha", type=float, default=0.3)
    parser.add_argument("--sampler_bins", type=int, default=20)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def channel_to_index(channel):

    mapping = {
        "440": 0,
        "555": 1,
        "740": 2
    }

    return mapping[channel]


def infer_teacher_architecture_from_checkpoint(checkpoint):
    """
    Infer ConfigurableCNNRegressor architecture directly from checkpoint weights.
    This is robust for Optuna checkpoints.
    """

    state_dict = checkpoint["model_state_dict"]

    conv_items = []

    for key, value in state_dict.items():

        if key.startswith("features.") and key.endswith(".weight"):

            if value.ndim == 4:

                layer_index = int(key.split(".")[1])
                out_channels = value.shape[0]
                kernel_size = value.shape[2]

                conv_items.append(
                    (
                        layer_index,
                        out_channels,
                        kernel_size
                    )
                )

    conv_items = sorted(
        conv_items,
        key=lambda x: x[0]
    )

    conv_channels = [
        item[1]
        for item in conv_items
    ]

    kernel_sizes = [
        item[2]
        for item in conv_items
    ]

    linear_items = []

    for key, value in state_dict.items():

        if key.startswith("regressor.") and key.endswith(".weight"):

            if value.ndim == 2:

                layer_index = int(key.split(".")[1])
                out_features = value.shape[0]

                linear_items.append(
                    (
                        layer_index,
                        out_features
                    )
                )

    linear_items = sorted(
        linear_items,
        key=lambda x: x[0]
    )

    dense_units = [
        item[1]
        for item in linear_items
        if item[1] != 1
    ]

    linear_indices = [
        item[0]
        for item in linear_items
    ]

    if len(linear_indices) >= 2:
        if linear_indices[1] - linear_indices[0] == 3:
            dropout_rate = 0.2
        else:
            dropout_rate = 0.0
    else:
        dropout_rate = 0.2

    config = checkpoint.get("config", None)

    if config is None:
        config = checkpoint.get("params", None)

    if config is not None:
        dropout_rate = config.get(
            "dropout_rate",
            dropout_rate
        )

    arch = {
        "in_channels": 3,
        "conv_channels": conv_channels,
        "kernel_sizes": kernel_sizes,
        "dense_units": dense_units,
        "dropout_rate": dropout_rate,
        "use_batch_norm": True,
        "embedding_dim": conv_channels[-1]
    }

    return arch


def build_teacher(checkpoint_path, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    arch = infer_teacher_architecture_from_checkpoint(
        checkpoint
    )

    teacher = ConfigurableCNNRegressor(
        in_channels=3,
        conv_channels=arch["conv_channels"],
        kernel_sizes=arch["kernel_sizes"],
        dense_units=arch["dense_units"],
        dropout_rate=arch["dropout_rate"],
        use_batch_norm=True,
        use_pooling=True
    )

    teacher.load_state_dict(
        checkpoint["model_state_dict"]
    )

    teacher.to(device)
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    return teacher, arch, checkpoint


class StudentKDModel(nn.Module):
    """
    Student model plus optional projection layer.

    The prediction uses the student's own regressor.
    Feature distillation uses projected student embedding so that
    student embedding dimension can differ from teacher embedding dimension.
    """

    def __init__(
        self,
        student,
        teacher_embedding_dim
    ):
        super().__init__()

        self.student = student

        student_embedding_dim = student.embedding_dim

        if student_embedding_dim == teacher_embedding_dim:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(
                student_embedding_dim,
                teacher_embedding_dim
            )

    def forward_with_embedding(self, x):

        pred, student_embedding = self.student.forward_with_embedding(
            x
        )

        projected_embedding = self.projection(
            student_embedding
        )

        return pred, student_embedding, projected_embedding


def suggest_student_hyperparameters(trial):
    """
    Student search space.

    Batch normalization is always True.
    """

    num_conv_layers = trial.suggest_int(
        "num_conv_layers",
        4,
        6
    )

    base_channels = trial.suggest_categorical(
        "base_channels",
        [16, 24, 32, 48, 64]
    )

    channel_multiplier = trial.suggest_categorical(
        "channel_multiplier",
        [1.5,2]
    )

    conv_channels = []

    current_channels = int(base_channels)

    for _ in range(num_conv_layers):

        conv_channels.append(
            int(current_channels)
        )

        current_channels = int(
            current_channels * channel_multiplier
        )

    conv_channels = [
        int(min(c, 256))
        for c in conv_channels
    ]

    kernel_size = 5 # kernel size = 5 seems to be the best option
    # kernel_size = trial.suggest_categorical(
    #     "kernel_size",
    #     [3, 5]
    # )

    kernel_sizes = [
        kernel_size
    ] * num_conv_layers

    dense_depth = 1 # 1 layer is sufficient for this task, no need to optimize for now
    # dense_depth = trial.suggest_int(
    #     "dense_depth",
    #     1,
    #     2
    # )

    dense_width = trial.suggest_categorical(
        "dense_width",
        [96, 112, 128, 144, 160, 176, 192]
    )

    dense_units = [
        dense_width // (2 ** i)
        for i in range(dense_depth)
    ]

    dense_units = [
        max(32, u)
        for u in dense_units
    ]

    dropout_rate = 0 # No overfitting observed, not needed for now
    # dropout_rate = trial.suggest_float(
    #     "dropout_rate",
    #     0.0,
    #     0.4
    # )

    lr = trial.suggest_float(
        "lr",
        1e-5,
        1e-3,
        log=True
    )

    batch_size = 128
    weight_decay = 10e-9
    use_batch_norm = True

    return {
        "num_conv_layers": num_conv_layers,
        "base_channels": base_channels,
        "channel_multiplier": channel_multiplier,
        "kernel_size": kernel_size,
        "dense_depth": dense_depth,
        "dense_width": dense_width,

        "conv_channels": conv_channels,
        "kernel_sizes": kernel_sizes,
        "dense_units": dense_units,
        "dropout_rate": dropout_rate,

        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "use_batch_norm": use_batch_norm
    }


def compute_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = np.mean(
        np.abs(y_pred - y_true)
    )

    mse = np.mean(
        (y_pred - y_true) ** 2
    )

    rmse = np.sqrt(
        mse
    )

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


def run_one_epoch(
    student_kd,
    dataloader,
    criterion,
    optimizer,
    device,
    student_channel_index,
    train=True
):

    if train:
        student_kd.train()
    else:
        student_kd.eval()

    total_loss = 0.0
    total_target_loss = 0.0
    total_pred_distill_loss = 0.0
    total_feature_distill_loss = 0.0
    num_batches = 0

    targets_all = []
    student_preds_all = []
    teacher_preds_all = []

    context = torch.enable_grad() if train else torch.no_grad()

    with context:

        for batch in dataloader:

            images = batch["image"].to(device)
            targets = batch["target"].to(device).float()

            teacher_pred = batch["teacher_prediction"].to(device).float()
            teacher_embedding = batch["teacher_embedding"].to(device).float()

            x_student = images[
                :,
                student_channel_index:student_channel_index + 1,
                :,
                :
            ]

            if train:
                optimizer.zero_grad()

            student_pred, student_embedding, projected_student_embedding = (
                student_kd.forward_with_embedding(
                    x_student
                )
            )

            loss_dict = criterion(
                student_pred=student_pred,
                student_embedding=projected_student_embedding,
                teacher_pred=teacher_pred,
                teacher_embedding=teacher_embedding,
                target=targets
            )

            loss = loss_dict["loss"]

            if train:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            total_target_loss += float(loss_dict["loss_target"].item())
            total_pred_distill_loss += float(loss_dict["loss_pred_distill"].item())
            total_feature_distill_loss += float(loss_dict["loss_feature_distill"].item())
            num_batches += 1

            targets_all.append(
                targets.detach().cpu().numpy()
            )

            student_preds_all.append(
                student_pred.detach().cpu().numpy()
            )

            teacher_preds_all.append(
                teacher_pred.detach().cpu().numpy()
            )

    y_true = np.concatenate(
        targets_all
    )

    y_student = np.concatenate(
        student_preds_all
    )

    y_teacher = np.concatenate(
        teacher_preds_all
    )

    student_metrics = compute_metrics(
        y_true,
        y_student
    )

    teacher_metrics = compute_metrics(
        y_true,
        y_teacher
    )

    metrics = {
        "loss": total_loss / num_batches,
        "loss_target": total_target_loss / num_batches,
        "loss_pred_distill": total_pred_distill_loss / num_batches,
        "loss_feature_distill": total_feature_distill_loss / num_batches,

        "student_MAE": student_metrics["MAE"],
        "student_MSE": student_metrics["MSE"],
        "student_RMSE": student_metrics["RMSE"],
        "student_R2": student_metrics["R2"],

        "teacher_MAE": teacher_metrics["MAE"],
        "teacher_MSE": teacher_metrics["MSE"],
        "teacher_RMSE": teacher_metrics["RMSE"],
        "teacher_R2": teacher_metrics["R2"]
    }

    return metrics


def save_checkpoint(
    student_kd,
    optimizer,
    epoch,
    train_metrics,
    val_metrics,
    output_dir,
    params,
    teacher_embedding_dim,
    args
):

    checkpoint = {
        "epoch": epoch,
        "student_state_dict": student_kd.student.state_dict(),
        "projection_state_dict": student_kd.projection.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "params": params,

        "teacher_output_dir": args.teacher_output_dir,
        "teacher_checkpoint": args.teacher_checkpoint,
        "teacher_embedding_dim": teacher_embedding_dim,

        "student_channel": args.student_channel,
        "alpha_target": args.alpha_target,
        "alpha_pred": args.alpha_pred,
        "alpha_feature": args.alpha_feature
    }

    torch.save(
        checkpoint,
        output_dir / "best_student_model.pt"
    )


def train_trial(
    teacher_embedding_dim,
    train_loader,
    val_loader,
    criterion,
    device,
    student_channel_index,
    trial_output_dir,
    params,
    args
):

    student = ConfigurableCNNRegressor(
        in_channels=1,
        conv_channels=params["conv_channels"],
        kernel_sizes=params["kernel_sizes"],
        dense_units=params["dense_units"],
        dropout_rate=params["dropout_rate"],
        use_batch_norm=True,
        use_pooling=True
    )

    student_kd = StudentKDModel(
        student=student,
        teacher_embedding_dim=teacher_embedding_dim
    )

    student_kd.to(device)

    optimizer = torch.optim.Adam(
        student_kd.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"]
    )

    history = []

    best_val_r2 = -float("inf")
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):

        train_metrics = run_one_epoch(
            student_kd=student_kd,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            student_channel_index=student_channel_index,
            train=True
        )   

        val_metrics = run_one_epoch(
            student_kd=student_kd,
            dataloader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            student_channel_index=student_channel_index,
            train=False
        )

        row = {
            "epoch": epoch,
            **{
                f"train_{k}": v
                for k, v in train_metrics.items()
            },
            **{
                f"val_{k}": v
                for k, v in val_metrics.items()
            }
        }

        history.append(row)

        current_val_r2 = val_metrics["student_R2"]

        if current_val_r2 > best_val_r2:

            best_val_r2 = current_val_r2
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0

            save_checkpoint(
                student_kd=student_kd,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                output_dir=trial_output_dir,
                params=params,
                teacher_embedding_dim=teacher_embedding_dim,
                args=args
            )   

        else:

            patience_counter += 1

            if patience_counter >= args.patience:
                break

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        trial_output_dir / "training_history.csv",
        index=False
    )

    return history_df, best_val_r2, best_val_loss, best_epoch


def build_study_name(args):

    sampler_tag = (
        f"_soft_sampler_alpha_{args.sampler_alpha:g}"
        if args.use_soft_balanced_sampler
        else ""
    )

    study_name = (
        f"kd_student_from_fusion_teacher"
        f"_v1_projection"
        f"_student_{args.student_channel}"
        f"_at_{args.alpha_target:g}"
        f"_ap_{args.alpha_pred:g}"
        f"_af_{args.alpha_feature:g}"
        f"{sampler_tag}"
    )

    return study_name


def objective(
    trial,
    args,
    teacher_embedding_dim,
    train_dataset,
    val_dataset,
    device,
    study_output_dir
):

    set_seed(
        args.seed + trial.number
    )

    params = suggest_student_hyperparameters(
        trial
    )

    trial_output_dir = (
        study_output_dir
        / f"trial_{trial.number:03d}"
    )

    trial_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(trial_output_dir / "params.json", "w") as f:
        json.dump(
            params,
            f,
            indent=4
        )

    if args.use_soft_balanced_sampler:

        train_sampler = create_soft_intensity_balanced_sampler(
            dataset=train_dataset,
            num_bins=args.sampler_bins,
            alpha=args.sampler_alpha,
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=params["batch_size"],
            sampler=train_sampler,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

    else:

        train_loader = DataLoader(
            train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    criterion = FusionTeacherStudentKDLoss(
        alpha_target=args.alpha_target,
        alpha_pred=args.alpha_pred,
        alpha_feature=args.alpha_feature
    )

    student_channel_index = channel_to_index(
        args.student_channel
    )

    history, best_val_r2, best_val_loss, best_epoch = train_trial(
        teacher_embedding_dim=teacher_embedding_dim,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        student_channel_index=student_channel_index,
        trial_output_dir=trial_output_dir,
        params=params,
        args=args
    )

    trial.set_user_attr(
        "best_epoch",
        int(best_epoch)
    )

    trial.set_user_attr(
        "best_val_R2",
        float(best_val_r2)
    )

    trial.set_user_attr(
        "best_val_loss",
        float(best_val_loss)
    )

    trial.set_user_attr(
        "trial_output_dir",
        str(trial_output_dir)
    )

    trial.set_user_attr(
        "conv_channels",
        params["conv_channels"]
    )

    trial.set_user_attr(
        "kernel_sizes",
        params["kernel_sizes"]
    )

    trial.set_user_attr(
        "dense_units",
        params["dense_units"]
    )

    trial.set_user_attr(
        "student_embedding_dim",
        int(params["conv_channels"][-1])
    )

    trial.set_user_attr(
        "teacher_embedding_dim",
        int(teacher_embedding_dim)
    )

    print(
        f"Trial {trial.number:03d} completed | "
        f"student_channel={args.student_channel} | "
        f"best_val_R2={best_val_r2:.6f} | "
        f"best_val_loss={best_val_loss:.6f} | "
        f"best_epoch={best_epoch} | "
        f"lr={params['lr']:.2e} | "
        f"conv={params['conv_channels']} | "
        f"dense={params['dense_units']} | "
        f"dropout={params['dropout_rate']:.3f}"
    )

    return best_val_r2

def get_teacher_embedding_dim(teacher_output_dir):

    summary_path = (
        Path(teacher_output_dir)
        / "teacher_output_summary.json"
    )

    if summary_path.exists():

        with open(summary_path, "r") as f:
            summary = json.load(f)

        return int(
            summary["teacher_arch"]["embedding_dim"]
        )

    emb_path = (
        Path(teacher_output_dir)
        / "train_teacher_embeddings.npy"
    )

    embeddings = np.load(
        emb_path,
        mmap_mode="r"
    )

    return int(
        embeddings.shape[1]
    )


def main():

    args = parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    study_name = build_study_name(
        args
    )

    study_output_dir = (
        Path("outputs")
        / "optuna"
        / study_name
    )

    study_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 80)
    print("Optuna Optimization: KD Student from Fusion Teacher")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Study name: {study_name}")
    print(f"Study output dir: {study_output_dir}")
    print(f"Teacher checkpoint: {args.teacher_checkpoint}")
    print(f"Student channel: {args.student_channel}")
    print(f"Alpha target: {args.alpha_target}")
    print(f"Alpha pred: {args.alpha_pred}")
    print(f"Alpha feature: {args.alpha_feature}")
    print(f"Use soft sampler: {args.use_soft_balanced_sampler}")
    print(f"Sampler alpha: {args.sampler_alpha}")
    print(f"Trials: {args.n_trials}")
    print(f"Epochs: {args.epochs}")
    print(f"Patience: {args.patience}")

    with open(study_output_dir / "optimization_config.json", "w") as f:
        json.dump(
            vars(args),
            f,
            indent=4
        )

    teacher_embedding_dim = get_teacher_embedding_dim(
        args.teacher_output_dir
    )

    print()
    print(f"Teacher output dir: {args.teacher_output_dir}")
    print(f"Teacher embedding dim: {teacher_embedding_dim}")

    train_dataset = KDTeacherOutputDataset(
        data_dir="data/processed",
        split="train",
        teacher_output_dir=args.teacher_output_dir,
        channels=[0, 1, 2],
        target_transform="cube_root",
        use_weights=False,
        mmap=True
    )

    val_dataset = KDTeacherOutputDataset(
        data_dir="data/processed",
        split="val",
        teacher_output_dir=args.teacher_output_dir,
        channels=[0, 1, 2],
        target_transform="cube_root",
        use_weights=False,
        mmap=True
    )

    storage_path = (
        study_output_dir.parent
        / f"{study_name}.db"
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True
    )

    study.optimize(
        lambda trial: objective(
            trial=trial,
            args=args,
            teacher_embedding_dim=teacher_embedding_dim,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            study_output_dir=study_output_dir
        ),
        n_trials=args.n_trials
    )

    trials_df = study.trials_dataframe()

    trials_df.to_csv(
        study_output_dir / "trials.csv",
        index=False
    )

    best_trial = study.best_trial

    best_trial_info = {
        "best_trial_number": best_trial.number,
        "objective": f"maximize_student_val_R2_{args.student_channel}",
        "best_val_R2": best_trial.value,
        "best_params": best_trial.params,
        "best_user_attrs": best_trial.user_attrs
    }

    with open(study_output_dir / "best_trial.json", "w") as f:
        json.dump(
            best_trial_info,
            f,
            indent=4
        )

    print()
    print("=" * 80)
    print("Optimization finished")
    print("=" * 80)
    print(f"Best trial: {best_trial.number}")
    print(f"Best validation R2: {best_trial.value}")
    print("Best params:")
    print(json.dumps(best_trial.params, indent=4))
    print("Best user attrs:")
    print(json.dumps(best_trial.user_attrs, indent=4))


if __name__ == "__main__":

    main()