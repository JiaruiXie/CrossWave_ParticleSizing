"""
Train a single-wavelength student using knowledge distillation
from a trained 3-channel fusion teacher.

Teacher:
    input channels = [440, 555, 740]
    trained fusion model

Student:
    input channel = selected wavelength only
    learns to match:
        1. true Cy3 target
        2. teacher prediction
        3. teacher embedding
"""

import sys
from pathlib import Path
import argparse
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.particle_dataset import ParticleDataset
from models.baseline_cnn import ConfigurableCNNRegressor
from training.kd_losses import FusionTeacherStudentKDLoss
from training.samplers import create_soft_intensity_balanced_sampler
from utils.seed import set_seed


def parse_args():

    parser = argparse.ArgumentParser(
        description="Train single-wavelength student from fusion teacher."
    )

    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--student_channel",
        type=str,
        required=True,
        choices=["440", "555", "740"]
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-8
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20
    )

    parser.add_argument(
        "--alpha_target",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--alpha_pred",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--alpha_feature",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--student_conv_channels",
        type=str,
        default="[16, 32, 64, 128]"
    )

    parser.add_argument(
        "--student_kernel_sizes",
        type=str,
        default="[3, 3, 3, 3]"
    )

    parser.add_argument(
        "--student_dense_units",
        type=str,
        default="[256, 128, 64]"
    )

    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--use_soft_balanced_sampler",
        action="store_true"
    )

    parser.add_argument(
        "--sampler_alpha",
        type=float,
        default=0.3
    )

    parser.add_argument(
        "--sampler_bins",
        type=int,
        default=20
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return parser.parse_args()


def channel_to_index(channel):

    mapping = {
        "440": 0,
        "555": 1,
        "740": 2
    }

    return mapping[channel]


def compute_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


def infer_teacher_architecture_from_checkpoint(checkpoint):
    """
    Infer ConfigurableCNNRegressor architecture directly from the checkpoint
    state_dict. This is more reliable for Optuna checkpoints.
    """

    state_dict = checkpoint["model_state_dict"]

    # --------------------------------------------------------
    # Infer convolutional architecture
    # --------------------------------------------------------

    conv_items = []

    for key, value in state_dict.items():

        if key.startswith("features.") and key.endswith(".weight"):

            # Conv2d weight shape: [out_channels, in_channels, kH, kW]
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

    # --------------------------------------------------------
    # Infer dense architecture
    # --------------------------------------------------------

    linear_items = []

    for key, value in state_dict.items():

        if key.startswith("regressor.") and key.endswith(".weight"):

            # Linear weight shape: [out_features, in_features]
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

    # Last linear layer is output layer with out_features = 1.
    dense_units = [
        item[1]
        for item in linear_items
        if item[1] != 1
    ]

    # --------------------------------------------------------
    # Infer dropout setting
    # --------------------------------------------------------
    # If the final Linear layer index is 4, 7, 10, etc.,
    # dropout was included after hidden dense layers.
    # Use 0.2 as safe default unless stored elsewhere.
    # Exact dropout rate does not affect checkpoint parameter shapes,
    # but dropout module presence affects layer indexing.
    # Therefore it must be > 0 if checkpoint has regressor.4 for one hidden layer.

    linear_indices = [
        item[0]
        for item in linear_items
    ]

    if len(linear_indices) >= 2:

        first_linear_index = linear_indices[0]
        second_linear_index = linear_indices[1]

        if second_linear_index - first_linear_index == 3:
            dropout_rate = 0.2
        else:
            dropout_rate = 0.0

    else:
        dropout_rate = 0.2

    # If checkpoint stores config/params, prefer exact dropout rate.
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
        "use_batch_norm": True
    }

    print()
    print("Inferred teacher architecture:")
    print(json.dumps(arch, indent=4))

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
        use_batch_norm=arch["use_batch_norm"],
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


def run_one_epoch(
    teacher,
    student,
    dataloader,
    criterion,
    optimizer,
    device,
    student_channel_index,
    train=True
):

    if train:
        student.train()
    else:
        student.eval()

    teacher.eval()

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

            x_student = images[
                :,
                student_channel_index:student_channel_index + 1,
                :,
                :
            ]

            with torch.no_grad():

                teacher_pred, teacher_embedding = teacher.forward_with_embedding(
                    images
                )

            if train:
                optimizer.zero_grad()

            student_pred, student_embedding = student.forward_with_embedding(
                x_student
            )

            loss_dict = criterion(
                student_pred=student_pred,
                student_embedding=student_embedding,
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

    y_true = np.concatenate(targets_all)
    student_pred = np.concatenate(student_preds_all)
    teacher_pred = np.concatenate(teacher_preds_all)

    student_metrics = compute_metrics(
        y_true,
        student_pred
    )

    teacher_metrics = compute_metrics(
        y_true,
        teacher_pred
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
    student,
    optimizer,
    epoch,
    train_metrics,
    val_metrics,
    output_dir,
    config
):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "config": config
    }

    torch.save(
        checkpoint,
        output_dir / "best_student_model.pt"
    )


def main():

    args = parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    student_channel_index = channel_to_index(
        args.student_channel
    )

    student_conv_channels = json.loads(
        args.student_conv_channels
    )

    student_kernel_sizes = json.loads(
        args.student_kernel_sizes
    )

    student_dense_units = json.loads(
        args.student_dense_units
    )

    if args.output_dir is None:

        sampler_tag = (
            f"_soft_sampler_alpha_{args.sampler_alpha:g}"
            if args.use_soft_balanced_sampler
            else ""
        )

        output_dir = Path(
            f"outputs/kd_from_fusion_teacher"
            f"_student_{args.student_channel}"
            f"_at_{args.alpha_target:g}"
            f"_ap_{args.alpha_pred:g}"
            f"_af_{args.alpha_feature:g}"
            f"{sampler_tag}"
        )

    else:

        output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 80)
    print("Knowledge Distillation: Fusion Teacher -> Single-Wavelength Student")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Teacher checkpoint: {args.teacher_checkpoint}")
    print(f"Student channel: {args.student_channel}")
    print(f"Output dir: {output_dir}")

    teacher, teacher_arch, teacher_checkpoint = build_teacher(
        checkpoint_path=args.teacher_checkpoint,
        device=device
    )

    student = ConfigurableCNNRegressor(
        in_channels=1,
        conv_channels=student_conv_channels,
        kernel_sizes=student_kernel_sizes,
        dense_units=student_dense_units,
        dropout_rate=args.dropout_rate,
        use_batch_norm=True,
        use_pooling=True
    )

    student.to(device)

    if student.embedding_dim != teacher.embedding_dim:
        raise ValueError(
            "Student and teacher embedding dimensions do not match. "
            f"Teacher embedding_dim={teacher.embedding_dim}, "
            f"Student embedding_dim={student.embedding_dim}. "
            "Use student_conv_channels with the same final channel size as the teacher."
        )

    criterion = FusionTeacherStudentKDLoss(
        alpha_target=args.alpha_target,
        alpha_pred=args.alpha_pred,
        alpha_feature=args.alpha_feature
    )

    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    train_dataset = ParticleDataset(
        data_dir="data/processed",
        split="train",
        channels=[0, 1, 2],
        target_transform="cube_root",
        use_weights=False,
        mmap=True
    )

    val_dataset = ParticleDataset(
        data_dir="data/processed",
        split="val",
        channels=[0, 1, 2],
        target_transform="cube_root",
        use_weights=False,
        mmap=True
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
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

    else:

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    config = {
        "teacher_checkpoint": args.teacher_checkpoint,
        "teacher_arch": teacher_arch,
        "student_channel": args.student_channel,
        "student_channel_index": student_channel_index,
        "student_conv_channels": student_conv_channels,
        "student_kernel_sizes": student_kernel_sizes,
        "student_dense_units": student_dense_units,
        "student_embedding_dim": student.embedding_dim,
        "dropout_rate": args.dropout_rate,
        "use_batch_norm": True,
        "alpha_target": args.alpha_target,
        "alpha_pred": args.alpha_pred,
        "alpha_feature": args.alpha_feature,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "use_soft_balanced_sampler": args.use_soft_balanced_sampler,
        "sampler_alpha": args.sampler_alpha,
        "sampler_bins": args.sampler_bins,
        "seed": args.seed
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(
            config,
            f,
            indent=4
        )

    history = []

    best_val_r2 = -float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):

        train_metrics = run_one_epoch(
            teacher=teacher,
            student=student,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            student_channel_index=student_channel_index,
            train=True
        )

        val_metrics = run_one_epoch(
            teacher=teacher,
            student=student,
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

        print(
            f"Epoch {epoch:03d} | "
            f"val_student_R2={val_metrics['student_R2']:.4f} | "
            f"val_student_MAE={val_metrics['student_MAE']:.4f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"target={val_metrics['loss_target']:.6f} | "
            f"predKD={val_metrics['loss_pred_distill']:.6f} | "
            f"featureKD={val_metrics['loss_feature_distill']:.6f} | "
            f"teacher_R2={val_metrics['teacher_R2']:.4f}"
        )

        if current_val_r2 > best_val_r2:

            best_val_r2 = current_val_r2
            best_epoch = epoch
            patience_counter = 0

            save_checkpoint(
                student=student,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                output_dir=output_dir,
                config=config
            )

            print(f"Best student model saved at epoch {epoch}.")

        else:

            patience_counter += 1

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        output_dir / "training_history.csv",
        index=False
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_student_R2": best_val_r2,
        "output_dir": str(output_dir)
    }

    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(
            summary,
            f,
            indent=4
        )

    print()
    print("=" * 80)
    print("KD training finished")
    print("=" * 80)
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":

    main()