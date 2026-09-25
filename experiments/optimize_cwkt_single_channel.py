"""
Optuna hyperparameter optimization for CWKT single-channel inference model.

CWKT idea:
    Training uses paired 440 / 555 / 740 nm patches.
    The shared encoder maps each wavelength into a latent embedding.
    Embeddings are aligned using mean alignment.
    Each wavelength predicts the same Cy3-derived target.

Inference:
    Use only one selected wavelength.

Optuna objective:
    Maximize validation R2 of the selected validation channel only.
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
from torch.utils.data import DataLoader

from datasets.particle_dataset import ParticleDataset
from models.cwkt_single_channel import CWKTSingleChannelRegressor
from training.cwkt_losses import CWKTMeanAlignmentLoss
from training.samplers import create_soft_intensity_balanced_sampler
from utils.seed import set_seed


def parse_args():

    parser = argparse.ArgumentParser(
        description="Optimize CWKT single-channel inference model."
    )

    parser.add_argument(
        "--validation_channel",
        type=str,
        required=True,
        choices=["440", "555", "740"],
        help="Use this wavelength only for validation R2 model selection."
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=50
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20
    )

    parser.add_argument(
        "--lambda_align",
        type=float,
        default=0.01
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
        "--seed",
        type=int,
        default=42
    )

    return parser.parse_args()


def suggest_hyperparameters(trial):
    """
    Hyperparameter search space requested by the user.
    """

    # --------------------------------------------------------
    # Encoder architecture
    # --------------------------------------------------------

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
        [1.5]
    )

    conv_channels = []

    current_channels = base_channels

    for _ in range(num_conv_layers):

        conv_channels.append(
            current_channels
        )

        current_channels = current_channels * channel_multiplier

    conv_channels = [
        min(c, 256)
        for c in conv_channels
    ]

    kernel_size = trial.suggest_categorical(
        "kernel_size",
        [3, 5]
    )

    kernel_sizes = [
        kernel_size
    ] * num_conv_layers

    # --------------------------------------------------------
    # Regressor architecture
    # --------------------------------------------------------

    dense_depth = trial.suggest_int(
        "dense_depth",
        1,
        2
    )

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

    # In this CWKT model, latent_dim is tied to dense_width.
    latent_dim = dense_width

    dropout_rate = trial.suggest_float(
        "dropout_rate",
        0.1,
        0.4
    )

    use_batch_norm = True

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    lr = trial.suggest_float(
        "lr",
        1e-5,
        1e-3,
        log=True
    )

    batch_size = 128

    weight_decay = 10e-9

    return {
        "num_conv_layers": num_conv_layers,
        "base_channels": base_channels,
        "channel_multiplier": channel_multiplier,
        "kernel_size": kernel_size,
        "dense_depth": dense_depth,
        "dense_width": dense_width,

        "conv_channels": conv_channels,
        "kernel_sizes": kernel_sizes,
        "regressor_hidden_units": dense_units,
        "latent_dim": latent_dim,

        "dropout_rate": dropout_rate,
        "use_batch_norm": use_batch_norm,
        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay
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


def select_prediction(outputs, channel):

    if channel == "440":
        return outputs["pred_440"]

    if channel == "555":
        return outputs["pred_555"]

    if channel == "740":
        return outputs["pred_740"]

    raise ValueError(
        f"Unknown channel: {channel}"
    )


def run_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    metric_channel,
    train=True
):

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_pred_loss = 0.0
    total_align_loss = 0.0
    num_batches = 0

    targets_all = []
    selected_pred_all = []

    context = torch.enable_grad() if train else torch.no_grad()

    with context:

        for batch in dataloader:

            images = batch["image"].to(device)
            targets = batch["target"].to(device).float()
            weights = batch["weight"].to(device).float()

            if train:
                optimizer.zero_grad()

            outputs = model(images)

            loss_dict = criterion(
                outputs=outputs,
                target=targets,
                weight=weights
            )

            loss = loss_dict["loss"]

            if train:
                loss.backward()
                optimizer.step()

            selected_pred = select_prediction(
                outputs=outputs,
                channel=metric_channel
            )

            total_loss += float(loss.item())
            total_pred_loss += float(loss_dict["loss_pred"].item())
            total_align_loss += float(loss_dict["loss_align"].item())
            num_batches += 1

            targets_all.append(
                targets.detach().cpu().numpy()
            )

            selected_pred_all.append(
                selected_pred.detach().cpu().numpy()
            )

    y_true = np.concatenate(
        targets_all
    )

    y_pred = np.concatenate(
        selected_pred_all
    )

    metrics = compute_metrics(
        y_true,
        y_pred
    )

    metrics.update(
        {
            "loss": total_loss / num_batches,
            "loss_pred": total_pred_loss / num_batches,
            "loss_align": total_align_loss / num_batches,
            "metric_channel": metric_channel
        }
    )

    return metrics


def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_metrics,
    val_metrics,
    output_dir,
    params,
    args
):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "params": params,
        "validation_channel": args.validation_channel,
        "lambda_align": args.lambda_align
    }

    torch.save(
        checkpoint,
        output_dir / "best_model.pt"
    )


def train_trial(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    trial_output_dir,
    params,
    args
):

    history = []

    best_val_r2 = -float("inf")
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):

        train_metrics = run_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            metric_channel=args.validation_channel,
            train=True
        )

        val_metrics = run_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            metric_channel=args.validation_channel,
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
            },
        }

        history.append(
            row
        )

        current_val_r2 = val_metrics["R2"]

        if current_val_r2 > best_val_r2:

            best_val_r2 = current_val_r2
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                output_dir=trial_output_dir,
                params=params,
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
        f"cwkt_single_channel"
        f"_valR2_{args.validation_channel}"
        f"_lambda_{args.lambda_align:g}"
        f"{sampler_tag}"
    )

    return study_name


def objective(
    trial,
    args,
    train_dataset,
    val_dataset,
    device,
    study_output_dir
):

    set_seed(
        args.seed + trial.number
    )

    params = suggest_hyperparameters(
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

    model = CWKTSingleChannelRegressor(
        conv_channels=params["conv_channels"],
        kernel_sizes=params["kernel_sizes"],
        latent_dim=params["latent_dim"],
        regressor_hidden_units=params["regressor_hidden_units"],
        dropout_rate=params["dropout_rate"],
        use_batch_norm=True,
        use_pooling=True
    )

    model.to(
        device
    )

    criterion = CWKTMeanAlignmentLoss(
        lambda_align=args.lambda_align,
        use_weights=False
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"]
    )

    history, best_val_r2, best_val_loss, best_epoch = train_trial(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
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
        "regressor_hidden_units",
        params["regressor_hidden_units"]
    )

    trial.set_user_attr(
        "latent_dim",
        params["latent_dim"]
    )

    print(
        f"Trial {trial.number:03d} completed | "
        f"val_channel={args.validation_channel} | "
        f"best_val_R2={best_val_r2:.6f} | "
        f"best_val_loss={best_val_loss:.6f} | "
        f"best_epoch={best_epoch} | "
        f"lr={params['lr']:.2e} | "
        f"conv={params['conv_channels']} | "
        f"regressor={params['regressor_hidden_units']} | "
        f"latent_dim={params['latent_dim']} | "
        f"dropout={params['dropout_rate']:.3f} | "
        f"batch_norm={params['use_batch_norm']}"
    )

    return best_val_r2


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
    print("Optuna Optimization: CWKT Single-Channel Inference")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Study name: {study_name}")
    print(f"Study output dir: {study_output_dir}")
    print(f"Validation channel: {args.validation_channel}")
    print(f"Lambda align: {args.lambda_align}")
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
        "objective": f"maximize_val_R2_{args.validation_channel}",
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
    print(
        f"Best validation R2 "
        f"({args.validation_channel}): {best_trial.value}"
    )
    print("Best params:")
    print(
        json.dumps(
            best_trial.params,
            indent=4
        )
    )
    print("Best user attrs:")
    print(
        json.dumps(
            best_trial.user_attrs,
            indent=4
        )
    )


if __name__ == "__main__":

    main()