"""
Evaluate every saved KD student Optuna trial checkpoint on the test set.

This script loads:
    trial_000/best_student_model.pt
    trial_001/best_student_model.pt
    ...

and records:
    R2, MAE, MSE, RMSE

For KD student evaluation, only the student model is needed.
The projection layer is not needed because prediction uses the student's own regressor.
"""

import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.particle_dataset import ParticleDataset
from models.baseline_cnn import ConfigurableCNNRegressor
from utils.seed import set_seed

print("Starting evaluate_all_kd_student_trials.py")

def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate all KD student Optuna trial checkpoints."
    )

    parser.add_argument(
        "--study_dir",
        type=str,
        required=True,
        help="Optuna study folder containing trial_XXX folders."
    )

    parser.add_argument(
        "--inference_channel",
        type=str,
        required=True,
        choices=["440", "555", "740"],
        help="Single wavelength used for KD student inference."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Evaluation batch size."
    )

    return parser.parse_args()


def get_channels(channel):

    if channel == "440":
        return [0]

    if channel == "555":
        return [1]

    if channel == "740":
        return [2]

    raise ValueError(f"Unknown channel: {channel}")


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


def predict(model, dataloader, device):

    model.eval()

    all_targets = []
    all_predictions = []

    with torch.no_grad():

        for batch in dataloader:

            images = batch["image"].to(device)
            targets = batch["target"].cpu().numpy()

            predictions = model(images).cpu().numpy()

            all_targets.append(
                targets
            )

            all_predictions.append(
                predictions
            )

    y_true = np.concatenate(
        all_targets
    )

    y_pred = np.concatenate(
        all_predictions
    )

    return y_true, y_pred


def get_trial_number_from_folder(trial_folder):

    return int(
        trial_folder.name.replace("trial_", "")
    )


def load_trials_csv(study_dir):

    trials_csv = study_dir / "trials.csv"

    if not trials_csv.exists():
        return None

    return pd.read_csv(
        trials_csv
    )


def get_trial_row(trials_df, trial_number):

    if trials_df is None:
        return None

    if "number" in trials_df.columns:

        row = trials_df[
            trials_df["number"] == trial_number
        ]

    elif "trial_id" in trials_df.columns:

        row = trials_df[
            trials_df["trial_id"] == trial_number
        ]

    else:

        return None

    if len(row) != 1:
        return None

    return row.iloc[0]


def build_student_from_checkpoint(checkpoint):

    params = checkpoint["params"]

    model = ConfigurableCNNRegressor(
        in_channels=1,
        conv_channels=params["conv_channels"],
        kernel_sizes=params["kernel_sizes"],
        dense_units=params["dense_units"],
        dropout_rate=params["dropout_rate"],
        use_batch_norm=True,
        use_pooling=True
    )

    architecture = {
        "conv_channels": params["conv_channels"],
        "kernel_sizes": params["kernel_sizes"],
        "dense_units": params["dense_units"],
        "dropout_rate": params["dropout_rate"],
        "use_batch_norm": True,
        "lr": params.get("lr", np.nan),
        "weight_decay": params.get("weight_decay", np.nan),
        "batch_size": params.get("batch_size", np.nan)
    }

    return model, architecture


def main():

    args = parse_args()

    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    study_dir = Path(
        args.study_dir
    )

    trials_df = load_trials_csv(
        study_dir
    )

    channels = get_channels(
        args.inference_channel
    )

    print(f"Study directory: {study_dir}")
    print(f"Inference channel: {args.inference_channel}")
    print(f"Channels: {channels}")

    # ========================================================
    # Test dataset
    # ========================================================

    test_dataset = ParticleDataset(
        data_dir="data/processed",
        split="test",
        channels=channels,
        target_transform="cube_root",
        use_weights=False,
        mmap=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # ========================================================
    # Evaluate all KD student trial checkpoints
    # ========================================================

    trial_folders = sorted(
        [
            p for p in study_dir.iterdir()
            if p.is_dir() and p.name.startswith("trial_")
        ],
        key=get_trial_number_from_folder
    )

    rows = []

    for trial_folder in trial_folders:

        checkpoint_path = trial_folder / "best_student_model.pt"

        if not checkpoint_path.exists():
            print(f"Skipping {trial_folder.name}: no best_student_model.pt")
            continue

        trial_number = get_trial_number_from_folder(
            trial_folder
        )

        try:

            checkpoint = torch.load(
                checkpoint_path,
                map_location=device
            )

            model, architecture = build_student_from_checkpoint(
                checkpoint
            )

            model.load_state_dict(
                checkpoint["student_state_dict"]
            )

            model.to(device)

            y_true, y_pred = predict(
                model=model,
                dataloader=test_loader,
                device=device
            )

            metrics = compute_metrics(
                y_true,
                y_pred
            )

            trial_row = get_trial_row(
                trials_df,
                trial_number
            )

            result_row = {
                "trial_number": trial_number,
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint.get("epoch", np.nan),

                "student_channel": checkpoint.get("student_channel", np.nan),
                "inference_channel": args.inference_channel,

                "optuna_value": (
                    trial_row.get("value", np.nan)
                    if trial_row is not None
                    else np.nan
                ),

                "val_student_R2": checkpoint.get("val_metrics", {}).get("student_R2", np.nan),
                "val_student_MAE": checkpoint.get("val_metrics", {}).get("student_MAE", np.nan),
                "val_loss": checkpoint.get("val_metrics", {}).get("loss", np.nan),

                "test_MAE": metrics["MAE"],
                "test_MSE": metrics["MSE"],
                "test_RMSE": metrics["RMSE"],
                "test_R2": metrics["R2"],

                "alpha_target": checkpoint.get("alpha_target", np.nan),
                "alpha_pred": checkpoint.get("alpha_pred", np.nan),
                "alpha_feature": checkpoint.get("alpha_feature", np.nan),

                "lr": architecture["lr"],
                "weight_decay": architecture["weight_decay"],
                "batch_size": architecture["batch_size"],

                "num_conv_layers": (
                    trial_row.get("params_num_conv_layers", np.nan)
                    if trial_row is not None
                    else np.nan
                ),
                "base_channels": (
                    trial_row.get("params_base_channels", np.nan)
                    if trial_row is not None
                    else np.nan
                ),
                "channel_multiplier": (
                    trial_row.get("params_channel_multiplier", np.nan)
                    if trial_row is not None
                    else np.nan
                ),
                "kernel_size": (
                    trial_row.get("params_kernel_size", np.nan)
                    if trial_row is not None
                    else np.nan
                ),
                "dense_depth": (
                    trial_row.get("params_dense_depth", np.nan)
                    if trial_row is not None
                    else np.nan
                ),
                "dense_width": (
                    trial_row.get("params_dense_width", np.nan)
                    if trial_row is not None
                    else np.nan
                ),

                "dropout_rate": architecture["dropout_rate"],
                "use_batch_norm": architecture["use_batch_norm"],
                "conv_channels": str(architecture["conv_channels"]),
                "kernel_sizes": str(architecture["kernel_sizes"]),
                "dense_units": str(architecture["dense_units"])
            }

            rows.append(
                result_row
            )

            print(
                f"Trial {trial_number:03d} | "
                f"test_R2={metrics['R2']:.4f} | "
                f"test_MAE={metrics['MAE']:.4f} | "
                f"test_MSE={metrics['MSE']:.4f} | "
                f"test_RMSE={metrics['RMSE']:.4f} | "
                f"val_R2={result_row['val_student_R2']:.4f}"
            )

        except Exception as e:

            print(
                f"Failed to evaluate {trial_folder.name}: {e}"
            )

    results = pd.DataFrame(
        rows
    )

    if len(results) == 0:
        raise RuntimeError(
            "No trials were successfully evaluated."
        )

    results = results.sort_values(
        "test_R2",
        ascending=False
    )

    output_path = (
        study_dir
        / f"test_metrics_all_kd_student_trials_{args.inference_channel}.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print()
    print("=" * 80)
    print("Top KD student trials by test R2")
    print("=" * 80)

    print(
        results.head(10)
    )

    print()
    print(f"Saved all KD student trial test metrics to: {output_path}")


if __name__ == "__main__":

    main()