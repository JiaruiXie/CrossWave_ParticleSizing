"""
Evaluate every saved Optuna trial checkpoint on the test set.

Example target folder:
    outputs/optuna/baseline_440_unweighted_soft_sampler_alpha_0.3/

This script loads:
    trial_000/best_model.pt
    trial_001/best_model.pt
    ...

and records:
    R2, MAE, MSE, RMSE

Important:
    This version reconstructs the model architecture directly from each
    checkpoint state_dict. This avoids regressor index mismatch caused by
    incorrect dropout assumptions from trials.csv.
"""

import sys
from pathlib import Path
import argparse
import ast

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


def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate all Optuna trial checkpoints."
    )

    parser.add_argument(
        "--study_dir",
        type=str,
        required=True,
        help="Optuna study folder containing trial_XXX folders and trials.csv."
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["440", "555", "740", "fusion"],
        help="Input mode."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Evaluation batch size."
    )

    return parser.parse_args()


def get_channels(mode):

    if mode == "440":
        return [0]

    if mode == "555":
        return [1]

    if mode == "740":
        return [2]

    if mode == "fusion":
        return [0, 1, 2]

    raise ValueError(f"Unknown mode: {mode}")


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


def parse_list(value):

    if isinstance(value, list):
        return value

    return ast.literal_eval(
        str(value)
    )


def get_value_from_row(row, possible_columns, default=None):

    if row is None:
        return default

    for column in possible_columns:

        if column in row.index:

            value = row[column]

            if pd.isna(value):
                continue

            return value

    return default


def infer_architecture_from_state_dict(state_dict):
    """
    Infer ConfigurableCNNRegressor architecture directly from checkpoint weights.

    This avoids dropout-index mismatch such as:
        checkpoint has regressor.3
        rebuilt model expects regressor.4
    """

    # ========================================================
    # Infer convolutional layers
    # ========================================================

    conv_items = []

    for key, value in state_dict.items():

        if key.startswith("features.") and key.endswith(".weight"):

            # Conv2d weight shape:
            # [out_channels, in_channels, kernel_height, kernel_width]
            if value.ndim == 4:

                layer_index = int(
                    key.split(".")[1]
                )

                out_channels = int(
                    value.shape[0]
                )

                kernel_size = int(
                    value.shape[2]
                )

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

    if len(conv_channels) == 0:
        raise ValueError(
            "Could not infer convolutional layers from checkpoint."
        )

    # ========================================================
    # Infer dense layers
    # ========================================================

    linear_items = []

    for key, value in state_dict.items():

        if key.startswith("regressor.") and key.endswith(".weight"):

            # Linear weight shape:
            # [out_features, in_features]
            if value.ndim == 2:

                layer_index = int(
                    key.split(".")[1]
                )

                out_features = int(
                    value.shape[0]
                )

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

    if len(linear_items) == 0:
        raise ValueError(
            "Could not infer dense layers from checkpoint."
        )

    # Remove the final output layer, whose out_features should be 1.
    dense_units = [
        item[1]
        for item in linear_items
        if item[1] != 1
    ]

    linear_indices = [
        item[0]
        for item in linear_items
    ]

    # ========================================================
    # Infer dropout presence from linear layer indices
    # ========================================================
    #
    # Example with one hidden layer:
    #
    # No dropout:
    #   regressor.0 = Flatten
    #   regressor.1 = Linear
    #   regressor.2 = ReLU
    #   regressor.3 = Output Linear
    #
    # With dropout:
    #   regressor.0 = Flatten
    #   regressor.1 = Linear
    #   regressor.2 = ReLU
    #   regressor.3 = Dropout
    #   regressor.4 = Output Linear
    #
    # For multiple hidden layers, the gap between consecutive Linear
    # layers is:
    #   2 if no Dropout
    #   3 if Dropout exists

    dropout_rate = 0.0

    if len(linear_indices) >= 2:

        gaps = [
            linear_indices[i + 1] - linear_indices[i]
            for i in range(len(linear_indices) - 1)
        ]

        if any(gap == 3 for gap in gaps):
            dropout_rate = 0.2
        else:
            dropout_rate = 0.0

    architecture = {
        "conv_channels": conv_channels,
        "kernel_sizes": kernel_sizes,
        "dense_units": dense_units,
        "dropout_rate": dropout_rate,
        "use_batch_norm": True
    }

    return architecture


def build_model_from_checkpoint(
    checkpoint,
    in_channels
):
    """
    Reconstruct ConfigurableCNNRegressor from the checkpoint itself.
    """

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    state_dict = checkpoint["model_state_dict"]

    architecture = infer_architecture_from_state_dict(
        state_dict
    )

    # If the checkpoint stores exact params/config, use exact dropout rate.
    # This is safe only when it is consistent with layer indexing.
    config = checkpoint.get("config", None)

    if config is None:
        config = checkpoint.get("params", None)

    if config is not None and "dropout_rate" in config:

        saved_dropout_rate = float(
            config["dropout_rate"]
        )

        # If architecture inference detected no Dropout layer,
        # force dropout_rate to 0.0 to match checkpoint module indices.
        if architecture["dropout_rate"] == 0.0:
            dropout_rate = 0.0
        else:
            dropout_rate = saved_dropout_rate

        architecture["dropout_rate"] = dropout_rate

    model = ConfigurableCNNRegressor(
        in_channels=in_channels,
        conv_channels=architecture["conv_channels"],
        kernel_sizes=architecture["kernel_sizes"],
        dense_units=architecture["dense_units"],
        dropout_rate=architecture["dropout_rate"],
        use_batch_norm=architecture["use_batch_norm"],
        use_pooling=True
    )

    return model, architecture


def load_trials_dataframe(study_dir):

    trials_csv = study_dir / "trials.csv"

    if not trials_csv.exists():

        print(
            f"Warning: Could not find trials.csv: {trials_csv}"
        )

        return None

    trials_df = pd.read_csv(
        trials_csv
    )

    return trials_df


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

    if not study_dir.exists():
        raise FileNotFoundError(
            f"Could not find study directory: {study_dir}"
        )

    trials_df = load_trials_dataframe(
        study_dir
    )

    channels = get_channels(
        args.mode
    )

    print(f"Study directory: {study_dir}")
    print(f"Mode: {args.mode}")
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
    # Evaluate all trial checkpoints
    # ========================================================

    trial_folders = sorted(
        [
            p for p in study_dir.iterdir()
            if p.is_dir() and p.name.startswith("trial_")
        ],
        key=get_trial_number_from_folder
    )

    rows = []

    failed_rows = []

    for trial_folder in trial_folders:

        checkpoint_path = trial_folder / "best_model.pt"

        if not checkpoint_path.exists():
            print(f"Skipping {trial_folder.name}: no best_model.pt")
            continue

        trial_number = get_trial_number_from_folder(
            trial_folder
        )

        try:

            row = get_trial_row(
                trials_df,
                trial_number
            )

            checkpoint = torch.load(
                checkpoint_path,
                map_location=device
            )

            model, architecture = build_model_from_checkpoint(
                checkpoint=checkpoint,
                in_channels=len(channels)
            )

            model.load_state_dict(
                checkpoint["model_state_dict"]
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

            result_row = {
                "trial_number": trial_number,
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint.get("epoch", np.nan),

                "checkpoint_train_loss": checkpoint.get("train_loss", np.nan),
                "checkpoint_val_loss": checkpoint.get("val_loss", np.nan),

                "optuna_value": (
                    row.get("value", np.nan)
                    if row is not None
                    else np.nan
                ),

                "test_MAE": metrics["MAE"],
                "test_MSE": metrics["MSE"],
                "test_RMSE": metrics["RMSE"],
                "test_R2": metrics["R2"],

                "lr": (
                    row.get("params_lr", np.nan)
                    if row is not None
                    else np.nan
                ),
                "num_conv_layers": (
                    row.get("params_num_conv_layers", np.nan)
                    if row is not None
                    else np.nan
                ),
                "base_channels": (
                    row.get("params_base_channels", np.nan)
                    if row is not None
                    else np.nan
                ),
                "channel_multiplier": (
                    row.get("params_channel_multiplier", np.nan)
                    if row is not None
                    else np.nan
                ),
                "kernel_size": (
                    row.get("params_kernel_size", np.nan)
                    if row is not None
                    else np.nan
                ),
                "dense_depth": (
                    row.get("params_dense_depth", np.nan)
                    if row is not None
                    else np.nan
                ),
                "dense_width": (
                    row.get("params_dense_width", np.nan)
                    if row is not None
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
                f"test_RMSE={metrics['RMSE']:.4f}"
            )

        except Exception as e:

            error_message = str(e)

            print(
                f"Failed to evaluate {trial_folder.name}: {error_message}"
            )

            failed_rows.append(
                {
                    "trial_number": trial_number,
                    "checkpoint": str(checkpoint_path),
                    "error": error_message
                }
            )

    results = pd.DataFrame(
        rows
    )

    failed_results = pd.DataFrame(
        failed_rows
    )

    failed_output_path = study_dir / "test_metrics_failed_trials.csv"

    if len(failed_results) > 0:

        failed_results.to_csv(
            failed_output_path,
            index=False
        )

        print()
        print(f"Saved failed trial details to: {failed_output_path}")

    if len(results) == 0 or "test_R2" not in results.columns:

        raise RuntimeError(
            "No trials were successfully evaluated. "
            "Check test_metrics_failed_trials.csv and the printed errors above."
        )

    results = results.sort_values(
        "test_R2",
        ascending=False
    )

    output_path = study_dir / "test_metrics_all_trials.csv"

    results.to_csv(
        output_path,
        index=False
    )

    print()
    print("=" * 80)
    print("Top trials by test R2")
    print("=" * 80)

    print(
        results.head(10)
    )

    print()
    print(f"Saved all trial test metrics to: {output_path}")


if __name__ == "__main__":

    main()