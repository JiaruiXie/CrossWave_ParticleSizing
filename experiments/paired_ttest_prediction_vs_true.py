"""
Paired t-test between predicted and true intensity.

This script evaluates a trained single-wavelength model and tests whether
the mean prediction error is significantly different from zero.

Equivalent tests:
    paired t-test between prediction and target
    one-sample t-test on difference = prediction - target

Example:

    & "C:\\Users\\jiaru\\.conda\\envs\\LNP_image_analysis\\python.exe" experiments/paired_ttest_prediction_vs_true.py --checkpoint "outputs\\optuna\\baseline_440_unweighted_soft_sampler_alpha_0.3\\trial_025\\best_model.pt" --mode 440 --split test
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
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy import stats

from datasets.particle_dataset import ParticleDataset
from models.baseline_cnn import ConfigurableCNNRegressor
from utils.seed import set_seed


def parse_args():

    parser = argparse.ArgumentParser(
        description="Run paired t-test between predicted and true intensity."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best_model.pt or best_student_model.pt."
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["440", "555", "740", "fusion"],
        help="Input mode used by the model."
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Evaluation batch size."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. If omitted, save next to checkpoint."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
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


def get_state_dict_from_checkpoint(checkpoint):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]
        checkpoint_type = "model_state_dict"

    elif "student_state_dict" in checkpoint:

        state_dict = checkpoint["student_state_dict"]
        checkpoint_type = "student_state_dict"

    else:

        raise KeyError(
            "Checkpoint must contain either 'model_state_dict' or 'student_state_dict'."
        )

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("student."):
            new_key = new_key.replace("student.", "", 1)

        if new_key.startswith("module."):
            new_key = new_key.replace("module.", "", 1)

        cleaned_state_dict[new_key] = value

    return cleaned_state_dict, checkpoint_type


def infer_architecture_from_state_dict(state_dict):
    """
    Infer ConfigurableCNNRegressor architecture directly from checkpoint.

    This avoids dropout mismatch issues such as:
        checkpoint has regressor.3.weight
        rebuilt model expects regressor.4.weight
    """

    conv_items = []

    for key, value in state_dict.items():

        if key.startswith("features.") and key.endswith(".weight"):

            if value.ndim == 4:

                layer_index = int(key.split(".")[1])
                out_channels = int(value.shape[0])
                kernel_size = int(value.shape[2])

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

    linear_items = []

    for key, value in state_dict.items():

        if key.startswith("regressor.") and key.endswith(".weight"):

            if value.ndim == 2:

                layer_index = int(key.split(".")[1])
                out_features = int(value.shape[0])

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
            "Could not infer regressor layers from checkpoint."
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


def build_model_from_checkpoint(checkpoint_path, in_channels, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    state_dict, checkpoint_type = get_state_dict_from_checkpoint(
        checkpoint
    )

    architecture = infer_architecture_from_state_dict(
        state_dict
    )

    model = ConfigurableCNNRegressor(
        in_channels=in_channels,
        conv_channels=architecture["conv_channels"],
        kernel_sizes=architecture["kernel_sizes"],
        dense_units=architecture["dense_units"],
        dropout_rate=architecture["dropout_rate"],
        use_batch_norm=architecture["use_batch_norm"],
        use_pooling=True
    )

    model.load_state_dict(
        state_dict
    )

    model.to(device)
    model.eval()

    return model, architecture, checkpoint, checkpoint_type


def predict(model, dataloader, device):

    model.eval()

    all_targets = []
    all_predictions = []
    all_particle_ids = []
    all_image_indices = []

    with torch.no_grad():

        for batch in dataloader:

            images = batch["image"].to(device)

            targets = batch["target"].detach().cpu().numpy()

            predictions = model(images).detach().cpu().numpy()

            all_targets.append(targets)
            all_predictions.append(predictions)

            all_particle_ids.extend(
                batch["particle_id"].numpy().tolist()
            )

            all_image_indices.extend(
                batch["image_index"].numpy().tolist()
            )

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)

    results = pd.DataFrame(
        {
            "particle_id": all_particle_ids,
            "image_index": all_image_indices,
            "target_cube_root": y_true,
            "prediction_cube_root": y_pred,
            "difference_cube_root": y_pred - y_true,
            "abs_error_cube_root": np.abs(y_pred - y_true)
        }
    )

    # Convert back to original intensity scale.
    results["target_original"] = results["target_cube_root"] ** 3
    results["prediction_original"] = results["prediction_cube_root"] ** 3
    results["difference_original"] = (
        results["prediction_original"] - results["target_original"]
    )
    results["abs_error_original"] = np.abs(
        results["difference_original"]
    )

    return results


def compute_regression_metrics(y_true, y_pred):

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


def run_paired_ttest(results, scale_name):

    if scale_name == "cube_root":

        y_true = results["target_cube_root"].values
        y_pred = results["prediction_cube_root"].values
        diff = results["difference_cube_root"].values

    elif scale_name == "original":

        y_true = results["target_original"].values
        y_pred = results["prediction_original"].values
        diff = results["difference_original"].values

    else:

        raise ValueError(
            f"Unknown scale_name: {scale_name}"
        )

    # Paired t-test between predictions and targets.
    paired_test = stats.ttest_rel(
        y_pred,
        y_true
    )

    # Equivalent one-sample t-test on differences.
    one_sample_test = stats.ttest_1samp(
        diff,
        popmean=0.0
    )

    n = len(diff)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    se_diff = std_diff / np.sqrt(n)

    confidence_level = 0.95
    alpha = 1.0 - confidence_level
    df = n - 1

    t_critical = stats.t.ppf(
        1.0 - alpha / 2.0,
        df=df
    )

    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff

    cohen_dz = mean_diff / std_diff if std_diff != 0 else np.nan

    metrics = compute_regression_metrics(
        y_true=y_true,
        y_pred=y_pred
    )

    result = {
        "scale": scale_name,
        "n": n,
        "mean_true": mean_true,
        "mean_prediction": mean_pred,
        "mean_difference_prediction_minus_true": mean_diff,
        "std_difference": std_diff,
        "se_difference": se_diff,
        "t_statistic_paired": paired_test.statistic,
        "p_value_paired": paired_test.pvalue,
        "t_statistic_one_sample_difference": one_sample_test.statistic,
        "p_value_one_sample_difference": one_sample_test.pvalue,
        "df": df,
        "confidence_level": confidence_level,
        "ci_lower_mean_difference": ci_lower,
        "ci_upper_mean_difference": ci_upper,
        "cohen_dz": cohen_dz,
        **metrics
    }

    return result


def main():

    args = parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Could not find checkpoint: {checkpoint_path}"
        )

    if args.output_dir is None:

        output_dir = (
            checkpoint_path.parent
            / f"paired_ttest_{args.mode}_{args.split}"
        )

    else:

        output_dir = Path(
            args.output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    channels = get_channels(
        args.mode
    )

    print("=" * 80)
    print("Paired t-test: predicted vs true intensity")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Mode: {args.mode}")
    print(f"Channels: {channels}")
    print(f"Split: {args.split}")
    print(f"Output dir: {output_dir}")

    model, architecture, checkpoint, checkpoint_type = build_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        in_channels=len(channels),
        device=device
    )

    print()
    print("Checkpoint type:", checkpoint_type)
    print("Architecture:")
    print(
        json.dumps(
            architecture,
            indent=4
        )
    )

    dataset = ParticleDataset(
        data_dir="data/processed",
        split=args.split,
        channels=channels,
        target_transform="cube_root",
        use_weights=False,
        mmap=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print()
    print(f"Loaded {args.split} dataset: {len(dataset)} samples")

    predictions_df = predict(
        model=model,
        dataloader=dataloader,
        device=device
    )

    predictions_df.to_csv(
        output_dir / "predictions_for_paired_ttest.csv",
        index=False
    )

    cube_root_test = run_paired_ttest(
        results=predictions_df,
        scale_name="cube_root"
    )

    original_test = run_paired_ttest(
        results=predictions_df,
        scale_name="original"
    )

    test_results_df = pd.DataFrame(
        [
            cube_root_test,
            original_test
        ]
    )

    test_results_df.to_csv(
        output_dir / "paired_ttest_results.csv",
        index=False
    )

    with open(output_dir / "paired_ttest_config.json", "w") as f:

        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_type": checkpoint_type,
                "mode": args.mode,
                "channels": channels,
                "split": args.split,
                "architecture": architecture,
                "test_description": (
                    "Paired t-test between prediction and true target; "
                    "equivalent to one-sample t-test on prediction minus true difference."
                )
            },
            f,
            indent=4
        )

    print()
    print("=" * 80)
    print("Paired t-test results")
    print("=" * 80)

    print(
        test_results_df
    )

    print()
    print("Interpretation:")
    print(
        "H0: mean(prediction - true) = 0. "
        "A small p-value means the model has a statistically significant mean bias."
    )

    print()
    print(f"Saved predictions to: {output_dir / 'predictions_for_paired_ttest.csv'}")
    print(f"Saved t-test results to: {output_dir / 'paired_ttest_results.csv'}")


if __name__ == "__main__":

    main()