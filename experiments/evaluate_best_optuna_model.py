"""
Evaluate the best Optuna trial model and generate plots.

Example:
    outputs/optuna/baseline_440_unweighted_soft_sampler_alpha_0.3/best_trial.json

This version keeps all five plots:
    1. Prediction scatter
    2. Prediction distribution
    3. Prediction heatmap with LogNorm
    4. Residual distribution
    5. MAPE by cube-root intensity

It also rebuilds the model directly from checkpoint state_dict to avoid
dropout-index mismatch such as:
    checkpoint has regressor.3.weight
    rebuilt model expects regressor.4.weight
"""

import sys
from pathlib import Path
import argparse
import json
import ast

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from datasets.particle_dataset import ParticleDataset
from models.baseline_cnn import ConfigurableCNNRegressor
from utils.seed import set_seed


def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate best Optuna model."
    )

    parser.add_argument(
        "--study_dir",
        type=str,
        required=True,
        help="Optuna study directory containing best_trial.json."
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["440", "555", "740", "fusion"],
        help="Input mode."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. If omitted, save inside study_dir/best_model_evaluation."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Evaluation batch size."
    )

    parser.add_argument(
        "--hist_min",
        type=float,
        default=0.0,
        help="Fixed minimum histogram edge for prediction distribution plots."
    )

    parser.add_argument(
        "--hist_max",
        type=float,
        default=45.0,
        help="Fixed maximum histogram edge for prediction distribution plots."
    )

    parser.add_argument(
        "--hist_bins",
        type=int,
        default=80,
        help="Number of histogram bins for prediction distribution plots."
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

    mae = np.mean(np.abs(y_pred - y_true))

    mse = np.mean((y_pred - y_true) ** 2)

    rmse = np.sqrt(mse)

    epsilon = 1e-8

    mape = np.mean(
        np.abs(
            (y_true - y_pred)
            /
            np.maximum(np.abs(y_true), epsilon)
        )
    ) * 100

    ss_res = np.sum((y_true - y_pred) ** 2)

    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }


def compute_binned_metrics(
    results,
    bin_edges=None,
    num_bins=10
):
    """
    Compute MAE, MSE, RMSE, MAPE, and R2 within target-intensity bins.
    Bins are based on true cube-root Cy3 max intensity.
    """

    df = results.copy()

    y_true = df["target"].values
    y_pred = df["prediction"].values

    if bin_edges is None:

        bin_edges = np.linspace(
            y_true.min(),
            y_true.max(),
            num_bins + 1
        )

    df["target_bin"] = pd.cut(
        df["target"],
        bins=bin_edges,
        include_lowest=True
    )

    rows = []

    epsilon = 1e-8

    for target_bin, group in df.groupby("target_bin", observed=True):

        bin_y_true = group["target"].values
        bin_y_pred = group["prediction"].values

        if len(group) == 0:
            continue

        mae = np.mean(
            np.abs(bin_y_pred - bin_y_true)
        )

        mse = np.mean(
            (bin_y_pred - bin_y_true) ** 2
        )

        rmse = np.sqrt(
            mse
        )

        mape = np.mean(
            np.abs(
                (bin_y_true - bin_y_pred)
                /
                np.maximum(np.abs(bin_y_true), epsilon)
            )
        ) * 100

        ss_res = np.sum(
            (bin_y_true - bin_y_pred) ** 2
        )

        ss_tot = np.sum(
            (bin_y_true - np.mean(bin_y_true)) ** 2
        )

        r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

        rows.append(
            {
                "target_bin": str(target_bin),
                "bin_left": target_bin.left,
                "bin_right": target_bin.right,
                "bin_center": (target_bin.left + target_bin.right) / 2,
                "num_samples": len(group),
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "MAPE": mape,
                "R2": r2
            }
        )

    binned_metrics_df = pd.DataFrame(rows)

    return binned_metrics_df


def plot_binned_mape(
    binned_metrics_df,
    save_path,
    title
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        binned_metrics_df["bin_center"],
        binned_metrics_df["MAPE"],
        marker="o",
        linewidth=1.8
    )

    plt.xlabel("True cube-root Cy3 max intensity")
    plt.ylabel("MAPE (%)")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=300
    )
    plt.close()


def parse_list(value):

    if isinstance(value, list):
        return value

    return ast.literal_eval(str(value))


def load_best_trial(study_dir):

    best_trial_path = study_dir / "best_trial.json"

    if not best_trial_path.exists():
        raise FileNotFoundError(
            f"Could not find best_trial.json: {best_trial_path}"
        )

    with open(best_trial_path, "r") as f:
        best_trial = json.load(f)

    return best_trial


def infer_architecture_from_state_dict(state_dict):
    """
    Infer ConfigurableCNNRegressor architecture directly from checkpoint weights.

    This avoids dropout-index mismatch such as:
        checkpoint has regressor.3
        rebuilt model expects regressor.4
    """

    conv_items = []

    for key, value in state_dict.items():

        if key.startswith("features.") and key.endswith(".weight"):

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

    linear_items = []

    for key, value in state_dict.items():

        if key.startswith("regressor.") and key.endswith(".weight"):

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


def build_model_from_checkpoint(
    checkpoint,
    in_channels
):
    """
    Reconstruct ConfigurableCNNRegressor directly from checkpoint.
    """

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    state_dict = checkpoint["model_state_dict"]

    architecture = infer_architecture_from_state_dict(
        state_dict
    )

    config = checkpoint.get("config", None)

    if config is None:
        config = checkpoint.get("params", None)

    if config is not None and "dropout_rate" in config:

        saved_dropout_rate = float(
            config["dropout_rate"]
        )

        if architecture["dropout_rate"] == 0.0:
            architecture["dropout_rate"] = 0.0
        else:
            architecture["dropout_rate"] = saved_dropout_rate

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


def predict(model, dataloader, device):

    model.eval()

    all_targets = []
    all_predictions = []
    all_particle_ids = []
    all_image_indices = []

    with torch.no_grad():

        for batch in dataloader:

            images = batch["image"].to(device)

            targets = batch["target"].cpu().numpy()

            predictions = model(images).cpu().numpy()

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
            "target": y_true,
            "prediction": y_pred,
            "residual": y_pred - y_true
        }
    )

    return results


def plot_prediction_scatter(results, metrics, save_path, title):

    y_true = results["target"].values
    y_pred = results["prediction"].values

    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())

    plt.figure(figsize=(6, 6))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.25,
        s=8
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        linewidth=1.5,
        color="black"
    )

    text = (
        f"R² = {metrics['R2']:.4f}\n"
        f"RMSE = {metrics['RMSE']:.4f}\n"
        f"MAE = {metrics['MAE']:.4f}\n"
        f"MSE = {metrics['MSE']:.4f}\n"
        f"MAPE = {metrics['MAPE']:.2f}%"
    )

    plt.text(
        0.05,
        0.95,
        text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            alpha=0.8
        )
    )

    plt.xlabel("True cube-root Cy3 max intensity")
    plt.ylabel("Predicted cube-root Cy3 max intensity")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_prediction_distribution(
    results,
    save_path,
    title,
    hist_min=0.0,
    hist_max=45.0,
    hist_bins=80
):

    y_true = results["target"].values
    y_pred = results["prediction"].values

    bins = np.linspace(
        hist_min,
        hist_max,
        hist_bins + 1
    )

    plt.figure(figsize=(8, 5))

    plt.hist(
        y_true,
        bins=bins,
        alpha=0.5,
        label="Ground truth",
        edgecolor="black"
    )

    plt.hist(
        y_pred,
        bins=bins,
        alpha=0.5,
        label="Prediction",
        edgecolor="black"
    )

    plt.xlim(
        hist_min,
        hist_max
    )

    plt.xlabel("Cube-root Cy3 max intensity")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_prediction_heatmap(results, save_path, title):

    y_true = results["target"].values
    y_pred = results["prediction"].values

    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())

    plt.figure(figsize=(6, 6))

    heatmap = plt.hist2d(
        y_true,
        y_pred,
        bins=100,
        range=[
            [min_value, max_value],
            [min_value, max_value]
        ],
        cmap="viridis",
        norm=LogNorm(),
        cmin=1
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        linewidth=1.5,
        color="black"
    )

    plt.xlabel("True cube-root Cy3 max intensity")
    plt.ylabel("Predicted cube-root Cy3 max intensity")
    plt.title(title)

    cbar = plt.colorbar(heatmap[3])
    cbar.set_label("Count, log scale")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_residuals(results, save_path, title):

    residuals = results["residual"].values

    plt.figure(figsize=(7, 5))

    plt.hist(
        residuals,
        bins=80,
        alpha=0.75,
        edgecolor="black"
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=1.5
    )

    plt.xlabel("Prediction residual")
    plt.ylabel("Frequency")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():

    args = parse_args()

    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    study_dir = Path(args.study_dir)

    if args.output_dir is None:
        output_dir = study_dir / "best_model_evaluation"
    else:
        output_dir = Path(args.output_dir)

    prediction_dir = output_dir / "predictions"
    figure_dir = output_dir / "figures"

    prediction_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    best_trial = load_best_trial(study_dir)

    channels = get_channels(args.mode)

    trial_output_dir = Path(
        best_trial["best_user_attrs"]["trial_output_dir"]
    )

    checkpoint_path = trial_output_dir / "best_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Could not find checkpoint: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model, architecture = build_model_from_checkpoint(
        checkpoint=checkpoint,
        in_channels=len(channels)
    )

    print(f"Study dir: {study_dir}")
    print(f"Mode: {args.mode}")
    print(f"Channels: {channels}")
    print(f"Best checkpoint: {checkpoint_path}")
    print("Architecture:")
    print(json.dumps(architecture, indent=4))

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

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    results = predict(
        model=model,
        dataloader=test_loader,
        device=device
    )

    metrics = compute_metrics(
        results["target"],
        results["prediction"]
    )

    bin_edges = np.array(
        [0, 5, 10, 15, 20, 25, 30, 35, 45]
    )

    binned_metrics_df = compute_binned_metrics(
        results=results,
        bin_edges=bin_edges
    )

    metrics_df = pd.DataFrame(
        [
            {
                "mode": args.mode,
                "study_dir": str(study_dir),
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint.get("epoch", np.nan),
                "checkpoint_train_loss": checkpoint.get("train_loss", np.nan),
                "checkpoint_val_loss": checkpoint.get("val_loss", np.nan),
                "optuna_best_value": best_trial.get(
                    "best_val_R2",
                    best_trial.get(
                        "best_value",
                        best_trial.get("value", np.nan)
                    )
                ),
                "num_test_samples": len(results),
                **metrics,
                "conv_channels": str(architecture["conv_channels"]),
                "kernel_sizes": str(architecture["kernel_sizes"]),
                "dense_units": str(architecture["dense_units"]),
                "dropout_rate": architecture["dropout_rate"],
                "use_batch_norm": architecture["use_batch_norm"]
            }
        ]
    )

    results.to_csv(
        prediction_dir / "test_predictions.csv",
        index=False
    )

    metrics_df.to_csv(
        output_dir / "test_metrics.csv",
        index=False
    )

    binned_metrics_df.to_csv(
        output_dir / "test_metrics_by_cube_root_intensity.csv",
        index=False
    )

    with open(output_dir / "best_trial_used.json", "w") as f:
        json.dump(best_trial, f, indent=4)

    plot_prediction_scatter(
        results,
        metrics,
        figure_dir / "test_prediction_scatter.png",
        title=f"Best Optuna {args.mode}: Predicted vs True"
    )

    plot_prediction_distribution(
        results,
        figure_dir / "test_prediction_distribution.png",
        title=f"Best Optuna {args.mode}: Prediction Distribution",
        hist_min=args.hist_min,
        hist_max=args.hist_max,
        hist_bins=args.hist_bins
    )

    plot_prediction_heatmap(
        results,
        figure_dir / "test_prediction_heatmap.png",
        title=f"Best Optuna {args.mode}: Prediction Heatmap"
    )

    plot_residuals(
        results,
        figure_dir / "test_residuals.png",
        title=f"Best Optuna {args.mode}: Residual Distribution"
    )

    plot_binned_mape(
        binned_metrics_df=binned_metrics_df,
        save_path=figure_dir / "test_mape_by_cube_root_intensity.png",
        title=f"Best Optuna {args.mode}: MAPE by Cube-root Intensity"
    )

    print()
    print("=" * 80)
    print("Best Optuna Model Test Metrics")
    print("=" * 80)
    print(metrics_df)

    print()
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":

    main()