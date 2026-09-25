"""
Evaluate the trained three-wavelength fusion baseline CNN.

Input:
    440 + 555 + 740 nm iSCAT channels

Target:
    Cube-root transformed Cy3 max intensity
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets.particle_dataset import ParticleDataset
from models.baseline_cnn import ConfigurableCNNRegressor
from utils.seed import set_seed

import argparse


# ============================================================
# Metrics
# ============================================================

def compute_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = np.mean(np.abs(y_pred - y_true))

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

    ss_res = np.sum((y_true - y_pred) ** 2)

    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1.0 - ss_res / ss_tot

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# Prediction
# ============================================================

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


# ============================================================
# Plotting
# ============================================================

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
        linewidth=1.5
    )

    plt.xlabel("True cube-root Cy3 max intensity")
    plt.ylabel("Predicted cube-root Cy3 max intensity")
    plt.title(title)

    text = (
        f"R² = {metrics['R2']:.4f}\n"
        f"RMSE = {metrics['RMSE']:.4f}\n"
        f"MAE = {metrics['MAE']:.4f}"
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

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_residuals(results, save_path, title):

    residuals = results["residual"].values

    plt.figure(figsize=(7, 5))

    plt.hist(
        residuals,
        bins=80,
        edgecolor="black",
        alpha=0.75
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


def plot_prediction_distribution(results, save_path, title):

    y_true = results["target"].values
    y_pred = results["prediction"].values

    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())

    bins = np.linspace(
        min_value,
        max_value,
        80
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
        cmap="viridis"
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        linewidth=1.5,
        color="white"
    )

    plt.xlabel("True cube-root Cy3 max intensity")
    plt.ylabel("Predicted cube-root Cy3 max intensity")
    plt.title(title)

    cbar = plt.colorbar(heatmap[3])
    cbar.set_label("Count")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate fusion baseline model."
    )

    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="outputs/baseline_fusion_unweighted",
        help="Path to experiment output folder."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="baseline_fusion_unweighted",
        help="Model name for metrics.csv."
    )

    return parser.parse_args()

# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()
    
    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    experiment_dir = Path(
    args.experiment_dir
    ) 

    checkpoint_path = experiment_dir / "best_model.pt"

    prediction_dir = experiment_dir / "predictions"

    figure_dir = experiment_dir / "figures"

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"Evaluating: {experiment_dir}")

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    val_dataset = ParticleDataset(
        data_dir="data/processed",
        split="val",
        channels=[0, 1, 2],
        target_transform="cube_root",
        use_weights=False,
        mmap=True
    )

    test_dataset = ParticleDataset(
        data_dir="data/processed",
        split="test",
        channels=[0, 1, 2],
        target_transform="cube_root",
        use_weights=False,
        mmap=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # Must match fusion training architecture exactly
    # --------------------------------------------------------

    model = ConfigurableCNNRegressor(
        in_channels=3,
        conv_channels=[16, 32, 64, 128],
        kernel_sizes=[3, 3, 3, 3],
        dense_units=[256, 128, 64],
        dropout_rate=0.2,
        use_batch_norm=True,
        use_pooling=True
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Best val loss: {checkpoint['val_loss']:.6f}")

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    val_results = predict(
        model,
        val_loader,
        device
    )

    test_results = predict(
        model,
        test_loader,
        device
    )

    val_metrics = compute_metrics(
        val_results["target"],
        val_results["prediction"]
    )

    test_metrics = compute_metrics(
        test_results["target"],
        test_results["prediction"]
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    val_results.to_csv(
        prediction_dir / "val_predictions.csv",
        index=False
    )

    test_results.to_csv(
        prediction_dir / "test_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = pd.DataFrame(
        [
            {
                "model": args.model_name,
                "input": "440+555+740 nm",
                "split": "val",
                **val_metrics
            },
            {
                "model": args.model_name,
                "input": "440+555+740 nm",
                "split": "test",
                **test_metrics
            }
        ]
    )

    metrics.to_csv(
        experiment_dir / "metrics.csv",
        index=False
    )

    print()
    print("=" * 60)
    print("Evaluation Metrics")
    print("=" * 60)
    print(metrics)

    # --------------------------------------------------------
    # Plots: validation
    # --------------------------------------------------------

    plot_prediction_scatter(
        val_results,
        val_metrics,
        figure_dir / "val_prediction_scatter.png",
        title="Validation: Fusion Predicted vs True"
    )

    plot_residuals(
        val_results,
        figure_dir / "val_residuals.png",
        title="Validation: Fusion Residual Distribution"
    )

    plot_prediction_distribution(
        val_results,
        figure_dir / "val_prediction_distribution.png",
        title="Validation: Fusion Prediction Distribution"
    )

    plot_prediction_heatmap(
        val_results,
        figure_dir / "val_prediction_heatmap.png",
        title="Validation: Fusion Prediction Heatmap"
    )

    # --------------------------------------------------------
    # Plots: test
    # --------------------------------------------------------

    plot_prediction_scatter(
        test_results,
        test_metrics,
        figure_dir / "test_prediction_scatter.png",
        title="Test: Fusion Predicted vs True"
    )

    plot_residuals(
        test_results,
        figure_dir / "test_residuals.png",
        title="Test: Fusion Residual Distribution"
    )

    plot_prediction_distribution(
        test_results,
        figure_dir / "test_prediction_distribution.png",
        title="Test: Fusion Prediction Distribution"
    )

    plot_prediction_heatmap(
        test_results,
        figure_dir / "test_prediction_heatmap.png",
        title="Test: Fusion Prediction Heatmap"
    )

    print()
    print("Saved predictions, metrics, and figures.")




if __name__ == "__main__":

    main()