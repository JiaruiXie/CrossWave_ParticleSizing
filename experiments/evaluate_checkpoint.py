"""
Load a saved model checkpoint and evaluate it on the test set.

Metrics:
    R2
    MAE
    MSE
    RMSE
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
from utils.seed import set_seed


def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate a saved CNN checkpoint on the test set."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best_model.pt."
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
        required=True,
        help="Directory to save predictions and metrics."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Evaluation batch size."
    )

    # Architecture arguments
    parser.add_argument(
        "--conv_channels",
        type=str,
        default="[16, 32, 64, 128]",
        help="List of convolution channels, e.g. '[16, 32, 64, 128]'."
    )

    parser.add_argument(
        "--kernel_sizes",
        type=str,
        default="[3, 3, 3, 3]",
        help="List of kernel sizes, e.g. '[3, 3, 3, 3]'."
    )

    parser.add_argument(
        "--dense_units",
        type=str,
        default="[256, 128, 64]",
        help="List of dense units, e.g. '[256, 128, 64]'."
    )

    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2,
        help="Dropout rate used by the model."
    )

    parser.add_argument(
        "--use_batch_norm",
        action="store_true",
        help="Use batch normalization. Add this flag if the trained model used batch norm."
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
    all_particle_ids = []
    all_image_indices = []

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

            all_particle_ids.extend(
                batch["particle_id"].numpy().tolist()
            )

            all_image_indices.extend(
                batch["image_index"].numpy().tolist()
            )

    y_true = np.concatenate(
        all_targets
    )

    y_pred = np.concatenate(
        all_predictions
    )

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


def main():

    args = parse_args()

    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    checkpoint_path = Path(
        args.checkpoint
    )

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

    conv_channels = json.loads(
        args.conv_channels
    )

    kernel_sizes = json.loads(
        args.kernel_sizes
    )

    dense_units = json.loads(
        args.dense_units
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Mode: {args.mode}")
    print(f"Channels: {channels}")
    print(f"Conv channels: {conv_channels}")
    print(f"Kernel sizes: {kernel_sizes}")
    print(f"Dense units: {dense_units}")
    print(f"Dropout rate: {args.dropout_rate}")
    print(f"Use batch norm: {args.use_batch_norm}")

    # ========================================================
    # Dataset
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
    # Model
    # ========================================================

    model = ConfigurableCNNRegressor(
        in_channels=len(channels),
        conv_channels=conv_channels,
        kernel_sizes=kernel_sizes,
        dense_units=dense_units,
        dropout_rate=args.dropout_rate,
        use_batch_norm=args.use_batch_norm,
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

    print()
    print("Loaded checkpoint.")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint train loss: {checkpoint.get('train_loss', 'unknown')}")
    print(f"Checkpoint val loss: {checkpoint.get('val_loss', 'unknown')}")

    # ========================================================
    # Predict
    # ========================================================

    results = predict(
        model=model,
        dataloader=test_loader,
        device=device
    )

    metrics = compute_metrics(
        results["target"],
        results["prediction"]
    )

    metrics_df = pd.DataFrame(
        [
            {
                "mode": args.mode,
                "checkpoint": str(checkpoint_path),
                "num_test_samples": len(results),
                **metrics
            }
        ]
    )

    # ========================================================
    # Save
    # ========================================================

    results.to_csv(
        output_dir / "test_predictions.csv",
        index=False
    )

    metrics_df.to_csv(
        output_dir / "test_metrics.csv",
        index=False
    )

    print()
    print("=" * 80)
    print("Test Metrics")
    print("=" * 80)
    print(metrics_df)

    print()
    print(f"Saved predictions and metrics to: {output_dir}")


if __name__ == "__main__":

    main()