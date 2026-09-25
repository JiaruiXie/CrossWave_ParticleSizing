"""
Train baseline CNN using three-wavelength iSCAT fusion.

Input:
    Channel 0 -> 440 nm
    Channel 1 -> 555 nm
    Channel 2 -> 740 nm

Target:
    Cube-root transformed Cy3 max intensity
"""

import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from datasets.particle_dataset import ParticleDataset
from models.baseline_cnn import ConfigurableCNNRegressor
from training.losses import RegressionLoss
from training.samplers import create_soft_intensity_balanced_sampler
from training.trainer import train_model
from utils.seed import set_seed


def parse_args():

    parser = argparse.ArgumentParser(
        description="Train baseline CNN using 3-channel iSCAT fusion."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate."
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay."
    )

    parser.add_argument(
        "--use_weights",
        action="store_true",
        help="Use sample weighting mechanism."
    )

    parser.add_argument(
        "--clip_weights",
        action="store_true",
        help="Clip sample weights."
    )

    parser.add_argument(
        "--max_weight",
        type=float,
        default=20.0,
        help="Maximum sample weight after clipping."
    )

    parser.add_argument(
        "--use_balanced_sampler",
        action="store_true",
        help="Use intensity-balanced sampler for training."
    )

    parser.add_argument(
        "--use_soft_balanced_sampler",
        action="store_true",
        help="Use soft intensity-balanced sampler for training."
    )

    parser.add_argument(
        "--sampler_alpha",
        type=float,
        default=0.5,
        help="Soft balancing strength. 0=random sampling, 1=strict balancing."
    )

    parser.add_argument(
        "--sampler_bins",
        type=int,
        default=20,
        help="Number of intensity bins for soft sampler."
    )

    return parser.parse_args()


def main():

    args = parse_args()

    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print("Input channels: 440 + 555 + 740 nm")
    print(f"Use weights: {args.use_weights}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")

    train_dataset = ParticleDataset(
        data_dir="data/processed",
        split="train",
        channels=[0, 1, 2],
        target_transform="cube_root",
        use_weights=args.use_weights,
        mmap=True,
        clip_weights=args.clip_weights,
        min_weight=0.1,
        max_weight=args.max_weight
    )

    val_dataset = ParticleDataset(
        data_dir="data/processed",
        split="val",
        channels=[0, 1, 2],
        target_transform="cube_root",
        use_weights=args.use_weights,
        mmap=True,
        clip_weights=args.clip_weights,
        min_weight=0.1,
        max_weight=args.max_weight
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

    model = ConfigurableCNNRegressor(
        in_channels=3,
        conv_channels=[16, 32, 64, 128],
        kernel_sizes=[3, 3, 3, 3],
        dense_units=[256, 128, 64],
        dropout_rate=0.2,
        use_batch_norm=True,
        use_pooling=True
    )

    criterion = RegressionLoss(
        loss_type="mse",
        use_weights=args.use_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    weight_tag = "weighted" if args.use_weights else "unweighted"

    if args.use_weights and args.clip_weights:
        weight_tag = f"weighted_clipped_{args.max_weight:g}"
    elif args.use_weights:
        weight_tag = "weighted"
    else:
        weight_tag = "unweighted"

    if args.use_soft_balanced_sampler:
        sampler_tag = f"_soft_sampler_alpha_{args.sampler_alpha:g}"
    else:
        sampler_tag = ""
        
    output_dir = f"outputs/baseline_fusion_{weight_tag}{sampler_tag}"

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        output_dir=output_dir,
        early_stopping_patience=15
    )

    print()
    print("Fusion training completed.")
    print(history.tail())


if __name__ == "__main__":

    main()