"""
Hyperparameter optimization for particle sizing baseline CNN.

Supports:
    440-only
    555-only
    740-only
    fusion: 440 + 555 + 740

Target:
    Cube-root transformed Cy3 max intensity

Optimization objective:
    Minimize validation RMSE / MSE loss on cube-root Cy3 max intensity.
"""

import sys
from pathlib import Path
import argparse
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import optuna
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
        description="Optimize baseline CNN hyperparameters."
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["440", "555", "740", "fusion"],
        help="Input setting to optimize."
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=30,
        help="Number of Optuna trials."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Maximum epochs per trial."
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early stopping patience per trial."
    )

    parser.add_argument(
        "--use_weights",
        action="store_true",
        help="Use sample weights during optimization."
    )

    parser.add_argument(
        "--use_soft_balanced_sampler",
        action="store_true",
        help="Use soft intensity-balanced sampler during training."
    )

    parser.add_argument(
        "--sampler_alpha",
        type=float,
        default=0.3,
        help="Soft balancing strength. 0=random sampling, 1=strict balancing."
    )

    parser.add_argument(
        "--sampler_bins",
        type=int,
        default=20,
        help="Number of intensity bins for the soft sampler."
    )

    return parser.parse_args()


def get_channels(mode):

    if mode == "440":
        return [0], "440"

    if mode == "555":
        return [1], "555"

    if mode == "740":
        return [2], "740"

    if mode == "fusion":
        return [0, 1, 2], "fusion"

    raise ValueError(f"Unknown mode: {mode}")

def compute_r2_score(y_true, y_pred):
    """
    Compute R2 score.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if ss_tot == 0:
        return np.nan

    r2 = 1.0 - ss_res / ss_tot

    return r2

def evaluate_r2(
    model,
    dataloader,
    device
):
    """
    Evaluate model R2 on a dataloader.
    """

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

    r2 = compute_r2_score(
        y_true,
        y_pred
    )

    return r2

def suggest_hyperparameters(trial):

    # --------------------------------------------------------
    # Architecture
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
        [1.5, 2]
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

    use_batch_norm = True  # Fixed for now
    # use_batch_norm = trial.suggest_categorical(
    #     "use_batch_norm",
    #     [True, False]
    # )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    lr = trial.suggest_float(
        "lr",
        1e-5,
        1e-3,
        log=True
    )

    # Fixed for now
    batch_size = 128

    weight_decay = 10e-9

    return {
        "num_conv_layers": num_conv_layers,
        "conv_channels": conv_channels,
        "kernel_sizes": kernel_sizes,
        "dense_units": dense_units,
        "dropout_rate": dropout_rate,
        "use_batch_norm": use_batch_norm,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size
    }


def objective(
    trial,
    args,
    channels,
    mode_name,
    device
):

    set_seed(
        42 + trial.number
    )

    params = suggest_hyperparameters(
        trial
    )

    # print()
    # print("=" * 80)
    # print(f"Trial {trial.number}")
    # print("=" * 80)
    # print(json.dumps(params, indent=4))

    # ========================================================
    # Dataset
    # ========================================================

    train_dataset = ParticleDataset(
        data_dir="data/processed",
        split="train",
        channels=channels,
        target_transform="cube_root",
        use_weights=args.use_weights,
        mmap=True
    )

    val_dataset = ParticleDataset(
        data_dir="data/processed",
        split="val",
        channels=channels,
        target_transform="cube_root",
        use_weights=args.use_weights,
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

    # ========================================================
    # Model
    # ========================================================

    model = ConfigurableCNNRegressor(
        in_channels=len(channels),
        conv_channels=params["conv_channels"],
        kernel_sizes=params["kernel_sizes"],
        dense_units=params["dense_units"],
        dropout_rate=params["dropout_rate"],
        use_batch_norm=params["use_batch_norm"],
        use_pooling=True
    )

    criterion = RegressionLoss(
        loss_type="mse",
        use_weights=args.use_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"]
    )

    # ========================================================
    # Output directory for this trial
    # ========================================================

    weight_tag = "weighted" if args.use_weights else "unweighted"

    if args.use_soft_balanced_sampler:
        sampler_tag = f"_soft_sampler_alpha_{args.sampler_alpha:g}"
    else:
        sampler_tag = ""

    trial_output_dir = (
        Path("outputs")
        / "optuna"
        / f"baseline_{mode_name}_{weight_tag}{sampler_tag}"
        / f"trial_{trial.number:03d}"
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        output_dir=trial_output_dir,
        early_stopping_patience=args.patience,
        verbose=False
    )

    best_val_loss = history["val_loss"].min()

    best_epoch = int(
        history.loc[
            history["val_loss"].idxmin(),
            "epoch"
        ]
    )

    # --------------------------------------------------------
    # Load best checkpoint before computing final validation R2
    # --------------------------------------------------------

    checkpoint_path = trial_output_dir / "best_model.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    val_r2 = evaluate_r2(
        model=model,
        dataloader=val_loader,
        device=device
    )

    trial.set_user_attr(
        "best_epoch",
        best_epoch
    )

    trial.set_user_attr(
        "best_val_loss",
        float(best_val_loss)
    )

    trial.set_user_attr(
        "val_r2",
        float(val_r2)
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

    print(
        f"Trial {trial.number:03d} completed | "
        f"best_val_loss={best_val_loss:.6f} | "
        f"best_epoch={best_epoch} | "
        f"lr={params['lr']:.2e} | "
        f"conv={params['conv_channels']} | "
        f"dense={params['dense_units']} | "
        f"dropout={params['dropout_rate']:.3f} | "
        f"batch_norm={params['use_batch_norm']}"
    )

    return val_r2


def main():

    args = parse_args()

    channels, mode_name = get_channels(
        args.mode
    )

    weight_tag = "weighted" if args.use_weights else "unweighted"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Channels: {channels}")
    print(f"Use weights: {args.use_weights}")

    study_dir = Path(
        "outputs/optuna"
    )

    study_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if args.use_soft_balanced_sampler:
        sampler_tag = f"_soft_sampler_alpha_{args.sampler_alpha:g}"
    else:
        sampler_tag = ""

    storage_path = (
        study_dir / f"baseline_{mode_name}_{weight_tag}{sampler_tag}.db"
    )

    study_name = f"baseline_{mode_name}_{weight_tag}{sampler_tag}"

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        direction="maximize",
        load_if_exists=True
    )

    study.optimize(
        lambda trial: objective(
            trial=trial,
            args=args,
            channels=channels,
            mode_name=mode_name,
            device=device
        ),
        n_trials=args.n_trials
    )

    print()
    print("=" * 80)
    print("Best Trial")
    print("=" * 80)

    best_trial = study.best_trial

    print(f"Best value: {best_trial.value}")
    print("Best params:")
    print(json.dumps(best_trial.params, indent=4))

    print("Best user attrs:")
    print(json.dumps(best_trial.user_attrs, indent=4))

    # Save best parameters
    if args.use_soft_balanced_sampler:
        sampler_tag = f"_soft_sampler_alpha_{args.sampler_alpha:g}"
    else:
        sampler_tag = ""

    output_dir = (
        Path("outputs")
        / "optuna"
        / f"baseline_{mode_name}_{weight_tag}{sampler_tag}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    best_summary = {
        "mode": args.mode,
        "channels": channels,
        "use_weights": args.use_weights,
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "best_user_attrs": best_trial.user_attrs
    }

    with open(
        output_dir / "best_trial.json",
        "w"
    ) as f:
        json.dump(
            best_summary,
            f,
            indent=4
        )

    trials_df = study.trials_dataframe()

    trials_df.to_csv(
        output_dir / "trials.csv",
        index=False
    )

    print()
    print(f"Saved Optuna results to: {output_dir}")


if __name__ == "__main__":

    main()