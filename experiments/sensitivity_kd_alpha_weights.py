"""
Sensitivity analysis for CrossWave distillation weights.

Varies ONLY:
    alpha_pred
    alpha_feature

while fixing:
    alpha_target = 1.0
    student architecture
    learning rate
    batch size
    weight decay
    sampler
    training schedule

The fixed architecture/hyperparameters for each wavelength are loaded
from the already-selected CrossWave checkpoint:

    440 nm -> trial_016
    555 nm -> trial_033
    740 nm -> trial_065

This isolates sensitivity to the KD loss weights rather than
confounding the analysis with another Optuna architecture search.

Grid:
    alpha_pred    = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    alpha_feature = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

Main heatmap:
    validation MAE

Also saves:
    validation R2
    validation RMSE

Optional:
    multiple random seeds

Example
-------
cd C:\\Users\\jiaru\\Documents\\code_temp_files\\ParticleSizing

& "C:\\Users\\jiaru\\.conda\\envs\\LNP_image_analysis\\python.exe" `
    experiments\\sensitivity_kd_alpha_weights.py `
    --epochs 100 `
    --patience 20 `
    --seeds 42
"""


# ============================================================
# 1. IMPORTS
# ============================================================

import sys
from pathlib import Path
import argparse
import json
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


from datasets.kd_teacher_dataset import KDTeacherOutputDataset

from models.baseline_cnn import ConfigurableCNNRegressor

from training.kd_losses import FusionTeacherStudentKDLoss

from training.samplers import (
    create_soft_intensity_balanced_sampler
)

from utils.seed import set_seed


# ============================================================
# 2. GRID
# ============================================================

ALPHA_VALUES = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.5,
    0.7,
    1.0,
]

ALPHA_TARGET = 1.0


# ============================================================
# 3. REFERENCE CROSSWAVE CHECKPOINTS
# ============================================================
#
# These determine the FIXED student architecture and optimizer
# hyperparameters for each wavelength.
#
# Only alpha_pred and alpha_feature will be changed.
# ============================================================

REFERENCE_CHECKPOINTS = {

    "440": Path(
        r"outputs\optuna"
        r"\kd_student_from_fusion_teacher_v1_projection_student_440_at_1_ap_0.5_af_0.1_soft_sampler_alpha_0.3"
        r"\trial_016"
        r"\best_student_model.pt"
    ),

    "555": Path(
        r"outputs\optuna"
        r"\kd_student_from_fusion_teacher_v1_projection_student_555_at_1_ap_0.3_af_0.3_soft_sampler_alpha_0.3"
        r"\trial_033"
        r"\best_student_model.pt"
    ),

    "740": Path(
        r"outputs\optuna"
        r"\kd_student_from_fusion_teacher_v1_projection_student_740_at_1_ap_0.5_af_0.1_soft_sampler_alpha_0.3"
        r"\trial_065"
        r"\best_student_model.pt"
    ),
}


WAVELENGTHS = [
    "440",
    "555",
    "740",
]


# ============================================================
# 4. OUTPUT
# ============================================================

OUTPUT_ROOT = Path(
    r"outputs\sensitivity_kd_alpha_weights"
)


# ============================================================
# 5. ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "CrossWave sensitivity analysis for "
            "alpha_pred and alpha_feature."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help=(
            "Random seeds. Example: "
            "--seeds 42 43 44"
        ),
    )

    parser.add_argument(
        "--sampler_alpha",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--sampler_bins",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help=(
            "Skip conditions for which result.json "
            "already exists."
        ),
    )

    return parser.parse_args()


# ============================================================
# 6. CHANNEL INDEX
# ============================================================

def channel_to_index(channel):

    mapping = {
        "440": 0,
        "555": 1,
        "740": 2,
    }

    return mapping[
        channel
    ]


# ============================================================
# 7. STUDENT + PROJECTION
# ============================================================

class StudentKDModel(nn.Module):

    def __init__(
        self,
        student,
        teacher_embedding_dim,
    ):

        super().__init__()

        self.student = student

        student_embedding_dim = (
            student.embedding_dim
        )

        if (
            student_embedding_dim
            == teacher_embedding_dim
        ):

            self.projection = nn.Identity()

        else:

            self.projection = nn.Linear(
                student_embedding_dim,
                teacher_embedding_dim,
            )


    def forward_with_embedding(
        self,
        x,
    ):

        pred, embedding = (
            self.student.forward_with_embedding(
                x
            )
        )

        projected_embedding = (
            self.projection(
                embedding
            )
        )

        return (
            pred,
            embedding,
            projected_embedding,
        )


# ============================================================
# 8. METRICS
# ============================================================

def compute_metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )


    mae = np.mean(
        np.abs(
            y_pred
            - y_true
        )
    )


    mse = np.mean(
        (
            y_pred
            - y_true
        ) ** 2
    )


    rmse = np.sqrt(
        mse
    )


    ss_res = np.sum(
        (
            y_true
            - y_pred
        ) ** 2
    )


    ss_tot = np.sum(
        (
            y_true
            - np.mean(
                y_true
            )
        ) ** 2
    )


    r2 = (
        np.nan
        if ss_tot == 0
        else 1.0
        - ss_res
        / ss_tot
    )


    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


# ============================================================
# 9. RUN ONE EPOCH
# ============================================================

def run_one_epoch(
    student_kd,
    dataloader,
    criterion,
    optimizer,
    device,
    student_channel_index,
    train=True,
):

    if train:
        student_kd.train()

    else:
        student_kd.eval()


    total_loss = 0.0
    total_target_loss = 0.0
    total_pred_loss = 0.0
    total_feature_loss = 0.0

    num_batches = 0

    targets_all = []
    predictions_all = []


    context = (
        torch.enable_grad()
        if train
        else torch.no_grad()
    )


    with context:

        for batch in dataloader:

            images = batch[
                "image"
            ].to(
                device
            )


            targets = batch[
                "target"
            ].to(
                device
            ).float()


            teacher_pred = batch[
                "teacher_prediction"
            ].to(
                device
            ).float()


            teacher_embedding = batch[
                "teacher_embedding"
            ].to(
                device
            ).float()


            x_student = images[
                :,
                student_channel_index:
                student_channel_index + 1,
                :,
                :
            ]


            if train:

                optimizer.zero_grad()


            (
                student_pred,
                student_embedding,
                projected_embedding,
            ) = student_kd.forward_with_embedding(
                x_student
            )


            loss_dict = criterion(

                student_pred=
                    student_pred,

                student_embedding=
                    projected_embedding,

                teacher_pred=
                    teacher_pred,

                teacher_embedding=
                    teacher_embedding,

                target=
                    targets,
            )


            loss = loss_dict[
                "loss"
            ]


            if train:

                loss.backward()

                optimizer.step()


            total_loss += float(
                loss.item()
            )

            total_target_loss += float(
                loss_dict[
                    "loss_target"
                ].item()
            )

            total_pred_loss += float(
                loss_dict[
                    "loss_pred_distill"
                ].item()
            )

            total_feature_loss += float(
                loss_dict[
                    "loss_feature_distill"
                ].item()
            )


            num_batches += 1


            targets_all.append(
                targets
                .detach()
                .cpu()
                .numpy()
            )

            predictions_all.append(
                student_pred
                .detach()
                .cpu()
                .numpy()
            )


    y_true = np.concatenate(
        targets_all
    )

    y_pred = np.concatenate(
        predictions_all
    )


    metrics = compute_metrics(
        y_true,
        y_pred,
    )


    return {

        "loss":
            total_loss
            / num_batches,

        "loss_target":
            total_target_loss
            / num_batches,

        "loss_pred_distill":
            total_pred_loss
            / num_batches,

        "loss_feature_distill":
            total_feature_loss
            / num_batches,

        "student_MAE":
            metrics[
                "MAE"
            ],

        "student_MSE":
            metrics[
                "MSE"
            ],

        "student_RMSE":
            metrics[
                "RMSE"
            ],

        "student_R2":
            metrics[
                "R2"
            ],
    }


# ============================================================
# 10. LOAD REFERENCE CONFIGURATION
# ============================================================

def load_reference_configuration(
    channel,
    device,
):

    checkpoint_path = (
        REFERENCE_CHECKPOINTS[
            channel
        ]
    )


    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"\nReference checkpoint not found:\n"
            f"{checkpoint_path}"
        )


    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )


    params = checkpoint[
        "params"
    ].copy()


    teacher_output_dir = Path(
        checkpoint[
            "teacher_output_dir"
        ]
    )


    teacher_embedding_dim = int(
        checkpoint[
            "teacher_embedding_dim"
        ]
    )


    print()
    print(
        "=" * 80
    )

    print(
        f"REFERENCE CONFIGURATION: "
        f"{channel} nm"
    )

    print(
        "=" * 80
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"Teacher outputs: "
        f"{teacher_output_dir}"
    )

    print(
        f"Teacher embedding dim: "
        f"{teacher_embedding_dim}"
    )

    print(
        "Fixed student parameters:"
    )

    print(
        json.dumps(
            params,
            indent=4,
        )
    )


    return (
        params,
        teacher_output_dir,
        teacher_embedding_dim,
    )


# ============================================================
# 11. BUILD DATASETS
# ============================================================

def build_datasets(
    teacher_output_dir,
):

    train_dataset = (
        KDTeacherOutputDataset(

            data_dir=
                "data/processed",

            split=
                "train",

            teacher_output_dir=
                teacher_output_dir,

            channels=[
                0,
                1,
                2,
            ],

            target_transform=
                "cube_root",

            use_weights=
                False,

            mmap=
                True,
        )
    )


    val_dataset = (
        KDTeacherOutputDataset(

            data_dir=
                "data/processed",

            split=
                "val",

            teacher_output_dir=
                teacher_output_dir,

            channels=[
                0,
                1,
                2,
            ],

            target_transform=
                "cube_root",

            use_weights=
                False,

            mmap=
                True,
        )
    )


    return (
        train_dataset,
        val_dataset,
    )


# ============================================================
# 12. BUILD LOADERS
# ============================================================

def build_loaders(
    train_dataset,
    val_dataset,
    params,
    sampler_alpha,
    sampler_bins,
):

    train_sampler = (
        create_soft_intensity_balanced_sampler(

            dataset=
                train_dataset,

            num_bins=
                sampler_bins,

            alpha=
                sampler_alpha,

            replacement=
                True,
        )
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=
            int(
                params[
                    "batch_size"
                ]
            ),

        sampler=
            train_sampler,

        shuffle=
            False,

        num_workers=
            0,

        pin_memory=
            True,
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=
            int(
                params[
                    "batch_size"
                ]
            ),

        shuffle=
            False,

        num_workers=
            0,

        pin_memory=
            True,
    )


    return (
        train_loader,
        val_loader,
    )


# ============================================================
# 13. BUILD STUDENT
# ============================================================

def build_student(
    params,
    teacher_embedding_dim,
    device,
):

    student = ConfigurableCNNRegressor(

        in_channels=
            1,

        conv_channels=
            params[
                "conv_channels"
            ],

        kernel_sizes=
            params[
                "kernel_sizes"
            ],

        dense_units=
            params[
                "dense_units"
            ],

        dropout_rate=
            params[
                "dropout_rate"
            ],

        use_batch_norm=
            True,

        use_pooling=
            True,
    )


    student_kd = StudentKDModel(

        student=
            student,

        teacher_embedding_dim=
            teacher_embedding_dim,
    )


    student_kd.to(
        device
    )


    return student_kd


# ============================================================
# 14. TRAIN ONE CONDITION
# ============================================================

def train_condition(
    channel,
    alpha_pred,
    alpha_feature,
    seed,
    params,
    teacher_embedding_dim,
    train_dataset,
    val_dataset,
    args,
    device,
    output_dir,
):

    set_seed(
        seed
    )


    train_loader, val_loader = (
        build_loaders(

            train_dataset=
                train_dataset,

            val_dataset=
                val_dataset,

            params=
                params,

            sampler_alpha=
                args.sampler_alpha,

            sampler_bins=
                args.sampler_bins,
        )
    )


    student_kd = build_student(

        params=
            params,

        teacher_embedding_dim=
            teacher_embedding_dim,

        device=
            device,
    )


    optimizer = torch.optim.Adam(

        student_kd.parameters(),

        lr=
            float(
                params[
                    "lr"
                ]
            ),

        weight_decay=
            float(
                params[
                    "weight_decay"
                ]
            ),
    )


    criterion = FusionTeacherStudentKDLoss(

        alpha_target=
            ALPHA_TARGET,

        alpha_pred=
            alpha_pred,

        alpha_feature=
            alpha_feature,
    )


    channel_index = channel_to_index(
        channel
    )


    best_val_r2 = -np.inf
    best_epoch = -1

    best_metrics = None

    patience_counter = 0

    history = []


    start_time = time.time()


    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_metrics = run_one_epoch(

            student_kd=
                student_kd,

            dataloader=
                train_loader,

            criterion=
                criterion,

            optimizer=
                optimizer,

            device=
                device,

            student_channel_index=
                channel_index,

            train=
                True,
        )


        val_metrics = run_one_epoch(

            student_kd=
                student_kd,

            dataloader=
                val_loader,

            criterion=
                criterion,

            optimizer=
                optimizer,

            device=
                device,

            student_channel_index=
                channel_index,

            train=
                False,
        )


        history.append(
            {
                "epoch":
                    epoch,

                **{
                    f"train_{k}": v
                    for k, v
                    in train_metrics.items()
                },

                **{
                    f"val_{k}": v
                    for k, v
                    in val_metrics.items()
                },
            }
        )


        current_r2 = val_metrics[
            "student_R2"
        ]


        if current_r2 > best_val_r2:

            best_val_r2 = (
                current_r2
            )

            best_epoch = (
                epoch
            )

            best_metrics = (
                val_metrics.copy()
            )

            patience_counter = 0


            torch.save(

                {
                    "epoch":
                        epoch,

                    "student_state_dict":
                        student_kd
                        .student
                        .state_dict(),

                    "projection_state_dict":
                        student_kd
                        .projection
                        .state_dict(),

                    "params":
                        params,

                    "student_channel":
                        channel,

                    "alpha_target":
                        ALPHA_TARGET,

                    "alpha_pred":
                        alpha_pred,

                    "alpha_feature":
                        alpha_feature,

                    "seed":
                        seed,

                    "val_metrics":
                        val_metrics,

                    "teacher_embedding_dim":
                        teacher_embedding_dim,
                },

                output_dir
                / "best_student_model.pt",
            )


        else:

            patience_counter += 1


            if (
                patience_counter
                >= args.patience
            ):

                break


    elapsed = (
        time.time()
        - start_time
    )


    history_df = pd.DataFrame(
        history
    )


    history_df.to_csv(

        output_dir
        / "training_history.csv",

        index=False,
    )


    result = {

        "channel":
            channel,

        "alpha_target":
            ALPHA_TARGET,

        "alpha_pred":
            alpha_pred,

        "alpha_feature":
            alpha_feature,

        "seed":
            seed,

        "best_epoch":
            best_epoch,

        "val_R2":
            best_metrics[
                "student_R2"
            ],

        "val_MAE":
            best_metrics[
                "student_MAE"
            ],

        "val_MSE":
            best_metrics[
                "student_MSE"
            ],

        "val_RMSE":
            best_metrics[
                "student_RMSE"
            ],

        "val_loss":
            best_metrics[
                "loss"
            ],

        "runtime_seconds":
            elapsed,
    }


    with open(
        output_dir
        / "result.json",
        "w",
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
        )


    return result


# ============================================================
# 15. RUN FULL GRID
# ============================================================

def run_grid(
    args,
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    all_results = []


    total_runs = (
        len(WAVELENGTHS)
        * len(ALPHA_VALUES)
        * len(ALPHA_VALUES)
        * len(args.seeds)
    )


    run_index = 0


    print(
        "=" * 80
    )

    print(
        "CROSSWAVE LOSS-WEIGHT SENSITIVITY"
    )

    print(
        "=" * 80
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Grid: "
        f"{len(ALPHA_VALUES)} x "
        f"{len(ALPHA_VALUES)}"
    )

    print(
        f"Seeds: {args.seeds}"
    )

    print(
        f"Total runs: {total_runs}"
    )


    for channel in WAVELENGTHS:

        (
            params,
            teacher_output_dir,
            teacher_embedding_dim,
        ) = load_reference_configuration(

            channel=
                channel,

            device=
                device,
        )


        (
            train_dataset,
            val_dataset,
        ) = build_datasets(

            teacher_output_dir=
                teacher_output_dir
        )


        for alpha_feature in ALPHA_VALUES:

            for alpha_pred in ALPHA_VALUES:

                for seed in args.seeds:

                    run_index += 1


                    run_name = (

                        f"channel_{channel}"
                        f"_ap_{alpha_pred:g}"
                        f"_af_{alpha_feature:g}"
                        f"_seed_{seed}"
                    )


                    output_dir = (
                        OUTPUT_ROOT
                        / channel
                        / run_name
                    )


                    output_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )


                    result_path = (
                        output_dir
                        / "result.json"
                    )


                    print()
                    print(
                        f"[{run_index}/{total_runs}] "
                        f"{channel} nm | "
                        f"ap={alpha_pred:g} | "
                        f"af={alpha_feature:g} | "
                        f"seed={seed}"
                    )


                    if (
                        args.skip_existing
                        and result_path.exists()
                    ):

                        with open(
                            result_path,
                            "r",
                        ) as f:

                            result = json.load(
                                f
                            )


                        print(
                            "  -> existing result loaded"
                        )


                    else:

                        result = train_condition(

                            channel=
                                channel,

                            alpha_pred=
                                alpha_pred,

                            alpha_feature=
                                alpha_feature,

                            seed=
                                seed,

                            params=
                                params,

                            teacher_embedding_dim=
                                teacher_embedding_dim,

                            train_dataset=
                                train_dataset,

                            val_dataset=
                                val_dataset,

                            args=
                                args,

                            device=
                                device,

                            output_dir=
                                output_dir,
                        )


                    all_results.append(
                        result
                    )


                    print(
                        f"  val R²   = "
                        f"{result['val_R2']:.4f}"
                    )

                    print(
                        f"  val MAE  = "
                        f"{result['val_MAE']:.4f}"
                    )

                    print(
                        f"  val RMSE = "
                        f"{result['val_RMSE']:.4f}"
                    )


                    pd.DataFrame(
                        all_results
                    ).to_csv(

                        OUTPUT_ROOT
                        / "sensitivity_results_partial.csv",

                        index=False,
                    )


    results_df = pd.DataFrame(
        all_results
    )


    results_df.to_csv(

        OUTPUT_ROOT
        / "sensitivity_results_all_runs.csv",

        index=False,
    )


    return results_df


# ============================================================
# 16. AGGREGATE SEEDS
# ============================================================

def aggregate_results(
    results_df,
):

    metric_columns = [
        "val_R2",
        "val_MAE",
        "val_RMSE",
    ]


    grouped = results_df.groupby(
        [
            "channel",
            "alpha_pred",
            "alpha_feature",
        ],
        as_index=False,
    )


    mean_df = grouped[
        metric_columns
    ].mean()


    std_df = grouped[
        metric_columns
    ].std()


    std_df = std_df.rename(
        columns={
            metric:
                f"{metric}_std"
            for metric
            in metric_columns
        }
    )


    aggregated = mean_df.merge(

        std_df,

        on=[
            "channel",
            "alpha_pred",
            "alpha_feature",
        ],

        how="left",
    )


    aggregated.to_csv(

        OUTPUT_ROOT
        / "sensitivity_results_aggregated.csv",

        index=False,
    )


    return aggregated


# ============================================================
# 17. PLOT STYLE
# ============================================================

plt.rcParams.update({

    "font.family":
        "Arial",

    "font.size":
        8,

    "axes.labelsize":
        8,

    "axes.titlesize":
        9,

    "xtick.labelsize":
        7,

    "ytick.labelsize":
        7,

    "axes.linewidth":
        0.8,

    "svg.fonttype":
        "none",

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,
})


WAVELENGTH_COLORS = {

    "440":
        "#2F5FD0",

    "555":
        "#2F8F3A",

    "740":
        "#D73027",
}


# ============================================================
# 18. THREE-PANEL HEATMAP
# ============================================================

def plot_three_panel_heatmap(
    aggregated,
    metric,
    filename,
    colorbar_label,
    lower_is_better,
):

    matrices = []


    for channel in WAVELENGTHS:

        subset = aggregated[
            aggregated[
                "channel"
            ].astype(str)
            == channel
        ]


        pivot = subset.pivot(

            index=
                "alpha_feature",

            columns=
                "alpha_pred",

            values=
                metric,
        )


        pivot = pivot.reindex(

            index=
                ALPHA_VALUES,

            columns=
                ALPHA_VALUES,
        )


        matrices.append(
            pivot.values
        )


    all_values = np.concatenate(
        [
            m.ravel()
            for m in matrices
        ]
    )


    vmin = np.nanmin(
        all_values
    )

    vmax = np.nanmax(
        all_values
    )


    # ========================================================
    # Figure
    # ========================================================

    fig, axes = plt.subplots(

        1,
        3,

        figsize=(
            7.2,
            2.55,
        ),
    )


    last_im = None


    for (
        ax,
        channel,
        matrix,
    ) in zip(
        axes,
        WAVELENGTHS,
        matrices,
    ):


        cmap = (
            "viridis_r"
            if lower_is_better
            else "viridis"
        )


        last_im = ax.imshow(

            matrix,

            cmap=
                cmap,

            vmin=
                vmin,

            vmax=
                vmax,

            aspect=
                "equal",

            interpolation=
                "nearest",

            origin=
                "lower",
        )


        # ----------------------------------------------------
        # ticks
        # ----------------------------------------------------

        ticks = np.arange(
            len(
                ALPHA_VALUES
            )
        )


        ax.set_xticks(
            ticks
        )

        ax.set_yticks(
            ticks
        )


        ax.set_xticklabels(
            [
                f"{v:g}"
                for v
                in ALPHA_VALUES
            ]
        )


        ax.set_yticklabels(
            [
                f"{v:g}"
                for v
                in ALPHA_VALUES
            ]
        )


        ax.set_xlabel(
            r"Prediction weight, $\alpha_p$"
        )


        if ax is axes[0]:

            ax.set_ylabel(
                r"Feature weight, $\alpha_f$"
            )


        ax.set_title(

            f"{channel} nm",

            color=
                WAVELENGTH_COLORS[
                    channel
                ],

            fontweight=
                "bold",

            pad=
                5,
        )


        # ----------------------------------------------------
        # cell values
        # ----------------------------------------------------

        midpoint = (
            vmin
            + vmax
        ) / 2


        for r in range(
            matrix.shape[0]
        ):

            for c in range(
                matrix.shape[1]
            ):

                value = matrix[
                    r,
                    c
                ]


                if not np.isfinite(
                    value
                ):

                    continue


                if metric == "val_R2":

                    label = (
                        f"{value:.3f}"
                    )

                else:

                    label = (
                        f"{value:.3f}"
                    )


                normalized = (
                    (
                        value
                        - vmin
                    )
                    /
                    max(
                        vmax
                        - vmin,
                        1e-12,
                    )
                )


                text_color = (
                    "white"
                    if (
                        normalized < 0.28
                        or normalized > 0.78
                    )
                    else "black"
                )


                ax.text(

                    c,
                    r,

                    label,

                    ha=
                        "center",

                    va=
                        "center",

                    fontsize=
                        5.0,

                    color=
                        text_color,
                )


        # ----------------------------------------------------
        # Mark best cell
        # ----------------------------------------------------

        if lower_is_better:

            best_flat = np.nanargmin(
                matrix
            )

        else:

            best_flat = np.nanargmax(
                matrix
            )


        best_r, best_c = (
            np.unravel_index(
                best_flat,
                matrix.shape,
            )
        )


        ax.scatter(

            best_c,
            best_r,

            marker=
                "s",

            s=
                115,

            facecolors=
                "none",

            edgecolors=
                "white",

            linewidths=
                1.4,

            zorder=
                10,
        )


    # ========================================================
    # Shared colorbar
    # ========================================================

    cbar = fig.colorbar(

        last_im,

        ax=
            axes,

        fraction=
            0.025,

        pad=
            0.025,
    )


    cbar.set_label(
        colorbar_label
    )


    fig.subplots_adjust(

        left=
            0.08,

        right=
            0.90,

        bottom=
            0.19,

        top=
            0.88,

        wspace=
            0.22,
    )


    svg_path = (
        OUTPUT_ROOT
        / f"{filename}.svg"
    )


    png_path = (
        OUTPUT_ROOT
        / f"{filename}.png"
    )


    fig.savefig(

        svg_path,

        format=
            "svg",

        bbox_inches=
            "tight",

        transparent=
            False,
    )


    fig.savefig(

        png_path,

        dpi=
            600,

        bbox_inches=
            "tight",

        transparent=
            False,
    )


    plt.show()


# ============================================================
# 19. GENERATE HEATMAPS
# ============================================================

def generate_heatmaps(
    aggregated,
):

    # --------------------------------------------------------
    # Main manuscript candidate
    # --------------------------------------------------------

    plot_three_panel_heatmap(

        aggregated=
            aggregated,

        metric=
            "val_MAE",

        filename=
            "Fig5_alpha_sensitivity_val_MAE",

        colorbar_label=
            r"Validation MAE ($\sqrt[3]{I_{\mathrm{Cy3}}}$)",

        lower_is_better=
            True,
    )


    # --------------------------------------------------------
    # Supplementary
    # --------------------------------------------------------

    plot_three_panel_heatmap(

        aggregated=
            aggregated,

        metric=
            "val_R2",

        filename=
            "alpha_sensitivity_val_R2",

        colorbar_label=
            r"Validation $R^2$",

        lower_is_better=
            False,
    )


    plot_three_panel_heatmap(

        aggregated=
            aggregated,

        metric=
            "val_RMSE",

        filename=
            "alpha_sensitivity_val_RMSE",

        colorbar_label=
            r"Validation RMSE ($\sqrt[3]{I_{\mathrm{Cy3}}}$)",

        lower_is_better=
            True,
    )


# ============================================================
# 20. MAIN
# ============================================================

def main():

    args = parse_args()


    results_df = run_grid(
        args
    )


    aggregated = aggregate_results(
        results_df
    )


    generate_heatmaps(
        aggregated
    )


    print()
    print(
        "=" * 80
    )

    print(
        "SENSITIVITY ANALYSIS COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Results saved to:\n"
        f"{OUTPUT_ROOT}"
    )


if __name__ == "__main__":

    main()