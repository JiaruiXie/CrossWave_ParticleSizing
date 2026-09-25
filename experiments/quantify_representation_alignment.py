"""
Quantify representation alignment between single-wavelength models
and the multi-wavelength Fusion model using linear CKA.

Main comparison:

    Baseline 440   vs Fusion
    CrossWave 440  vs Fusion

    Baseline 555   vs Fusion
    CrossWave 555  vs Fusion

    Baseline 740   vs Fusion
    CrossWave 740  vs Fusion


The analysis uses the ORIGINAL latent embeddings returned by:

    model.extract_embedding(x)

NOT the 2D PCA/t-SNE coordinates.

Models are compared using exactly matched test particles based on:

    particle_id
    image_index


Linear CKA:

                      || X^T Y ||_F^2
    CKA(X, Y) = --------------------------------
                 || X^T X ||_F || Y^T Y ||_F


Run from project root, for example:

cd C:\\Users\\jiaru\\Documents\\code_temp_files\\ParticleSizing

& "C:\\Users\\jiaru\\.conda\\envs\\LNP_image_analysis\\python.exe" `
    experiments/quantify_representation_alignment.py `
    --data_dir "data/processed" `
    --output_dir "outputs/representation_alignment" `
    --batch_size 256 `
    --bootstrap_reps 100 `
    --bootstrap_fraction 0.8
"""


# ============================================================
# 1. IMPORTS
# ============================================================

import sys
from pathlib import Path
import argparse
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from datasets.particle_dataset import ParticleDataset
from models.baseline_cnn import ConfigurableCNNRegressor
from utils.seed import set_seed


# ============================================================
# 2. MODEL SPECIFICATIONS
# ============================================================

MODEL_SPECS = [

    {
        "name": "Baseline 440 nm",
        "model_type": "baseline",
        "mode": "440",
        "checkpoint":
            r"outputs\optuna"
            r"\baseline_440_unweighted_soft_sampler_alpha_0.3002"
            r"\trial_046"
            r"\best_model.pt"
    },

    {
        "name": "Baseline 555 nm",
        "model_type": "baseline",
        "mode": "555",
        "checkpoint":
            r"outputs\optuna"
            r"\baseline_555_unweighted_soft_sampler_alpha_0.3"
            r"\trial_013"
            r"\best_model.pt"
    },

    {
        "name": "Baseline 740 nm",
        "model_type": "baseline",
        "mode": "740",
        "checkpoint":
            r"outputs\optuna"
            r"\baseline_740_unweighted_soft_sampler_alpha_0.3"
            r"\trial_009"
            r"\best_model.pt"
    },

    {
        "name": "Fusion",
        "model_type": "baseline",
        "mode": "fusion",
        "checkpoint":
            r"outputs\optuna"
            r"\baseline_fusion_unweighted_soft_sampler_alpha_0.3"
            r"\trial_046"
            r"\best_model.pt"
    },

    {
        "name": "CrossWave 440 nm",
        "model_type": "kd_student",
        "mode": "440",
        "checkpoint":
            r"outputs\optuna"
            r"\kd_student_from_fusion_teacher_v1_projection_student_440_at_1_ap_0.5_af_0.1_soft_sampler_alpha_0.3"
            r"\trial_016"
            r"\best_student_model.pt"
    },

    {
        "name": "CrossWave 555 nm",
        "model_type": "kd_student",
        "mode": "555",
        "checkpoint":
            r"outputs\optuna"
            r"\kd_student_from_fusion_teacher_v1_projection_student_555_at_1_ap_0.3_af_0.3_soft_sampler_alpha_0.3"
            r"\trial_033"
            r"\best_student_model.pt"
    },

    {
        "name": "CrossWave 740 nm",
        "model_type": "kd_student",
        "mode": "740",
        "checkpoint":
            r"outputs\optuna"
            r"\kd_student_from_fusion_teacher_v1_projection_student_740_at_1_ap_0.5_af_0.1_soft_sampler_alpha_0.3"
            r"\trial_065"
            r"\best_student_model.pt"
    },
]


# ============================================================
# 3. ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Quantify latent representation alignment "
            "with the multi-wavelength Fusion model."
        )
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/representation_alignment",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--bootstrap_reps",
        type=int,
        default=100,
        help=(
            "Number of bootstrap repetitions for 95% confidence intervals. "
            "Set to 0 to disable bootstrapping."
        ),
    )

    parser.add_argument(
        "--bootstrap_fraction",
        type=float,
        default=0.8,
        help=(
            "Fraction of matched test particles sampled in each bootstrap "
            "replicate."
        ),
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help=(
            "Optional maximum number of matched test samples used for CKA. "
            "Default uses all matched samples."
        ),
    )

    return parser.parse_args()


# ============================================================
# 4. CHANNEL HELPERS
# ============================================================

def get_channels(mode):

    if mode == "440":
        return [0]

    if mode == "555":
        return [1]

    if mode == "740":
        return [2]

    if mode == "fusion":
        return [0, 1, 2]

    raise ValueError(
        f"Unknown mode: {mode}"
    )


def channel_to_index(mode):

    if mode == "440":
        return 0

    if mode == "555":
        return 1

    if mode == "740":
        return 2

    raise ValueError(
        f"KD student only supports single wavelength mode. Got: {mode}"
    )


# ============================================================
# 5. INFER MODEL ARCHITECTURE
# ============================================================

def infer_architecture_from_state_dict(state_dict):

    conv_items = []

    for key, value in state_dict.items():

        if (
            key.startswith("features.")
            and key.endswith(".weight")
            and value.ndim == 4
        ):

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
                    kernel_size,
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

        if (
            key.startswith("regressor.")
            and key.endswith(".weight")
            and value.ndim == 2
        ):

            layer_index = int(
                key.split(".")[1]
            )

            out_features = int(
                value.shape[0]
            )

            linear_items.append(
                (
                    layer_index,
                    out_features,
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
            linear_indices[i + 1]
            - linear_indices[i]

            for i
            in range(
                len(linear_indices) - 1
            )
        ]

        if any(
            gap == 3
            for gap in gaps
        ):
            dropout_rate = 0.2


    architecture = {

        "conv_channels":
            conv_channels,

        "kernel_sizes":
            kernel_sizes,

        "dense_units":
            dense_units,

        "dropout_rate":
            dropout_rate,

        "use_batch_norm":
            True,
    }

    return architecture


# ============================================================
# 6. CHECKPOINT HELPERS
# ============================================================

def get_state_dict_from_checkpoint(
    checkpoint,
    model_type,
):

    if model_type == "baseline":

        if "model_state_dict" not in checkpoint:

            raise KeyError(
                "Baseline checkpoint does not contain model_state_dict."
            )

        return checkpoint[
            "model_state_dict"
        ]


    if model_type == "kd_student":

        if "student_state_dict" not in checkpoint:

            raise KeyError(
                "KD checkpoint does not contain student_state_dict."
            )

        return checkpoint[
            "student_state_dict"
        ]


    raise ValueError(
        f"Unknown model_type: {model_type}"
    )


def build_model_from_checkpoint(
    checkpoint_path,
    model_type,
    in_channels,
    device,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_dict = get_state_dict_from_checkpoint(
        checkpoint=checkpoint,
        model_type=model_type,
    )

    architecture = infer_architecture_from_state_dict(
        state_dict
    )


    config = checkpoint.get(
        "config",
        None,
    )

    if config is None:

        config = checkpoint.get(
            "params",
            None,
        )


    if (
        config is not None
        and "dropout_rate" in config
    ):

        saved_dropout_rate = float(
            config[
                "dropout_rate"
            ]
        )

        if architecture[
            "dropout_rate"
        ] > 0:

            architecture[
                "dropout_rate"
            ] = saved_dropout_rate


    model = ConfigurableCNNRegressor(

        in_channels=in_channels,

        conv_channels=
            architecture[
                "conv_channels"
            ],

        kernel_sizes=
            architecture[
                "kernel_sizes"
            ],

        dense_units=
            architecture[
                "dense_units"
            ],

        dropout_rate=
            architecture[
                "dropout_rate"
            ],

        use_batch_norm=
            architecture[
                "use_batch_norm"
            ],

        use_pooling=True,
    )


    model.load_state_dict(
        state_dict
    )

    model.to(
        device
    )

    model.eval()

    return model, architecture


# ============================================================
# 7. EXTRACT EMBEDDINGS
# ============================================================

def extract_embeddings_for_model(
    model_spec,
    data_dir,
    batch_size,
    device,
):

    model_name = model_spec[
        "name"
    ]

    model_type = model_spec[
        "model_type"
    ]

    mode = model_spec[
        "mode"
    ]

    checkpoint_path = Path(
        model_spec[
            "checkpoint"
        ]
    )


    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n"
            f"{checkpoint_path}"
        )


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    if model_type == "baseline":

        channels = get_channels(
            mode
        )

        dataset = ParticleDataset(

            data_dir=data_dir,

            split="test",

            channels=channels,

            target_transform="cube_root",

            use_weights=False,

            mmap=True,
        )

        in_channels = len(
            channels
        )

        student_channel_index = None


    elif model_type == "kd_student":

        # Load all wavelengths, then retain only
        # the student's wavelength.
        dataset = ParticleDataset(

            data_dir=data_dir,

            split="test",

            channels=[
                0,
                1,
                2,
            ],

            target_transform="cube_root",

            use_weights=False,

            mmap=True,
        )

        in_channels = 1

        student_channel_index = (
            channel_to_index(
                mode
            )
        )


    else:

        raise ValueError(
            f"Unknown model_type: {model_type}"
        )


    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0,
    )


    model, architecture = (
        build_model_from_checkpoint(

            checkpoint_path=
                checkpoint_path,

            model_type=
                model_type,

            in_channels=
                in_channels,

            device=
                device,
        )
    )


    all_embeddings = []

    all_particle_ids = []

    all_image_indices = []

    all_targets = []


    with torch.no_grad():

        for batch in loader:

            images = batch[
                "image"
            ].to(
                device
            )


            if model_type == "kd_student":

                images = images[
                    :,
                    student_channel_index:
                    student_channel_index + 1,
                    :,
                    :
                ]


            embeddings = (
                model.extract_embedding(
                    images
                )
            )


            all_embeddings.append(
                embeddings
                .detach()
                .cpu()
                .numpy()
            )


            all_particle_ids.extend(
                batch[
                    "particle_id"
                ]
                .cpu()
                .numpy()
                .tolist()
            )


            all_image_indices.extend(
                batch[
                    "image_index"
                ]
                .cpu()
                .numpy()
                .tolist()
            )


            all_targets.extend(
                batch[
                    "target"
                ]
                .cpu()
                .numpy()
                .tolist()
            )


    embeddings = np.concatenate(
        all_embeddings,
        axis=0,
    ).astype(
        np.float64
    )


    metadata = pd.DataFrame(
        {
            "particle_id":
                all_particle_ids,

            "image_index":
                all_image_indices,

            "target":
                all_targets,

            "row_index":
                np.arange(
                    len(
                        all_particle_ids
                    )
                ),
        }
    )


    print()
    print("=" * 80)
    print(
        f"{model_name}"
    )
    print("=" * 80)

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        "Architecture:"
    )

    print(
        json.dumps(
            architecture,
            indent=4,
        )
    )


    return embeddings, metadata


# ============================================================
# 8. LINEAR CKA
# ============================================================

def center_features(X):
    """
    Center each latent feature across samples.
    """

    return (
        X
        - X.mean(
            axis=0,
            keepdims=True,
        )
    )


def linear_cka(
    X,
    Y,
    eps=1e-12,
):
    """
    Feature-space linear CKA.

    X shape:
        N x D1

    Y shape:
        N x D2

    The feature dimensions D1 and D2 do NOT have to match.

    This implementation avoids constructing an N x N Gram matrix,
    which is important for large test sets.
    """

    if X.shape[0] != Y.shape[0]:

        raise ValueError(
            "X and Y must contain the same number of matched samples."
        )


    Xc = center_features(
        X.astype(
            np.float64,
            copy=False,
        )
    )

    Yc = center_features(
        Y.astype(
            np.float64,
            copy=False,
        )
    )


    cross_cov = (
        Xc.T
        @ Yc
    )

    x_cov = (
        Xc.T
        @ Xc
    )

    y_cov = (
        Yc.T
        @ Yc
    )


    numerator = (
        np.linalg.norm(
            cross_cov,
            ord="fro",
        ) ** 2
    )


    denominator = (
        np.linalg.norm(
            x_cov,
            ord="fro",
        )
        *
        np.linalg.norm(
            y_cov,
            ord="fro",
        )
    )


    if denominator < eps:

        return np.nan


    return float(
        numerator
        / denominator
    )


# ============================================================
# 9. MATCH IDENTICAL TEST PARTICLES
# ============================================================

def match_embeddings(
    embedding_a,
    metadata_a,
    embedding_b,
    metadata_b,
):

    keys = [
        "particle_id",
        "image_index",
    ]


    a = metadata_a[
        keys
        + [
            "row_index"
        ]
    ].rename(
        columns={
            "row_index":
                "row_a"
        }
    )


    b = metadata_b[
        keys
        + [
            "row_index"
        ]
    ].rename(
        columns={
            "row_index":
                "row_b"
        }
    )


    matched = a.merge(

        b,

        on=keys,

        how="inner",

        validate="one_to_one",
    )


    if len(matched) == 0:

        raise ValueError(
            "No matched test particles were found."
        )


    X = embedding_a[
        matched[
            "row_a"
        ].to_numpy()
    ]


    Y = embedding_b[
        matched[
            "row_b"
        ].to_numpy()
    ]


    return (
        X,
        Y,
        matched,
    )


# ============================================================
# 10. BOOTSTRAP CKA
# ============================================================

def bootstrap_cka(
    X,
    Y,
    n_reps,
    fraction,
    seed,
):
    """
    Estimate a 95% bootstrap interval for CKA.

    Sampling is paired:
    the same row indices are applied to X and Y.
    """

    if n_reps <= 0:

        return (
            np.nan,
            np.nan,
            np.array([]),
        )


    rng = np.random.default_rng(
        seed
    )


    n = X.shape[0]


    n_boot = max(
        2,
        int(
            np.round(
                n * fraction
            )
        ),
    )


    bootstrap_values = []


    for rep in range(
        n_reps
    ):

        indices = rng.choice(

            n,

            size=n_boot,

            replace=True,
        )


        value = linear_cka(

            X[
                indices
            ],

            Y[
                indices
            ],
        )


        bootstrap_values.append(
            value
        )


    bootstrap_values = np.asarray(
        bootstrap_values
    )


    ci_lower = np.nanpercentile(
        bootstrap_values,
        2.5,
    )


    ci_upper = np.nanpercentile(
        bootstrap_values,
        97.5,
    )


    return (
        float(ci_lower),
        float(ci_upper),
        bootstrap_values,
    )


# ============================================================
# 11. OPTIONAL SAMPLE LIMIT
# ============================================================

def subsample_matched(
    X,
    Y,
    max_samples,
    seed,
):

    if max_samples is None:

        return X, Y


    if X.shape[0] <= max_samples:

        return X, Y


    rng = np.random.default_rng(
        seed
    )


    indices = rng.choice(

        X.shape[0],

        size=max_samples,

        replace=False,
    )


    return (
        X[
            indices
        ],
        Y[
            indices
        ],
    )


# ============================================================
# 12. FIGURE STYLE
# ============================================================

WAVELENGTHS = [
    "440 nm",
    "555 nm",
    "740 nm",
]


WAVELENGTH_COLORS = {
    "440 nm": "#2F5FD0",
    "555 nm": "#2F8F3A",
    "740 nm": "#D73027",
}


BASELINE_COLOR = "#BDBDBD"


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
        8,

    "ytick.labelsize":
        8,

    "legend.fontsize":
        7,

    "axes.linewidth":
        0.8,

    "xtick.major.width":
        0.7,

    "ytick.major.width":
        0.7,

    "xtick.major.size":
        3,

    "ytick.major.size":
        3,

    "svg.fonttype":
        "none",

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,
})


# ============================================================
# 13. PLOT CKA
# ============================================================

def plot_cka_results(
    results_df,
    output_dir,
):

    plt.close("all")

    # ========================================================
    # FIGURE: NARROWER + TALLER
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(2.75, 3.60)
    )


    # ========================================================
    # X POSITIONS
    # ========================================================

    GROUP_SPACING = 0.82

    x = np.array([
        0.00,
        GROUP_SPACING,
        2 * GROUP_SPACING,
    ])

    BAR_WIDTH = 0.20
    PAIR_OFFSET = 0.13


    # ========================================================
    # COLLECT VALUES
    # ========================================================

    baseline_values = []
    crosswave_values = []

    baseline_lower = []
    baseline_upper = []

    crosswave_lower = []
    crosswave_upper = []


    for wavelength in WAVELENGTHS:

        b = results_df[
            (
                results_df["wavelength"] == wavelength
            )
            &
            (
                results_df["model_family"] == "Baseline"
            )
        ].iloc[0]


        cw = results_df[
            (
                results_df["wavelength"] == wavelength
            )
            &
            (
                results_df["model_family"] == "CrossWave"
            )
        ].iloc[0]


        baseline_values.append(
            b["cka"]
        )

        crosswave_values.append(
            cw["cka"]
        )


        baseline_lower.append(
            b["cka"] - b["ci_lower"]
        )

        baseline_upper.append(
            b["ci_upper"] - b["cka"]
        )


        crosswave_lower.append(
            cw["cka"] - cw["ci_lower"]
        )

        crosswave_upper.append(
            cw["ci_upper"] - cw["cka"]
        )


    baseline_values = np.array(
        baseline_values
    )

    crosswave_values = np.array(
        crosswave_values
    )


    # ========================================================
    # BASELINE BARS
    # ========================================================

    baseline_bars = ax.bar(

        x - PAIR_OFFSET,

        baseline_values,

        width=BAR_WIDTH,

        color=BASELINE_COLOR,

        edgecolor="none",

        label="Baseline",

        zorder=3,
    )


    # ========================================================
    # CROSSWAVE BARS
    # ========================================================

    crosswave_bars = []


    for i, wavelength in enumerate(
        WAVELENGTHS
    ):

        bar = ax.bar(

            x[i] + PAIR_OFFSET,

            crosswave_values[i],

            width=BAR_WIDTH,

            color=WAVELENGTH_COLORS[
                wavelength
            ],

            edgecolor="none",

            zorder=3,
        )

        crosswave_bars.append(
            bar
        )


    # ========================================================
    # ERROR BARS
    # ========================================================

    if np.all(
        np.isfinite(
            baseline_lower
        )
    ):

        ax.errorbar(

            x - PAIR_OFFSET,

            baseline_values,

            yerr=np.vstack(
                [
                    baseline_lower,
                    baseline_upper,
                ]
            ),

            fmt="none",

            ecolor="#444444",

            elinewidth=0.8,

            capsize=2,

            capthick=0.8,

            zorder=10,
        )


    if np.all(
        np.isfinite(
            crosswave_lower
        )
    ):

        ax.errorbar(

            x + PAIR_OFFSET,

            crosswave_values,

            yerr=np.vstack(
                [
                    crosswave_lower,
                    crosswave_upper,
                ]
            ),

            fmt="none",

            ecolor="#444444",

            elinewidth=0.8,

            capsize=2,

            capthick=0.8,

            zorder=10,
        )


    # ========================================================
    # NUMERIC LABELS
    # ========================================================

    LABEL_OFFSET = 0.010


    for bar, value in zip(
        baseline_bars,
        baseline_values,
    ):

        ax.text(

            bar.get_x()
            + bar.get_width() / 2,

            value + LABEL_OFFSET,

            f"{value:.3f}",

            ha="center",
            va="bottom",

            fontsize=6.5,

            color="#555555",
        )


    for bar_container, value in zip(
        crosswave_bars,
        crosswave_values,
    ):

        bar = bar_container[0]

        ax.text(

            bar.get_x()
            + bar.get_width() / 2,

            value + LABEL_OFFSET,

            f"{value:.3f}",

            ha="center",
            va="bottom",

            fontsize=6.5,

            color="#222222",
        )


    # ========================================================
    # X AXIS
    # ========================================================

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        WAVELENGTHS
    )


    for tick, wavelength in zip(
        ax.get_xticklabels(),
        WAVELENGTHS,
    ):

        tick.set_color(
            WAVELENGTH_COLORS[
                wavelength
            ]
        )

        tick.set_fontweight(
            "bold"
        )


    # Tight horizontal crop
    ax.set_xlim(
        x[0] - 0.34,
        x[-1] + 0.34,
    )


    # ========================================================
    # Y AXIS
    # ========================================================

    ax.set_ylabel(
        "Similarity to Fusion (linear CKA)"
    )


    # Start at 0.50
    ax.set_ylim(
        0.50,
        1.02,
    )


    # Longer / clearer vertical scale
    ax.set_yticks(
        [
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            1.00,
        ]
    )


    # ========================================================
    # TITLE
    # ========================================================

    ax.set_title(

        "(e) Representation similarity to Fusion",

        fontweight="bold",

        pad=36,
    )


    # ========================================================
    # STYLE
    # ========================================================

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        direction="out"
    )

    ax.grid(
        False
    )


    # ========================================================
    # LEGEND
    # ========================================================

    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerTuple


    baseline_patch = Patch(
        facecolor=BASELINE_COLOR,
        edgecolor="none",
    )


    cw_patches = tuple(

        Patch(
            facecolor=WAVELENGTH_COLORS[w],
            edgecolor="none",
        )

        for w in WAVELENGTHS
    )


    ax.legend(

        [
            baseline_patch,
            cw_patches,
        ],

        [
            "Baseline",
            "CrossWave",
        ],

        handler_map={
            tuple: HandlerTuple(
                ndivide=None,
                pad=0.15,
            )
        },

        frameon=False,

        loc="lower center",

        bbox_to_anchor=(
            0.5,
            1.02,
        ),

        ncol=2,

        handlelength=1.25,

        columnspacing=1.0,

        handletextpad=0.45,

        borderaxespad=0.0,
    )


    # ========================================================
    # LAYOUT
    # ========================================================

    fig.subplots_adjust(

        left=0.23,
        right=0.98,

        bottom=0.15,
        top=0.77,
    )


    # ========================================================
    # SAVE
    # ========================================================

    svg_path = (
        output_dir
        / "Fig4e_representation_similarity_CKA.svg"
    )

    png_path = (
        output_dir
        / "Fig4e_representation_similarity_CKA.png"
    )


    fig.savefig(

        svg_path,

        format="svg",

        bbox_inches="tight",

        transparent=False,
    )


    fig.savefig(

        png_path,

        dpi=600,

        bbox_inches="tight",

        transparent=False,
    )


    plt.show()


    print(
        "\nSaved:"
    )

    print(
        svg_path
    )

    print(
        png_path
    )


# ============================================================
# 14. MAIN
# ============================================================

def main():

    args = parse_args()


    set_seed(
        args.seed
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    print(
        "=" * 80
    )

    print(
        "Quantitative Representation Alignment"
    )

    print(
        "=" * 80
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Data directory: {args.data_dir}"
    )

    print(
        f"Bootstrap repetitions: {args.bootstrap_reps}"
    )


    # ========================================================
    # Extract all embeddings
    # ========================================================

    embeddings_by_model = {}

    metadata_by_model = {}


    for model_spec in MODEL_SPECS:

        embeddings, metadata = (
            extract_embeddings_for_model(

                model_spec=
                    model_spec,

                data_dir=
                    args.data_dir,

                batch_size=
                    args.batch_size,

                device=
                    device,
            )
        )


        embeddings_by_model[
            model_spec[
                "name"
            ]
        ] = embeddings


        metadata_by_model[
            model_spec[
                "name"
            ]
        ] = metadata


    # ========================================================
    # Fusion reference
    # ========================================================

    fusion_embeddings = (
        embeddings_by_model[
            "Fusion"
        ]
    )

    fusion_metadata = (
        metadata_by_model[
            "Fusion"
        ]
    )


    # ========================================================
    # Comparisons
    # ========================================================

    comparisons = [

        (
            "440 nm",
            "Baseline",
            "Baseline 440 nm",
        ),

        (
            "440 nm",
            "CrossWave",
            "CrossWave 440 nm",
        ),

        (
            "555 nm",
            "Baseline",
            "Baseline 555 nm",
        ),

        (
            "555 nm",
            "CrossWave",
            "CrossWave 555 nm",
        ),

        (
            "740 nm",
            "Baseline",
            "Baseline 740 nm",
        ),

        (
            "740 nm",
            "CrossWave",
            "CrossWave 740 nm",
        ),
    ]


    result_rows = []


    for comparison_idx, (
        wavelength,
        model_family,
        model_name,
    ) in enumerate(
        comparisons
    ):

        print()
        print(
            "=" * 80
        )

        print(
            f"{model_name} vs Fusion"
        )

        print(
            "=" * 80
        )


        model_embeddings = (
            embeddings_by_model[
                model_name
            ]
        )

        model_metadata = (
            metadata_by_model[
                model_name
            ]
        )


        X, Y, matched = (
            match_embeddings(

                embedding_a=
                    model_embeddings,

                metadata_a=
                    model_metadata,

                embedding_b=
                    fusion_embeddings,

                metadata_b=
                    fusion_metadata,
            )
        )


        # ----------------------------------------------------
        # Optional fixed subsampling
        # ----------------------------------------------------

        X, Y = subsample_matched(

            X,
            Y,

            max_samples=
                args.max_samples,

            seed=
                args.seed,
        )


        # ----------------------------------------------------
        # Full CKA
        # ----------------------------------------------------

        cka_value = linear_cka(
            X,
            Y,
        )


        # ----------------------------------------------------
        # Bootstrap CI
        # ----------------------------------------------------

        ci_lower, ci_upper, boot_values = (
            bootstrap_cka(

                X,
                Y,

                n_reps=
                    args.bootstrap_reps,

                fraction=
                    args.bootstrap_fraction,

                seed=
                    args.seed
                    + comparison_idx,
            )
        )


        print(
            f"Matched samples: "
            f"{X.shape[0]}"
        )

        print(
            f"{model_name} embedding dim: "
            f"{X.shape[1]}"
        )

        print(
            f"Fusion embedding dim: "
            f"{Y.shape[1]}"
        )

        print(
            f"Linear CKA: "
            f"{cka_value:.4f}"
        )


        if args.bootstrap_reps > 0:

            print(
                f"95% bootstrap CI: "
                f"[{ci_lower:.4f}, "
                f"{ci_upper:.4f}]"
            )


        # ----------------------------------------------------
        # Save bootstrap distribution
        # ----------------------------------------------------

        if len(
            boot_values
        ) > 0:

            bootstrap_df = pd.DataFrame(
                {
                    "cka":
                        boot_values
                }
            )

            safe_name = (
                model_name
                .replace(
                    " ",
                    "_"
                )
            )

            bootstrap_df.to_csv(

                output_dir
                / (
                    f"{safe_name}"
                    f"_vs_Fusion"
                    f"_bootstrap_CKA.csv"
                ),

                index=False,
            )


        result_rows.append(
            {
                "wavelength":
                    wavelength,

                "model_family":
                    model_family,

                "model_name":
                    model_name,

                "reference_model":
                    "Fusion",

                "n_samples":
                    X.shape[0],

                "model_embedding_dim":
                    X.shape[1],

                "fusion_embedding_dim":
                    Y.shape[1],

                "cka":
                    cka_value,

                "ci_lower":
                    ci_lower,

                "ci_upper":
                    ci_upper,
            }
        )


    # ========================================================
    # Results table
    # ========================================================

    results_df = pd.DataFrame(
        result_rows
    )


    results_path = (
        output_dir
        / "representation_alignment_CKA.csv"
    )


    results_df.to_csv(
        results_path,
        index=False,
    )


    print()
    print(
        "=" * 80
    )

    print(
        "CKA RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        results_df[
            [
                "wavelength",
                "model_family",
                "cka",
                "ci_lower",
                "ci_upper",
            ]
        ].to_string(
            index=False
        )
    )


    # ========================================================
    # CrossWave improvement over baseline
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "CHANGE IN REPRESENTATION SIMILARITY"
    )

    print(
        "=" * 80
    )


    improvement_rows = []


    for wavelength in WAVELENGTHS:

        baseline_cka = float(
            results_df.loc[
                (
                    results_df[
                        "wavelength"
                    ]
                    == wavelength
                )
                &
                (
                    results_df[
                        "model_family"
                    ]
                    == "Baseline"
                ),
                "cka"
            ].iloc[
                0
            ]
        )


        crosswave_cka = float(
            results_df.loc[
                (
                    results_df[
                        "wavelength"
                    ]
                    == wavelength
                )
                &
                (
                    results_df[
                        "model_family"
                    ]
                    == "CrossWave"
                ),
                "cka"
            ].iloc[
                0
            ]
        )


        delta = (
            crosswave_cka
            - baseline_cka
        )


        relative_change = (
            delta
            / baseline_cka
            * 100
        )


        improvement_rows.append(
            {
                "wavelength":
                    wavelength,

                "baseline_cka":
                    baseline_cka,

                "crosswave_cka":
                    crosswave_cka,

                "delta_cka":
                    delta,

                "relative_change_percent":
                    relative_change,
            }
        )


        print(
            f"{wavelength:7s} | "
            f"Baseline={baseline_cka:.4f} | "
            f"CrossWave={crosswave_cka:.4f} | "
            f"ΔCKA={delta:+.4f} | "
            f"relative={relative_change:+.1f}%"
        )


    improvement_df = pd.DataFrame(
        improvement_rows
    )


    improvement_df.to_csv(

        output_dir
        / "representation_alignment_CKA_improvement.csv",

        index=False,
    )


    # ========================================================
    # Plot
    # ========================================================

    plot_cka_results(

        results_df=
            results_df,

        output_dir=
            output_dir,
    )


    print()
    print(
        "=" * 80
    )

    print(
        "DONE"
    )

    print(
        "=" * 80
    )

    print(
        f"Results:\n{results_path}"
    )


if __name__ == "__main__":

    main()