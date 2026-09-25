"""
Visualize latent space of baseline single-wavelength, fusion, and KD student models.

The test-set particles are split into 8 cube-root Cy3 intensity bins:

    [0, 5)
    [5, 10)
    [10, 15)
    [15, 20)
    [20, 25)
    [25, 30)
    [30, 35)
    [35, 45]

The script extracts latent embeddings from each model using:

    model.extract_embedding(x)

Then it projects the embeddings to 2D using PCA or t-SNE.

Important:
    This script is for the original 64 x 64 multi-wavelength dataset:
        data/processed
        image shape: N x 3 x 64 x 64

Example:

    cd C:\\Users\\jiaru\\Documents\\code_temp_files\\ParticleSizing

    & "C:\\Users\\jiaru\\.conda\\envs\\LNP_image_analysis\\python.exe" experiments/visualize_latent_space_by_intensity_bins.py `
        --output_dir "outputs\\latent_space_visualization" `
        --reducer pca `
        --max_per_bin 1000
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
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from datasets.particle_dataset import ParticleDataset
from models.baseline_cnn import ConfigurableCNNRegressor
from utils.seed import set_seed


# ============================================================
# User configuration
# ============================================================
# Edit these paths to match your actual model folders.
#
# model_type:
#   "baseline" for ordinary single-wavelength or fusion models
#   "kd_student" for KD student models
#
# mode:
#   "440", "555", "740", or "fusion"
#
# For baseline/fusion models:
#   checkpoint should point to best_model.pt
#
# For KD student models:
#   checkpoint should point to best_student_model.pt
# ============================================================

MODEL_SPECS = [
    {
        "name": "Baseline 440 nm",
        "model_type": "baseline",
        "mode": "440",
        "checkpoint": r"outputs\optuna\baseline_440_unweighted_soft_sampler_alpha_0.3002\trial_046\best_model.pt"
    },
    {
        "name": "Baseline 555 nm",
        "model_type": "baseline",
        "mode": "555",
        "checkpoint": r"outputs\optuna\baseline_555_unweighted_soft_sampler_alpha_0.3\trial_013\best_model.pt"
    },
    {
        "name": "Baseline 740 nm",
        "model_type": "baseline",
        "mode": "740",
        "checkpoint": r"outputs\optuna\baseline_740_unweighted_soft_sampler_alpha_0.3\trial_049\best_model.pt"
    },
    {
        "name": "Fusion",
        "model_type": "baseline",
        "mode": "fusion",
        "checkpoint": r"outputs\optuna\baseline_fusion_unweighted_soft_sampler_alpha_0.3\trial_046\best_model.pt"
    },
    {
        "name": "CrossWave 440 nm",
        "model_type": "kd_student",
        "mode": "440",
        "checkpoint": r"outputs\optuna\kd_student_from_fusion_teacher_v1_projection_student_440_at_1_ap_0.5_af_0.1_soft_sampler_alpha_0.3\trial_016\best_student_model.pt"
    },
    {
        "name": "CrossWave 555 nm",
        "model_type": "kd_student",
        "mode": "555",
        "checkpoint": r"outputs\optuna\kd_student_from_fusion_teacher_v1_projection_student_555_at_1_ap_0.3_af_0.3_soft_sampler_alpha_0.3\trial_033\best_student_model.pt"
    },
    {
        "name": "CrossWave 740 nm",
        "model_type": "kd_student",
        "mode": "740",
        "checkpoint": r"outputs\optuna\kd_student_from_fusion_teacher_v1_projection_student_740_at_1_ap_0.5_af_0.1_soft_sampler_alpha_0.3\trial_065\best_student_model.pt"
    }
]


BIN_EDGES = np.array(
    [0, 5, 10, 15, 20, 25, 30, 35, 45],
    dtype=float
)

BIN_LABELS = [
    "0-5",
    "5-10",
    "10-15",
    "15-20",
    "20-25",
    "25-30",
    "30-35",
    "35-45"
]


def parse_args():

    parser = argparse.ArgumentParser(
        description="Visualize model latent spaces by cube-root Cy3 intensity bins."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Processed dataset directory for the original 64 x 64 dataset."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/latent_space_visualization",
        help="Directory to save latent-space figures and CSV files."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for embedding extraction."
    )

    parser.add_argument(
        "--reducer",
        type=str,
        default="pca",
        choices=["pca", "tsne"],
        help="2D dimensionality reduction method."
    )

    parser.add_argument(
        "--max_per_bin",
        type=int,
        default=1000,
        help=(
            "Maximum number of test samples per intensity bin per model. "
            "Use a smaller value for t-SNE."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    parser.add_argument(
        "--point_size",
        type=float,
        default=7.0,
        help="Scatter point size."
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Scatter point transparency."
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


def infer_architecture_from_state_dict(state_dict):
    """
    Infer ConfigurableCNNRegressor architecture from checkpoint weights.

    This avoids dropout-index mismatch.
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

    architecture = {
        "conv_channels": conv_channels,
        "kernel_sizes": kernel_sizes,
        "dense_units": dense_units,
        "dropout_rate": dropout_rate,
        "use_batch_norm": True
    }

    return architecture


def get_state_dict_from_checkpoint(checkpoint, model_type):

    if model_type == "baseline":

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "Baseline checkpoint does not contain model_state_dict."
            )

        return checkpoint["model_state_dict"]

    if model_type == "kd_student":

        if "student_state_dict" not in checkpoint:
            raise KeyError(
                "KD checkpoint does not contain student_state_dict."
            )

        return checkpoint["student_state_dict"]

    raise ValueError(
        f"Unknown model_type: {model_type}"
    )


def build_model_from_checkpoint(
    checkpoint_path,
    model_type,
    in_channels,
    device
):
    """
    Build ConfigurableCNNRegressor and load weights.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    state_dict = get_state_dict_from_checkpoint(
        checkpoint=checkpoint,
        model_type=model_type
    )

    architecture = infer_architecture_from_state_dict(
        state_dict
    )

    config = checkpoint.get(
        "config",
        None
    )

    if config is None:
        config = checkpoint.get(
            "params",
            None
        )

    if config is not None and "dropout_rate" in config:

        saved_dropout_rate = float(
            config["dropout_rate"]
        )

        if architecture["dropout_rate"] > 0:
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

    model.load_state_dict(
        state_dict
    )

    model.to(
        device
    )

    model.eval()

    return model, architecture


def assign_intensity_bins(targets):

    """
    Assign targets to the 8 requested cube-root intensity bins.
    """

    targets = np.asarray(
        targets
    )

    bin_ids = np.digitize(
        targets,
        BIN_EDGES[1:-1],
        right=False
    )

    valid_mask = (
        (targets >= BIN_EDGES[0])
        &
        (targets <= BIN_EDGES[-1])
    )

    bin_names = np.array(
        [
            BIN_LABELS[i]
            if valid_mask[j] and i < len(BIN_LABELS)
            else "outside"
            for j, i in enumerate(bin_ids)
        ]
    )

    return bin_ids, bin_names, valid_mask


def sample_balanced_by_bin(
    df,
    max_per_bin,
    seed
):
    """
    Downsample the test set so each bin contributes at most max_per_bin samples.
    """

    sampled_groups = []

    for bin_name, group in df.groupby("bin_name"):

        if bin_name == "outside":
            continue

        n = min(
            len(group),
            max_per_bin
        )

        sampled = group.sample(
            n=n,
            random_state=seed
        )

        sampled_groups.append(
            sampled
        )

    if len(sampled_groups) == 0:
        raise ValueError(
            "No samples remained after bin filtering."
        )

    return pd.concat(
        sampled_groups,
        axis=0
    ).reset_index(
        drop=True
    )


def extract_embeddings_for_model(
    model_spec,
    data_dir,
    batch_size,
    device
):
    """
    Extract latent embeddings and metadata for one model.
    """

    model_name = model_spec["name"]
    model_type = model_spec["model_type"]
    mode = model_spec["mode"]
    checkpoint_path = Path(
        model_spec["checkpoint"]
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for {model_name}: {checkpoint_path}"
        )

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
            mmap=True
        )

        in_channels = len(
            channels
        )

        student_channel_index = None

    elif model_type == "kd_student":

        channels = [0, 1, 2]

        dataset = ParticleDataset(
            data_dir=data_dir,
            split="test",
            channels=channels,
            target_transform="cube_root",
            use_weights=False,
            mmap=True
        )

        in_channels = 1

        student_channel_index = channel_to_index(
            mode
        )

    else:

        raise ValueError(
            f"Unknown model_type: {model_type}"
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    model, architecture = build_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        in_channels=in_channels,
        device=device
    )

    all_embeddings = []
    all_targets = []
    all_particle_ids = []
    all_image_indices = []

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(
                device
            )

            if model_type == "kd_student":

                images = images[
                    :,
                    student_channel_index:student_channel_index + 1,
                    :,
                    :
                ]

            embeddings = model.extract_embedding(
                images
            )

            targets = batch["target"].cpu().numpy()

            all_embeddings.append(
                embeddings.cpu().numpy()
            )

            all_targets.append(
                targets
            )

            all_particle_ids.extend(
                batch["particle_id"].cpu().numpy().tolist()
            )

            all_image_indices.extend(
                batch["image_index"].cpu().numpy().tolist()
            )

    embeddings = np.concatenate(
        all_embeddings,
        axis=0
    )

    targets = np.concatenate(
        all_targets,
        axis=0
    )

    bin_ids, bin_names, valid_mask = assign_intensity_bins(
        targets
    )

    metadata = pd.DataFrame(
        {
            "model_name": model_name,
            "model_type": model_type,
            "mode": mode,
            "particle_id": all_particle_ids,
            "image_index": all_image_indices,
            "target": targets,
            "bin_id": bin_ids,
            "bin_name": bin_names,
            "within_requested_bins": valid_mask
        }
    )

    print()
    print("=" * 80)
    print(f"Extracted embeddings: {model_name}")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Mode: {mode}")
    print(f"Model type: {model_type}")
    print(f"Embedding shape: {embeddings.shape}")
    print("Architecture:")
    print(json.dumps(architecture, indent=4))
    print("Bin counts:")
    print(metadata["bin_name"].value_counts().sort_index())

    return embeddings, metadata


def reduce_embeddings(
    embeddings,
    reducer,
    seed
):
    """
    Reduce high-dimensional embeddings to 2D.
    """

    if reducer == "pca":

        reducer_model = PCA(
            n_components=2,
            random_state=seed
        )

        coords = reducer_model.fit_transform(
            embeddings
        )

        explained = reducer_model.explained_variance_ratio_

        return coords, {
            "method": "pca",
            "pc1_explained_variance": float(explained[0]),
            "pc2_explained_variance": float(explained[1])
        }

    if reducer == "tsne":

        reducer_model = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            random_state=seed
        )

        coords = reducer_model.fit_transform(
            embeddings
        )

        return coords, {
            "method": "tsne"
        }

    raise ValueError(
        f"Unknown reducer: {reducer}"
    )


def plot_latent_space(
    df,
    output_path,
    title,
    point_size,
    alpha
):
    """
    Plot 2D latent space colored by intensity bin.
    """

    plt.figure(
        figsize=(7.5, 6.5)
    )

    cmap = plt.get_cmap(
        "tab10"
    )

    for bin_idx, bin_label in enumerate(BIN_LABELS):

        group = df[
            df["bin_name"] == bin_label
        ]

        if len(group) == 0:
            continue

        plt.scatter(
            group["z1"],
            group["z2"],
            s=point_size,
            alpha=alpha,
            color=cmap(bin_idx),
            label=f"{bin_label} ({len(group)})"
        )

    plt.xlabel(
        "Latent dimension 1"
    )

    plt.ylabel(
        "Latent dimension 2"
    )

    plt.title(
        title
    )

    plt.legend(
        title="Cube-root Cy3 bin",
        fontsize=8,
        title_fontsize=9,
        markerscale=2,
        loc="best"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


def plot_all_models_grid(
    all_plot_dfs,
    output_path,
    point_size,
    alpha
):
    """
    Plot latent spaces in a fixed 2 x 4 layout.

    Layout:
        Row 1:
            legend | Baseline 440 nm | Baseline 555 nm | Baseline 740 nm

        Row 2:
            Fusion | CrossWave 440 nm | CrossWave 555 nm | CrossWave 740 nm

    Note:
        Each model is reduced independently, so axes are not directly comparable
        across panels. Compare within-panel bin organization.
    """

    layout = {
        "Baseline 440 nm": (0, 1),
        "Baseline 555 nm": (0, 2),
        "Baseline 740 nm": (0, 3),
        "Fusion": (1, 0),
        "CrossWave 440 nm": (1, 1),
        "CrossWave 555 nm": (1, 2),
        "CrossWave 740 nm": (1, 3)
    }

    plot_df_by_name = {
        df["model_name"].iloc[0]: df
        for df in all_plot_dfs
    }

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18, 8.5),
        constrained_layout=True
    )

    cmap = plt.get_cmap(
        "tab10"
    )

    # ========================================================
    # Top-left legend panel
    # ========================================================

    legend_ax = axes[0, 0]
    legend_ax.axis("off")

    legend_handles = []

    for bin_idx, bin_label in enumerate(BIN_LABELS):

        handle = plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=cmap(bin_idx),
            markeredgecolor="none",
            label=bin_label
        )

        legend_handles.append(
            handle
        )

    legend_ax.legend(
        handles=legend_handles,
        title="Cube-root Cy3 bin",
        loc="center",
        frameon=True,
        fontsize=10,
        title_fontsize=11
    )

    legend_ax.set_title(
        "Intensity bins",
        fontsize=12
    )

    # ========================================================
    # Model panels
    # ========================================================

    for model_name, position in layout.items():

        row_idx, col_idx = position

        ax = axes[
            row_idx,
            col_idx
        ]

        if model_name not in plot_df_by_name:

            ax.axis("off")
            ax.set_title(
                f"{model_name}\nmissing",
                fontsize=11
            )
            continue

        plot_df = plot_df_by_name[
            model_name
        ]

        for bin_idx, bin_label in enumerate(BIN_LABELS):

            group = plot_df[
                plot_df["bin_name"] == bin_label
            ]

            if len(group) == 0:
                continue

            ax.scatter(
                group["z1"],
                group["z2"],
                s=point_size,
                alpha=alpha,
                color=cmap(bin_idx),
                label=bin_label
            )

        ax.set_title(
            model_name,
            fontsize=12
        )

        ax.set_xlabel(
            "Latent dim 1",
            fontsize=10
        )

        ax.set_ylabel(
            "Latent dim 2",
            fontsize=10
        )

        ax.tick_params(
            axis="both",
            labelsize=9
        )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():

    args = parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 80)
    print("Latent Space Visualization by Intensity Bins")
    print("=" * 80)
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Reducer: {args.reducer}")
    print(f"Max per bin: {args.max_per_bin}")
    print(f"Device: {device}")

    all_plot_dfs = []
    summary_rows = []

    for model_spec in MODEL_SPECS:

        model_name_safe = (
            model_spec["name"]
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        embeddings, metadata = extract_embeddings_for_model(
            model_spec=model_spec,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            device=device
        )

        # Save full embeddings metadata without z1/z2.
        metadata.to_csv(
            output_dir / f"{model_name_safe}_test_embedding_metadata.csv",
            index=False
        )

        # Balanced sampling by target-intensity bin.
        sampled_metadata = sample_balanced_by_bin(
            df=metadata,
            max_per_bin=args.max_per_bin,
            seed=args.seed
        )

        sampled_indices = sampled_metadata.index.values

        # Important:
        # sampled_metadata index was reset, so recover rows through particle/image matching.
        key_df = metadata.reset_index().rename(
            columns={"index": "original_row_index"}
        )

        sampled_with_index = sampled_metadata.merge(
            key_df[
                [
                    "original_row_index",
                    "particle_id",
                    "image_index"
                ]
            ],
            on=[
                "particle_id",
                "image_index"
            ],
            how="left"
        )

        original_indices = sampled_with_index[
            "original_row_index"
        ].values

        sampled_embeddings = embeddings[
            original_indices
        ]

        coords, reducer_info = reduce_embeddings(
            embeddings=sampled_embeddings,
            reducer=args.reducer,
            seed=args.seed
        )

        plot_df = sampled_with_index.copy()

        plot_df["z1"] = coords[
            :,
            0
        ]

        plot_df["z2"] = coords[
            :,
            1
        ]

        plot_df.to_csv(
            output_dir / f"{model_name_safe}_latent_2d_{args.reducer}.csv",
            index=False
        )

        with open(output_dir / f"{model_name_safe}_reducer_info.json", "w") as f:
            json.dump(
                reducer_info,
                f,
                indent=4
            )

        plot_title = (
            f"{model_spec['name']} latent space by intensity bin"
        )

        plot_latent_space(
            df=plot_df,
            output_path=output_dir / f"{model_name_safe}_latent_space_{args.reducer}.png",
            title=plot_title,
            point_size=args.point_size,
            alpha=args.alpha
        )

        all_plot_dfs.append(
            plot_df
        )

        for bin_label, group in plot_df.groupby("bin_name"):

            summary_rows.append(
                {
                    "model_name": model_spec["name"],
                    "model_type": model_spec["model_type"],
                    "mode": model_spec["mode"],
                    "bin_name": bin_label,
                    "num_samples_plotted": len(group),
                    "target_mean": group["target"].mean(),
                    "target_min": group["target"].min(),
                    "target_max": group["target"].max(),
                    **reducer_info
                }
            )

    plot_all_models_grid(
        all_plot_dfs=all_plot_dfs,
        output_path=output_dir / f"all_models_latent_space_grid_{args.reducer}.png",
        point_size=args.point_size,
        alpha=args.alpha
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        output_dir / "latent_space_bin_summary.csv",
        index=False
    )

    print()
    print("=" * 80)
    print("Saved latent-space outputs")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print()
    print("Main figure:")
    print(output_dir / f"all_models_latent_space_grid_{args.reducer}.png")
    print()
    print("Individual model figures:")
    for model_spec in MODEL_SPECS:
        model_name_safe = (
            model_spec["name"]
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
        print(output_dir / f"{model_name_safe}_latent_space_{args.reducer}.png")


if __name__ == "__main__":

    main()