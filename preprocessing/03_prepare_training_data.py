"""
Prepare the extracted particle dataset for model training.

This script:

1. Loads the intermediate particle dataset.
2. Splits particles into train / validation / test sets.
3. Fits one scaler using all three iSCAT wavelengths from the training set only.
4. Applies the scaler to train, validation, and test data.
5. Computes Effective Number sample weights based on Cy3 intensity.
6. Saves the processed training dataset.

Author: Jiarui Xie
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from configs.config import (
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    EFFECTIVE_NUMBER_BETA,
    NUM_INTENSITY_BINS
)

from utils.seed import set_seed
from utils.io import (
    load_numpy,
    load_csv,
    load_json,
    save_numpy,
    save_csv,
    save_pickle,
    save_json,
    create_directory
)


# ============================================================
# Load Intermediate Dataset
# ============================================================

def load_intermediate_dataset():

    input_dir = Path("data/intermediate")

    iscat_images = load_numpy(
        input_dir / "iscat_images.npy"
    )

    cy3_images = load_numpy(
        input_dir / "cy3_images.npy"
    )

    metadata = load_csv(
        input_dir / "metadata.csv"
    )

    dataset_info = load_json(
        input_dir / "dataset_info.json"
    )

    print("=" * 60)
    print("Loaded intermediate dataset")
    print("=" * 60)

    print(f"iSCAT images shape : {iscat_images.shape}")
    print(f"Cy3 images shape   : {cy3_images.shape}")
    print(f"Metadata shape     : {metadata.shape}")

    assert len(iscat_images) == len(metadata), (
        "Number of iSCAT images does not match metadata rows."
    )

    assert len(cy3_images) == len(metadata), (
        "Number of Cy3 images does not match metadata rows."
    )

    return iscat_images, cy3_images, metadata, dataset_info


# ============================================================
# Split Dataset
# ============================================================

def split_dataset(metadata):

    print()
    print("=" * 60)
    print("Splitting dataset")
    print("=" * 60)

    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO

    assert abs(total_ratio - 1.0) < 1e-6, (
        "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1."
    )

    indices = np.arange(len(metadata))

    train_indices, temp_indices = train_test_split(
        indices,
        train_size=TRAIN_RATIO,
        random_state=SEED,
        shuffle=True
    )

    val_size_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_size_adjusted,
        random_state=SEED,
        shuffle=True
    )

    metadata["split"] = ""

    metadata.loc[train_indices, "split"] = "train"
    metadata.loc[val_indices, "split"] = "val"
    metadata.loc[test_indices, "split"] = "test"

    print(f"Train samples : {len(train_indices)}")
    print(f"Val samples   : {len(val_indices)}")
    print(f"Test samples  : {len(test_indices)}")

    return metadata, train_indices, val_indices, test_indices


# ============================================================
# Normalize iSCAT Images
# ============================================================

def fit_scaler_on_training_data(
    iscat_images,
    train_indices
):
    """
    Fit one StandardScaler using all pixels from all three wavelengths
    in the training set only.

    This creates one global scaler for 440, 555, and 740 together.
    """

    print()
    print("=" * 60)
    print("Fitting scaler on training data")
    print("=" * 60)

    train_images = iscat_images[train_indices]

    train_values = train_images.reshape(-1, 1)

    scaler = StandardScaler()

    scaler.fit(train_values)

    print(f"Scaler mean : {scaler.mean_[0]:.6f}")
    print(f"Scaler std  : {np.sqrt(scaler.var_[0]):.6f}")

    return scaler


def normalize_images(
    iscat_images,
    scaler
):
    """
    Normalize all iSCAT images using the fitted training scaler.
    """

    print()
    print("=" * 60)
    print("Normalizing all iSCAT images")
    print("=" * 60)

    original_shape = iscat_images.shape

    normalized = scaler.transform(
        iscat_images.reshape(-1, 1)
    )

    normalized = normalized.reshape(
        original_shape
    )

    normalized = normalized.astype(
        np.float32
    )

    print(f"Normalized image shape : {normalized.shape}")

    return normalized


# ============================================================
# Effective Number Weights
# ============================================================

def compute_effective_number_weights(
    metadata,
    train_indices,
    beta,
    num_bins
):
    """
    Compute sample weights using the Effective Number of Samples.

    The bin frequencies are computed using the training set only.
    Validation and test samples are assigned weights using the same
    training-derived bins and bin weights.

    Formula:
        effective_num = 1 - beta^n
        weight = (1 - beta) / effective_num

    We then normalize the weights so that the mean training weight is 1.
    """

    print()
    print("=" * 60)
    print("Computing Effective Number weights")
    print("=" * 60)

    targets = metadata["cy3_max"].values.astype(float)

    train_targets = targets[train_indices]

    bin_edges = np.linspace(
        train_targets.min(),
        train_targets.max(),
        num_bins + 1
    )

    train_bin_ids = np.digitize(
        train_targets,
        bin_edges[1:-1],
        right=False
    )

    bin_counts = np.bincount(
        train_bin_ids,
        minlength=num_bins
    )

    bin_weights = np.zeros(
        num_bins,
        dtype=np.float64
    )

    for bin_id in range(num_bins):

        count = bin_counts[bin_id]

        if count > 0:

            effective_num = 1.0 - (beta ** count)

            bin_weights[bin_id] = (
                (1.0 - beta) / effective_num
            )

        else:

            bin_weights[bin_id] = 0.0

    all_bin_ids = np.digitize(
        targets,
        bin_edges[1:-1],
        right=False
    )

    sample_weights = bin_weights[
        all_bin_ids
    ]

    train_weights = sample_weights[
        train_indices
    ]

    mean_train_weight = train_weights.mean()

    sample_weights = sample_weights / mean_train_weight

    metadata["intensity_bin"] = all_bin_ids
    metadata["weight"] = sample_weights

    print(f"Number of bins       : {num_bins}")
    print(f"Beta                 : {beta}")
    print(f"Train target min     : {train_targets.min():.4f}")
    print(f"Train target max     : {train_targets.max():.4f}")
    print(f"Weight min           : {sample_weights.min():.4f}")
    print(f"Weight max           : {sample_weights.max():.4f}")
    print(f"Mean train weight    : {metadata.loc[train_indices, 'weight'].mean():.4f}")

    return metadata, bin_edges, bin_counts, bin_weights


# ============================================================
# Save Processed Dataset
# ============================================================

def save_processed_dataset(
    normalized_iscat_images,
    cy3_images,
    metadata,
    scaler,
    dataset_info,
    bin_edges,
    bin_counts,
    bin_weights
):

    output_dir = Path("data/processed")

    create_directory(output_dir)

    print()
    print("=" * 60)
    print("Saving processed dataset")
    print("=" * 60)

    save_numpy(
        normalized_iscat_images,
        output_dir / "iscat_images_normalized.npy"
    )

    save_numpy(
        cy3_images.astype(np.float32),
        output_dir / "cy3_images.npy"
    )

    save_csv(
        metadata,
        output_dir / "metadata.csv"
    )

    save_pickle(
        scaler,
        output_dir / "scaler.pkl"
    )

    processed_info = dict(dataset_info)

    processed_info.update(
        {
            "dataset_stage": "processed",
            "description": (
                "Particle-level dataset after train/val/test split, "
                "iSCAT normalization, and Effective Number weighting."
            ),
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "num_train": int((metadata["split"] == "train").sum()),
            "num_val": int((metadata["split"] == "val").sum()),
            "num_test": int((metadata["split"] == "test").sum()),
            "normalization": {
                "method": "StandardScaler",
                "fit_on": "training iSCAT pixels from all three wavelengths",
                "mean": float(scaler.mean_[0]),
                "std": float(np.sqrt(scaler.var_[0]))
            },
            "weighting": {
                "method": "effective_number",
                "beta": float(EFFECTIVE_NUMBER_BETA),
                "num_bins": int(NUM_INTENSITY_BINS),
                "bin_edges": bin_edges.tolist(),
                "bin_counts": bin_counts.astype(int).tolist(),
                "bin_weights": bin_weights.tolist(),
                "normalized_to_mean_train_weight": 1.0
            }
        }
    )

    save_json(
        processed_info,
        output_dir / "dataset_info.json"
    )

    print(f"Saved normalized iSCAT images : {output_dir / 'iscat_images_normalized.npy'}")
    print(f"Saved Cy3 images              : {output_dir / 'cy3_images.npy'}")
    print(f"Saved metadata                : {output_dir / 'metadata.csv'}")
    print(f"Saved scaler                  : {output_dir / 'scaler.pkl'}")
    print(f"Saved dataset info            : {output_dir / 'dataset_info.json'}")


# ============================================================
# Main
# ============================================================

def main():

    set_seed()

    iscat_images, cy3_images, metadata, dataset_info = load_intermediate_dataset()

    metadata, train_indices, val_indices, test_indices = split_dataset(
        metadata
    )

    scaler = fit_scaler_on_training_data(
        iscat_images,
        train_indices
    )

    normalized_iscat_images = normalize_images(
        iscat_images,
        scaler
    )

    metadata, bin_edges, bin_counts, bin_weights = compute_effective_number_weights(
        metadata=metadata,
        train_indices=train_indices,
        beta=EFFECTIVE_NUMBER_BETA,
        num_bins=NUM_INTENSITY_BINS
    )

    save_processed_dataset(
        normalized_iscat_images=normalized_iscat_images,
        cy3_images=cy3_images,
        metadata=metadata,
        scaler=scaler,
        dataset_info=dataset_info,
        bin_edges=bin_edges,
        bin_counts=bin_counts,
        bin_weights=bin_weights
    )

    print()
    print("=" * 60)
    print("Training dataset preparation completed.")
    print("=" * 60)


if __name__ == "__main__":

    main()