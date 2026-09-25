"""
Validate the raw dataset before preprocessing.

This script:
1. Loads the raw dataset.
2. Checks dataset integrity.
3. Builds the list of valid FoVs.
4. Prints a dataset summary.

No files are modified.
"""

from pathlib import Path

import numpy as np

from configs.config import (
    RAW_DATA_FILE,
    CHANNEL_NAMES,
    EXCLUDED_FOVS,
)

from utils.io import load_pickle
from utils.seed import set_seed


# ============================================================
# Validation
# ============================================================

def validate_dataset(all_ch_list):
    """
    Validate the raw dataset.

    Parameters
    ----------
    all_ch_list : list
    """

    print("=" * 60)
    print("Validating raw dataset")
    print("=" * 60)

    # --------------------------------------------------------
    # Number of channels
    # --------------------------------------------------------

    assert len(all_ch_list) == len(CHANNEL_NAMES), (
        f"Expected {len(CHANNEL_NAMES)} channels, "
        f"found {len(all_ch_list)}."
    )

    # --------------------------------------------------------
    # Number of FoVs
    # --------------------------------------------------------

    total_fovs = len(all_ch_list[0])

    for channel in all_ch_list:

        assert len(channel) == total_fovs

    # --------------------------------------------------------
    # Image size
    # --------------------------------------------------------

    image_shape = np.squeeze(
        all_ch_list[0][0]
    ).shape

    for channel in all_ch_list:

        for image in channel:

            image = np.squeeze(image)

            assert image.shape == image_shape

    # --------------------------------------------------------
    # Valid FoVs
    # --------------------------------------------------------

    valid_fov_indices = [

        idx

        for idx in range(total_fovs)

        if idx not in EXCLUDED_FOVS

    ]

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(f"Channels              : {len(CHANNEL_NAMES)}")
    print(f"Channel names         : {CHANNEL_NAMES}")

    print()

    print(f"Total FoVs            : {total_fovs}")
    print(f"Excluded FoVs         : {EXCLUDED_FOVS}")
    print(f"Valid FoVs            : {len(valid_fov_indices)}")

    print()

    print(f"Image size            : {image_shape}")

    print()

    print("-" * 60)
    print("Channel statistics")
    print("-" * 60)

    for name, channel in zip(
        CHANNEL_NAMES,
        all_ch_list
    ):

        stacked = np.stack([

            np.squeeze(channel[idx])

            for idx in valid_fov_indices

        ])

        print(f"\n{name}")

        print(f"  dtype : {stacked.dtype}")

        print(f"  shape : {stacked.shape}")

        print(f"  min   : {stacked.min():.4f}")

        print(f"  max   : {stacked.max():.4f}")

        print(f"  mean  : {stacked.mean():.4f}")

        print(f"  std   : {stacked.std():.4f}")

    print()

    print("=" * 60)
    print("Validation passed.")
    print("=" * 60)

    return valid_fov_indices


# ============================================================
# Main
# ============================================================

def main():

    set_seed()

    data_path = Path("data/raw") / RAW_DATA_FILE

    print("\nLoading dataset...\n")

    all_ch_list = load_pickle(data_path)

    valid_fov_indices = validate_dataset(
        all_ch_list
    )

    return all_ch_list, valid_fov_indices


if __name__ == "__main__":

    all_ch_list, valid_fov_indices = main()