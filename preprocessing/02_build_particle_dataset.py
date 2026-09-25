"""
Build the particle dataset from the raw microscopy images.

This script

1. Loads the raw data.
2. Detects particles.
3. Extracts multi-wavelength patches.
4. Builds the metadata table.
5. Saves the intermediate dataset.

Author: Jiarui Xie
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from configs.config import (
    RAW_DATA_FILE,
    CHANNEL_NAMES,
    PATCH_SIZE,
    GAUSSIAN_SIGMA,
    CY3_THRESHOLD,
    ISCAT_THRESHOLD,
    EXCLUDED_FOVS
)

from utils.seed import set_seed
from utils.io import (
    load_pickle,
    save_numpy,
    save_csv,
    save_json,
    create_directory
)

from utils.image import (
    apply_gaussian_filter,
    calculate_filtered_maxima,
    extract_patch_with_padding
)

# ============================================================
# Load Raw Data
# ============================================================

def load_raw_dataset():

    print("=" * 60)
    print("Loading raw dataset")
    print("=" * 60)

    data_path = Path("data/raw") / RAW_DATA_FILE

    all_ch_list = load_pickle(data_path)

    total_fovs = len(all_ch_list[0])

    valid_fov_indices = [

        idx

        for idx in range(total_fovs)

        if idx not in EXCLUDED_FOVS

    ]

    print(f"Total FoVs : {total_fovs}")
    print(f"Valid FoVs : {len(valid_fov_indices)}")

    return all_ch_list, valid_fov_indices

# ============================================================
# Prepare Output Folder
# ============================================================

def prepare_output_directory():

    output_dir = Path("data/intermediate")

    create_directory(output_dir)

    return output_dir

# ============================================================
# Initialize Dataset
# ============================================================

def initialize_particle_list():

    return []

# ============================================================
# Main
# ============================================================

def main():

    set_seed()

    all_ch_list, valid_fov_indices = load_raw_dataset()

    output_dir = prepare_output_directory()

    particles = initialize_particle_list()

    particle_id = 0

    print()

    print("=" * 60)
    print("Scanning FoVs")
    print("=" * 60)

    for fov_id in tqdm(valid_fov_indices):

        # --------------------------------------------------
        # Load four channels
        # --------------------------------------------------

        iscat_440 = np.squeeze(all_ch_list[0][fov_id]).astype(np.float32)

        iscat_555 = np.squeeze(all_ch_list[1][fov_id]).astype(np.float32)

        iscat_740 = np.squeeze(all_ch_list[2][fov_id]).astype(np.float32)

        cy3 = np.squeeze(all_ch_list[3][fov_id]).astype(np.float32)

        # --------------------------------------------------
        # Gaussian filtering
        # --------------------------------------------------

        cy3_filtered = apply_gaussian_filter(
            cy3,
            sigma=GAUSSIAN_SIGMA
        )

        # --------------------------------------------------
        # Detect particle centers
        # --------------------------------------------------

        particle_mask = calculate_filtered_maxima(
            cy3_patch=cy3_filtered,
            iscat_patch=iscat_555,
            cy3_threshold=CY3_THRESHOLD,
            iscat_threshold=ISCAT_THRESHOLD
        )

        particle_y, particle_x = np.where(
            particle_mask
        )

        num_particles = len(particle_x)

        # Optional logging
        if fov_id % 20 == 0:
            print(
                f"FoV {fov_id}: {num_particles} particles"
            )

        # --------------------------------------------------
        # Extract every particle
        # --------------------------------------------------

        for center_x, center_y in zip(
            particle_x,
            particle_y
        ):

            patch440 = extract_patch_with_padding(
                iscat_440,
                center_x,
                center_y,
                PATCH_SIZE
            )

            patch555 = extract_patch_with_padding(
                iscat_555,
                center_x,
                center_y,
                PATCH_SIZE
            )

            patch740 = extract_patch_with_padding(
                iscat_740,
                center_x,
                center_y,
                PATCH_SIZE
            )

            patch_cy3 = extract_patch_with_padding(
                cy3,
                center_x,
                center_y,
                PATCH_SIZE
            )

            # ----------------------------------------------
            # Metadata
            # ----------------------------------------------

            particle = {

                "particle_id": particle_id,

                "fov_id": fov_id,

                "center_x": center_x,

                "center_y": center_y,

                "patch440": patch440,

                "patch555": patch555,

                "patch740": patch740,

                "patch_cy3": patch_cy3,

                "cy3_max": float(
                    cy3[
                        center_y,
                        center_x
                    ]
                ),

                "wl440_max": float(
                    patch440.max()
                ),

                "wl555_max": float(
                    patch555.max()
                ),

                "wl740_max": float(
                    patch740.max()
                )

            }

            particles.append(
                particle
            )

            particle_id += 1

    print()

    print("=" * 60)
    print("FoV scan completed.")
    print("=" * 60)

    print(f"Total particles: {len(particles)}")

    if len(particles) > 0:

        sample = particles[0]

        print()

        print("Sample particle")

        print("---------------------------")

        print(f"Particle ID : {sample['particle_id']}")

        print(f"FoV         : {sample['fov_id']}")

        print(f"Center      : ({sample['center_x']}, {sample['center_y']})")

        print(f"Cy3 Max     : {sample['cy3_max']:.2f}")

        print(f"440 Patch   : {sample['patch440'].shape}")

        print(f"555 Patch   : {sample['patch555'].shape}")

        print(f"740 Patch   : {sample['patch740'].shape}")

        print(f"Cy3 Patch   : {sample['patch_cy3'].shape}")

    # ========================================================
    # Build Dataset Arrays
    # ========================================================

    if len(particles) == 0:
        raise RuntimeError(
            "No particles were detected. "
            "Please check the Cy3 and iSCAT thresholds."
        )

    print()

    print("=" * 60)
    print("Building dataset arrays")
    print("=" * 60)

    iscat_images = np.stack(
        [
            np.stack(
                [
                    particle["patch440"],
                    particle["patch555"],
                    particle["patch740"]
                ],
                axis=0
            )
            for particle in particles
        ],
        axis=0
    ).astype(np.float32)

    cy3_images = np.stack(
        [
            particle["patch_cy3"]
            for particle in particles
        ],
        axis=0
    ).astype(np.float32)

    print(f"iSCAT images shape : {iscat_images.shape}")
    print(f"Cy3 images shape   : {cy3_images.shape}")

    # ========================================================
    # Build Metadata
    # ========================================================

    print()

    print("=" * 60)
    print("Building metadata")
    print("=" * 60)

    metadata = pd.DataFrame(
        [
            {
                "particle_id": particle["particle_id"],
                "image_index": image_index,
                "fov_id": particle["fov_id"],
                "center_x": int(particle["center_x"]),
                "center_y": int(particle["center_y"]),
                "cy3_max": particle["cy3_max"],
                "cy3_integrated": float(
                    particle["patch_cy3"].sum()
                ),
                "wl440_max": particle["wl440_max"],
                "wl555_max": particle["wl555_max"],
                "wl740_max": particle["wl740_max"]
            }
            for image_index, particle in enumerate(particles)
        ]
    )

    print(metadata.head())

    # ========================================================
    # Dataset Information
    # ========================================================

    dataset_info = {
        "dataset_stage": "intermediate",
        "description": (
            "Particle-level dataset extracted from raw FoVs. "
            "No train/val/test split, normalization, or weighting has been applied."
        ),
        "num_particles": int(len(metadata)),
        "num_fovs": int(len(valid_fov_indices)),
        "patch_size": int(PATCH_SIZE),
        "gaussian_sigma": float(GAUSSIAN_SIGMA),
        "cy3_threshold": float(CY3_THRESHOLD),
        "iscat_threshold": float(ISCAT_THRESHOLD),
        "excluded_fovs": EXCLUDED_FOVS,
        "channel_names": CHANNEL_NAMES,
        "iscat_image_shape": list(iscat_images.shape),
        "cy3_image_shape": list(cy3_images.shape)
    }

    # ========================================================
    # Save Intermediate Dataset
    # ========================================================

    print()

    print("=" * 60)
    print("Saving intermediate dataset")
    print("=" * 60)

    save_numpy(
        iscat_images,
        output_dir / "iscat_images.npy"
    )

    save_numpy(
        cy3_images,
        output_dir / "cy3_images.npy"
    )

    save_csv(
        metadata,
        output_dir / "metadata.csv"
    )

    save_json(
        dataset_info,
        output_dir / "dataset_info.json"
    )

    print(f"Saved iSCAT images to : {output_dir / 'iscat_images.npy'}")
    print(f"Saved Cy3 images to   : {output_dir / 'cy3_images.npy'}")
    print(f"Saved metadata to     : {output_dir / 'metadata.csv'}")
    print(f"Saved dataset info to : {output_dir / 'dataset_info.json'}")

    print()

    print("=" * 60)
    print("Intermediate dataset creation completed.")
    print("=" * 60)

if __name__ == "__main__":

    main()