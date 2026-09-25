"""
PyTorch Dataset for particle-level iSCAT particle sizing.

This dataset loads the processed particle dataset and supports
single-wavelength, multi-wavelength fusion, and future CWKT experiments.

Author: Jiarui Xie
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ParticleDataset(Dataset):
    """
    Particle-level dataset.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing processed dataset files.

    split : str
        One of {"train", "val", "test"}.

    channels : list of int
        Which iSCAT channels to use.

        Channel mapping:
            0 -> 440 nm
            1 -> 555 nm
            2 -> 740 nm

    target_transform : str or None
        Optional target transformation.
        Supported:
            None
            "cube_root"
            "log"

    use_weights : bool
        Whether to return sample weights.

    mmap : bool
        Whether to memory-map the image array.
    """

    def __init__(
        self,
        data_dir="data/processed",
        split="train",
        channels=[0, 1, 2],
        target_transform=None,
        use_weights=True,
        mmap=True,
        clip_weights=False,
        min_weight=0.1,
        max_weight=20.0
    ):

        self.data_dir = Path(data_dir)

        self.split = split

        self.channels = channels

        self.target_transform = target_transform

        self.use_weights = use_weights

        self.mmap = mmap

        self.clip_weights = clip_weights
        self.min_weight = min_weight
        self.max_weight = max_weight

        assert split in ["train", "val", "test"], (
            "split must be one of: train, val, test"
        )

        assert all(
            channel in [0, 1, 2]
            for channel in channels
        ), (
            "channels must be selected from [0, 1, 2]"
        )

        # ----------------------------------------------------
        # Load data
        # ----------------------------------------------------

        self.images = np.load(
            self.data_dir / "iscat_images_normalized.npy",
            mmap_mode="r" if mmap else None
        )

        self.metadata = pd.read_csv(
            self.data_dir / "metadata.csv"
        )

        # ----------------------------------------------------
        # Select split
        # ----------------------------------------------------

        self.metadata_split = self.metadata[
            self.metadata["split"] == split
        ].reset_index(drop=True)

        self.indices = self.metadata_split[
            "image_index"
        ].values.astype(int)

        print(
            f"Loaded {split} dataset: "
            f"{len(self.indices)} samples, "
            f"channels={channels}"
        )

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):
        """
        Get one particle sample.
        """

        # --------------------------------------------------------
        # Get metadata row
        # --------------------------------------------------------

        metadata_row = self.metadata.iloc[idx]

        image_index = int(metadata_row["image_index"])

        # --------------------------------------------------------
        # Get image
        # --------------------------------------------------------

        image = self.images[
            image_index,
            self.channels,
            :, :
        ]

        image = np.asarray(
            image,
            dtype=np.float32
        )

        # If only one channel, keep shape as (1, H, W)
        if image.ndim == 2:
            image = image[np.newaxis, :, :]

        # --------------------------------------------------------
        # Get target
        # --------------------------------------------------------

        target = float(
            metadata_row["cy3_max"]
        )

        if self.target_transform == "cube_root":

            target = np.cbrt(target)

        elif self.target_transform == "log":

            target = np.log1p(target)

        elif self.target_transform is None:

            pass

        else:

            raise ValueError(
                f"Unknown target_transform: {self.target_transform}"
            )

        # --------------------------------------------------------
        # Get sample weight
        # --------------------------------------------------------

        if self.use_weights:

            weight = float(
                metadata_row["weight"]
            )

            if self.clip_weights:

                weight = np.clip(
                    weight,
                    self.min_weight,
                    self.max_weight
                )

        else:

            weight = 1.0

        # --------------------------------------------------------
        # Return sample
        # --------------------------------------------------------

        sample = {
            "image": torch.tensor(
                image,
                dtype=torch.float32
            ),
            "target": torch.tensor(
                target,
                dtype=torch.float32
            ),
            "weight": torch.tensor(
                weight,
                dtype=torch.float32
            ),
            "particle_id": int(
                metadata_row["particle_id"]
            ),
            "image_index": int(
                metadata_row["image_index"]
            )
        }

        return sample