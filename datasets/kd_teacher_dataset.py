"""
Dataset wrapper for KD training with precomputed teacher outputs.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.particle_dataset import ParticleDataset


class KDTeacherOutputDataset(Dataset):
    """
    Wrap ParticleDataset and attach precomputed teacher prediction/embedding.

    The order must match the original ParticleDataset order with shuffle=False
    during teacher precomputation.
    """

    def __init__(
        self,
        data_dir,
        split,
        teacher_output_dir,
        channels=(0, 1, 2),
        target_transform="cube_root",
        use_weights=False,
        mmap=True
    ):

        self.base_dataset = ParticleDataset(
            data_dir=data_dir,
            split=split,
            channels=list(channels),
            target_transform=target_transform,
            use_weights=use_weights,
            mmap=mmap
        )

        self.split = split
        self.teacher_output_dir = Path(
            teacher_output_dir
        )

        pred_path = (
            self.teacher_output_dir
            / f"{split}_teacher_predictions.npy"
        )

        emb_path = (
            self.teacher_output_dir
            / f"{split}_teacher_embeddings.npy"
        )

        if not pred_path.exists():
            raise FileNotFoundError(
                f"Could not find teacher predictions: {pred_path}"
            )

        if not emb_path.exists():
            raise FileNotFoundError(
                f"Could not find teacher embeddings: {emb_path}"
            )

        self.teacher_predictions = np.load(
            pred_path,
            mmap_mode="r"
        )

        self.teacher_embeddings = np.load(
            emb_path,
            mmap_mode="r"
        )

        if len(self.base_dataset) != len(self.teacher_predictions):
            raise ValueError(
                f"Length mismatch for {split}: "
                f"base_dataset={len(self.base_dataset)}, "
                f"teacher_predictions={len(self.teacher_predictions)}"
            )

        if len(self.base_dataset) != len(self.teacher_embeddings):
            raise ValueError(
                f"Length mismatch for {split}: "
                f"base_dataset={len(self.base_dataset)}, "
                f"teacher_embeddings={len(self.teacher_embeddings)}"
            )

    def __len__(self):

        return len(
            self.base_dataset
        )

    def __getitem__(self, idx):

        sample = self.base_dataset[idx]

        sample["teacher_prediction"] = torch.tensor(
            self.teacher_predictions[idx],
            dtype=torch.float32
        )

        sample["teacher_embedding"] = torch.tensor(
            self.teacher_embeddings[idx],
            dtype=torch.float32
        )

        return sample