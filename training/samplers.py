"""
Sampling utilities for imbalanced intensity distributions.
"""

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def create_soft_intensity_balanced_sampler(
    dataset,
    num_bins=20,
    alpha=0.5,
    replacement=True
):
    """
    Create a soft intensity-balanced sampler.

    This sampler reduces the dominance of overrepresented intensity bins
    without making all bins strictly balanced.

    Parameters
    ----------
    dataset : ParticleDataset
        Training dataset.

    num_bins : int
        Number of target intensity bins.

    alpha : float
        Strength of balancing.

        alpha = 0.0:
            original random sampling

        alpha = 1.0:
            strict inverse-frequency balancing

        alpha between 0 and 1:
            soft balancing

    replacement : bool
        Whether to sample with replacement.

    Returns
    -------
    sampler : WeightedRandomSampler
    """

    targets = []

    for idx in range(len(dataset)):

        sample = dataset[idx]

        targets.append(
            float(sample["target"])
        )

    targets = np.asarray(
        targets
    )

    bin_edges = np.linspace(
        targets.min(),
        targets.max(),
        num_bins + 1
    )

    bin_ids = np.digitize(
        targets,
        bin_edges[1:-1]
    )

    bin_counts = np.bincount(
        bin_ids,
        minlength=num_bins
    )

    bin_counts = np.maximum(
        bin_counts,
        1
    )

    sample_weights = 1.0 / (
        bin_counts[bin_ids] ** alpha
    )

    sample_weights = sample_weights / sample_weights.mean()

    sample_weights = torch.DoubleTensor(
        sample_weights
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=replacement
    )

    return sampler