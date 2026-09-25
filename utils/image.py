"""
utils/image.py

Image processing utilities for particle extraction.

Author: Jiarui Xie
"""

import numpy as np

from scipy.ndimage import (
    gaussian_filter,
    maximum_filter
)


# ============================================================
# Gaussian Filter
# ============================================================

def apply_gaussian_filter(
    image,
    sigma=1
):
    """
    Apply Gaussian smoothing.

    Parameters
    ----------
    image : ndarray

    sigma : float

    Returns
    -------
    ndarray
    """

    return gaussian_filter(
        image,
        sigma=sigma
    )


# ============================================================
# Local Maxima Detection
# ============================================================

# def calculate_filtered_maxima(
#     cy3_patch,
#     iscat_patch,
#     cy3_threshold#,
#     #iscat_threshold
# ):
#     """
#     Detect valid local maxima.

#     Parameters
#     ----------
#     cy3_patch : ndarray

#     iscat_patch : ndarray

#     cy3_threshold : float

#     iscat_threshold : float

#     Returns
#     -------
#     ndarray(bool)
#     """

#     cy3_patch = np.squeeze(cy3_patch)
#     iscat_patch = np.squeeze(iscat_patch)

#     footprint = np.ones((3, 3))

#     max_filtered = maximum_filter(
#         cy3_patch,
#         footprint=footprint
#     )

#     local_max_mask = (
#         cy3_patch == max_filtered
#     )

#     valid_mask = (
#         cy3_patch >= cy3_threshold
#     )

#     # valid_mask = (
#     #     (cy3_patch >= cy3_threshold)
#     #     &
#     #     (iscat_patch >= iscat_threshold)
#     # )

#     final_mask = (
#         local_max_mask
#         &
#         valid_mask
#     )

#     return final_mask

def calculate_filtered_maxima(
    cy3_patch,
    iscat_patch=None,
    cy3_threshold=39,
    iscat_threshold=None
):
    """
    Detect valid local maxima using Cy3 only.
    iSCAT threshold is ignored.
    """

    cy3_patch = np.squeeze(cy3_patch)

    footprint = np.ones((3, 3))

    max_filtered = maximum_filter(
        cy3_patch,
        footprint=footprint
    )

    local_max_mask = (
        cy3_patch == max_filtered
    )

    valid_mask = (
        cy3_patch >= cy3_threshold
    )

    final_mask = (
        local_max_mask
        &
        valid_mask
    )

    return final_mask


# ============================================================
# Patch Extraction
# ============================================================

def extract_patch_with_padding(
    image,
    center_x,
    center_y,
    patch_size=64
):
    """
    Extract a square patch centered at (center_x, center_y).

    If the patch extends beyond the image boundary, the missing
    region is filled with zeros.

    This implementation avoids padding the full image, so it is
    much more memory-efficient.

    Parameters
    ----------
    image : ndarray
        Input 2D image.

    center_x : int
        Center x-coordinate.

    center_y : int
        Center y-coordinate.

    patch_size : int
        Size of the output patch.

    Returns
    -------
    patch : ndarray
        Extracted patch with shape (patch_size, patch_size).
    """

    image = np.squeeze(image)

    assert image.ndim == 2, (
        f"Expected a 2D image, but got shape {image.shape}."
    )

    assert patch_size % 2 == 0, (
        "patch_size must be even."
    )

    half = patch_size // 2

    height, width = image.shape

    # --------------------------------------------------------
    # Desired patch region in the original image coordinates
    # --------------------------------------------------------

    x_start = center_x - half
    x_end = center_x + half

    y_start = center_y - half
    y_end = center_y + half

    # --------------------------------------------------------
    # Valid region inside the image
    # --------------------------------------------------------

    src_x_start = max(x_start, 0)
    src_x_end = min(x_end, width)

    src_y_start = max(y_start, 0)
    src_y_end = min(y_end, height)

    # --------------------------------------------------------
    # Corresponding region inside the output patch
    # --------------------------------------------------------

    dst_x_start = src_x_start - x_start
    dst_x_end = dst_x_start + (src_x_end - src_x_start)

    dst_y_start = src_y_start - y_start
    dst_y_end = dst_y_start + (src_y_end - src_y_start)

    # --------------------------------------------------------
    # Create zero-padded patch
    # --------------------------------------------------------

    patch = np.zeros(
        (patch_size, patch_size),
        dtype=image.dtype
    )

    patch[
        dst_y_start:dst_y_end,
        dst_x_start:dst_x_end
    ] = image[
        src_y_start:src_y_end,
        src_x_start:src_x_end
    ]

    return patch