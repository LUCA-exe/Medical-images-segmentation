"""Module containig utils function for the cell counts prototype
"""
import numpy as np
import os
from glob import glob
from skimage.io import imread


def load_masks(folder_path: str) -> list[np.ndarray]:
    """
    Loads mask images from a specified folder.

    This function reads all TIFF (.tiff or .tif) image files from the given folder path and returns them as a list
    of NumPy arrays. It adheres to best practices for:

    - Error Handling:
        - Raises `FileNotFoundError` if the folder path does not exist.
        - Raises `ValueError` if no TIFF mask images are found in the folder.
    - Data Validation:
        - Validates input path as a string for clarity.

    Args:
        folder_path (str): The path to the folder containing mask images.

    Returns:
        list[np.ndarray]: A list of loaded mask images as NumPy arrays, or an empty list if no masks are found.

    Raises:
        FileNotFoundError: If the folder path does not exist.
        ValueError: If no TIFF mask images are found in the folder.
    """

    if not isinstance(folder_path, str):
        raise ValueError("folder_path must be a string")

    # Check if folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Find all TIFF mask images
    mask_files = glob(os.path.join(folder_path, "*.tif"))
    mask_files.extend(glob(os.path.join(folder_path, "*.tiff")))  # Include both .tif and .tiff extensions

    # In-place sorting
    mask_files.sort()
    
    if not mask_files:
        raise ValueError(f"No TIFF mask images found in folder: {folder_path}")

    # Load mask images using skimage.io.imread for flexibility
    masks = [imread(mask_file) for mask_file in mask_files]

    # NOTE: Atemption to the memory consumption
    masks = [mask[400: 1400, :1000] for mask in masks]
    return masks

# TODO: assert type of the loaded masks - single channel and unsidegned int 16