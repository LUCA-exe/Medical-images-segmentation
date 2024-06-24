"""
Module for testing post-processing functionality in cell counting.

This module provides helper functions and classes to facilitate unit testing
of post-processing steps involved in cell counting workflows. It is designed
to be adaptable to various cell counting algorithms and image processing pipelines.

**Key Features:** (to change)

- **MockDataGenerator:** Creates mock image and mask data for testing purposes.
- **CellCounter:** Provides a base class for cell counting algorithms with testing utilities.
- **TestResult:** A class to represent the outcome of a cell counting test case.
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

    if not mask_files:
        raise ValueError(f"No TIFF mask images found in folder: {folder_path}")

    # Load mask images using skimage.io.imread for flexibility
    masks = [imread(mask_file) for mask_file in mask_files]
    return masks

# assert type of loaded masks (form tiff extension)

def get_centroids_map(image: np.ndarray) -> dict:
    # Get a single-channle image and computed centroids coord: label and rerturn as hashmap


    pass

if __name__ == '__main__':
    # Testing new functions
     
    masks_path = "/content/Medical-images-segmentation/training_data/Fluo-E2DV-test/01_GT/SEG"
    gt_masks = load_masks(masks_path)
    print(gt_masks)



    # FIRST MASK AS CENTROIDS COORD SET-up

    # ITERATE OVER THE MAPS

    # For every maps, recompute centroids, assignt he neanrest label with the gorund truth  centroids, fill the dict (label: EVs as ordere tuple)







