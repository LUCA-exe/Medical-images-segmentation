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
from collections import defaultdict
from skimage import measure
from skimage.morphology import binary_dilation
from copy import deepcopy

from ext_modules.utils import *
from net_utils.utils import save_image

# TODO: Move to typeVerifierObject - avoid constants
MASK_IMAGES_SUPPORTED_TYPE = [np.uint16]

def get_centroids_map(labeled_image: np.ndarray, dim_filter: int = 4000) -> dict:
    """
    Calculates centroids for connected components in a single-channel image.

    This function takes a grayscale image (`image`) and a minimum size filter (`dim_filter`) as input. 
    It performs the following steps:

    1. Binarizes the image using thresholding (assuming background is 0).
    2. Identifies connected components (objects) in the binary image.
    3. Filters out components smaller than the specified `dim_filter`.
    4. Calculates the centroid (center of mass) for each remaining connected component.
    5. Returns a dictionary where keys are unique IDs (incrementing integers)
        and values are NumPy arrays representing the (row, col) coordinates of the centroids.

    Args:
        image (np.ndarray): A single-channel labeled image (assumed to be 2D).
        dim_filter (int, optional): Minimum size (number of pixels) for a connected component to be included. 
                                    Defaults to 4000.

    Returns:
        dict: A dictionary containing centroids as key-value pairs. Keys are unique IDs (integers),
                and values are NumPy arrays with shape (2,) representing (row, col) coordinates.

    Raises:
        ValueError: If the input image is not a single-channel grayscale image.
    """

    # NOTE: Temporary fixed list of possible types
    if len(labeled_image.shape) != 2 or labeled_image.dtype not in MASK_IMAGES_SUPPORTED_TYPE:
        raise ValueError("Input image must be a single-channel grayscale image (2D).")
    region_props = measure.regionprops(labeled_image)

    # Extract centroids and store in dictionary
    centroids_map = {}
    for prop in region_props:

        # Just take the Cells
        if prop['area'] > dim_filter:
            centroid = prop['centroid']

            # Store the tuple with the original id of the connected components
            centroids_map[prop['label']] = centroid
    return centroids_map


def get_nearer_centroid(labeled_centroids: dict[int, tuple], centroid_coord: tuple) -> int:
    """
    Finds the label of the nearest centroid from a given reference point.

    This function takes a dictionary of labeled centroids (`labeled_centroids`) and a reference coordinate 
    (`centroid_coord`) as input. It iterates through the centroids in the dictionary and calculates the 
    Euclidean distance between each centroid and the reference point. It then returns the label (key) 
    associated with the nearest centroid.

    Args:
        labeled_centroids (dict[int, tuple]): A dictionary where keys are integer labels and values are 
                                                tuples representing centroid coordinates (row, col).
        centroid_coord (tuple): A tuple representing the reference coordinate (row, col) for distance calculation.

    Returns:
        int: The label (key) of the nearest centroid in the dictionary.

    Raises:
        ValueError: If the labeled_centroids dictionary is empty.
    """

    if not labeled_centroids:
        raise ValueError("labeled_centroids dictionary cannot be empty")

    nearest_label = None
    min_distance = np.inf  # Initialize minimum distance to positive infinity

    for label, centroid in labeled_centroids.items():
        # Calculate Euclidean distance between reference point and current centroid
        distance = np.linalg.norm(np.array(centroid_coord) - np.array(centroid))
        if distance < min_distance:
            min_distance = distance
            nearest_label = label
    return nearest_label


def count_evs(masked_image: np.ndarray, labeled_image: np.ndarray, expand_value: float, dim_filter: int) -> int:
    """
    Counts the number of connected components within a dilated masked area.

    This function takes a binary mask image (`masked_image`), a labeled image (`labeled_image`), and an expansion 
    value (`expand_value`) as input. It performs the following steps:

    1. Creates a dilated mask by applying binary dilation with the specified `expand_value`.
    2. Filters the labeled image by selecting only the connected components that overlap with the dilated mask.
    3. Returns the number of unique labels (connected components) in the filtered labeled image.

    Args:
        masked_image (np.ndarray): A binary mask image (boolean NumPy array).
        labeled_image (np.ndarray): A labeled image containing connected components (uint16).
        expand_value (float): The dilation factor (positive value) to expand the masked area.

    Returns:
        int: The number of connected components within the dilated masked area.

    Raises:
        ValueError: 
            - If the masked image is not a binary NumPy array.
            - If the labeled image data type is not uint16.
            - If the expansion value is non-positive.
    """

    if not masked_image.dtype == bool:
        raise ValueError("masked_image must be a binary NumPy array (boolean).")
    if labeled_image.dtype != np.uint16:
        raise ValueError("labeled_image data type must be uint16.")
    if expand_value <= 0:
        raise ValueError("Expansion value must be a positive float.")

    # Dilate the mask to capture components in the vicinity
    dilated_mask = binary_dilation(masked_image, footprint=np.ones((expand_value, expand_value)))

    save_image(dilated_mask > 0, "./tmp", f"Current dilated cell in account")  

    # Filter labeled image based on overlap with dilated mask
    filtered_labeled_image = labeled_image[dilated_mask]

    # Count unique labels (connected components)
    num_evs = np.unique(filtered_labeled_image.flat)
    # NOTE: Exclude the background '0' value
    print(len(num_evs) - 1)
    exit(1)

    # Count unique labels (connected components)
    return total_evs


"""Visualization function
Additional visualization functions (figure saves in the ./tmp folder as visualization for debugging purpose)
""" 


if __name__ == '__main__':
     
    # TODO: Testing the features
    masks_path = "/content/Medical-images-segmentation/training_data/Fluo-E2DV-test/01_GT/SEG"
    cells_area = 5000
    expanding_val = 150
    gt_masks = load_masks(masks_path)

    # Get centroids from a chosen frame - for now the first frame is taken as truth
    centroids_map = get_centroids_map(gt_masks[0], dim_filter = cells_area)
    print(f"Current identified centroids: {centroids_map}")

    exit(1)

    id_evs_map = defaultdict(list)
    for idx, mask in enumerate(gt_masks):
        # Iterate over all the images
        print(f".. Analyzing mask {idx} ..")
        
        # Keep just the EVs for the current mask
        evs_labeled_image = deepcopy(mask)

        for reg in measure.regionprops(evs_labeled_image):
            if reg["area"] > cells_area:

                # Remove the current label
                current_cell_mask = evs_labeled_image == reg["label"]
                evs_labeled_image[current_cell_mask] = 0

        # Debug
        save_image(evs_labeled_image > 0, "./tmp", f"Current labeled evs mask of mask {idx}") 

        for label, centroid in centroids_map.items():
            # Iterate over all the centroids in the earlier computation

            current_centroids_map = get_centroids_map(mask)
            # Take label of the current centroids nearer to the original centroids
            current_cell_label = get_nearer_centroid(current_centroids_map, centroid)

            # Take the mask of the current cell
            current_cell_mask = mask == current_cell_label
            
            # Debug plot
            save_image(current_cell_mask > 0, "./tmp", f"Current label referenced: {current_cell_label}") 

            current_evs = count_evs(current_cell_mask, evs_labeled_image, expand_value = expanding_val, dim_filter = cells_area) 
            id_evs_map[label].append(current_evs)

        print(id_evs_map)
 






