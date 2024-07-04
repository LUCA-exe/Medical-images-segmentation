"""
Module for testing post-processing functionality in cell counting.

This module provides helper functions and classes to facilitate unit testing
of post-processing steps involved in cell counting workflows. It is designed
to be adaptable to various cell counting algorithms and image processing pipelines.

**Key Features:** (to refactor)

- **CellCounter:** Provides a base class for cell counting algorithms with testing utilities.
- **TestResult:** A class to represent the outcome of a cell counting test case.
"""
import numpy as np
from typing import Union, Optional
from collections import defaultdict
from skimage import measure
from skimage.morphology import binary_dilation, binary_erosion
from copy import deepcopy
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from skimage.io import imread
from PIL import Image

from ext_modules.utils import *
from net_utils.utils import save_image

# TODO: Move to typeVerifierObject - avoid constants
MASK_IMAGES_SUPPORTED_TYPE = [np.uint16]

# Const to filter out unwanted cells for the counting
CONSIDERED_LABEL = [5]

def get_centroids_map(labeled_image: np.ndarray, dim_filter: int = 5000) -> dict:
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


def get_nearer_centroid_label(labeled_centroids: dict[int, tuple], centroid_coord: tuple, max_distance: Union[float, int] = 200) -> int:
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
        if (distance < min_distance) and (distance < max_distance):
            min_distance = distance
            nearest_label = label

    # Return None in case of centroids to distant to the current one - possible not identified cell in the current frame
    return nearest_label


def count_evs(masked_image: np.ndarray, labeled_image: np.ndarray, expand_value: float, dim_filter: int, rgb_image: Union[None, np.ndarray], idx: int) -> int:
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

    # NOTE: Requested the debug images for the technical report during the counting
    if not rgb_image is None:

        # Additional operations just for the debug print
        eroded_mask = binary_erosion(dilated_mask, footprint=np.ones((6, 6)))
        delimiting_area = dilated_mask & (~eroded_mask)

        #save_image(dilated_mask > 0, "./tmp", f"Current dilated cell in account") 
        #save_image(delimiting_area > 0, "./tmp", f"Current delimiting area") 
        rgb_line = plot_rgb_image_from_mask(delimiting_area > 0, idx)
        overlap_images(rgb_image, rgb_line, f'./tmp/overlapped_line_{idx}.png')

        #plot_image_with_highlighted_mask(image = rgb_image, mask = delimiting_area, idx = idx)

    #save_image(dilated_mask > 0, "./tmp", f"Current dilated cell in account")  

    # Filter labeled image based on overlap with dilated mask
    filtered_labeled_image = labeled_image[dilated_mask]

    # Count unique labels (connected components) - exclude the background '0' value
    num_evs = len(np.unique(filtered_labeled_image.flat)) - 1
    return num_evs


"""Visualization function
Additional visualization functions (figure saves in the ./tmp folder as visualization for debugging purpose)

""" 

def plot_image_with_dots(image: np.ndarray, dots: list[tuple], file_path: str) -> None:

    """
    Plots a single-channel image with red dots at specified coordinates.

    This function takes a grayscale image (`image`) and a list of coordinates (`dots`) as input. 
    It plots the image using matplotlib and superimposes red dots at each coordinate in the `dots` list.

    Args:
        image (np.ndarray): A single-channel grayscale image (assumed to be 2D).
        dots (list[tuple]): A list of tuples representing coordinates (row, col) where red dots should be placed.

    Returns:
        None: This function plots the image and does not return a value.

    Raises:
        ValueError: If the input image is not a single-channel grayscale image.
    """

    #if len(image.shape) != 2 or image.dtype not in MASK_IMAGES_SUPPORTED_TYPE:
        #raise ValueError("Input image must be a single-channel grayscale image (2D).")

    # Plot the grayscale image
    plt.imshow(image)

    # Plot red dots at specified coordinates
    for dot in dots:
        # NOTE: The centroid "points" have inverted coordinates
        plt.plot(dot[1], dot[0], marker='v', color="red") 

    # Remove unnecessary elements from the plot (optional)
    plt.axis('off')  # Hide axes
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def plot_rgb_image_from_mask(mask, idx, resolution=(2048, 2048)):

    # Ensure the mask is of the correct resolution
    if mask.shape != resolution:
        raise ValueError(f"Mask must be of shape {resolution}")

    # Create a blank RGB image with all pixels set to black
    image = np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8)

    # Set the pixels to red where the mask is True
    image[mask] = [255, 0, 0]

    # Plot and save the image
    plt.imshow(image)
    plt.axis('off')  # Turn off the axis
    plt.savefig(f'./tmp/delimiting_line_{idx}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Return the value
    return image


def overlap_images(image1: np.ndarray, image2, output_path: str):
    """
    Overlap two RGB images and save the result.
    
    Parameters:
    image1 (np.ndarray): First image as a 3D numpy array.
    image2_path (str): Path to the second image.
    output_path (str): Path where the output image will be saved.
    """
    # Load the second image
    #image2 = Image.open(image2_path)
    
    # Ensure the second image is in RGB mode
    #if image2.mode != 'RGB':
        #image2 = image2.convert('RGB')
    #image2 = imread(image2_path)
    
    # Convert the second image to a numpy array
    #image2 = np.array(image2)
    
    # Ensure both images have the same dimensions
    if image1.shape != image2.shape:
        raise ValueError("The two images must have the same dimensions.")
    
    # Blend the images (simple average here, can be adjusted)
    blended_image = image1.astype(np.float32) + image2.astype(np.float32)
    blended_image = blended_image.astype(np.uint8)  # Convert back to uint8
    
    # Convert the blended image to a PIL Image
    blended_image_pil = Image.fromarray(blended_image)
    
    # Save the result
    blended_image_pil.save(output_path)


### Wrapper function ###
def count_small_connected_components(rgb_images_paths, masks_path: str, cells_area: int, expanding_val: float) -> dict:

    """
    Counts small connected components (EVs) around identified cells in labeled masks.

    This function takes a folder path containing labeled masks (`masks_path`), a minimum cell area (`cells_area`),
    two expansion values for counting EVs (`hard_expanding_val` and `soft_expanding_val`), and returns a dictionary
    mapping cell labels to lists containing the number of EVs counted using each expansion value.

    Args:
        masks_path (str): Path to the folder containing labeled masks (assumed to be single-channel grayscale images).
        cells_area (int): Minimum area (number of pixels) to consider a component a cell.
        hard_expanding_val (float): Expansion value (positive) for stricter EV counting around cells.
        soft_expanding_val (float): Expansion value (positive) for looser EV counting around cells.

    Returns:
        dict: A dictionary where keys are cell labels (integers) and values are lists containing the number of EVs
                counted using `hard_expanding_val` and `soft_expanding_val`, respectively.

    Raises:
        ValueError: If `hard_expanding_val` is greater than or equal to `soft_expanding_val`.
    """

    # Load all masks from the folder
    gt_masks = load_masks(masks_path)
    rgb_images = load_masks(rgb_images_paths)

    # Ensure debug folder exists (optional)
    debug_folder_path = os.path.join(os.getcwd(), "tmp")  # Use current working directory for debug folder
    os.makedirs(debug_folder_path, exist_ok=True)

    # Get centroids from the first mask (assuming first mask represents cells)
    centroids_map = get_centroids_map(gt_masks[0], dim_filter=cells_area)
    print(f"Current identified centroids: {centroids_map}")

    # Visualize centroids (taken from mask) on the first original image (optional)
    #centroids_list = centroids_map[14].values()
    current_centroid = centroids_map[5]
    plot_image_with_dots(rgb_images[0], [current_centroid], os.path.join(debug_folder_path, "first_frame_example"))

    # Initialize dictionary to store cell labels and corresponding EV counts
    id_evs_map = defaultdict(list)

    # Process each mask (image)
    for idx, mask in tqdm(enumerate(gt_masks)):
        print(f".. Analyzing mask {idx} ..")

        # Keep only small components (potential EVs)
        evs_labeled_image = mask.copy()  # Avoid modifying the original mask
        for reg in measure.regionprops(evs_labeled_image):
            if reg["area"] > cells_area:
                # Remove regions exceeding the cell area threshold
                current_cell_mask = evs_labeled_image == reg["label"]
                evs_labeled_image[current_cell_mask] = 0

        # Visualize mask with EVs (optional)
        save_image(evs_labeled_image > 0, debug_folder_path, f"Current labeled EVs mask of mask {idx}")

        # Get centroids for the current mask
        current_centroids_map = get_centroids_map(mask, dim_filter=cells_area)

        # Iterate over centroids from the first mask
        for label, centroid in centroids_map.items():

            # Filter out unwanted cells
            if label in CONSIDERED_LABEL:
            
                # Take label of the current centroid nearer the original centroids
                current_cell_label = get_nearer_centroid_label(current_centroids_map, centroid)

                # Take the mask of the current cell label
                current_cell_mask = mask == current_cell_label
                
                # Debug plot
                #save_image(current_cell_mask > 0, "./tmp", f"Current label referenced: {current_cell_label} - frame: {idx}") 
                
                # Make two times the count of EVs using different expansion values
                current_evs = count_evs(current_cell_mask, evs_labeled_image, expand_value = expanding_val, dim_filter = cells_area, rgb_image = rgb_images[idx], idx = idx )
                id_evs_map[label].append(current_evs)
    return id_evs_map
 

if __name__ == '__main__':
     
    # TODO: Testing the features
    masks_path = "/content/Medical-images-segmentation/training_data/Fluo-E2DV-count/01_RES_Fluo-E2DV-train_GT_01_320_kit-ge_original-dual-unet_02_0.02_0.01"
    # NOTE: Temorary position
    rgb_images_paths = "/content/Medical-images-segmentation/training_data/Fluo-E2DV-count/01"

    # Add original path to load the original RGB image for delinneation
    cells_area = 5000
    # NOTE: Soft and hard expanding values are use to count EVs - two values to provide more information about circultaing EVs
    expanding_val = 200

    evs_counter = count_small_connected_components(rgb_images_paths, masks_path, cells_area, expanding_val)
    print(evs_counter)
    # Save teh dict in a jsopn format for clarity - add utils function in this module







