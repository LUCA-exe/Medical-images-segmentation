from inference.postprocessing import *
from net_utils.utils import save_image

import os
import numpy as np
from pathlib import Path
from typing import Dict, Union
from PIL import Image  # Assuming PIL (Pillow) for image processing


def load_npy_arrays_by_label(folder_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Loads NumPy arrays from a folder into a dictionary with labels as keys.

    This function iterates through a specified folder (`folder_path`) and loads
    all NumPy array files (*.npy). It creates a dictionary where the array label
    (filename without extension) is used as the key, and the loaded NumPy array
    is stored as the value.

    Args:
        folder_path (str or pathlib.Path): The path to the folder containing NumPy files.

    Returns:
        dict: A dictionary containing loaded NumPy arrays with labels as keys.
            If no files are found or errors occur, an empty dictionary is returned.

    Raises:
        ValueError: If the provided folder path is not a directory.
        IOError: If an error occurs while loading a NumPy file.
    """

    if not isinstance(folder_path, (str, Path)):
        raise ValueError("folder_path must be a string or pathlib.Path object")

    folder_path = Path(folder_path)  # Ensure consistent Path object
    if not folder_path.is_dir():
        raise ValueError(f"Invalid path: {folder_path} is not a directory")

    arrays = {}
    for file in folder_path.glob("*.npy"):
        try:
            label = file.stem  # Get filename without extension
            array = np.load(file)
            arrays[label] = array.astype(np.uint16)
        except (IOError, OSError) as e:
            print(f"Error loading NumPy file: {file} ({e})")
    return arrays


def try_fusion_approach(images: dict) -> None:
    """
    Call functions from the post-processing technique for easy testing the visual results.

    This function will create multiple "debug's" plot in the temporary folder.
    """ 

    # Parsing the orginal data structure
    prediction_instance = images["prediction"]
    save_image(prediction_instance > 0, "./tmp", f"All particles mask prediction")    
    sc_prediction_instance = images["single_channel_prediction_EVs"]
    save_image(sc_prediction_instance > 0, "./tmp", f"Single-channel mask EVs prediction")  
    nuclei_prediction_instance = images["single_channel_prediction_nuclei"]
    nuclei_prediction_instance = remove_smaller_areas(nuclei_prediction_instance, area_threshold = 1000) 
    save_image(nuclei_prediction_instance > 0, "./tmp", f"Single-channel mask nuclei prediction")
    
    # Two different wrapper function to add nuclei and EVs from single channel prediciton
    prediction_instance = add_nuclei_by_overlapping(prediction_instance, nuclei_prediction_instance)

    # Refine/remove the connected-components
    processed_prediction = refine_objects_by_overlapping(prediction_instance, sc_prediction_instance)
    save_image(processed_prediction > 0, "./tmp", f"Refined mask overlapped image with EVs")        

    # Optimized for metrics.
    refined_evs_prediction = add_objects_by_overlapping(processed_prediction, sc_prediction_instance)
    save_image(refined_evs_prediction, "./tmp", f"Final image with additional objects") 
    return None


def filter_images_by_name(images: dict[str, np.ndarray], excluding_value: str) -> dict[str, np.ndarray]:
    """
    Filters a dictionary of images based on a name exclusion pattern.

    Args:
        images: A dictionary where keys are image names (strings) and values are NumPy ndarrays.
        excluding_value: A string pattern to exclude images from the output.

    Returns:
        A new dictionary containing only images whose names do not include the `excluding_value` pattern.
    """
    return {key: value for key, value in images.items() if excluding_value not in key}


if __name__ == '__main__':
    
    images = load_npy_arrays_by_label(Path("./net_utils/images_sample"))
    updated_images = filter_images_by_name(images, excluding_value = "E2DV")
    try_fusion_approach(updated_images)
















