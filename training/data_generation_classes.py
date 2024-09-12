"""
This file contains the implementation of the 
generation of data split by different concrete classes
inheriting from general interface.
"""

from typing import Any, Dict, Tuple, Union, List
import abc
from abc import abstractmethod, ABCMeta
import numpy as np
import copy
import scipy.ndimage as ndimage

from training.train_data_representations import distance_label_2d

class data_generation_factory_interface(metaclass = ABCMeta):
    """
    Interface for creating the factory to generate images based
    on the needed type of labels.
    """

    @abstractmethod
    def create_training_data(labels: List[str], img: np.ndarray, mask: np.ndarray, tra: np.ndarray):
        """
        For every label generate the requested images.

        Args:
            Labels: Variable-length argument list containing the labels to provide the correspondent images.
            img: Original image.
            mask: Loaded annotate mask (array of type uint16).
            tra: Loaded annotate mask for tracking objects (array of type uint16).
        """
        raise NotImplementedError
    

class data_generation_factory(data_generation_factory_interface):
    """
    Implementing the interface to provide computed images based on the labels needed.
    """

    def create_training_data(labels: List[str], img: np.ndarray, mask: np.ndarray,
                             td_settings: Dict) -> Dict[str, np.ndarray]:
        """
        For every label generate the requested images.

        Args:
            Labels: Variable-length argument list containing the labels to provide the correspondent images.
            img: Original image.
            mask: Loaded annotate mask (array of type uint16).
            tra: Loaded annotate mask for tracking objects (array of type uint16).
            td_settings: hasmap containing different key-value pair with processing
                        informations
        
        Returns:
            Key-value pairs of computed images.
        """
        # Pre-conditions for the 'correctness' labels strings inside 'labels'
        images = {}
        for label in labels:

            # The following images are created toghether reducing the complexity.
            if label == "dist_cell_and_neighbor":
                # Calculate train data representations
                images["cell_dist"], images["neighbor_dist"] = distance_label_2d(label=mask,
                                                            cell_radius=int(np.ceil(0.5 * td_settings['max_mal'])),
                                                            neighbor_radius=td_settings['search_radius'], 
                                                            disk_radius = td_settings['disk_radius'])

            if label == "binary_mask_label":
                images["binary_mask_label"] = extract_binary_mask_label(mask)

            if label == "binary_border_label":
                images["binary_border_label"] = extract_binary_border_label(mask)
        return images
    

def extract_binary_border_label(mask: np.ndarray, border_width: int = 4) -> np.ndarray:
    """
    Extracts binary borders of cells from a binary segmentation mask.

    Args:
        mask: Binary mask (2D Array of uint16 type) with cells represented as 1 and background as 0.
        border_width: Width of the border to extract from the annotated particles.

    Returns:
        np.ndarray: Mask with only cell borders remaining.
    """
    # pre-conditions - np.ndarray of uint16
    if not mask.dtype == np.uint16:
        raise ValueError(f"The current cropped mask passed is of a type class {mask.dtype}")
    
    mask = copy.copy(mask)
    # NOTE: Currently reating all the cells as the same class (e.g. not distinguish between the EVs and Cells)
    mask[mask > 0] = 1 

    # Work with the boolean array to invert the original mask
    inverted_mask = ~mask.astype(bool)
    inverted_mask = inverted_mask.astype(int)

    # Dilate the mask to slightly enlarge the borders (handling very thin borders)
    dil_inverted_mask = ndimage.binary_dilation(inverted_mask, iterations = border_width)
    
    # Obtain the borders directly by difference
    cell_border = dil_inverted_mask ^ inverted_mask

    # Post-conditions
    cell_border = cell_border.astype(mask.dtype)
    return cell_border

def extract_binary_mask_label(mask: np.ndarray) -> np.ndarray:
        """
        Prepare the segmentation mask for the binary cross entropy.

        Args:
            mask: Integer np.ndarray with every cell represented
            by a different number

        Returns:
            Binary annotated mask - not distringuishing between different cell types.
        """ 

        mask_picture = copy.copy(mask) # Work on the copy of the object - no reference
        mask_picture[mask_picture > 0] = 1 
        return mask_picture
