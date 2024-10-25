""" This module generates images data for the training loop.

It implements a factory class to return the requested processed images
for further processing before the saving of the files in memory.
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
        """For every label generate the requested images.
        """
        raise NotImplementedError
    

class data_generation_factory(data_generation_factory_interface):
    """It provides the method to get the requested images based on the 
    provided architecture arg.
    """

    def create_training_data(labels: Tuple[str,], img: np.ndarray, mask: np.ndarray,
                             tra_gt: np.ndarray, td_settings: Dict) -> Dict[str, np.ndarray]:
        """
        Args:
            Labels: Variable-length argument list containing the labels to provide the correspondent images.
            img: Original image.
            mask: Loaded annotate mask (array of type uint16).
            tra_gt: Loaded annotate mask for tracking objects (array of type uint16).
            td_settings: hasmap containing different key-value pair with processing
                        informations
        
        Returns:
            Key-value pairs of computed images.
        """
        # Pre-conditions for the 'correctness' labels strings inside 'labels'
        images = {}
        images["img"] = img
        images["mask"] = mask
        images["tra_gt"] = tra_gt
        for label in labels:
            if label == "dist_cell_and_neighbor":
                images["dist_cell"], images["dist_neighbor"] = distance_label_2d(label=mask,
                                                            cell_radius=int(np.ceil(0.5 * td_settings['max_mal'])),
                                                            neighbor_radius=td_settings['search_radius'], 
                                                            disk_radius = td_settings['disk_radius'])
                
                if not ((str(images["dist_cell"].dtype) == 'float32') and (str(images["dist_neighbor"].dtype) == 'float32')):
                    raise TypeError(f"The dist_cell and dist_neighbor images computed are not the expected type!")
            
            # NOTE: Temporary coupling of two processing method.
            if label == "dist_cell":
                images["dist_cell"], _ = distance_label_2d(label=mask,
                                                            cell_radius=int(np.ceil(0.5 * td_settings['max_mal'])),
                                                            neighbor_radius=td_settings['search_radius'], 
                                                            disk_radius = td_settings['disk_radius'])
                
                if not (str(images["dist_cell"].dtype) == 'float32'):
                    raise TypeError(f"The dist_cell image computed are not the expected type!")


            if label == "mask_label":
                images["mask_label"] = extract_binary_mask_label(mask)

                if not str(images["mask_label"].dtype) == 'uint16':
                    raise TypeError(f"The mask_label computed is not the expected type!")
            
            if label == "binary_border_label":
                images["binary_border_label"] = extract_binary_border_label(mask)

                if not str(images["binary_border_label"].dtype) == 'uint16':
                    raise TypeError(f"The binary_border_label computed is not the expected type!")
        
        # Check on correctly computed labels.   
        len_comparison = len(labels)
        # The following label add two processed images to the dict.
        if "dist_cell_and_neighbor" in labels: len_comparison += 1
        if len(images.keys()) != 3 + len_comparison:
            raise ValueError(f"The dict. contains just {images.keys()}: erroneous labels passed: {labels}")
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