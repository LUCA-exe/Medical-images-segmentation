import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
from skimage.segmentation import watershed
from skimage import measure
from skimage.feature import peak_local_max, canny
from skimage.morphology import binary_closing
from skimage.measure import label
from skimage.morphology import square
from skimage.morphology import dilation
import torch
import copy

from net_utils.utils import get_nucleus_ids, save_image


def foi_correction(mask, cell_type): # TODO: Implement option for my dataset ..
    """ Field of interest correction for Cell Tracking Challenge data (see
    https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf and
    https://public.celltrackingchallenge.net/documents/Annotation%20procedure.pdf )

    :param mask: Segmented cells.
        :type mask:
    :param cell_type: Cell Type.
        :type cell_type: str
    :return: FOI corrected segmented cells.
    """

    if cell_type in ['DIC-C2DH-HeLa', 'Fluo-C2DL-Huh7', 'Fluo-C2DL-MSC', 'Fluo-C3DH-H157', 'Fluo-N2DH-GOWT1',
                     'Fluo-N3DH-CE', 'Fluo-N3DH-CHO', 'PhC-C2DH-U373']:
        E = 50
    elif cell_type in ['BF-C2DL-HSC', 'BF-C2DL-MuSC', 'Fluo-C3DL-MDA231', 'Fluo-N2DL-HeLa', 'PhC-C2DL-PSC']:
        E = 25
    else:
        E = 0

    if len(mask.shape) == 2:
        foi = mask[E:mask.shape[0] - E, E:mask.shape[1] - E]
    else:
        foi = mask[:, E:mask.shape[1] - E, E:mask.shape[2] - E]

    ids_foi = get_nucleus_ids(foi)
    ids_prediction = get_nucleus_ids(mask)
    for id_prediction in ids_prediction:
        if id_prediction not in ids_foi:
            mask[mask == id_prediction] = 0

    return mask


def border_cell_post_processing(border_prediction, cell_prediction, args):
    """ Post-processing WT for distance label (cell distance + neighbor distance continuos tensors - KIT-GE solution) prediction.

    :param border_prediction: Neighbor distance prediction.
        :type border_prediction:
    :param cell_prediction: Cell distance prediction.
        :type cell_prediction:
    :param args: Post-processing settings (th_cell, th_seed, n_splitting, fuse_z_seeds).
        :type args:
    :return: Instance segmentation mask.
    """

    # Smooth predictions slightly + clip border prediction (to avoid negative values being positive after squaring) - Fixed parameters
    sigma_cell = 0.5
    cell_prediction = gaussian_filter(cell_prediction, sigma=sigma_cell)
    border_prediction = np.clip(border_prediction, 0, 1)

    th_seed = args.th_seed
    th_cell = args.th_cell
    th_local = 0.25

    # Get mask for watershed - straight up eliminations of low intensity pixel.
    mask = cell_prediction > th_cell

    # Get seeds for marker-based watershed
    borders = np.tan(border_prediction ** 2)

    # Empirical threhsolds of border pixel values
    borders[borders < 0.05] = 0
    borders = np.clip(borders, 0, 1)
    
    # Try to clean/thin the cell_prediction
    cell_prediction_cleaned = (cell_prediction - borders)
    seeds = cell_prediction_cleaned > th_seed
    seeds = measure.label(seeds, background=0)

    min_area = get_minimum_area_to_remove(seeds)
    seeds = remove_smaller_areas(seeds, min_area)

    # Avoid empty predictions (there needs to be at least one cell)
    while np.max(seeds) == 0 and th_seed > 0.05:
        th_seed -= 0.1
        seeds = cell_prediction_cleaned > th_seed
        seeds = measure.label(seeds, background=0)
        props = measure.regionprops(seeds)
        for i in range(len(props)):
            if props[i].area <= 4:
                seeds[seeds == props[i].label] = 0
        seeds = measure.label(seeds, background=0)

    # Marker-based watershed - markers (seeds) should be smaller than the image cells (cells distance)
    prediction_instance = watershed(image=-cell_prediction, markers=seeds, mask=mask, watershed_line=False)

    if args.apply_merging and np.max(prediction_instance) < 255:
        # Get borders between touching cells
        label_bin = prediction_instance > 0
        pred_boundaries = cv2.Canny(prediction_instance.astype(np.uint8), 1, 1) > 0
        pred_borders = cv2.Canny(label_bin.astype(np.uint8), 1, 1) > 0
        pred_borders = pred_boundaries ^ pred_borders
        pred_borders = measure.label(pred_borders)
        for border_id in get_nucleus_ids(pred_borders):
            pred_border = (pred_borders == border_id)
            if np.sum(border_prediction[pred_border]) / np.sum(pred_border) < 0.075:  # very likely splitted due to shape
                # Get ids to merge
                pred_border_dilated = binary_dilation(pred_border, np.ones(shape=(3, 3), dtype=np.uint8))
                merge_ids = get_nucleus_ids(prediction_instance[pred_border_dilated])
                if len(merge_ids) == 2:
                    prediction_instance[prediction_instance == merge_ids[1]] = merge_ids[0]
        prediction_instance = measure.label(prediction_instance)
    return np.squeeze(prediction_instance.astype(np.uint16)), np.squeeze(borders)


def get_minimum_area_to_remove(connected_components, percentage=0.1):
    """
    Calculates the minimum area allowed in an array of connected components,
    based on a percentage of the average area, and removes smaller components.

    Args:
        connected_components: A NumPy array of connected components.
        percentage: A percentage of the average area to use as the minimum threshold
                (default: 0.1).

    Returns:
        The minimum area allowed in the array.

    Raises:
        ValueError: If connected_components is empty.
    """

    if not connected_components.size:
        raise ValueError("connected_components cannot be empty")

    # Calculate areas of all components
    areas = np.array([prop.area for prop in measure.regionprops(connected_components)])

    # Calculate minimum area based on percentage of average area
    if np.any(areas):
        min_area = percentage * np.mean(areas)
    else:
        min_area = 0

    # Set a minimum threshold
    min_area = np.maximum(min_area, 4)

    # NOTE: Just override the min area component in case of my dataset for the first testing with a fixed value
    min_area = 30
    return min_area


def simple_binary_mask_post_processing(mask, original_image, args, denoise=True):
    """Performs simple post-processing on a binary mask prediction.

    This function takes a binary mask prediction, post-processes it using a
    threshold and optional noise removal, and returns an instance segmentation mask.

    Args:
        mask (numpy.ndarray): The binary mask prediction, typically with a shape of
            (C, H, W) where C is the number of channels, H is the height, and W is
            the width. It's assumed the binary mask is in the first channel (C=1).
        original_image (numpy.ndarray): The original image that the mask corresponds to.
            This is not used in this function but might be useful for other
            post-processing steps in the future.
        args (object): Additional arguments for post-processing configuration.
            The specific contents of this argument might depend on the implementation
            details but could include the threshold value or noise removal parameters.
        denoise (bool, optional): Flag indicating whether to perform noise removal on
            small objects in the mask. Defaults to True.

    Returns:
        numpy.ndarray: The instance segmentation mask with integer labels for each
            object. The mask has the same shape (H, W) as the original binary mask.
    """

    # Fixed parameters (consider moving these to the args dictionary for flexibility)
    threshold = 0.4  # Threshold value for binarization (can be fine-tuned)
    binary_ch = 1 # Binary channel used in the further processing

    # Process the binary mask
    processed_mask = np.squeeze(mask[binary_ch, :, :] > threshold)
    prediction_instance = measure.label(processed_mask, background=0)

    if denoise:
        # Remove small objects (area less than 30 pixels) for noise reduction
        prediction_instance = remove_smaller_areas(prediction_instance, 30)

    # Convert to uint16 for memory efficiency (assuming instance IDs fit in 16 bits)
    return np.squeeze(prediction_instance.astype(np.uint16))


# NOTE: Finish prototype (refactoring/creation of new object) - throw away code and fill documentation
def complex_binary_mask_post_processing(mask, binary_border, cell_prediction, original_image, args):
    """ 
    Assignining different IDs in the final segmentation mask prediction using a complex watershed algorithm (enahcned solution respect to the base thresholding).
    """

    # Adapting the data structure
    binary_channel = 1
    mask = np.squeeze(mask[binary_channel, :, :])
    binary_border = np.squeeze(binary_border[binary_channel, :, :])
    cell_prediction = np.squeeze(cell_prediction)
    original_image = np.squeeze(original_image)

    # NOTE: Temporary debug prints
    save_image(mask, "./tmp", f"Sigmoid layer ouput for mask")
    save_image(binary_border, "./tmp", f"Sigmoid layer ouput for binary border")
    save_image(cell_prediction, "./tmp", f"Sigmoid layer ouput for cell distance")
    save_image(original_image, "./tmp", f"Original input image")

    # NOTE: Refine borders - unusued for now
    th_borders = 0.1
    binary_border[binary_border < th_borders] = 0

    # Increased smoothness on the regressed image  
    sigma_cell = 0.5
    smoothed_cell_prediction = gaussian_filter(cell_prediction, sigma = sigma_cell)

    # Processing the binary mask with simple thresholding
    th_mask = 0.2
    processed_mask = mask > th_mask
    
    # More strict contraint for the seed
    th_seed = th_mask + 0.5
    seeds = mask > th_seed
    seeds = measure.label(seeds, background = 0)
    seeds = remove_smaller_areas(seeds, 30)

    # Apply watershed
    prediction_instance = watershed(image=-smoothed_cell_prediction, markers=seeds, mask=processed_mask, watershed_line=False)
    
    save_image(prediction_instance, "./tmp", f"Final output")
    return np.squeeze(prediction_instance.astype(np.uint16))


def fusion_post_processing(prediction_dict, sc_prediction_dict, args, just_evs=True):
    """ Post-processing WT enhanced with Fusion prediction for distance label (cell + neighbor distances continuos tensors) plus single-channel prediction.    
    :return: Instance segmentation mask.
    """
    # Choose predefined channel when working with prediciton from sigomid layers
    binary_channel = 1

    # Unpack the dictionary
    original_image, sc_original_image = prediction_dict["original_image"], sc_prediction_dict["original_image"]
    mask, sc_mask = prediction_dict["mask"], sc_prediction_dict["mask"]

    prediction_instance = simple_binary_mask_post_processing(mask, original_image, args)
    save_image(prediction_instance, "./tmp", f"Clean prediction for multi-channel image")

    if just_evs == True:
        sc_prediction_instance = simple_binary_mask_post_processing(sc_mask, sc_original_image, args)
        save_image(sc_prediction_instance, "./tmp", f"Clean prediction for single-channel image")
        processed_prediction = refine_objects_by_overlapping(prediction_instance, sc_prediction_instance)        
        # NOTE: Work in progress
        refined_evs_prediction = add_objects_by_overlapping(processed_prediction, sc_prediction_instance)
    return refined_evs_prediction, None


def remove_smaller_areas(seeds, area_threshold):
    """
    Removes connected components in a 2D array with areas smaller than a given threshold.

    Args:
        seeds (np.ndarray): A 2D NumPy array of connected components (labeled).
        area_threshold (int): The minimum area allowed for a component to remain.

    Returns:
        np.ndarray: The modified 3D array with smaller components removed.

    Raises:
        ValueError: If seeds is empty or area_threshold is negative.
    """

    if not seeds.size:
        raise ValueError("seeds cannot be empty")

    if area_threshold < 0:
        raise ValueError("area_threshold must be a non-negative integer")

    # Extract properties of each component
    props = measure.regionprops(seeds)

    # Filter and remove components based on area
    filtered_seeds = seeds.copy()
    for prop in props:
        if prop.area <= area_threshold:
            filtered_seeds[filtered_seeds == prop.label] = 0

    # Re-label the remaining components
    return measure.label(filtered_seeds, background=0)


def refine_objects_by_overlapping(base_image, refiner_image, max_cell_area=4000):
    """
    Refines a segmentation mask based on overlap with another segmentation mask.

    This function refines a base segmentation mask (`base_image`) by considering
    overlaps with a refiner segmentation mask (`refiner_image`). It removes small
    objects (less than `max_cell_area`) from the base image unless they overlap
    with the refiner image.

    Args:
        base_image (numpy.ndarray): The base segmentation mask to be refined.
            Assumed to be a 2D array where each pixel value represents the
            corresponding object label.
        refiner_image (numpy.ndarray): The refiner segmentation mask used for
            overlap analysis. Assumed to be a 2D binary mask where non-zero
            pixels indicate the presence of an object.
        max_cell_area (int, optional): The maximum allowed area (in pixels) for a
            connected component in the base image to be kept. Defaults to 4000.

    Returns:
        numpy.ndarray: The refined segmentation mask where small objects in the
            base image are removed unless they overlap with the refiner image. The
            output has the same shape and data type as the base image.
    """
    # Deep copy the base image to avoid modifying the original
    refined_image = base_image.copy()
    # Create a mask for non-zero pixels in the refiner image
    refiner_mask = refiner_image > 0

    # Loop over connected components in the base image
    for region_prop in measure.regionprops(refined_image):
        # Check if the area is less than the threshold
        if region_prop.area < max_cell_area:
            # Get the mask for the current connected component
            current_label_mask = refined_image == region_prop.label

            # Calculate the overlap mask between the component and refiner
            overlap_mask = current_label_mask * refiner_mask

            # Check if there's any overlap
            if np.any(overlap_mask):
                # If there's overlap, keep only the overlapped region
                refined_image[current_label_mask] = 0
                refined_image[overlap_mask] = region_prop.label
            else:
                # If no overlap, remove the small object
                refined_image[current_label_mask] = 0
    return refined_image


def add_objects_by_overlapping(base_image, single_channel_image, cells_overlap = 0.95, min_cell_area = 4000):
    """
    Adds objects from a single-channel image to a base image based on overlap conditions.

    This function takes a base image containing labeled connected components and
    a single-channel image representing potential objects. It adds objects from
    the single-channel image to the base image if they meet certain overlap
    criteria.

    Args:
        base_image (numpy.ndarray): The base image containing labeled connected
            components (assumed to be a 2D array).
        single_channel_image (numpy.ndarray): The single-channel image representing
            potential objects (assumed to be a 2D array).
        cells_overlap (float, optional): The minimum overlap ratio between an
            object and a connected component in the base image to be considered
            overlapping (defaults to 0.95).
        min_cell_area (int, optional): The minimum area (in pixels) for a connected
            component in the base image to be considered a cell (defaults to 4000).

    Returns:
        numpy.ndarray: The modified base image with objects from the single-channel
            image added based on overlap conditions. The output has the same shape
            and data type as the base image.
    """


    # Deep copy the original prediction
    image = copy.deepcopy(base_image)
    image_mask = image > 0
    mask = image > 0
    
    save_image(mask, "./tmp", f"Mask of true-false of the base prediction")

    # Get the next label from the already used ones
    next_usable_label = get_maximum_label(base_image) + 1

    for reg_prop in measure.regionprops(single_channel_image):
        
        # take the overlap etween a region and the mask of the orignial image
        current_mask = single_channel_image == reg_prop.label
        overlap_mask = current_mask & mask

        if np.any(overlap_mask):
            # Manage overlap with connected components in the base image
            component_label, component_mask, overlap_ratio = get_overlapping_components(image, current_mask)

            if overlap_ratio <= 0.95:
                # Add the Evs on the cells, contrary doesn't add the EVs in the cells

                save_image(current_mask, "./tmp", f"Current EVs in the single_image")
                save_image(component_mask, "./tmp", f"Current cells overlapped in the original_image")

                # Dilate the current mask to separate the future object fromt he connected components overalpped
                dilated_current_mask = dilation(current_mask.astype(int), square(3))
                image[component_mask] = 0
                # Ensure the not toruching regions before adding the current component
                component_mask = component_mask ^ dilated_current_mask
                image[component_mask] = component_label
                image[current_mask] =  next_usable_label
                next_usable_label += 1
                
                save_image(image, "./tmp", f"Current image updated")
                
        else:
            # The current object is not present nor overlapped iwth the connected componesnts in the base image
            image[current_mask] = next_usable_label
            next_usable_label += 1 # udaprte the label fopr next insertion

    image = measure.label(image, background=0)
    return image

    
def get_maximum_label(image):
    """
    Calculates the maximum label value present in a labeled image.

    This function takes a labeled image (assumed to be a 2D NumPy array) and
    returns the maximum label value (plus 1) found in the image. This is useful
    when working with connected component analysis where labels are assigned
    to each connected component.

    Args:
        image (numpy.ndarray): The labeled image. Assumed to be a 2D array
            where each pixel value represents the label of the corresponding
            connected component.

    Returns:
        int: The maximum label value present in the image.
            This value can be used for creating new unique labels during image
            processing tasks.
    """

    # Get the unique label values in the image
    unique_labels = np.unique(image)
    # Calculate the maximum label value
    max_label = np.max(unique_labels)
    return max_label


def get_overlapping_components(original_image, marker):
    """
    Extracts labeled connected components that overlap a marker mask.

    This function takes an original image containing labeled connected components,
    a marker mask representing a small object, and an optional overlap threshold
    (defaults to 1.0 for complete overlap). It returns the label and mask of
    connected components in the original image that have a certain level of overlap
    with the marker mask.

    Args:
        original_image (numpy.ndarray): The original image containing labeled
            connected components (assumed to be a 2D array where each pixel value
            represents the label of the corresponding connected component).
        marker (numpy.ndarray): The marker mask representing the small object
            (assumed to be a 2D binary mask where non-zero pixels indicate the
            presence of the marker).
        overlap_threshold (float, optional): The minimum overlap ratio between
            the marker and a connected component to consider it overlapping.
            Defaults to 1.0 (complete overlap).

    Returns:
        tuple: A tuple containing two elements:
            - int: The label for the overlapping connected component.
            - numpy.ndarray: A mask of the searched connected component.
            The mask has the same shape as the original image and is a boolean 
            NumPy array.

    Raises:
        ValueError: If the marker is not a NumPy array of booleans.
"""

    # Ensure marker has boolean data type (efficient overlap calculation)
    if marker.dtype != bool:
        raise ValueError("Mask of the marker has to be boolean.")

    # Label connected components in the original image
    labels = label(original_image)
    # Get unique labels present in the image
    unique_labels = np.unique(labels)

    # Loop through unique labels
    for label_value in unique_labels:
        if label_value == 0:  # Skip background label
            continue

        # Create mask for the current label
        current_mask = (labels == label_value)

        # Calculate the overlap ratio between the marker and the current component
        overlap_ratio = np.sum(current_mask & marker) / np.sum(marker)
        return label_value, current_mask, overlap_ratio

    return None, None, None


def get_partially_covered_regions(labeled_image, mask):   
    """
    Identifies connected regions in an image that are partially covered by a mask.

    Args:
        image: A numpy array representing the image containing labeled components (integers).
                Background should be represented by 0.
        mask: A numpy array representing the mask (boolean or integer, True/1 for mask region).

    Returns:
        A list of numpy arrays, where each array represents a connected region in the
        image that is partially covered by the mask.
    """

    # Store partially covered regions
    partially_covered_regions = []

    # Iterate through unique labels
    for label in np.unique(labeled_image):

        if label == 0:  # Skip background label
            continue

        # Get mask for current label
        mask_current_label = labeled_image == label

        # Check for any overlap with the mask
        overlap = np.any(mask_current_label & mask)

        # Extract entire region if there's overlap
        if overlap:
            return mask_current_label
    return None


def filter_regions_by_size(mask, min_dim = 1, max_dim = 3000):
    """
    Filters a labeled mask image, removing regions with size outside a specified range.

    Args:
        mask: A 2D NumPy array representing the labeled mask image (positive integer labels).
        min_dim: The minimum allowed dimension (pixels) for a region to be kept (inclusive).
        max_dim: The maximum allowed dimension (pixels) for a region to be kept (inclusive).

    Returns:
        A new 2D NumPy array representing the filtered mask image, where regions
        violating the size constraints are removed.
    """

    # Ensure consistent data types
    #mask = mask.astype(np.int32)  # Ensure integer type for calculations

    # Find connected components and get their labels and counts
    labels, counts = np.unique(mask, return_counts=True)

    # Validate input parameters (optional but recommended)
    if min_dim < 1:
        raise ValueError("Minimum dimension must be greater than or equal to 1.")
    if max_dim < min_dim:
        raise ValueError("Maximum dimension must be greater than or equal to minimum dimension.")

    # Filter labels based on size constraints
    filtered_counts = []
    for count in counts:
        if count > min_dim and count < max_dim:
            filtered_counts.append(True)
        else:
            filtered_counts.append(False)

    # Create a mask with only the filtered labels
    filtered_mask = np.zeros_like(mask)
    for idx, val in enumerate(filtered_counts):

        if val == True:
            filtered_mask[mask == labels[idx]] = labels[idx]
    return filtered_mask
