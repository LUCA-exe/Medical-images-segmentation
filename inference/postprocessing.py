import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
from skimage.segmentation import watershed
from skimage import measure
from skimage.feature import peak_local_max, canny
from skimage.morphology import binary_closing
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

  # NOTE: Just override the min area component in case of my dataset for the firt testing
  min_area = 10
  return min_area


def border_cell_post_processing(border_prediction, cell_prediction, args):
    """ Post-processing WT for distance label (cell + neighbor distances continuos tensors) prediction.

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


def simple_binary_border_mask_post_processing(mask, binary_border, original_image, cell_distance, args, diff_between_channels = 0.2):
    """ Assignining different IDs in the final segmentation mask prediction just thresholded without watershed.

    :param mask: Binary mask prediction.

    :param args: Post-processing settings.
        :type args:
    :return: Instance segmentation mask.
    """
    # Simple parameters to control the thresholdings of markers and mask
    
    processed_mask = torch.diff(mask, dim=0)
    th_processed_mask = processed_mask > diff_between_channels
    binary_channel = 1
    
    # Qualitative debug
    save_image(np.squeeze(cell_distance), "./tmp", f"Original cell distances image")
    save_image(np.squeeze(mask[binary_channel, :, :]), "./tmp", f"Mask prediction (channel {binary_channel})")
    save_image(np.squeeze(binary_border[binary_channel, :, :]), "./tmp", f"Binary border prediciton (channel {binary_channel})")
    save_image(processed_mask, "./tmp", f"Processed mask")
    save_image(th_processed_mask, "./tmp", f"Thresholded Processed mask")
    exit(1)

    # Processing the binary mask with simple thresholding
    processed_mask = np.squeeze(mask[binary_channel, :, :] > th_mask)
    prediction_instance = measure.label(processed_mask)
    return np.squeeze(prediction_instance.astype(np.uint16))


# NOTE: Simple solution using a binary mask prediction for a post processing phase - evaluate if you have to use watershed or not
def seg_mask_post_processing(mask, binary_border, original_image, cell_distance, args):
    """ Assignining different IDs in the final segmentation mask prediction.

    :param mask: Binary mask prediction.

    :param args: Post-processing settings.
        :type args:
    :return: Instance segmentation mask.
    """

    # Simple parameters to control the thresholdings of markers and mask
    th_mask, th_seeds = 0.2, 0.4
    binary_channel = 1

    # Processing the binary mask with simple thresholding
    processed_mask = np.squeeze(mask[binary_channel, :, :] > th_mask)
    seeds = np.squeeze(mask[binary_channel, :, :] > th_seeds)


    # Apply watershed
    prediction_instance = watershed(image=-np.squeeze(cell_distance), markers=seeds, mask=processed_mask, watershed_line=False)
    prediction_instance = measure.label(prediction_instance)

    # Temporary - Debug 
    save_image(np.squeeze(cell_distance), "./tmp", f"Original cell distances image")
    save_image(np.squeeze(mask[binary_channel, :, :]), "./tmp", f"Mask prediction (channel {binary_channel})")
    save_image(np.squeeze(binary_border[binary_channel, :, :]), "./tmp", f"Binary border prediciton (channel {binary_channel})")
    save_image(processed_mask, "./tmp", f"Processed mask")
    save_image(seeds, "./tmp", f"Seeds")
    #save_image(np.squeeze(original_image), "./tmp", f"Original image")
    #save_image(prediction_instance, "./tmp", f"Final image using Watershed")
    exit(1)
    return np.squeeze(prediction_instance.astype(np.uint16))


def remove_false_positive_by_overlapping(prediction, single_channel_prediction, min_cell_area = 2000):
    # Take two images as numpy array

    # Deep copy the original prediction
    prediction = copy.deepcopy(prediction)
    sc_mask = single_channel_prediction > 0
    # Loop over every area in the original input images
    for reg_prop in measure.regionprops(prediction):

        if reg_prop.area < min_cell_area:
            # It is usually an EVs
            
            # Get mask for the current position of the "predicted" EVs
            curr_mask = prediction == reg_prop.label 

            # Check if there's any overlap with objects in image2
            overlap = np.any(curr_mask * sc_mask)

            # Update the prediction if not overlap is present
            if not overlap:
                prediction[curr_mask] = 0  # Remove the region in the predicted image in the predicted image
    # Adjusted prediction
    return prediction


# WORK IN PROGRESS: This function can be seen as wrapper and feature fusion functions
def sc_border_cell_post_processing(border_prediction, cell_prediction, sc_border_prediction, sc_cell_prediction, args):
    """ Post-processing WT enhanced with Fusion prediction for distance label (cell + neighbor distances continuos tensors) plus single-channgel prediction.

    :param border_prediction: Neighbor distance prediction.
        :type border_prediction:
    :param cell_prediction: Cell distance prediction.
        :type cell_prediction:
    :param args: Post-processing settings (th_cell, th_seed, n_splitting, fuse_z_seeds).
        :type args:
    :return: Instance segmentation mask.
    """
    prediction_instance, borders = border_cell_post_processing(border_prediction, cell_prediction, args)
    sc_prediction_instance, sc_borders = border_cell_post_processing(sc_border_prediction, sc_cell_prediction, args)

    # DEBUG
    #prediction_instance[prediction_instance > 0] = 200
    #sc_prediction_instance[sc_prediction_instance > 0] = 200
    save_image(prediction_instance, "./tmp", f"Original final results")
    save_image(sc_prediction_instance, "./tmp", f"Single channel results")

    # TO TEST
    processed_prediction = remove_false_positive_by_overlapping(prediction_instance, sc_prediction_instance)
    # TO implement adding new regions from the single channel 
    save_image(processed_prediction, "./tmp", f"Original final results")
    exit(1)
    return None
