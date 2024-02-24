import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.segmentation import watershed
from skimage import measure
from skimage.feature import peak_local_max, canny
from skimage.morphology import binary_closing

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
  return min_area


def border_cell_distance_post_processing(border_prediction, cell_prediction, args):
    """ Post-processing for distance label (cell + neighbor distances) prediction.

    :param border_prediction: Neighbor distance prediction.
        :type border_prediction:
    :param cell_prediction: Cell distance prediction.
        :type cell_prediction:
    :param args: Post-processing settings (th_cell, th_seed, n_splitting, fuse_z_seeds).
        :type args:
    :param input_3d: True (3D data), False (2D data).
        :type input_3d: bool
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

    # Marker-based watershed
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


