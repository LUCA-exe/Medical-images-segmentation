import hashlib
import json
import math
import numpy as np
import os
import tifffile as tiff

from pathlib import Path
from random import shuffle, random
from scipy.ndimage import gaussian_filter
import shutil
from skimage.measure import regionprops
from skimage.morphology import binary_closing, binary_opening
from skimage.transform import rescale

from training.train_data_representations import distance_label_2d
from net_utils.utils import get_nucleus_ids, write_file


def adjust_dimensions(crop_size, *imgs):
    """ Adjust dimensions so that only 'complete' crops are generated.

    :param crop_size: Size of the (square) crops.
        :type crop_size: int
    :param imgs: Images to adjust the dimensions.
        :type imgs:
    :return: img with adjusted dimension.
    """

    img_adj = []

    # Add pseudo color channels
    for img in imgs:
        img = np.expand_dims(img, axis=-1)

        pads = []
        for i in range(2):
            if img.shape[i] < crop_size:
                pads.append((0, crop_size - (img.shape[i] % crop_size)))
            elif img.shape[i] == crop_size:
                pads.append((0, 0))
            else:
                if (img.shape[i] % crop_size) < 0.075 * img.shape[i]:
                    idx_start = (img.shape[i] % crop_size) // 2
                    idx_end = img.shape[i] - ((img.shape[i] % crop_size) - idx_start)
                    if i == 0:
                        img = img[idx_start:idx_end, ...]
                    else:
                        img = img[:, idx_start:idx_end, ...]
                    pads.append((0, 0))
                else:
                    pads.append((0, crop_size - (img.shape[i] % crop_size)))

        img = np.pad(img, (pads[0], pads[1], (0, 0)), mode='constant')

        img_adj.append(img)

    return img_adj


def close_mask(mask, apply_opening=False, kernel_closing=np.ones((10, 10)), kernel_opening=np.ones((10, 10))):
    """ Morphological closing of STs.

    :param mask: Segmentation mask (gold truth or silver truth).
        :type mask: numpy array.
    :param apply_opening: Apply opening or not (basically needed to correct slices from 3D silver truth).
        :type apply_opening: bool.
    :param kernel_closing: Kernel for closing.
        :type kernel_closing: numpy array.
    :param kernel_opening: Kernel for opening.
        :type kernel_opening: numpy array.
    :return: Closed (and opened) mask.
    """

    # Get nucleus ids and close/open the nuclei separately
    nucleus_ids = get_nucleus_ids(mask)
    hlabel = np.zeros(shape=mask.shape, dtype=mask.dtype)
    for nucleus_id in nucleus_ids:
        nucleus = mask == nucleus_id
        # Close nucleus gaps
        nucleus = binary_closing(nucleus, kernel_closing)
        # Remove small single not connected pixels
        if apply_opening:
            nucleus = binary_opening(nucleus, kernel_opening)
        hlabel[nucleus] = nucleus_id.astype(mask.dtype)

    return hlabel


def copy_train_data(source_path, target_path, idx):
    """  Copy generated training data crops.

    :param source_path: Directory containing the training data crops.
        :type source_path: pathlib Path object
    :param target_path: Directory to copy the training data crops into.
        :type target_path: pathlib Path Object
    :param idx: path/id of the training data crops to copy.
        :type idx: pathlib Path Object
    :return: None
    """
    shutil.copyfile(str(source_path / "img_{}.tif".format(idx)), str(target_path / "img_{}.tif".format(idx)))
    shutil.copyfile(str(source_path / "dist_cell_{}.tif".format(idx)), str(target_path / "dist_cell_{}.tif".format(idx)))
    shutil.copyfile(str(source_path / "dist_neighbor_{}.tif".format(idx)), str(target_path / "dist_neighbor_{}.tif".format(idx)))
    shutil.copyfile(str(source_path / "mask_{}.tif".format(idx)), str(target_path / "mask_{}.tif".format(idx)))
    return


def copy_train_set(source_path, target_path, mode='GT'):
    """  Copy generated training data sets (train and val).

    :param source_path: Directory containing the training data sets.
        :type source_path: pathlib Path object.
    :param target_path: Directory to copy the training data sets into.
        :type target_path: pathlib Path Object
    :param mode: 'GT' deletes possibly existing train and val directories.
        :type mode: str
    :return: None
    """

    if mode == 'GT':
        os.rmdir(str(target_path / 'train'))
        os.rmdir(str(target_path / 'val'))
        shutil.copytree(str(source_path / 'train'), str(target_path / 'train'))
        shutil.copytree(str(source_path / 'val'), str(target_path / 'val'))
    else:
        shutil.copytree(str(source_path / 'train'), str(target_path / 'train_st'))
        shutil.copytree(str(source_path / 'val'), str(target_path / 'val_st'))


def downscale(img, scale, order=2, aa=None):
    """ Downscale image and segmentation ground truth.

    :param img: Image to downscale
        :type img:
    :param scale: Scale for downscaling.
        :type scale: float
    :param order: Order of the polynom used.
        :type order: int
    :param aa: apply anti-aliasing (not recommended for the masks).
        :type aa: bool
    :return: downscale images.
    """
    if len(img.shape) == 3:
        scale_img = (1, scale, scale)
    else:
        scale_img = (scale, scale)
    img = rescale(img, scale=scale_img, order=order, anti_aliasing=aa, preserve_range=True).astype(img.dtype)

    return img


def foi_correction_train(cell_type, mode, *imgs):
    """ Field of interest correction (see
    https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf and
    https://public.celltrackingchallenge.net/documents/Annotation%20procedure.pdf )

    For the GTs, the foi correction differs a bit for some cell types since some GT training data sets were already
    fixed before we have seen the need for the foi correction. However, this should not affect the results
    but affects the cropping and fully annotated crop selection and therefore is needed for reproducibility of our sets.

    :param cell_type: Cell type / dataset (needed for filename).
        :type cell_type: str
    :param mode: 'GT' gold truth, 'ST' silver truth.
        :type mode: str
    :param imgs: Images to adjust the dimensions.
        :type imgs:
    :return: foi corrected images.
    """

    if mode == 'GT':
        if cell_type in ['Fluo-C2DL-Huh7', 'Fluo-N2DH-GOWT1', 'Fluo-N3DH-CHO', 'PhC-C2DH-U373']:
            E = 50
        elif cell_type in ['Fluo-N2DL-HeLa', 'PhC-C2DL-PSC', 'Fluo-C3DL-MDA231']:
            E = 25
        else:
            E = 0
    else:
        if cell_type in ['Fluo-C2DL-Huh7', 'Fluo-N2DH-GOWT1', 'Fluo-N3DH-CHO', 'PhC-C2DH-U373', 'Fluo-C3DH-H157',
                         'Fluo-N3DH-CHO']:
            E = 50
        elif cell_type in ['Fluo-N2DL-HeLa', 'PhC-C2DL-PSC', 'Fluo-C3DL-MDA231']:
            E = 25
        else:
            E = 0

    img_corr = []

    for img in imgs:
        if len(img.shape) == 2:
            img_corr.append(img[E:img.shape[0] - E, E:img.shape[1] - E])
        else:
            img_corr.append(img[:, E:img.shape[1] - E, E:img.shape[2] - E])

    return img_corr


def generate_data(img, mask, tra_gt, td_settings, cell_type, mode, subset, frame, path, crop_idx=0, slice_idx=None):
    """ Calculate cell and neighbor distances and create crops.

    :param img: Image.
        :type img: numpy array
    :param mask: (Segmentation) Mask / label image (intensity coded).
        :type mask: numpy array
    :param tra_gt: Tracking ground truth (needed to evaluate if all cells are annotated).
        :type tra_gt: numpy array
    :param td_settings: training data creation settings (search radius, crop size ...).
        :type td_settings: dict
    :param cell_type: Cell type / dataset (needed for filename).
        :type cell_type: str
    :param mode: 'GT' gold truth, 'ST' silver truth.
        :type mode: str
    :param subset: Subset from which the image and segmentation mask come from (needed for filename) ('01', '02').
        :type subset: str
    :param frame: Frame of the time series (needed for filename).
        :type frame: str
    :param path: Path to save the generated crops.
        :type path: Pathlib Path object
    :crop_idx: Index to count generated crops and break the data generation if the maximum ammount is reached (for STs)
        :type crop_idx: int
    :param slice_idx: Slice index (for 3D data).
        :type slice_idx: int
    :return: None
    """

    # Calculate train data representations
    cell_dist, neighbor_dist = distance_label_2d(label=mask,
                                                 cell_radius=int(np.ceil(0.5 * td_settings['max_mal'])),
                                                 neighbor_radius=td_settings['search_radius'], 
                                                 disk_radius = td_settings['radius_disk'])

    # Adjust image dimensions for appropriate cropping
    img, mask, cell_dist, neighbor_dist, tra_gt = adjust_dimensions(td_settings['crop_size'], img, mask, cell_dist,
                                                                    neighbor_dist, tra_gt)

    # Cropping
    nx, ny = math.floor(img.shape[1] / td_settings['crop_size']), math.floor(img.shape[0] / td_settings['crop_size'])
    for y in range(ny):
        for x in range(nx):

            # Crop
            img_crop, mask_crop, cell_dist_crop, neighbor_dist_crop, tra_gt_crop = get_crop(x, y, td_settings['crop_size'],
                                                                                            img, mask, cell_dist,
                                                                                            neighbor_dist, tra_gt)
            # Get crop name
            if slice_idx is not None:
                crop_name = '{}_{}_{}_{}_{:02d}_{:02d}_{:02d}.tif'.format(cell_type, mode, subset, frame, slice_idx, y, x)
            else:
                crop_name = '{}_{}_{}_{}_{:02d}_{:02d}.tif'.format(cell_type, mode, subset, frame, y, x)

            # Check cell number TRA/SEG
            tr_ids, mask_ids = get_nucleus_ids(tra_gt_crop), get_nucleus_ids(mask_crop)
            if np.sum(mask_crop[10:-10, 10:-10, 0] > 0) < td_settings['min_area']:  # only cell parts / no cell
                continue
            if len(mask_ids) == 1:  # neighbor may be cut from crop --> set dist to 0
                neighbor_dist_crop = np.zeros_like(neighbor_dist_crop)
            if np.sum(img_crop == 0) > (0.66 * img_crop.shape[0] * img_crop.shape[1]):  # almost background
                # For (e.g.) GOWT1 cells a lot of 0s are in the image
                if np.min(img_crop[:100, :100, ...]) == 0:
                    if np.sum(gaussian_filter(np.squeeze(img_crop), sigma=1) == 0) > (0.66 * img_crop.shape[0] * img_crop.shape[1]):
                        continue
                else:
                    continue
            if np.max(cell_dist_crop) < 0.8:
                continue

            # Remove only partially visible cells in mask for better comparison with tra_gt
            props_crop, n_part = regionprops(mask_crop), 0
            for cell in props_crop:
                if mode == 'GT' and cell.area <= 0.1 * td_settings['min_area'] and td_settings['scale'] == 1:  # needed since tra_gt seeds are smaller
                    n_part += 1
            if (len(mask_ids) - n_part) >= len(tr_ids):  # A: all cells annotated
                crop_quality = 'A'
            elif (len(mask_ids) - n_part) >= 0.8 * len(tr_ids):  # >= 80% of the cells annotated
                crop_quality = 'B'
            else:
                continue

            # Save only needed crops for kit-sch-ge split
            if td_settings['used_crops']:
                if isinstance(slice_idx, int):
                    if not ([subset, frame, '{:02d}'.format(slice_idx), '{:02d}'.format(y), '{:02d}'.format(x), 'train'] in td_settings['used_crops']) \
                            and not ([subset, frame, '{:02d}'.format(slice_idx), '{:02d}'.format(y), '{:02d}'.format(x), 'val'] in td_settings['used_crops']):
                        continue
                else:
                    if not ([subset, frame, '{:02d}'.format(y), '{:02d}'.format(x), 'train'] in td_settings['used_crops']) \
                            and not ([subset, frame, '{:02d}'.format(y), '{:02d}'.format(x), 'val'] in td_settings['used_crops']):
                        continue

            # Save the images
            tiff.imsave(str(path / crop_quality / 'img_{}'.format(crop_name)), img_crop)
            tiff.imsave(str(path / crop_quality / 'mask_{}'.format(crop_name)), mask_crop)
            tiff.imsave(str(path / crop_quality / 'dist_cell_{}'.format(crop_name)), cell_dist_crop)
            tiff.imsave(str(path / crop_quality / 'dist_neighbor_{}'.format(crop_name)), neighbor_dist_crop)

            # Increase crop counter
            crop_idx += 1

            if mode == 'ST' and crop_idx > td_settings['st_limit']:
                # In the first release of this code, for 3D ST data all possible crops have been created and selected
                # afterwards. So, there is a small discrepancy but the old version needed too much time.
                return crop_idx

    return crop_idx


def get_crop(x, y, crop_size, *imgs):
    """ Get crop from image.

    :param x: Grid position (x-dim).
        :type x: int
    :param y: Grid position (y-dim).
        :type y: int
    :param crop_size: size of the (square) crop
        :type crop_size: int
    :param imgs: Images to crop.
        :type imgs:
    :return: img crop.
    """

    imgs_crop = []

    for img in imgs:
        img_crop = img[y * crop_size:(y + 1) * crop_size, x * crop_size:(x + 1) * crop_size, :]
        imgs_crop.append(img_crop)

    return imgs_crop


def get_annotated_gt_frames(path_train_set):
    """ Get GT frames (so that these frames are not used for STs in GT+ST setting).

    :param path_train_set: path to Cell Tracking Challenge training data sets.
        :type path_train_set: pathlib Path object
    :return: List of available GT frames
    """

    seg_gt_ids_01 = sorted((path_train_set / '01_GT' / 'SEG').glob('*.tif'))
    seg_gt_ids_02 = sorted((path_train_set / '02_GT' / 'SEG').glob('*.tif'))

    annotated_gt_frames = []
    for seg_gt_id in seg_gt_ids_01:
        if len(seg_gt_id.stem.split('_')) > 2:  # only slice annotated
            annotated_gt_frames.append("01_{}".format(seg_gt_id.stem.split('_')[2]))
        else:
            annotated_gt_frames.append("01_{}".format(seg_gt_id.stem.split('man_seg')[-1]))
    for seg_gt_id in seg_gt_ids_02:
        if len(seg_gt_id.stem.split('_')) > 2:  # only slice annotated
            annotated_gt_frames.append("02_{}".format(seg_gt_id.stem.split('_')[2]))
        else:
            annotated_gt_frames.append("02_{}".format(seg_gt_id.stem.split('man_seg')[-1]))

    return annotated_gt_frames


def get_file(path):
    """ Load json file.

    :param path: Path to the json file to load.
        :type path: pathlib Path object
    """
    with open(path) as f:
        file = json.load(f)
    return file


def get_kernel(cell_type):
    """ Get kernel for morphological closing and opening operation.

    :param cell_type: Cell type/dataset for which the kernel is needed.
        :type cell_type: str
    :return: kernel for closing, kernel for opening
    """

    # Larger cells need larger kernels (could be coupled with mean major axis length in future)
    if cell_type in ['Fluo-C3DH-H157', 'Fluo-N3DH-CHO']:
        kernel_closing = np.ones((20, 20))
        kernel_opening = np.ones((20, 20))
    elif cell_type == 'Fluo-C3DL-MDA231':
        kernel_closing = np.ones((3, 3))
        kernel_opening = np.ones((3, 3))
    elif cell_type == 'Fluo-N3DH-CE':
        kernel_closing = np.ones((15, 15))
        kernel_opening = np.ones((15, 15))
    else:
        kernel_closing = np.ones((10, 10))
        kernel_opening = np.ones((10, 10))

    return kernel_closing, kernel_opening


def get_mask_ids(log, path_data, ct, mode, split, st_limit):
    """ Get ids of the masks of a specific cell type/dataset.

    :param path_data: Path to the directory containing the Cell Tracking Challenge training sets.
        :type path_data: Pathlib Path object.
    :param ct: cell type/dataset.
        :type ct: str
    :param mode: 'GT' use gold truth, 'ST' use silver truth.
        :type mode: str
    :param split: Use a single ('01'/'02) or both subsets ('01+02') to select the training data from. 'kit-sch-ge'
            reproduces the training data of the Cell Tracking Challenge team KIT-Sch-GE.
        :type split: str
    :param st_limit: Maximum amount of ST crops to create.
        :type st_limit: int
    :return: mask ids, increment for selecting slices.
    """
    log.debug(f"Preparing mask ids ..")

    # Get mask ids
    mask_ids_01, mask_ids_02 = [], []
    if '01' in split or split == 'kit-sch-ge':
        mask_ids_01 = sorted((path_data / ct / '01_{}'.format(mode) / 'SEG').glob('*.tif'))
    if '02' in split or split == 'kit-sch-ge':
        mask_ids_02 = sorted((path_data / ct / '02_{}'.format(mode) / 'SEG').glob('*.tif'))
    mask_ids = mask_ids_01 + mask_ids_02

    # NOTE: In the original work it is present processing 'ST' and 3D annotations.
    if not split == 'kit-sch-ge':
        shuffle(mask_ids)
    log.debug(f"Mask collected (shuffled for training purpose) are: {mask_ids}")
    return mask_ids

# NOTE: Adjusted 'td_settings' for my dataset ('search_radius' it is interesting to study)
def get_td_settings(log, mask_id_list, crop_size, evs_presence = False):
    """ Get settings for the training data generation.

    :param mask_id_list: List of all segmentation GT ids (list of pathlib Path objects).
        :type mask_id_list: list
    :return: dict with keys 'search_radius', 'min_area', 'max_mal', 'scale', 'crop_size'.
    """
    log.debug(f"Computing the training data properties ..")
    # Load all GT and get cell parameters to adjust parameters for the distance transform calculation
    diameters, major_axes, areas = [], [], []
    for mask_id in mask_id_list:
        mask = tiff.imread(str(mask_id))
        props = regionprops(mask)
        # NOTE: In the original repository It was supported the computation on 3D images slices.
        for cell in props:
            major_axes.append(cell.major_axis_length)
            diameters.append(cell.equivalent_diameter)
            areas.append(cell.area)

    # Get maximum and minimum diameter and major axis length and set search radius for distance transform
    max_diameter, min_diameter = int(np.ceil(np.max(np.array(diameters)))), int(np.ceil(np.min(np.array(diameters))))
    mean_diameter, std_diameter = int(np.ceil(np.mean(np.array(diameters)))), int(np.std(np.array(diameters)))
    max_mal = int(np.ceil(np.max(np.array(major_axes)))) # Take the approximated 'ellipse' maximum axe.
    min_area = int(0.95 * np.floor(np.min(np.array(areas))))
    search_radius = mean_diameter + std_diameter

    # Some simple heuristics for large cells. If enough data are available scale=1 should work in most cases
    if max_diameter > 200 and min_diameter > 35: # TODO: Study this function, can be tweaked accordingly/optimized/automated
        if max_mal > 2 * max_diameter:  # very longish and long cells not made for neighbor distance
            scale = 0.5
            search_radius = min_diameter + 0.5 * std_diameter
        elif max_diameter > 300 and min_diameter > 60:
            scale = 0.5
        elif max_diameter > 250 and min_diameter > 50:
            scale = 0.6
        else:
            scale = 0.7
        min_area = (scale ** 2) * min_area
        max_mal = int(np.ceil(scale * max_mal))
        search_radius = int(np.ceil(scale * search_radius))

    else:
        scale = 1

    # NOTE: Check if there is evs in the data: if yes increment/decrement the search radius by a fixed factor - can be optimized
    if evs_presence: search_radius = search_radius * 1.5 # More the increment, more the neighbor considered.

    properties_dict = {'search_radius': search_radius,
            'radius_disk': 3, #TODO: Automatize in respect to the cells dimensions - fixed for now.
            'min_area': min_area,
            'max_mal': max_mal,
            'scale': scale,
            'crop_size': crop_size}
    
    log.debug(f"Current suggested properties gathered for data generation: {properties_dict}")
    return properties_dict


def get_train_val_split(img_idx_list, b_img_idx_list):
    """ Split generated training data crops into training and validation set.
    :param img_idx_list: List of image indices/paths (list of Pathlib Path objects).
        :type img_idx_list: list
    :param b_img_idx_list: List of image indices/paths which were classified as 'B' (list of Pathlib Path objects).
        :type b_img_idx_list: list
    :return: dict with ids for training and ids for validation.
    """

    img_ids_stem = []
    for idx in img_idx_list:
        img_ids_stem.append(idx.stem.split('img_')[-1])
    # Random 80%/20% split
    shuffle(img_ids_stem)
    train_ids = img_ids_stem[0:int(np.floor(0.8 * len(img_ids_stem)))]
    val_ids = img_ids_stem[int(np.floor(0.8 * len(img_ids_stem))):]
    # Add "B" quality only to train
    for idx in b_img_idx_list:
        train_ids.append(idx.stem.split('img_')[-1])
    # Train/val split
    train_val_ids = {'train': train_ids, 'val': val_ids}

    return train_val_ids


def get_used_crops(train_val_ids, mode='GT'):
    """ Get frames used in given training/validation split.

    :param train_val_ids: Training/validation split ids.
        :type train_val_ids: dict
    :param mode: 'GT' use gold truth, 'ST' use silver truth, 'GT+ST' use mixture of gold and silver truth.
        :type mode: str
    :return: used crops [subset, frame, (slice for 3D data), y grid position, x grid position].
    """

    used_crops = []

    if mode == 'GT+ST':
        # Only ST ids are saved since the GTs are just copied from the corresponding ground truth set
        sets = ['train_st', 'val_st']  # bad selection of train/val key names ...
    else:
        sets = ['train', 'val']
    for split_mode in sets:
        for idx in train_val_ids[split_mode]:
            if '2D' in idx:
                used_crops.append([idx.split('_')[-4], idx.split('_')[-3], idx.split('_')[-2], idx.split('_')[-1],
                                   split_mode])
            else:
                if idx.split('_')[-5] in ['GT', 'ST', "GT+ST"]:  # only frame annotated --> slice info has not been saved
                    used_crops.append([idx.split('_')[-4], idx.split('_')[-3], idx.split('_')[-2], idx.split('_')[-1],
                                       split_mode])
                else:
                    used_crops.append([idx.split('_')[-5], idx.split('_')[-4], idx.split('_')[-3], idx.split('_')[-2],
                                       idx.split('_')[-1], split_mode])

    return used_crops


def make_train_dirs(path):
    """ Make directories to save the created training data into.

    :param path: Path to the created training data sets.
        :type path: pathlib Path object.
    :return: None
    """

    Path.mkdir(path / 'A', parents=True, exist_ok=True)  # for high quality crops
    Path.mkdir(path / 'B', exist_ok=True)  # for good quality crops
    Path.mkdir(path / 'train', exist_ok=True)
    Path.mkdir(path / 'val', exist_ok=True)

    return None


def remove_st_with_gt_annotation(st_ids, annotated_gt_frames):
    """ Remove ST crops which are taken from frames which have also a GT available.

    :param st_ids: List of pathlib Path objects.
        :type st_ids: list
    :param annotated_gt_frames: Annotated GT frames.
        :type annotated_gt_frames: list
    return: None
    """

    # gt_frames: "subset_frame"
    for st_id in st_ids:
        frame = "{}_{}".format(st_id.stem.split('ST_')[-1].split('_')[0], st_id.stem.split('ST_')[-1].split('_')[1])
        if frame in annotated_gt_frames:
            files_to_remove = list(st_id.parent.glob("*{}".format(st_id.name.split('img')[-1])))
            for idx in files_to_remove:
                os.remove(idx)
    return None


def create_ctc_training_sets(log, path_data, mode, cell_type, split='01+02', crop_size=320, st_limit=280,
                             n_max_train_gt_st=150, n_max_val_gt_st=30, min_a_images=30):
    """ Create training sets for Cell Tracking Challenge data.

    In the new version of this code, 2 Fluo-C3DL-MDA231 crops and 1 Fluo-C3DH-H157 crop differ slightly from the
    original kit-sch-ge training sets. This should not make any difference for the model training: in the
    Fluo-C3DH-H157 case only one pixel differs in the segmentation mask, and in the other case just the next slice is
    taken instead of the original slice.

    :param path_data: Path to the directory containing the Cell Tracking Challenge training sets.
        :type path_data: Pathlib Path object.
    :param mode: 'GT' gold truth, 'ST' silver truth, 'GT+ST': mixture of gold truth and silver truth.
        :type mode: str
    :param cell_type_list: List of cell types to include in the training data set. If more than 1 cell type a unique
            name of the training is built (hash). Special case: cell_type_list = ['all'] (see code below).
        :type cell_type_list: list
    :param split: Use a single ('01'/'02) or both subsets ('01+02') to select the training data from. 'kit-sch-ge'
            reproduces the training data of the Cell Tracking Challenge team KIT-Sch-GE.
        :type split: str
    :param crop_size: Size of the generated crops (square).
        :type crop_size: int
    :param st_limit: Maximum amount of ST crops to create (reduces computation time.
        :type st_limit: int
    :param n_max_train_gt_st: Maximum number of gold truths per cell type in the training sets of training datasets
            consisting of multiple cell types.
        :type n_max_train_gt_st: int
    :param n_max_val_gt_st: Maximum number of gold truths per cell type in the validation sets of training datasets
            consisting of multiple cell types.
        :type n_max_val_gt_st: int
    :return: None
    """

    if split == 'kit-sch-ge':
        st_limit = 280  # ORIGINAL NOTE: needed for reproducibility (smaller values will just not work to reproduce that split)

    # Implemented for one cell_type (my 'dataset' variable)
    trainset_name = cell_type

    # Create needed training data sets - check if data set already exists
    path_trainset = path_data / "{}_{}_{}_{}".format(cell_type, mode, split, crop_size) # Name of the generated train set in the chosen dataset folder
    
    if len(list((path_trainset / 'train').glob('*.tif'))) > 0: # Check if the generated 'train set' already exist

        log.info(f"Training set {path_trainset.stem} already generated, returning to the main training loop")
        return None

    log.info('   ... create {} training set ...'.format(path_trainset.stem))
    make_train_dirs(path=path_trainset)

    # From original work: Load split if original 'kit-sch-ge' training sets should be reproduced.
    if split == 'kit-sch-ge':
        train_val_ids = get_file(path=Path(__file__).parent/'splits'/'ids_{}_{}.json'.format(cell_type, mode)) # File not present in my repository
        used_crops = get_used_crops(train_val_ids, mode)
    else:
        used_crops = []

    # Get ids of segmentation ground truth masks (GTs may not be fully annotated and STs may be erroneous)
    mask_ids = get_mask_ids(log, path_data=path_data, ct=cell_type, mode=mode, split=split, st_limit=st_limit)

    # Get settings for distance map creation
    td_settings = get_td_settings(log, mask_id_list=mask_ids, crop_size=crop_size)
    td_settings['used_crops'],  td_settings['st_limit'], td_settings['cell_type'] = used_crops, st_limit, str(cell_type) # Gathering additional information in the 'properties' dict.

    # Iterate through files and load images and masks (and TRA GT for GT mode)
    log.info(f"Starting loop over the loaded masks {mask_ids}")
    if td_settings['scale'] !=1: log.debug(f"Downsampling operation will be performed due to the suggested 'scale' value {td_settings['scale']}")
    for mask_id in mask_ids: # TODO: To parallelize
 
        frame = mask_id.stem.split('man_seg')[-1] # Just 2D images
        # Check if frame is needed to reproduce the kit-sch-ge training sets
        if used_crops and not any(e[1] == frame for e in used_crops): # TODO: To remove
            print(f"Should NOT JUST get here.. should exit the program")

        # Load image and mask and get subset from which they are
        log.debug(f"... working with {mask_id} mask ...")
        mask = tiff.imread(str(mask_id))
        subset = mask_id.parents[1].stem.split('_')[0]
        img = tiff.imread(str(mask_id.parents[2] / subset / "t{}.tif".format(frame)))

        # Adjust the number of channel of my 2D images.
        if len(img.shape) > 2: 
            img = np.sum(img, axis=2) # NOTE: Keep all object (Both EVs, cell boundaries and cell nucleus)

        # NOTE: TRA GT (fully annotated, no region information) to detect fully annotated mask GTs later
        if 'GT' in mode:
            tra_gt = tiff.imread(str(mask_id.parents[1] / 'TRA' / "man_track{}.tif".format(frame)))
        else:  # Do not use TRA GT to detect high quality STs (to be able to compare ST and GT results)
            raise TypeError("For now just 'GT' is supported!")

        # FOI correction: followin the standard CTC requests for the pixels near the border.
        img, mask, tra_gt = foi_correction_train(cell_type, mode, img, mask, tra_gt)

        # Downsampling
        if td_settings['scale'] != 1:
            img = downscale(img=img, scale=td_settings['scale'], order=2)
            mask = downscale(img=mask, scale=td_settings['scale'], order=0, aa=False)
            tra_gt = downscale(img=tra_gt, scale=td_settings['scale'], order=0, aa=False)

        # Normalization: min-max normalize image to [0, 65535] - Can be automated optimized
        img = 65535 * (img.astype(np.float32) - img.min()) / (img.max() - img.min())
        img = np.clip(img, 0, 65535).astype(np.uint16) # Kept the requested type 'unsigned' integer 16 bits

        # Calculate distance transforms, crop and classify crops into 'A' (fully annotated) and 'B' (>80% annotated)
        _ = generate_data(img=img, mask=mask, tra_gt=tra_gt, td_settings=td_settings, cell_type=cell_type,
                                      mode=mode, subset=subset, frame=frame, path=path_trainset)

    td_settings.pop('used_crops')
    if mode == 'GT':
        td_settings.pop('st_limit')
    else: 
        raise TypeError("For now just 'GT' is supported!")

    log.info(f"Saving suggested data generation settings in '{path_trainset}'")
    # Saved the 'info.json' file with settings for data tranformations
    write_file(td_settings, path_trainset / 'info.json')

    # Create train/val split
    img_ids, b_img_ids = sorted((path_trainset / 'A').glob('img*.tif')), []
    log.info(f"Number of 'A' quality patches are {len(img_ids)}")
    if mode == 'GT' and len(img_ids) <= min_a_images:  # Use also "B" quality images when too few "A" quality images are available - default kept as 30 elements
        log.debug(f"Using {len(b_img_ids)} 'B' quality images, the 'A' quality patches are less than {min_a_images}")
        b_img_ids = sorted((path_trainset / 'B').glob('img*.tif'))

    if not split == 'kit-sch-ge':
        log.info(f"Splitting train/val elements for training with standard 80% / 20%") # TODO: To add as argument to the main parser and train specific parser
        train_val_ids = get_train_val_split(img_ids, b_img_ids) # Get simple shuffled dict with train/val patches ids.
    
    log.debug(f"Train patches ids ({len(train_val_ids['train'])}): {train_val_ids['train']}")
    log.debug(f"Val patches ids ({len(train_val_ids['val'])}): {train_val_ids['val']}")

    # Copy images to train/val
    for train_mode in ['train', 'val']:

        for idx in train_val_ids[train_mode]:

            if (path_trainset / "A" / ("img_{}.tif".format(idx))).exists():
                source_path = path_trainset / "A"
            else:
                source_path = path_trainset / "B"
            copy_train_data(source_path, path_trainset / train_mode, idx)

    # NOTE: The original repository contained a beatiful method to merge multiple dataset/cell type patches avoiding a final 'unbalanced' train/val split
    log.info(f"Trainig dataset created correctly")
    return None
