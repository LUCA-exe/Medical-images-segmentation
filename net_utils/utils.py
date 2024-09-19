"""
This file provides additional util functions for both train/eval pipelines
as laoding the model weights, save/overwrite the metrics and results.
"""
import logging
import json
import numpy as np
import shutil
import os
from pathlib import Path
import tifffile as tiff
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import torch
from torch import nn
from typing import Dict, Union, List
import pandas as pd

from net_utils import unets


def create_model_architecture(log: logging.Logger, model_config: Dict[str, Union[str, int]], device: int, num_gpus: int, pre_train: bool = False) -> nn.Module:
    """
    Call the public method in the './unets.py' module to return the correct
    NN class.

    Returns:
        Custom neural networks builded inheriting from the 'nn.Module'
        and using 'nn.layers/functionals'. 
    """
    
    if pre_train:
        unet_type = 'AutoU' # Get CNN (U-Net without skip connections)
    else:
        unet_type = model_config['architecture'][0]

    net = unets.build_unet(log, 
                           unet_type = unet_type,
                           act_fun = model_config['architecture'][2],
                           pool_method = model_config['architecture'][1],
                           normalization = model_config['architecture'][3],
                           device = device,
                           num_gpus = num_gpus,  # FIXME: Use the 'num_gpus' key-value pair in the 'model_config' dict.
                           ch_in = 1,
                           ch_out = 1,
                           filters = model_config['architecture'][4],
                           detach_fusion_layers = model_config['architecture'][5],
                           softmax_layer = model_config['architecture'][6])
    return net


def save_training_loss(loss_labels: List[str], 
                       train_loss: List[List[float]], 
                       val_loss: List[List[float]], 
                       second_run: bool, 
                       path_models: Path, 
                       config: Dict, 
                       tot_time: float, 
                       tot_epochs: int):
    """
    Get the training loss and save it in a formatted '*.txt' file.
    In this function key-value are added to the 'config' dict. containig the 
    current specifics of the model/trainig phase.

    Args:
        loss_labels: List of columns correspondent to the loss values passed.
        train_loss: Every i-th entry of this list correspond to the value of 
        that i-th epochs of the total losses gathered during training.
        val_loss: Every i-th entry of this list correspond to the value of 
        that i-th epochs of the losses gathered during validation.
        second_run: Temporary flag to differentiate between end-to-end 
        trainig and fine-tuning.
        path_models: Folder path in which to save the current model.
        config: Hashmap containing the additional information on the 
        current model in training.
        tot_time: Floating value indicating the total time used for the overall
        training phase.
        tot_epochs: Integer representings the total number of epochs. Usefull
        to record in case of early stops to avoid overfitting.
    """

    # Un-pack the losses in the original variables
    train_total_loss, train_loss_border, train_loss_cell, train_loss_mask, train_loss_binary_border = zip(*train_loss) 
    val_total_loss, val_loss_border, val_loss_cell, val_loss_mask, val_loss_binary_border = zip(*val_loss) 
    
    stats = np.transpose(np.array([list(range(1, len(train_loss) + 1)), train_total_loss, train_loss_border, train_loss_cell, train_loss_mask, train_loss_binary_border, val_total_loss, val_loss_border, val_loss_cell, val_loss_mask, val_loss_binary_border]))
    try:

        if second_run:
            np.savetxt(fname=str(path_models / (config['run_name'] + '_2nd_loss.txt')), X=stats,
                    fmt=['%3i', '%2.5f', '%2.5f', '%2.5f', '%2.5f', '%2.5f', '%2.5f', '%2.5f', '%2.5f', '%2.5f', '%2.5f'],
                    header='Epoch, training total loss, training border loss, training cell loss, training mask loss, training binary border loss, validation total loss, validation border loss, validation cell loss, validation mask loss, validation binary border loss', delimiter=',')
            config['training_time_run_2'], config['trained_epochs_run2'] = tot_time, tot_epochs + 1

        else:
            np.savetxt(fname=str(path_models / (config['run_name'] + '_loss.txt')), X=stats,
                    fmt=['%3i', '%2.5f', '%2.5f', '%2.5f', '%2.5f', '%2.5f', '%2.5f', '%2.5f', '%2.5f', '%2.5f', '%2.5f'],
                    header='Epoch, training total loss, training border loss, training cell loss, training mask loss, training binary border loss, validation total loss, validation border loss, validation cell loss, validation mask loss, validation binary border loss', delimiter=',')
            
            # FIXME: Move to the calling function.
            config['training_time'], config['trained_epochs'] = tot_time, tot_epochs + 1
        print(f"Training losses saved corretly and current configuration parameters updated!")

    except Exception as e:
        raise Exception(f"Unexpected error: {e}")
    return None


def save_inference_final_images(result_path, file_id, prediction_instance):
    # Save final inferred image.

    tiff.imwrite(str(result_path / ('mask' + file_id)), prediction_instance)
    return None


def save_inference_raw_images(result_path, file_id, prediction_cell_batch, prediction_border_batch, border):
    # Called if "args.save_raw_pred" is true: save the raw outputs of the model during inference time
    
    tiff.imwrite(str(result_path / ('cell' + file_id)), prediction_cell_batch[h, ..., 0].astype(np.float32))
    tiff.imwrite(str(result_path / ('raw_border' + file_id)), prediction_border_batch[h, ..., 0].astype(np.float32))
    tiff.imwrite(str(result_path / ('border' + file_id)), border.astype(np.float32))
    return None


def save_current_model_state(config, net, path_models):
    # The state dict of data parallel (multi GPU) models need to get saved in a way that allows to
    # load them also on single GPU or CPU

    try:
        if config['num_gpus'] > 1:
            torch.save(net.module.state_dict(), str(path_models / (config['run_name'] + '.pth')))
        else:
            torch.save(net.state_dict(), str(path_models / (config['run_name'] + '.pth')))
        print(f".. current model state correctly saved !")
        
    except Exception as e:
        raise Exception(f"Unexpected error: {e}")
    return None
    

def save_current_branch_state(config, net, path_models, device):
    # Util function to upgrade just'part' of the neural-network by upgrading certain tensors

    current_model_name = config['run_name'] + '.pth'
    weights_path = str(path_models / current_model_name)

    # Read the saved dict. from the previous loops
    old_weights_dict = read_weights_dict(weights_path, device)

    # Fetch the current model weights depending on the number of GPU in usage
    current_weights_dict = read_model_state(config, net)

    # NOTE: For now the only branch to update is in the 'triple-unet' config.
    upgraded_weights_dict = upgrade_weights_dict(old_weights_dict, current_weights_dict, ["decoder", "2"])
    try:
        if config['num_gpus'] > 1:
            torch.save(upgraded_weights_dict, weights_path)
        else:
            torch.save(upgraded_weights_dict, weights_path)
        print(f".. current model state correctly upgraded !")
        
    except Exception as e:
        raise Exception(f"Unexpected error: {e}")
    return None


def read_weights_dict(file_path: str, device: torch.device) -> dict:
    """
    Reads a model weights dictionary from a file path and maps it to the specified device.

    This function loads the weights dictionary from the provided `file_path` (assumed to be a PyTorch model file)
    and maps the loaded tensors to the specified `device` (CPU or GPU).

    Args:
        file_path (str): The path to the model weights file.
        device (torch.device): The device (CPU or GPU) to which the tensors should be mapped.

    Returns:
        dict: The loaded model weights dictionary with tensors mapped to the specified device.

    Raises:
        FileNotFoundError: If the model file is not found.
        RuntimeError: If there's an error loading the model (potentially due to file corruption).
    """

    try:
        # Load weights dictionary and map tensors to the device
        return torch.load(file_path, map_location=device)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model file '{file_path}' not found.") from e
    except RuntimeError as e:
        raise RuntimeError(f"Error loading model from '{file_path}': {e}") from e


def read_model_state(config: dict, net) -> dict:
    """
    Reads the model state dictionary considering potential data parallelism.

    Args:
        config (dict): The configuration dictionary (assumed to contain a key 'num_gpus').
        net (Any): The PyTorch model object.

    Returns:
        dict: The retrieved model state dictionary.

    Raises:
        ValueError: If the `config` dictionary doesn't contain the key 'num_gpus'.
    """

    # Validate configuration key existence
    if 'num_gpus' not in config:
        raise ValueError("Configuration dictionary must contain the key 'num_gpus'.")

    # Access state dictionary based on number of GPUs
    if config['num_gpus'] > 1:
        try:
            # Access state dictionary from wrapped module for data parallelism
            return net.module.state_dict()
        except AttributeError:
            raise ValueError("Data parallelism structure not found in the model.")
    else:
        return net.state_dict()


def upgrade_weights_dict(old_dict: dict, current_dict: dict, identifiers: list[str]) -> dict:

    """
    Upgrades weights in a dictionary based on layer names containing specific identifiers.

    Args:
        old_dict (dict): The dictionary containing the older weights to be potentially upgraded.
        current_dict (dict): The dictionary containing the newer weights to be used for upgrades.
        identifiers (list[str]): A list of strings (identifiers) to match within layer names for upgrades.

    Returns:
        dict: The modified `old_dict` with weights upgraded for matching layers.

    Raises:
        ValueError: If the `identifiers` list is empty or contains non-string elements.
    """

    if not identifiers:
        raise ValueError("`identifiers` list cannot be empty.")

    if not all(isinstance(identifier, str) for identifier in identifiers):
        raise ValueError("`identifiers` list must contain only strings.")

    for key, item in old_dict.items():
        layer_name = key.split('.')[0]
        if (identifiers[0] in layer_name) and (identifiers[1] in layer_name):
            # Update the value (wieght tensor) just in the right keys
            old_dict[key] = current_dict[key]
    return old_dict


def get_num_workers(device):
    # Called from training.py in train(...) to get the num workers for the dataloader 

    if device.type == "cpu":
        num_workers = 0
    else:
        try:
            num_workers = cpu_count() // 2
        except AttributeError:
            num_workers = 4
    if num_workers <= 2:  # Probably Google Colab --> use 0
        num_workers = 0
    return num_workers


def load_weights(net, model, device, num_gpus):
    # Get number of GPUs to use and load weights

    try:
        model_path = str(model)  # Ensure valid file path

        if num_gpus > 1:
            net.module.load_state_dict(torch.load(model_path, map_location=device))
        else:
            net.load_state_dict(torch.load(model_path, map_location=device))

    except FileNotFoundError as e:

        print(f"Error: Model file '{model_path}' not found.")
    except Exception as e:

        print(f"Unexpected error while loading model: {e}")
    return None


def get_det_score(path):
    """  Get DET metric score.

    :param path: path to the DET score file.
        :type path: pathlib Path object.
    :return: DET score.
    """

    with open(path) as det_file:
        read_file = True
        while read_file:
            det_file_line = det_file.readline()
            if 'DET' in det_file_line:
                det_score = float(det_file_line.split('DET measure: ')[-1].split('\n')[0])
                read_file = False

    return det_score


def get_seg_score(path):
    """  Get SEG metric score.

    :param path: path to the SEG score file.
        :type path: pathlib Path object.
    :return: SEG score.
    """

    with open(path) as det_file:
        read_file = True
        while read_file:
            det_file_line = det_file.readline()
            if 'SEG' in det_file_line:
                seg_score = float(det_file_line.split('SEG measure: ')[-1].split('\n')[0])
                read_file = False

    return seg_score


def get_nucleus_ids(img: np.ndarray) -> List[int,]:
    """Get nucleus ids in intensity-coded label image
    skipping the background values (0).

    Args:
        img: Intensity-coded nuclei image.
    
    Returns: 
        List of nucleus ids as integer values.
    """
    values = np.unique(img)
    values = values[values > 0]
    return values


def min_max_normalization(img, min_value=None, max_value=None):
    """ Minimum maximum normalization.

    :param img: Image (uint8, uint16 or int)
        :type img:
    :param min_value: minimum value for normalization, values below are clipped.
        :type min_value: int
    :param max_value: maximum value for normalization, values above are clipped.
        :type max_value: int
    :return: Normalized image (float32)
    """

    if max_value is None:
        max_value = img.max()

    if min_value is None:
        min_value = img.min()

    # Clip image to filter hot and cold pixels
    img = np.clip(img, min_value, max_value)

    # Apply min-max-normalization
    img = 2 * (img.astype(np.float32) - min_value) / (max_value - min_value) - 1

    return img.astype(np.float32)


def unique_path(directory, name_pattern):
    """ Get unique file name to save trained model.

    :param directory: Path to the model directory
        :type directory: pathlib path object.
    :param name_pattern: Pattern for the file name
        :type name_pattern: str
    :return: pathlib path
    """
    counter = 0
    while True:
        counter += 1
        # .format is called here with the prepared string in the calling function
        path = directory / Path(name_pattern.format(counter))
        if not path.exists():
            return path


def write_train_info(configs, path):
    """ Write training configurations into a json file.

    :param configs: Dictionary with configurations of the training process.
        :type configs: dict
    :param path: path to the directory to store the json file.
        :type path: pathlib Path object
    :return: None
    """

    with open(path / (configs['run_name'] + '.json'), 'w', encoding='utf-8') as outfile:
        json.dump(configs, outfile, ensure_ascii=False, indent=2)

    print(f"Model configuration saved corretly!")
    return None


def copy_best_model(path_models, path_best_models, best_model, best_settings):
    """ Copy best models to KIT-Sch-GE_2021/SW and the best (training data set) results.

    :param path_models: Path to all saved models.
        :type path_models: pathlib Path object.
    :param path_best_models: Path to the best models.
        :type path_best_models: pathlib Path object.
    :param best_model: Name of the best model.
        :type best_model: str
    :param best_settings: Best post-processing settings (th_cell, th_mask, ...)
        :type best_settings: dict
    :return: None.
    """

    new_model_name = best_model[:-3]

    # Copy and rename model
    shutil.copy(str(path_models / "{}.pth".format(best_model)),
                str(path_best_models / "{}.pth".format(new_model_name)))
    shutil.copy(str(path_models / "{}.json".format(best_model)),
                str(path_best_models / "{}.json".format(new_model_name)))

    # Add best settings to model info file
    with open(path_best_models / "{}.json".format(new_model_name)) as f:
        settings = json.load(f)
    settings['best_settings'] = best_settings

    with open(path_best_models / "{}.json".format(new_model_name), 'w', encoding='utf-8') as outfile:
        json.dump(settings, outfile, ensure_ascii=False, indent=2)

    return None


def get_best_model(metric_scores, mode, subset, th_cells, th_seeds):
    """ Get best model and corresponding settings.

    :param metric_scores: Scores of corresponding models.
        :type metric_scores: dict
    :param mode: Mode ('all', 'single')
        :type mode: str
    :param subset: Evaluate on dataset '01', '02' or on both ('01+02').
        :type subset: str
    :param th_cells: Cell/mask thresholds which are evaluated
        :type th_cells: list
    :param th_seeds: Seed/marker thresholds which are evaluated.
        :type th_seeds: list
    :return:
    """

    best_th_cell, best_th_seed, best_model = 0, 0, ''

    subsets = [subset]
    if subset == '01+02':
        subsets = ['01', '02']

    if "all" in mode:

        best_op_csb = 0

        for model in metric_scores:

            for th_seed in th_seeds:

                for th_cell in th_cells:

                    op_csb = 0

                    for cell_type in metric_scores[model]:

                        # Exclude too different cell types (goal: better model on other data sets)
                        if cell_type in ['Fluo-C2DL-MSC', 'Fluo-C3DH-H157']:
                            continue

                        for train_set in subsets:
                            op_csb += metric_scores[model][cell_type][train_set][str(th_seed)][str(th_cell)]['OP_CSB']

                    op_csb /= len(metric_scores[model]) * len(subsets)

                    if op_csb > best_op_csb:
                        best_op_csb = op_csb
                        best_th_cell = th_cell
                        best_th_seed = th_seed
                        best_model = model

    else:

        best_op_csb = 0

        for model in metric_scores:

            for cell_type in metric_scores[model]:

                for th_seed in th_seeds:

                    for th_cell in th_cells:

                        op_csb = 0

                        for train_set in subsets:

                            op_csb += metric_scores[model][cell_type][train_set][str(th_seed)][str(th_cell)]['OP_CSB']

                        op_csb /= len(subsets)

                        if op_csb > best_op_csb:
                            best_op_csb = op_csb
                            best_th_cell = th_cell
                            best_th_seed = th_seed
                            best_model = model

    return best_op_csb, float(best_th_cell), float(best_th_seed), best_model


def zero_pad_model_input(img, pad_val=0):
    """ Zero-pad model input to get for the model needed sizes (more intelligent padding ways could easily be
        implemented but there are sometimes cudnn errors with image sizes which work on cpu ...).

    :param img: Model input image.
        :type:
    :param pad_val: Value to pad.
        :type pad_val: int.

    :return: (zero-)padded img, [0s padded in y-direction, 0s padded in x-direction]
    """

    # Tested shapes
    tested_img_shapes = [64, 128, 256, 320, 512, 768, 1024, 1280, 1408, 1600, 1920, 2048, 2240, 2560, 3200, 4096,
                         4480, 6080, 8192]

    if len(img.shape) == 3:  # 3D image (z-dimension needs no pads)
        img = np.transpose(img, (2, 1, 0))

    # More effective padding (but may lead to cuda errors)
    # y_pads = int(np.ceil(img.shape[0] / 64) * 64) - img.shape[0]
    # x_pads = int(np.ceil(img.shape[1] / 64) * 64) - img.shape[1]

    pads = []
    for i in range(2):
        for tested_img_shape in tested_img_shapes:
            if img.shape[i] <= tested_img_shape:
                pads.append(tested_img_shape - img.shape[i])
                break

    if not pads:
        raise Exception('Image too big to pad. Use sliding windows')

    if len(img.shape) == 3:  # 3D image
        img = np.pad(img, ((pads[0], 0), (pads[1], 0), (0, 0)), mode='constant', constant_values=pad_val)
        img = np.transpose(img, (2, 1, 0))
    else:
        img = np.pad(img, ((pads[0], 0), (pads[1], 0)), mode='constant', constant_values=pad_val)

    return img, [pads[0], pads[1]]


def save_metrics(log, metrics, dataset_path, name = 'results', ext = '.json'):
    # Save/Update the final metrics dict after the post_processing pipeline.

    file_name = name + ext
    file_path = os.path.join(dataset_path, file_name)

    # Check if the file already exists - TODO: Implement update of existing result file.
    if os.path.exists(file_path):

        log.info("The file {file_name} already exists in {dataset_path}: It will be updated.")
        update_evaluation_metrics(file_path, metrics)
    else:

        # Save the dict in input in a '*.json' file in the dataset folder.
        '''with open(file_path, "w") as outfile:
            json.dump(metrics, fp=outfile, indent = 4, sort_keys=True)'''
        save_dict_to_json(metrics, file_path)
    log.info(f"File '{name + ext}' in '{dataset_path}' saved correctly!")
    return None


def update_evaluation_metrics(file_path, new_data_dict):
    # In case of same "*.json" file containing results, update it.

    old_data = read_json_file(file_path) # Read existing results.
    old_df = pd.DataFrame.from_dict(old_data, orient="index") # tranform the dict data into a dataframe.
    new_df = pd.DataFrame.from_dict(new_data_dict, orient="index")

    # Merge the two dict 
    merged_df = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates() # Check for duplicates results - drop during the assignement.
    merged_df.index = merged_df.index.astype("object") # Cast the index to object type before tranforming into a dict
    final_dict = merged_df.to_dict('index')
    save_dict_to_json(final_dict, file_path)
    return None


def read_json_file(file_path):
    """
    Reads a JSON file and returns the contents as a dictionary.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        dict: The contents of the JSON file as a dictionary.

    Raises:
        ValueError: If the file cannot be opened or if the JSON is invalid.
    """

    try:
        with open(file_path, 'r') as f:

            data = json.load(f)
    except FileNotFoundError:

        raise ValueError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:

        raise ValueError(f"Invalid JSON format: {e}")
    return data


def save_dict_to_json(data, filepath, *, indent=4, ensure_ascii=True):
    """
    Saves a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to save.
        filepath (str): The path to the JSON file.
        indent (int, optional): The indentation level for human-readable output.
        ensure_ascii (bool, optional): Whether to ensure ASCII encoding (default: True).

    Raises:
        TypeError: If the data is not a dictionary.
        IOError: If there is an error writing to the file.
    """

    if not isinstance(data, dict):
        raise TypeError("Data must be a dictionary.")
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

    except IOError as e:
        raise IOError(f"Error writing to file: {e}")
    return None


# Moved from the 'create_training_sets.py' module - function used by 'training' methods.
def write_file(file, path):
    with open(path, 'w', encoding='UTF-8') as f:
        json.dump(file, f, ensure_ascii=False, indent=2)
    return


# Util to show specific pipeline's training sets
def show_training_set_images(pipeline, dataset_path, cell_type, mode, split, n_samples, crop_size):

    if pipeline == 'kit-ge':

        # Get train set cell distance maps
        path_data = Path(dataset_path)
        # As default fetch from the selected 'train' images
        img_ids = (path_data / f"{cell_type}_{mode}_{split}_{crop_size}" / 'train').glob('img*')
        # Show 'n_samples' examples (image, mask, cell distance, neighbor distance)
        images = []
        
        for img_idx in img_ids:
            fname = img_idx.name.split('img')[-1]
            img = tiff.imread(str(img_idx))
            mask = tiff.imread(str(img_idx.parent / f"mask{fname}"))
            cell_dist = tiff.imread(str(img_idx.parent / f"dist_cell{fname}"))
            neighbor_dist = tiff.imread(str(img_idx.parent / f"dist_neighbor{fname}"))
            images.append([img, mask, cell_dist, neighbor_dist])
            if len(images) == n_samples:
                break

        print(f"Original images patches available {len([id for id in img_ids])}")
        n_patch = 4
        cmap_mask = plt.get_cmap("prism")
        cmap_mask.set_bad(color="k")
        fig, axs = plt.subplots(len(images), n_patch, figsize=(12, 12)) # NOTE: 'kit-ge' pipeline use 4 patches for every training sample
        fig.suptitle('Exemplary training data (image, mask, cell distance, neighbor distance)')
        for i in range(len(images)):
            for j in range(n_patch):
                if j == 1:
                    mask = np.ma.masked_array(images[i][j], images[i][j]==0)
                    axs[i, j].imshow(np.squeeze(mask), cmap=cmap_mask)
                else:
                    axs[i, j].imshow(np.squeeze(images[i][j]), cmap='gray')
            axs[i, j].axis('off')
        fig.tight_layout()

    return None


# Used in train/validation phase to perform further analysis - e.g. in the 'border_label_2d' to debug visually the highlighted borders.
def save_image(img, path, title, use_cmap = False):
    
    # Ensure the folder exists
    os.makedirs(path, exist_ok=True)
    
    # Create the full path for saving the image
    image_path = os.path.join(path, f"{title}.png")

    # Plot the image using matplotlib
    plt.imshow(img, cmap='gray' if use_cmap else None)
    plt.title(title)
    plt.axis('off')  # Turn off axis labels
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # Close the plot to free up resources
    return None


def save_segmentation_image(img, path, title, use_cmap = False):
    
    # Additional settings for mask plotting
    cmap_mask = plt.get_cmap("prism")
    cmap_mask.set_bad(color="k")
    mask = np.ma.masked_array(img, img==0)

    # Ensure the folder exists
    os.makedirs(path, exist_ok=True)
    
    # Create the full path for saving the image
    image_path = os.path.join(path, f"{title}.png")

    # Plot the image using matplotlib
    plt.imshow(mask, cmap=cmap_mask if use_cmap else None)
    plt.title(title)
    plt.axis('off')  # Turn off axis labels
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # Close the plot to free up resources
    return None


def log_final_images_properties(log, image):
    # util function to debug the final instance prediction

    n_region = np.unique(image)
    log.debug(f".. the current image has {len(n_region)} labeled cells ..")
    return None


def show_training_dataset_samples(log, dataset: torch.utils.data.Dataset, n_sample: int = 10) -> None:
    """
    It plots a defined number of images to provide a sample of data from the group used
    for the training.

    Args:
        dataset: A split ('train' or 'val') of type 'CellSegDataset'.
        n_sample: Number of sample to show - every sample is all the different
        format/transformation of the original image patch.   
    """

    # Pre-conditions
    if n_sample > len(dataset):
        log.debug(f"""The original {n_sample} sample requested are too much, the 
                  current dataset contains {len(dataset)} samples""")
        n_sample = len(dataset)

    count_neighbor_distance = 0
    log.debug(f"Visually inspect the first {n_sample} samples of images from the training Dataset")
    folder = os.getenv("TEMPORARY_PATH")
    for idx in range(n_sample):

        image_dict = dataset[idx]
        for pos, (key, image) in enumerate(image_dict.items()):

            curr_title = "Sample " + str(idx) + f" type ({key})"
            save_image(np.squeeze(image), folder, curr_title, use_cmap=True)

            # Keep track of not-blank neighbor tranform
            if key == "border_label" and np.squeeze(image).sum(axis = None) > 0:
                count_neighbor_distance += 1
    print(f"Between the shown example, the {round(count_neighbor_distance/n_sample, 2)}% of distance tranform are not-blanck!")
    log.debug(f"Images correctly saved in {folder} before the training phase!")

def show_inference_dataset_samples(log, dataset, samples = 3):
    # Visual debug for the images used in the inference phase.

    log.debug(f"Visually inspect the first {samples} samples of images from the inference CTC Dataset")
    folder = os.getenv("TEMPORARY_PATH")
    for idx in range(samples):

        image_dict = dataset[idx]
        for pos, (key, image) in enumerate(image_dict.items()):

            if key in ["image", "single_channel_image", "nuclei_channel_image"]:
                curr_title = "Sample " + str(idx) + f" type ({key})"
                save_image(np.squeeze(image), folder, curr_title, use_cmap=True)
    log.debug(f"Images correctly saved in {folder} before the inference phase!")
    return True
