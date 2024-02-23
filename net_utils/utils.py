''' ./net_utils/utils.py
This file contains all the functions to store/save results (both models/metrics).
'''
import json
import numpy as np
import shutil
import os
from pathlib import Path
import tifffile as tiff
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import torch


def save_training_loss(loss_labels, train_loss, val_loss, second_run, path_models, config, tot_time, tot_epochs):
    # Get the training loss and save it in a formatted '*.txt' file.

    # Un-pack the losses in the original variables
    train_total_loss, train_loss_border, train_loss_cell, train_loss_mask = zip(*train_loss) 
    val_total_loss, val_loss_border, val_loss_cell, val_loss_mask = zip(*val_loss) 
    
    stats = np.transpose(np.array([list(range(1, len(train_loss) + 1)), train_total_loss, train_loss_border, train_loss_cell, train_loss_mask, val_total_loss, val_loss_border, val_loss_cell, val_loss_mask]))
    try:
        if second_run:
            np.savetxt(fname=str(path_models / (config['run_name'] + '_2nd_loss.txt')), X=stats,
                    fmt=['%3i', '%2.5f', '%2.5f', '%2.5f', '%2.5f', '%2.5f', '%2.5f', '%2.5f', '%2.5f'],
                    header='Epoch, training total loss, training border loss, training cell loss, training mask loss, validation total loss, validation border loss, validation cell loss, validation mask loss,', delimiter=',')
            config['training_time_run_2'], config['trained_epochs_run2'] = tot_time, tot_epochs + 1

        else:
            np.savetxt(fname=str(path_models / (config['run_name'] + '_loss.txt')), X=stats,
                    fmt=['%3i', '%2.5f', '%2.5f', '%2.5f', '%2.5f', '%2.5f', '%2.5f', '%2.5f', '%2.5f'],
                    header='Epoch, training total loss, training border loss, training cell loss, training mask loss, validation total loss, validation border loss, validation cell loss, validation mask loss,', delimiter=',')
            config['training_time'], config['trained_epochs'] = tot_time, tot_epochs + 1
        
        print(f"Training losses saved corretly and current configuration parameters updated")

    except Exception as e:
        raise Exception(f"Unexpected error: {e}")
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


def get_num_gpus_and_set_weights(net, model, device):
    # Get number of GPUs to use and load weights

    try:
        model_path = str(model)  # Ensure valid file path
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            net.module.load_state_dict(torch.load(model_path, map_location=device))
        else:
            net.load_state_dict(torch.load(model_path, map_location=device))

    except FileNotFoundError as e:

        print(f"Error: Model file '{model_path}' not found.")
    except Exception as e:

        print(f"Unexpected error while loading model: {e}")
    return num_gpus


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


def get_nucleus_ids(img):
    """ Get nucleus ids in intensity-coded label image.

    :param img: Intensity-coded nuclei image.
        :type:
    :return: List of nucleus ids.
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


def get_evaluation_dict(args, path_data):
    # Get the dictionary containing the information for the current evaluation run.

    result_dict = { 'model': args.model_pipeline,
                'post_processing': args.post_processing_pipeline, # NOTE: It is possible to add more fixed args in the dict
                'data': path_data,
                'results': {}} # NOTE: This field will contains specific post-processing args for the current pipeline.
    return result_dict


# Custom saving/loading metrics functions
def save_metrics(log, metrics, dataset_path, name = 'results', ext = '.json'):
    
    file_name = name + ext
    file_path = os.path.join(dataset_path, file_name)

    # Check if the file already exists
    if os.path.exists(file_path):
        log.info("The file {file_name} already exists in {dataset_path}: It will be subscribed.")

    # Save the dict in input in a 'results.json' file in the dataset folder
    with open(os.path.join(dataset_path, name + ext), "w") as outfile:
        json.dump(metrics, fp=outfile, indent = 4, sort_keys=True)

    log.info(f"File '{name + ext}' in '{dataset_path}' saved correctly!")
    return None


# TODO: Implement as aggregation of results for visualization purpose.
def aggregate_metrics():
    pass


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


# Used in train/validation phase to perform further analysis - e.g. in the 'border_label_2d' to debug visually the highlighted borders
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


def show_training_dataset_samples(log, dataset, samples = 10):
    # Visual debug for the images used in the training set.
    log.debug(f"Visually inspect the first {samples} samples of images from the training Dataset")
    folder = os.getenv("TEMPORARY_PATH")
    for idx in range(samples):

        image_list = dataset[idx]

        for pos, image in enumerate(image_list): # Plot all images except the last one: the mask tensor.

            curr_title = "Sample " + str(idx) + f" image {str(pos)}"
            save_image(np.squeeze(image), folder, curr_title)

    log.debug(f"Images correctly saved in {folder}!")
    return True













