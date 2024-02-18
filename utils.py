"""utils.py

- Logger functions 
- Parsing functions
- Containing specific class args parsing
- Set the device for train/eval
"""

import torch
import os
from os.path import join, exists
import logging
from datetime import datetime
import json
import requests
import zipfile
import tifffile as tiff
from dotenv import load_dotenv

from img_processing.imageUtils import *


# TODO: Move this to a 'config file' inside the repository.
# TODO: If my dataset will become public, manage the download from this script as additional option.
DATASETS = ["BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa", "Fluo-C2DL-Huh7", "Fluo-C2DL-MSC", "Fluo-N2DH-GOWT1", "Fluo-N2DL-HeLa",
            "PhC-C2DH-U373", "PhC-C2DL-PSC", "Fluo-N2DH-SIM+"]

GT_FOLDERS = ['SEG', 'TRA'] # Folders expected in the annotated time lapses (e.g. '01_GT')
TRAINING_FOLDER = "training_data"


def create_logging():
  """ Function to set up the 'INFO' and 'DEBUG' log file
  """ 
  os.makedirs(os.getenv('LOGS_PATH', None), exist_ok=True)

  # Get the current date and time
  current_datetime = datetime.now()

  # Extract the date and  hour, and minute components for the sub-folder
  current_date = current_datetime.strftime("%Y-%m-%d") 
  day_log_path = os.path.join(os.getenv('LOGS_PATH', None), current_date) # Folder for the day's logs
  os.makedirs(day_log_path, exist_ok=True)

  current_time = current_datetime.strftime("%H-%M-%S") 
  run_log_path = os.path.join(day_log_path, current_time) # Specific folder for a run
  os.makedirs(run_log_path, exist_ok=True)
  
  info_log_file = f"info_{current_time}.log"
  debug_log_file = f"debug_{current_time}.log"
  
  # Complete paths for the logs file
  info_log_path = os.path.join(run_log_path, info_log_file)
  debug_log_path = os.path.join(run_log_path, debug_log_file)
  
  # Set up handlers
  info_handler = logging.FileHandler(info_log_path)
  info_handler.setLevel(logging.INFO)
  debug_handler = logging.FileHandler(debug_log_path)
  debug_handler.setLevel(logging.DEBUG)

  # Set up single logger (info and debug)
  logger = logging.getLogger('logger') # Name will remain fixed 
  logger.setLevel(logging.DEBUG) # Lower level for the two handlers

  # Set up the format messages
  log_info_format = logging.Formatter('%(asctime)s - %(levelname)s   %(message)s')
  log_debug_format = logging.Formatter('%(asctime)s - %(levelname)s  %(message)s')
  info_handler.setFormatter(log_info_format)
  debug_handler.setFormatter(log_debug_format)

  # Add the handlers to the loggers
  logger.addHandler(info_handler)
  logger.addHandler(debug_handler)

  return logger


# For now the repository is implemented for single-gpu usage.
def set_device():
    # Set device for using CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == 'cuda':
        torch.backends.cudnn.benchmark = True
        num_gpus = 1
    else:
        num_gpus = 0
    return device, num_gpus


# NOTE: Updating to be used by all modules in this project
def set_environment_paths():

    # Load the paths on the environment
    load_dotenv()

    # Set-up folders
    os.makedirs(os.getenv('TEMPORARY_PATH', None), exist_ok=True)
    return None


def check_and_download_evaluation_software(log, software_path):
    # Check if eval. software is already downloaded: if not download it.
    
    # Check if the evaluation folder is already contained
    files = [name for name in os.listdir(software_path) if not name.startswith(".")]
    
    # Download evaluation software if it is not already donwloaded (there is just one file in the '*/evaluation software' folder)
    if len(files) <= 1:
        software_url = os.getenv("CTC_SOFTWARE_URL", None)
        software_name = os.getenv("CTC_SOFTWARE_NAME", None).split(".")[0]
        log.info(f"Downloading evaluation software to '{software_path}' ..")
        __download_data(log=None, url=software_url, target=software_path)
        unzip_donwloaded_file(log, os.path.join(software_path, software_name), target = software_path)

    else:
        log.info(f"Evaluation software already set-up in '{software_path}'")
        return False # Eval software already set up.

    log.info(f"Evaluation software correctly set up")
    return None # Evaluation folder correctly set up


def __download_data(log, url, target): 
    # Download from the provided url with a specific chunk size

    local_filename = os.path.join(target, url.split('/')[-1])
    # TODO: Try a couple of times if raise status give error. 
    with requests.get(url, stream=True) as r:

        if not r.raise_for_status() is None: # If the method doesn't return NoneType It is bad
            if not log is None: 
                log.debug(f"Status for the url {url}: {r.raise_for_status()}") # Help debugging error in case of status different from 200
            else: # In case called from the notebook
                print(f"Status for the url {url}: {r.raise_for_status()}")

        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


def unzip_donwloaded_file(log, path, target = 'none'):
    # Unzip the file and remove the original zipped version.

    log.info(f"Unzipping {path + '.zip'} ..")
    with zipfile.ZipFile(path + '.zip', 'r') as z: # Extract zipped dataset
        if target == 'none':
            z.extractall(path.split('/')[0])
        else:
            z.extractall(target)
        
    os.remove(path + '.zip') # Remove orginal zip
    log.info(f"The file {path + '.zip'} correctly removed")
    return True


def check_downloaded_images(log, images_path):
    # As default will be used '01' ('01_GT/SEG' and '01_GT/TRA') with the respective first files - first check.

    # Read first images
    images = os.listdir(images_path)
    images.sort()
    images = [img for img in images if img.endswith('.tif')] # Avoid additional file not '*.tif', as the 'man_track.txt'.
    image = tiff.imread(os.path.join(images_path, images[0]))
    log_image_characteristics(log, image, images[0].split('.')[0]) # Log characteristics of images
    
    return None


def __check_and_download_dataset(log, dataset_path, url): 
    # Check if It is alredy available, if not download it (function used both for test/train splits).
    
    dataset = dataset_path.split('/')[-1] # Extract the name.
    if not os.path.isdir(dataset_path): # Check if It's already present

        log.info(f"Downloading {dataset_path.split('/')[0].split('_')[0]} {dataset} dataset ..")
        try: # even if some datasets are not downlodable continue the code

            dataset_url = os.path.join(url, dataset + '.zip') # Set up the url for the download 
            __download_data(log, dataset_url, dataset_path.split('/')[0]) # Donwload the data on the target folder
            unzip_donwloaded_file(log, dataset_path)
        except:

            log.info(f"Failed the download of the {dataset_path.split('/')[0].split('_')[0]} split of {dataset}")
        
        check_downloaded_images(log, os.path.join(dataset_path, '01')) # Original time-lapses images.
        if dataset_path.split('/')[-2] == TRAINING_FOLDER: # TODO: should be passed directly from args
            
            for folder in GT_FOLDERS:# Analyze both first mask in SEG and TRA.
                current_folder_path = os.path.join(dataset_path, '01_GT', folder)
                
                if os.path.exists(current_folder_path):
                    check_downloaded_images(log, current_folder_path)
                else:
                    log.info("Path {current_folder_path} does not exist ..")
                            
    else:
        log.info(f"'{dataset_path}' exists already!")
    return None


def download_datasets(log, args): # Main function to download the chosen datasets and set up their utilization
    """ Function to download the images in the path passed by arguments - dataset folder structure is fixed.

    Args:
        args (dict): Arguments usefull for the download and creation of the images folder

    Returns:
        None

    """
    log.info(f"Preparing the datasets download ..")
    log.info(f"> The folders used will be respectively '{args.train_images_path}' and '{args.test_images_path}'")
    
    # Set up the split folders
    os.makedirs(args.train_images_path, exist_ok=True)
    os.makedirs(args.test_images_path, exist_ok=True)
    # Get url from the .env loaded paths.
    train_data_url = os.getenv("CTC_TRAINDATA_URL", None)
    test_data_url = os.getenv("CTC_TESTDATA_URL", None)

    if args.dataset_to_download == 'all': # TODO: Provide other options

        log.info(f"All datasets will be downloaded")
        for dataset in DATASETS:
            current_train_path = os.path.join(args.train_images_path, dataset)
            current_test_path = os.path.join(args.test_images_path, dataset)

            __check_and_download_dataset(log, current_train_path, train_data_url)
            __check_and_download_dataset(log, current_test_path, test_data_url)

    else: # Download single dataset if it is present in the list

        if args.dataset_to_download in DATASETS:
            log.info(f"Dataset that will be downloaded: {args.dataset_to_download}")
            current_train_path = os.path.join(args.train_images_path, args.dataset_to_download)
            current_test_path = os.path.join(args.test_images_path, args.dataset_to_download)

            __check_and_download_dataset(log, current_train_path,  train_data_url)
            __check_and_download_dataset(log, current_test_path, test_data_url)
            log.info(f"Data downloads completed!")
        
        else:
            log.info(f"Dataset {args.download_dataset} not found in the available list!")
    log.info(f"Program terminated")
    return None


def check_path(log, path):
    # Util function to check for existing path - it will raise an Error.

    if not exists(path):
        log.info(f"Warning: the '{path}' provided is not existent! Interrupting the program...")
        raise ValueError("The '{path}' provided is not existent")
    return True


# TODO: this class offer a customized EvaluationParser for every implemented pipeline.
class EvalArgs(object): # Class containings inference and post-processing parameters.
    """ Class with post-processing parameters.
    """

    def __init__(self, post_processing_pipeline, th_cell, th_seed, apply_clahe, scale, cell_type,
                 save_raw_pred,artifact_correction, apply_merging):
        """
        (kit-ge post-processing params)
        :param th_cell: Mask / cell size threshold.
            :type th_cell: float
        :param th_seed: Seed / marker threshold.
            :type th_seed: float
        :param apply_clahe: Apply contrast limited adaptive histogram equalization (CLAHE).
            :type apply_clahe: bool
        :param scale: Scale factor for downsampling.
            :type scale: float
        :param cell_type: Cell type.
            :type cell_type: str
        :param save_raw_pred: Save (some) raw predictions.
            :type save_raw_pred: bool
        :param artifact_correction: Apply artifact correction post-processing.
            :type artifact_correction: bool
        """
        if post_processing_pipeline == 'kit-ge':
            self.th_cell = th_cell
            self.th_seed = th_seed
            self.apply_clahe = apply_clahe
            self.scale = scale
            self.cell_type = cell_type
            self.save_raw_pred = save_raw_pred
            self.artifact_correction = artifact_correction
            self.apply_merging = apply_merging

    # Override default class function to print parameters
    def __str__(self):
        attributes = ', '.join(f'{key}={value}' for key, value in vars(self).items())
        return f"EvalArgs({attributes})"


# TODO: this class offer a customized TrainingParser for every implemented pipeline.
class TrainArgs(object):
    """ Class with training creation parameters.
    """

    def __init__(self, model_pipeline, act_fun, batch_size, filters, iterations,
    loss, norm_method, optimizer, pool_method, pre_train, retrain, split):
        """ kit-ge training params implemented for now.
        """
        if model_pipeline == 'kit-ge':
            self.arch = 'DU'
            self.act_fun = act_fun
            self.batch_size = batch_size
            self.filters = filters
            self.iterations = iterations
            self.loss = loss
            self.norm_method = norm_method
            self.optimizer = optimizer
            self.pool_method = pool_method
            self.pre_train = pre_train
            self.retrain = retrain
            self.split = split

        elif model_pipeline == 'dual-unet': # NOTE: Params for the work "Dual U-Net for the segmentation of Overlapping Glioma Nuclei"
            self.arch = 'TU'
            self.act_fun = act_fun
            self.batch_size = batch_size
            self.filters = filters
            self.iterations = iterations
            self.loss = loss
            self.norm_method = norm_method
            self.optimizer = optimizer
            self.pool_method = pool_method
            self.pre_train = pre_train
            self.retrain = retrain
            self.split = split
        else:
            raise Exception('Model architecture "{}" is not known'.format(model_pipeline))


    # Override default class function
    def __str__(self):
        attributes = ', '.join(f'{key}={value}' for key, value in vars(self).items())
        return f"TrainArgs({attributes})"


        