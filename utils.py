"""utils.py

- Logger functions 
- Parsing functions
- Containing specific class args parsing
- Set the device for train/eval
"""

import torch
import os
import logging
from datetime import datetime
import json
import requests
import zipfile
import tifffile as tiff
from dotenv import load_dotenv

from img_processing.imageUtils import *

LOGS_PATH = "logs" # TODO: Const as the path is fixed. Move this in a 'ENV' dict 

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


# NOTE: Utils functions for the data_download.py (main script to download the data)

# TODO: Move this to a 'config file' inside the repository.
# TODO: If my dataset will become public, manage the download from this script as additional option.
DATASETS = ["BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa", "Fluo-C2DL-Huh7", "Fluo-C2DL-MSC", "Fluo-N2DH-GOWT1", "Fluo-N2DL-HeLa",
            "PhC-C2DH-U373", "PhC-C2DL-PSC", "Fluo-N2DH-SIM+"]
# Folders expected in the annotated time lapses (e.g. '01')
GT_FOLDERS = ['SEG', 'TRA']


# NOTE: For now download just the train/test Datasets from CTC - move this to the '.env. file
TRAINDATA_URL = 'http://data.celltrackingchallenge.net/training-datasets/'
TESTDATA_URL = 'http://data.celltrackingchallenge.net/test-datasets/'
SOFTWARE_URL = 'http://public.celltrackingchallenge.net/software/EvaluationSoftware.zip'


# Check if eval. software is already downloaded: if not download it.
def check_evaluation_software(log, software_path):
    
    # Check if the evaluation folder is already contained
    files = [name for name in os.listdir(software_path) if not name.startswith(".")]
    
    # Download evaluation software if it is not already donwloaded (there is just one file in the '*/evaluation software' folder)
    if len(files) <= 1:
        log.info(f"Downloading evaluation software to '{software_path}' ..")
        __download_data(log=None, url=SOFTWARE_URL, target=software_path)
    else:
        log.info(f"Evaluation software already set-up in '{software_path}'")
        return False # Eval software already set up.

    # Unzip evaluation software
    log.info('Unzip evaluation software ..')
    with zipfile.ZipFile(os.path.join(software_path, SOFTWARE_URL.split('/')[-1]), 'r') as z:
        z.extractall(software_path)
    
    log.info(f"Evaluation software correctly set up")
    return None # Evaluation folder correctly set up


# Download from the provided url with a specific chunk size
def __download_data(log, url, target): 

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


def __check_dataset(log, dataset_path, url): # Check if It is alredy available, if not download it (function used both for test/train splits)
    
    dataset = dataset_path.split('/')[-1] # Extract the name

    if not os.path.isdir(dataset_path): # Check if It's already present

        log.info(f"Downloading {dataset_path.split('/')[0].split('_')[0]} {dataset} dataset ..")
        try: # even if some datasets are not downlodable continue the code

            dataset_url = os.path.join(url, dataset + '.zip') # Set up the url for the download 
            __download_data(log, dataset_url, dataset_path.split('/')[0]) # Donwload the data on the target folder

        except:
            log.info(f"Failed the download of the {dataset_path.split('/')[0].split('_')[0]} split of {dataset}")
        
        # Continue the operations
        log.info(f"Unzipping {dataset_path + '.zip'} ..")
        with zipfile.ZipFile(dataset_path + '.zip', 'r') as z: # Extract zipped dataset
            z.extractall(dataset_path.split('/')[0])
        os.remove(dataset_path + '.zip') # Remove orginal zip
        
        # As default will be used '01' ('01_GT/SEG' and '01_GT/TRA') with the respective first files
        images = os.listdir(os.path.join(dataset_path, '01'))
        images.sort()
        image = tiff.imread(os.path.join(dataset_path, '01', images[0]))
        log_image_characteristics(log, image, 'image') # Log characteristics of images

        if dataset_path.split('/')[0] == 'training_data': # TODO: should be passed directly from args
            
            for folder in GT_FOLDERS:# Analyze both first mask in SEG and TRA.
                masks = os.listdir(os.path.join(dataset_path, '01_GT', folder))
                if folder == 'TRA': # Avoid the requested '.txt' file.
                    masks = [mask for mask in masks if mask.endswith('.tif')]
                masks.sort()
                mask = tiff.imread(os.path.join(dataset_path, '01_GT', folder, masks[0]))
                log_image_characteristics(log, mask, 'mask')
        
    else:
        log.info(f"'{dataset_path}' exists already!")

    return None


def download_datasets(log, args): # Main function to download the chosen datasets and set up their utilization
    """ Function to download the images in the path passed by arguments.
        Dataset folder structure is fixed.

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

    if args.download_dataset == 'all': # TODO: Provide other options

        log.info(f"All datasets will be downloaded")
        for dataset in DATASETS:
            current_train_path = os.path.join(args.train_images_path, dataset)
            current_test_path = os.path.join(args.test_images_path, dataset)

            __check_dataset(log, current_train_path, TRAINDATA_URL)
            __check_dataset(log, current_test_path, TESTDATA_URL)

    else: # Download single dataset if it is present in the list

        if args.download_dataset in DATASETS:
            log.info(f"Dataset that will be downloaded: {args.download_dataset}")
            current_train_path = os.path.join(args.train_images_path, args.download_dataset)
            current_test_path = os.path.join(args.test_images_path, args.download_dataset)

            __check_dataset(log, current_train_path, TRAINDATA_URL)
            __check_dataset(log, current_test_path, TESTDATA_URL)
            log.info(f"Data downloads completed!")
        
        else:
            log.info(f"Dataset {args.download_dataset} not found in the available list!")

    log.info(f"Program terminated")
    return None


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


        