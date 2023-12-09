"""download_data.py

This file contains the function to download and set up the train/test data.

"""

import os
import requests
import zipfile
import tifffile as tiff

from img_processing.imageUtils import *

# TODO: Move this to a 'config file' inside the repository.
# TODO: If my dataset will become public, manage the download from this script as additional option.
DATASETS = ["BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa", "Fluo-C2DL-Huh7", "Fluo-C2DL-MSC", "Fluo-N2DH-GOWT1", "Fluo-N2DL-HeLa",
            "PhC-C2DH-U373", "PhC-C2DL-PSC", "Fluo-N2DH-SIM+"]


# NOTE: For now download just the train/test Datasets from CTC
TRAINDATA_URL = 'http://data.celltrackingchallenge.net/training-datasets/'
TESTDATA_URL = 'http://data.celltrackingchallenge.net/test-datasets/'


def __download_data(log, url, target): # Download the datasets chosen with a specific chunk size

    local_filename = os.path.join(target, url.split('/')[-1])
    
    # TODO: Try a couple of times if raise status give error. 
    with requests.get(url, stream=True) as r:

        if not r.raise_for_status() is None: # If the method doesn't return NoneType It is bad
            log.debug(f"Status for the url {url}: {r.raise_for_status()}") # Help debugging error in case of status different from 200
        
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                
    return local_filename


def __check_dataset(log, dataset_path, url): # Check if It is alredy available, if not download it (function used both for test/train splits)
    
    dataset = dataset_path.split('/')[-1] # Extract the name

    if not os.path.isdir(dataset_path): # Check if It's already present

        log.info(f"Downloading {dataset_path.split('/')[0].split('_')[0]} {dataset} dataset ..")
        try: # even if some datasets are not downloadable continue the code

            dataset_url = os.path.join(url, dataset + '.zip') # Set up the url for the download 
            __download_data(log, dataset_url, dataset_path.split('/')[0]) # Donwload the data on the target folder

            log.info(f"Unzipping {dataset_path + '.zip'} ..")
            with zipfile.ZipFile(dataset_path + '.zip', 'r') as z: # Extract zipped dataset
                z.extractall(dataset_path.split('/')[0])
            os.remove(dataset_path + '.zip') # Remove orginal zip
            
            # As default will be used '01' and '01_GT/SEG' with the respective first files
            images = os.listdir(os.path.join(dataset_path, '01'))
            images.sort()
            image = tiff.imread(os.path.join(dataset_path, '01', images[0]))
            log_image_characteristics(log, image, 'image') # Log characteristics of images

            if dataset_path.split('/')[0] == 'training_data': # TODO: should be passed directly from args
                masks = os.listdir(os.path.join(dataset_path, '01_GT/SEG'))
                masks.sort()
                mask = tiff.imread(os.path.join(dataset_path, '01_GT/SEG', masks[0]))
                log_image_characteristics(log, mask, 'mask')

        except:
            log.info(f"Failed the download of the {dataset_path.split('/')[0].split('_')[0]} split of {dataset}")
        
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








