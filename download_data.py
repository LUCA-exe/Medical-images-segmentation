"""download_data.py

This file contains the function to download and set up the train/test data.

"""

import os
import requests
import zipfile

# TODO: Move this to a 'config file' inside the repository
DATASETS = ["BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa", "Fluo-C2DL-Huh7", "Fluo-C2DL-MSC", "Fluo-N2DH-GOWT1", "Fluo-N2DL-HeLa",
            "PhC-C2DH-U373", "PhC-C2DL-PSC", "Fluo-N2DH-SIM+"]

# NOTE: For now donwload just the train/test Datasets from CTC
TRAINDATA_URL = 'http://data.celltrackingchallenge.net/training-datasets/'
TESTDATA_URL = 'http://data.celltrackingchallenge.net/challenge-datasets/'

def __download_data(url, target): # Download the datasets chosen

    #DEBUG 
    print(url)
    print(target)

    local_filename = target / url.split('/')[-1]

    with requests.get(url, stream=True) as r:
        r.raise_for_status() # Help debugging error in case of status different from 200
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                
    return local_filename


def download_datasets(log, args): # Main function to download the chosen datasets and set up their utilization
    """ Function to download the images in the path passed by arguments.
        Dataset folder structure is fixed.

    Args:
        args (dict): Arguments usefull for the download and creation of the images folder

    Returns:
        None

    """
    log.info(f"Preparing the datasets download ..")
    log.info(f"The folders used will be respectively '{args.train_images_path}' and '{args.test_images_path}'")
    # Set up the split folders
    os.makedirs(args.train_images_path, exist_ok=True)
    os.makedirs(args.test_images_path, exist_ok=True)

    if args.download_dataset == 'all': # TODO: Provide other options
        
        for dataset in DATASETS:
            current_train_path = os.path.join(args.train_images_path, dataset)
            current_test_path = os.path.join(args.test_images_path, dataset)
            
            if not os.path.isdir(current_train_path): # Check if It's already present

                try: # even if some datasets are not downloadable continue the code
                    log.info(f"Downloading train/test {dataset} dataset ..")
                    train_url = os.path.join(TRAINDATA_URL, dataset + '.zip') # Set up the url for the download 
                    test_url = os.path.join(TESTDATA_URL, dataset + '.zip')

                    __download_data(train_url, args.train_images_path) # Donwload training data
                    __download_data(test_url, args.test_images_path) # Donwload test data
                
                except:
                    log.info(f"Failed the donwload of {dataset}")

                

    else: # Donwload single dataset if it is present in the list

        pass


    return None








