"""
This testing module provide the following test features (WORK IN PROGRESS):
- Dataset creation.
- Training Dataloader.
- Train pipeline.

To test just the sub-functions:
...> python -m pytest -v --run-sub tests/test_train_pipelines.py

To test just the entire pipeline:
...> python -m pytest -v --run-pipeline tests/test_train_pipelines.py

In some tests it will uses the *.npy files listed in the ./tests/README.txt file.
"""

import pytest
import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path
from shutil import rmtree
import torch
import os
import numpy as np
from os.path import join, exists

from training.data_generation_classes import data_generation_factory
from train import set_up_training_loops
from net_utils.utils import create_model_architecture
from utils import load_environment_variables, set_current_run_folders, \
    create_logging, read_json_file, set_device, check_path, \
        train_factory
from training.create_training_sets import create_ctc_training_sets

def mock_training_dataset_creation_pipeline(args: Dict) -> Tuple[logging.Logger, int, torch.device, str, str, str]:
    """This function provide a 'mock' train pipeline to 
    test the overall functions from the set up of the loggers,
    folders to the building of the neural networks.

    Specifically it will call all the main functions executed
    during a proper model training.

    Args:
        args: Arguments already parsed from calling function.

    Returns:
        A tuple of any types for the further training loop execution.
    """
    # Set up the current environment/folders
    load_environment_variables()
    set_current_run_folders()
    log = create_logging()
    device, num_gpus = set_device()
    log.info(f"System detected {device} device and {num_gpus} GPUs available.")

    # Store local variables in 'Path' type obj.
    path_data = Path(args["train_images_path"])
    path_models = Path(args["models_folder"])
    cell_type = Path(args["dataset"])

    # Using the 'os.path.join' method with 'Path' type variables.
    if check_path(log, path_data) and check_path(log, join(path_data, cell_type)):
        trainset_name = args["dataset"] # 'args.dataset' used as cell type

    # FIXME: Duplicated for now - testing purposes.
    train_args_cls = train_factory.create_arg_class(args["model_pipeline"],
                                                         args["act_fun"],
                                                         args["batch_size"],
                                                         args["filters"],
                                                         args["detach_fusion_layers"],
                                                         args["iterations"],
                                                         args["loss"],
                                                         args["norm_method"],
                                                         args["optimizer"],
                                                         args["pool_method"],
                                                         args["pre_train"],
                                                         args["retrain"],
                                                         args["split"],
                                                         args["crop_size"],
                                                         args["mode"],
                                                         args["pre_processing_pipeline"],
                                                         args["softmax_layer"],
                                                         args["classification_loss"])

    # Originally in another function.
    if args["pre_processing_pipeline"] == 'kit-ge':
        log.info(f"Creation of the training dataset using {args['pre_processing_pipeline']} pipeline for {args['crop_size']} crops")
        
        create_ctc_training_sets(log, 
                                    path_data = path_data,
                                    mode = args["mode"],
                                    cell_type = cell_type,
                                    split = args["split"],
                                    min_a_images = args["min_a_images"],
                                    crop_size = args["crop_size"],
                                    train_arg_class = train_args_cls)
    else:
        raise ValueError("This argument support just 'kit-ge' as pre-processing pipeline")
    
    # FIXME: Temporary passing the 'logging.Logger()' object around
    return log, num_gpus, device, path_data, trainset_name, path_models

def mock_training_loop_pipeline(log: logging.Logger, args: Dict, num_gpus: int, device: torch.device,
                                path_data: str, trainset_name: str, path_models: str) -> None:
    """
    It starts the parsing of the arguments to create the architecture and start the training
    loop if the dataset is craeted correctly in the previous steps.
    """
    # FIXME: Adjust temporary the args to be 'iterables' (e.g 'crop_size).
    args["crop_size"] = [args["crop_size"]]
    
    # NOTE: it was originally a single function - adjust after first tests
    train_args_cls = train_factory.create_arg_class(args["model_pipeline"],
                                                         args["act_fun"],
                                                         args["batch_size"],
                                                         args["filters"],
                                                         args["detach_fusion_layers"],
                                                         args["iterations"],
                                                         args["loss"],
                                                         args["norm_method"],
                                                         args["optimizer"],
                                                         args["pool_method"],
                                                         args["pre_train"],
                                                         args["retrain"],
                                                         args["split"],
                                                         args["crop_size"],
                                                         args["mode"],
                                                         args["pre_processing_pipeline"],
                                                         args["softmax_layer"],
                                                         args["classification_loss"])

    # Training parameters used for all the iterations/crop options given.                         
    log.info(train_args_cls)
    
    # FIXME: Another sub-dictionary - see if the 'train_args' classes make sense, maybe redundant.
    model_config = {'architecture': train_args_cls.get_arch_args(),
                    'batch_size': train_args_cls.batch_size,
                    'batch_size_auto': 2,
                    'label_type': "distance", # NOTE: Fixed param.
                    'loss': train_args_cls.loss,
                    'classification_loss': train_args_cls.classification_loss,
                    'num_gpus': num_gpus,
                    'optimizer': train_args_cls.optimizer,
                    'max_epochs': 1  # NOTE: Set to 1 for testing purposes 
                    }  # TODO: Add the device obj. directly inside the 'model_config' hashmap. 
    
    log.info(f"Model configuration: {model_config}")
    net = create_model_architecture(log, model_config, device, num_gpus)

    # Call the original methods - method to refactor.
    set_up_training_loops(log, train_args_cls, path_data, trainset_name, path_models, model_config, net, num_gpus, device)

def mock_processed_images(args: Dict) -> Dict[str, np.ndarray]:
    """It use the images generation factory class and return the computed
    dictionary.
    """
    processed_images = data_generation_factory.create_training_data(args["labels"], args["img"], args["mask"], 
                                                                    args["tra_gt"], args["td_settings"])
    return processed_images

def update_default_args(default_args: Dict, new_args: Dict) -> Dict[str, Any]:
    """
    It updates/adds key-value pairs to the 'default_args' from
    the provided 'new_args'.

    It changes the 'default_args' in-place and returns it.

    Args:
        default_args: Arguments read from a parameters file or similar.
        new_args: Key-Value pairs to add to the parameters hasmap.

    Returns:
        Hasmap encompassing the args updated.
    """
    for key, value in new_args.items():
        default_args[key] = value
    return default_args

def check_created_training_set_structure(folder_path: str, expected_folders: List[str], expected_files: List[str]) -> bool:
    """
    Checking the integrity/characteristics of the created folders
    prior to the training loops.
    """ 
    dir_list = os.listdir(folder_path)
    
    # The expected folders/files structure is not respected
    if sorted(dir_list) != sorted(expected_files + expected_folders):
        return False 
    return True

def load_images(folder_path: str) -> list[np.ndarray, np.ndarray, np.ndarray]:
    """Load saved images (.npy format) such us image, mask and tracking
    mask.
    
    The dtype of this numpy arrays should be uint8 (image) and uint16 (the masks).

    Args:
        folder_path: String corresponding to the folder
        path containg the files to load.

    Returns:
        A list containing respectively the image, mask and the
        tracking mask.
    """
    file_names = ["Mock-E2DV-train-t000.npy", "Mock-E2DV-train-man_seg000.npy", "Mock-E2DV-train-man_track000.npy"]
    images = []
    for file_name in file_names:

        # Load the images saved as '.npy' format
        images.append(np.load(os.path.join(folder_path, file_name)))
    return images

def check_created_images(dataset_folder: str, folder_path: str, expected_image_prefixes: List[str], expected_image_number: int) -> bool:
    """
    Checks the number of created images and the format to be suitable
    for train/test the deep learning models.

    Args:
        folder_path: Path to check for the image files.
        expected_image_prefixes: List of substring that the image files have to cover.
        expected_image_number: Every 'group' of images with a determined
        prefix has to have the provided number of files created.
    """
    dir_list = os.listdir(folder_path)
    for file in dir_list:

        # Fetch the initial part of the name for each file
        curr_file_name = file.split(dataset_folder)[0]
        if not curr_file_name in expected_image_prefixes:
            return False
    
    # For each group of images count the number of occurrences
    for prefix in expected_image_prefixes:
        current_files = [file for file in dir_list if file.split(dataset_folder)[0] == prefix]
        if len(current_files) != expected_image_number:
            return False
    return True


class TestMockTrainPipelines:
    """
    This class contains functions to simulate the overall training
    pipeline, from the creation of the training set to the instantiation
    and training of the deep learning models.
    """

    @pytest.mark.sub
    def test_images_creation(self):
        """Load image and masks to compute the hasmap containing the images
        to crop and save.

        For example the creation of neighbor and distance tranformation.
        """
        images_folder_path = "tests"
        
        # Usually computed dinamically based on the current dataset.
        mock_settings = {'search_radius': 100,
            'disk_radius': 3, 
            'min_area': 20,
            'max_mal': 200,
            'scale': 1,
            'crop_size': 320}
        
        test_arguments = [
            {"td_settings": mock_settings, "labels": tuple(["dist_cell_and_neighbor", ]), "expected_keys": ["dist_cell", "dist_neighbor", "img", "mask", "tra_gt"],
             "expected_types": ['uint8', 'uint16', 'float32']},
             {"td_settings": mock_settings, "labels": tuple(["mask_label", "binary_border_label"]), "expected_keys": ["mask_label", "binary_border_label", "img", "mask", "tra_gt"],
             "expected_types": ['uint8', 'uint16', 'float32']}
        ]

        for test_args in test_arguments:
            img, seg_mask, tra_gt = load_images(folder_path=images_folder_path)
            test_args["img"] = img
            test_args["mask"] = seg_mask
            test_args["tra_gt"] = tra_gt
            processed_images = mock_processed_images(test_args)

            # Assert the type requested by the challenge guidelines before the labels.
            assert test_args["expected_types"][0] == str(processed_images["img"].dtype)
            for label, value in processed_images.items():
                if label in ["mask_label", "binary_border_label", "mask"]:
                    assert test_args["expected_types"][1] == str(value.dtype)
                elif label in ["dist_cell", "dist_neighbor"]:
                    assert test_args["expected_types"][2] == str(value.dtype)

            # Assert for the returned images labels from the factory
            assert sorted(test_args["expected_keys"]) == sorted(processed_images.keys())
            
    @pytest.mark.sub
    def test_training_dataset_creation(self):
        """
        Set environment folders and run the creation of the training dataset folder.
        """
        expected_folders = ["A", "B", "train", "val"]
        expected_files = ["info.json"]
        default_args = read_json_file("./tests/mock_train_args.json")
        test_arguments = [
            {"model_pipeline": "dual-unet", "dataset": "Mock-E2DV-train", "crop_size": 320, "min_a_images": 30, "folder_to_check": ["A"], "expected_images": [38], "expected_images_prefix": ["dist_cell_", "dist_neighbor_", "img_", "mask_"]},
            {"model_pipeline": "original-dual-unet", "dataset": "Mock-E2DV-train", "crop_size": 320, "min_a_images": 30, "folder_to_check": ["A"], "expected_images": [38], "expected_images_prefix": ["dist_cell_", "binary_border_label_", "mask_label_", "img_", "mask_"]}
        ]

        for test_args in test_arguments:
            run_parameters = update_default_args(default_args, test_args)
            
            # Compose current dataset folder.
            dataset_folder = f"{run_parameters['dataset']}_{run_parameters['mode']}_{run_parameters['split']}_{run_parameters['crop_size']}"
            
            # Remove the 'mock' created dataset folder if already exists.
            if os.path.isdir(join(run_parameters["train_images_path"], dataset_folder)):
                rmtree(join(run_parameters["train_images_path"], dataset_folder))

            mock_training_dataset_creation_pipeline(run_parameters)
            # Compose the current dataset folder absolut path.
            dataset_folder_path = join(run_parameters["train_images_path"], dataset_folder)
            structure_result = check_created_training_set_structure(folder_path = dataset_folder_path,
                                                                    expected_folders = expected_folders,
                                                                    expected_files = expected_files)
            assert structure_result == True
            
            # Check the number of occurrences and the names of the images.
            for folder, expected_image_number in zip(run_parameters["folder_to_check"], run_parameters["expected_images"]):
                current_folder_to_check = join(dataset_folder_path, folder)
                images_integrity = check_created_images(test_args["dataset"],
                                     current_folder_to_check,
                                     test_args["expected_images_prefix"],
                                     expected_image_number)
                assert images_integrity == True

    # FIXME: Main pipeline to refactor - both for the dataloder and for the images of the training set created.
    @pytest.mark.pipeline
    def test_training_loop(self):
        """
        Set environment folders, run the creation of the training dataset folder and 
        execute the training loop.
        """
        
        default_args = read_json_file("./tests/mock_train_args.json")
        test_arguments = [
            {"dataset": "Mock-E2DV-train", "crop_size": 640}
        ]

        for test_args in test_arguments:
            run_parameters = update_default_args(default_args, test_args)
            
            # Compose current dataset folder.
            dataset_folder = f"{run_parameters['dataset']}_{run_parameters['mode']}_{run_parameters['split']}_{run_parameters['crop_size']}"
            
            # Remove the 'mock' created dataset folder if already exists.
            if os.path.isdir(join(run_parameters["train_images_path"], dataset_folder)):
                rmtree(join(run_parameters["train_images_path"], dataset_folder))

            log, num_gpus, device, path_data, trainset_name, path_models = mock_training_dataset_creation_pipeline(run_parameters)
            
            # TODO: Just compose an hasmap with all the path split/unified and pass that.
            mock_training_loop_pipeline(log, run_parameters, num_gpus, device, path_data,
                                        trainset_name, path_models)
            