from typing import Dict, Any, List
from pathlib import Path
from shutil import rmtree
import os
from os.path import join, exists

from utils import load_environment_variables, set_current_run_folders, \
    create_logging, read_json_file, set_device, check_path
from training.create_training_sets import create_ctc_training_sets

def mock_training_dataset_creation_pipeline(args: Dict) -> None:
    """
    This function provide a 'mock' train pipeline to 
    test the overall functions from the set up of the loggers,
    folders to the building of the neural networks.

    Specifically it will call all the main functions executed
    during a proper model training.

    Args:
        args: Arguments already parsed from calling function.

    Returns:
        None
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

    # Originally in another function.
    if args["pre_processing_pipeline"] == 'kit-ge':
        log.info(f"Creation of the training dataset using {args['pre_processing_pipeline']} pipeline for {args['crop_size']} crops")
        
        create_ctc_training_sets(log, 
                                    path_data = path_data,
                                    mode = args["mode"],
                                    cell_type = cell_type,
                                    split = args["split"],
                                    min_a_images = args["min_a_images"],
                                    crop_size = args["crop_size"])
    else:
        raise ValueError("This argument support just 'kit-ge' as pre-processing pipeline")

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
        current_files = [file for file in dir_list if file.startswith(prefix)]
        if len(current_files) != expected_image_number:
            return False
    return True


class TestMockTrainPipelines:
    """
    This class contains functions to simulate the overall training
    pipeline, from the creation of the training set to the instantiation
    and training of the deep learning models.
    """

    # FIXME: For now just implement the dataset creation with a 'mock' dataset folder.
    def test_training_dataset_creation(self):
        """
        Set environment folders and run the creation of the training dataset folder.
        """
        expected_folders = ["A", "B", "train", "val"]
        expected_files = ["info.json"]
        expected_images_prefix = ["dist_cell_", "dist_neighbor_", "img_", "mask_"]
        default_args = read_json_file("./tests/mock_train_args.json")
        test_arguments = [
            {"dataset": "Mock-E2DV-train", "crop_size": 320, "min_a_images": 30, "folder_to_check": ["A"], "expected_images": [38]}
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
                result = check_created_images(test_args["dataset"],
                                     current_folder_to_check,
                                     expected_images_prefix,
                                     expected_image_number)
                assert result == True