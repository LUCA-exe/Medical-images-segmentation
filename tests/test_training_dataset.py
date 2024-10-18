"""Module to test the consistency of the dataloader returned images.

It employes a function for setting the training dataset on disk
from the ./test_train_pipelines.py module.
"""
from typing import Dict, Any, List, Tuple
from shutil import rmtree
import torch
import os
from os.path import join
from pathlib import Path

from training.mytransforms import augmentors
from utils import read_json_file

from tests.test_train_pipelines import mock_training_dataset_creation_pipeline, update_default_args
from training.cell_segmentation_dataset import CellSegDataset

class TestTrainingDataset:
    """This class contains functions to set local training dataset folders and tests 
    custom Dataset methods and classes for training.
    """

    # NOTE: Refactoring in process.
    def test_training_custom_dataset(self):
        """Set the tranform function and a custom torch.utils.data.Dataset class.
        """
        path_data = Path("training_data")
        default_args = read_json_file("./tests/mock_train_args.json")
        test_arguments = [
            {"dataset": "Mock-E2DV-train", "crop_size": 320, "min_a_images": 3 }
        ]

        for test_args in test_arguments:
            run_parameters = update_default_args(default_args, test_args)
            
            # Compose current dataset folder.
            dataset_folder = f"{run_parameters['dataset']}_{run_parameters['mode']}_{run_parameters['split']}_{run_parameters['crop_size']}"
            
            # Remove the 'mock' created dataset folder if already exists.
            if os.path.isdir(join(run_parameters["train_images_path"], dataset_folder)):
                rmtree(join(run_parameters["train_images_path"], dataset_folder))
            mock_training_dataset_creation_pipeline(run_parameters)
            
            data_transforms = augmentors(label_type = "distance", min_value=0, max_value=65535) # NOTE: min_value and max_value fixed params.
            dataset_name = "{}_{}_{}_{}".format(test_args["dataset"], "GT", "01", test_args["crop_size"])
            
            # In the original script it was implemented the 'all' dataset plus ST option.
            datasets = {x: CellSegDataset(root_dir = path_data / Path(dataset_name), mode=x, transform=data_transforms[x])
                        for x in ['train', 'val']}
            
            for key, dataset in datasets.items():
                first_sample = dataset[0]
            

