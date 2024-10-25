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

        This function will call the method to save processed images on the local disk
        as trial dataset respecting the expected structure.
        """
        # Hard-coded values for the expected labels.
        expected_label_type = {"image": {"dtype": torch.float32},
                               "cell_label": {"dtype": torch.float32},
                               "border_label": {"dtype": torch.float32},
                               "mask_label": {"dtype": torch.int64, "unique_values": 2},
                               "binary_border_label": {"dtype": torch.int64, "unique_values": 2}
                               }
        path_data = Path("training_data")
        default_args = read_json_file("./tests/mock_train_args.json")
        test_arguments = [
            {"dataset": "Mock-E2DV-train", "crop_size": 320, "min_a_images": 3}
        ]

        for test_args in test_arguments:
            run_parameters = update_default_args(default_args, test_args)
            
            # Compose current dataset folder.
            dataset_folder = f"{run_parameters['dataset']}_{run_parameters['mode']}_{run_parameters['split']}_{run_parameters['crop_size']}"
            
            # Remove the 'mock' created dataset folder if already exists.
            if os.path.isdir(join(run_parameters["train_images_path"], dataset_folder)):
                rmtree(join(run_parameters["train_images_path"], dataset_folder))
            mock_training_dataset_creation_pipeline(run_parameters)

            # NOTE: min_value and max_value fixed params.
            data_transforms = augmentors(label_type = "distance", min_value=0, max_value=65535)
            dataset_name = "{}_{}_{}_{}".format(test_args["dataset"], "GT", "01", test_args["crop_size"])
            datasets = {x: CellSegDataset(root_dir = path_data / Path(dataset_name), mode=x, transform=data_transforms[x])
                        for x in ['train', 'val']}
            
            for key, dataset in datasets.items():

                # NOTE: Temporary testing just the first sample.
                first_sample = dataset[0]
                for key, item in first_sample.items():
                    assert expected_label_type[key]["dtype"] == item.dtype

                    # Checks on expected unique values just for category data (e.g. mask).
                    if "unique_values" in expected_label_type[key]:
                         assert expected_label_type[key]["unique_values"] == len(torch.unique(item))

            

