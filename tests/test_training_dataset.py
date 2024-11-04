"""Module to test the consistency of the dataloader returned images.

It employes a function for setting the training dataset on disk
from the ./test_train_pipelines.py module.
it involves the testing of the Dataset class custom dataset and 
for the transformation functions.

To test just the transformation functions:
...> python -m pytest -v --run-sub tests/test_training_dataset.py

To test just the entire pipeline:
...> python -m pytest -v --run-pipeline tests/test_training_dataset.py

In some tests it will uses the *.npy files listed in the ./tests/README.txt file.
"""
from typing import Dict, Any, List, Tuple
from shutil import rmtree
import torch
import os
from os.path import join
from pathlib import Path, PurePath
import numpy as np
import pytest
from copy import copy, deepcopy

from training.mytransforms import augmentors
from utils import read_json_file
from tests.test_train_pipelines import mock_training_dataset_creation_pipeline, update_default_args
from tests.test_loss import load_images
from training.cell_segmentation_dataset import CellSegDataset

def create_mock_dict_for_tranformations(float_image: np.array, 
                                        categorical_image: np.array,
                                        original_image_labels: List[str] = ["image"],
                                        float_labels: List[str] = ["cell_label", "border_label"],
                                        categorical_labels: List[str] = ["mask_label", "binary_border_label"]) -> Dict[str, np.array]:
    """It generates a mock dictionary containing compatible images for the Compose obj.

    For the further processing (in the Compose class) the categorical images have
    to be in the [H, W, C] shape.
    """
    categorical_image = np.expand_dims(copy(categorical_image), axis = 2)
    mock_sample = {}

    for label in original_image_labels:
        mock_sample[label] = float_image.astype(np.uint16)

    for label in float_labels:
        mock_sample[label] = float_image.astype(np.float32)

    for label in categorical_labels:
        mock_sample[label] = categorical_image.astype(np.bool_)
        mock_sample[label] = categorical_image.astype(np.uint8)
    return mock_sample

class TestCustomDataset:
    """This class contains functions to set local training dataset folders and tests 
    custom Dataset methods and classes for training.
    """

    # NOTE: Refactoring in process.
    @pytest.mark.pipeline
    def test_custom_dataset(self):
        """Set the tranform function and a custom torch.utils.data.Dataset class.

        This function will call the method to save processed images on the local disk
        as trial dataset respecting the expected structure.
        """
        # Hard-coded values for the expected labels.
        expected_label_type = {"image": {"dtype": torch.float32},
                               "cell_label": {"dtype": torch.float32},
                               "border_label": {"dtype": torch.float32},
                               "mask_label": {"dtype": torch.int64, "unique_values": 2},
                               "binary_border_label": {"dtype": torch.int64, "unique_values": 2},
                               "id": {"dtype": str}
                               }
        path_data = Path("training_data")
        default_args = read_json_file("./tests/mock_train_args.json")
        test_arguments = [
            {"model_pipeline": "dual-unet", "dataset": "Mock-E2DV-train", "crop_size": 320, "min_a_images": 3, "expected_keys": ["image", "id", "cell_label", "border_label"]},
            {"model_pipeline": "original-dual-unet", "dataset": "Mock-E2DV-train", "crop_size": 320, "min_a_images": 3, "expected_keys": ["image", "id", "cell_label", "mask_label", "binary_border_label"]}
        ]

        for test_args in test_arguments:
            run_parameters = update_default_args(default_args, test_args)
            
            # Compose current dataset folder.
            dataset_folder = f"{run_parameters['dataset']}_{run_parameters['mode']}_{run_parameters['split']}_{run_parameters['crop_size']}"
            
            # Remove the 'mock' created dataset folder if already exists.
            if os.path.isdir(join(run_parameters["train_images_path"], dataset_folder)):
                rmtree(join(run_parameters["train_images_path"], dataset_folder))
            _, _, _, _, _, _, train_args_cls = mock_training_dataset_creation_pipeline(run_parameters)

            # NOTE: min_value and max_value fixed params.
            data_transforms = augmentors(label_type = "distance", min_value=0, max_value=65535)
            dataset_name = "{}_{}_{}_{}".format(test_args["dataset"], "GT", "01", test_args["crop_size"])
            
            # Set-up the CellSegDataset class with the current needed images.
            labels = train_args_cls.get_requested_image_labels()
            datasets = {x: CellSegDataset(root_dir = path_data / Path(dataset_name), labels = labels, mode=x, transform=data_transforms[x])
                        for x in ['train', 'val']}
            
            for key, dataset in datasets.items():
                # Testing just the first sample.
                first_sample = dataset[0]

                # Test keys content.
                assert sorted(test_args["expected_keys"]) == sorted(first_sample.keys())
                for key, item in first_sample.items():

                    # Differentiate between tensor and built-in types.
                    if key != "id": assert expected_label_type[key]["dtype"] == item.dtype
                    else: assert expected_label_type[key]["dtype"] == type(item)

                    # Checks on expected unique values just for category data (e.g. mask).
                    if "unique_values" in expected_label_type[key]:
                        assert expected_label_type[key]["unique_values"] == len(torch.unique(item))

    @pytest.mark.sub
    def test_transformation_functions(self):
        """It tests the Compose class in case of val and training datasets.

        The Compose object is used inside the CellSegDataset class. 
        This function will load a set of mock images present in the ./tests
        folder to test the integrity of the tranformations.
        """ 
        images_folder_path = "tests"
        img, seg_mask = load_images(folder_path = images_folder_path)

        # Get mock sample for the current labels transformation
        mock_sample = create_mock_dict_for_tranformations(img, seg_mask)
        data_transform = augmentors(label_type = "null", min_value=0, max_value=65535)
        for mode in ['train', 'val']:
            tranform_functions = data_transform[mode]
            _ = tranform_functions(deepcopy(mock_sample))
