"""This module tests the losses configurations and computation only.
"""
from training.losses import get_loss
from typing import Dict, Union
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss, SmoothL1Loss
from torch import nn
import torch
from types import NoneType
import numpy as np
import os

from utils import set_device
from training.losses import LossComputator, WeightedCELoss, CrossEntropyDiceLoss, MultiClassJLoss

# NOTE: DEPRECATED
def mock_set_criterion(args: dict) -> Dict[str, nn.Module]:
    """Function to retrieve the set of losses corresponding
    to the architecture.
    """
    criterion = get_loss(args["config"], args["device"])
    return criterion

def load_images(folder_path: str) -> list[np.ndarray, np.ndarray]:
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
    file_names = ["Mock-E2DV-train-t000.npy", "Mock-E2DV-train-man_seg000.npy"]
    images = []
    for file_name in file_names:

        # Load the images saved as '.npy' format
        images.append(np.load(os.path.join(folder_path, file_name)))
    return images

def add_gaussian_noise(image: np.array,
                       mean: Union[float, int] = 20,
                       var: Union[float, int] = 3):
    """Add gaussian noise to the current image using the built-in numpy method.
    """
    row,col,ch= image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    return image + gauss



class TestMockCriterionComputation:
    """This class contains functions to simulate the overall losses retrieval.

    It instantiates the losses and provide mock computation for every configuration.
    """

    def test_criterion_retrieval(self):
        """Load the loss classes based on the architecture configurations.
        """

        test_arguments = [
            {"config": {"loss": "l1"}, "device": torch.device("cpu")}
        ]

        for test_args in test_arguments:
            raise NotImplementedError
        
    def test_loss_computator(self):
        """Test the correct instantiations of the loss computator.

        Specifically, it will be tested the internal attributes containing
        the losses objects.
        """
        device = torch.device("cpu")
        class_image_labels = ["binary_border", "mask"]
        regr_image_labels = ["border", "cell"]
        test_arguments = [
            {"regression_loss": "l1", "expected_regression_loss_class": L1Loss, "classification_loss": "cross-entropy", "expected_classification_loss_class": CrossEntropyLoss},
            {"regression_loss": "smooth_l1", "expected_regression_loss_class": SmoothL1Loss, "classification_loss": "cross-entropy-dice", "expected_classification_loss_class": CrossEntropyDiceLoss},
            {"regression_loss": None, "expected_regression_loss_class": NoneType, "classification_loss": "weighted-cross-entropy", "expected_classification_loss_class": WeightedCELoss}
        ]

        for test_args in test_arguments:
            computator = LossComputator(test_args["regression_loss"], test_args["classification_loss"], device)
            for key, loss_object in computator.losses_criterion.items():

                # Assertions based on the image label.
                if key in class_image_labels:
                    assert isinstance(loss_object, test_args["expected_classification_loss_class"]) == True
                if key in regr_image_labels:
                    assert isinstance(loss_object, test_args["expected_regression_loss_class"]) == True

    def test_loss_computator_criterions(self):
        """Test the criterions computations using the LossComputator istance.
        """
        device = torch.device("cpu")
        images_folder_path = "tests"
        img, seg_mask = load_images(folder_path=images_folder_path)

        # Add artifacts on the images for the loss computation.
        img = add_gaussian_noise(img)
        seg_mask = add_gaussian_noise(seg_mask)
        img = torch.from_numpy(img)
        seg_mask = torch.from_numpy(seg_mask)

        test_arguments = [
            {"regression_loss": "l1", "expected_regression_loss_class": L1Loss, "classification_loss": "cross-entropy", "expected_classification_loss_class": CrossEntropyLoss},
        ]

        for test_args in test_arguments:
            computator = LossComputator(test_args["regression_loss"], test_args["classification_loss"], device)
            
            # TODO: Finish to test the losses with the noised original mask and images.
            raise NotImplementedError
