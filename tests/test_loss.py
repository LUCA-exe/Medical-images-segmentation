"""This module tests the losses configurations and computation only.
"""
from training.losses import get_loss
from typing import Dict, Union, Optional
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss, SmoothL1Loss
from torch import nn
import torch
from types import NoneType
import numpy as np
import os
from copy import copy

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
    """Load saved images (npy format) such us image, mask and tracking
    mask.
    
    The dtype of this numpy arrays should be uint8 (image) and uint16 (the masks).
    The mask are in the shape of (H, W) while the original image are (H, W, C). 

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

def add_gaussian_noise(image: np.ndarray,
                       mean: Union[float, int] = 20,
                       var: Union[float, int] = 3):
    """Add gaussian noise to the current image using the built-in numpy method.
    """
    image = copy(image)
    
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, image.shape)
    gauss = gauss.reshape(image.shape)
    return image + gauss

def add_class_label(mask: np.ndarray,
                    label: Union[float, int] = 1,
                    p: float = 0.7) -> np.ndarray:
    """Add the provided label class to the mask in random positions.

    The mask in input will be converted to boolean mask.

    Args:
        mask: Categorical image of shape (H, W).
        label: Class to add randomly in the image.
        p: Probability to keep the original pixel value
            in the image.
    """
    mask = copy(mask)
    random_index = np.random.choice(a = [False, True], size = mask.shape, p = [p, 1-p])
    mask[random_index] = label
    return mask

def get_mock_float_batch(image: np.ndarray, n_batch: int = 4) -> torch.Tensor:
    """Get multiple images batch from a starting RGB image.

    The original image will be transposed, sum along the C 
    axis and converted to a tensor for batch suitability. 

    Args:
        image: Image array of shape (H, W, C).
        n_batch: Number of images contained in the batch.

    Returns:
        It returns a tensor of shape (N, C = 1, H, W).
    """
    image = np.transpose(image, (2, 0, 1))
    image = np.sum(image, axis = 0)
    image = torch.from_numpy(image)

    # Add a dimension to create mock batch data
    images = [image] * n_batch
    batch = torch.stack(images, dim = 0)
    return batch

def get_mock_categorical_batch(mask: np.ndarray, n_batch: int = 4, gt: bool = True) -> torch.Tensor:
    """Get multiple images batch from a starting mask.

    The original image will be reshaped and converted to a tensor
    for batch suitability. 

    Args:
        image: Image array of shape (H, W).
        n_batch: Number of images contained in the batch.
        gt: Flag to add a new dim. If the categorical image
            is used in the 'prediction' batch, then a new
            dim. has to be added as C axis.

    Returns:
        It returns a tensor of shape (N, C = 1, H, W).
    """
    mask = copy(mask)
    mask = np.expand_dims(mask, axis = 0)
    if not gt:
        mask = torch.from_numpy(mask)
    else:
        mask = torch.from_numpy(mask)
        #mask = torch.tensor(mask, dtype=torch.long)
    # Add a dimension to create mock batch data
    masks = [mask] * n_batch
    batch = torch.stack(masks, dim = 0)
    return batch


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
            for key, loss_object in computator.loss_criterions.items():

                # Assertions based on the image label.
                if key in class_image_labels:
                    assert isinstance(loss_object, test_args["expected_classification_loss_class"]) == True
                if key in regr_image_labels:
                    assert isinstance(loss_object, test_args["expected_regression_loss_class"]) == True

    def test_loss_computator_criterions(self):
        """Test the criterions computations using the LossComputator istance.

        It will uses some default image saved in *.npy format for the computation. The
        original image and mask are converted to a float64 dtype for successive processing.
        """
        device = torch.device("cpu")
        images_folder_path = "tests"

        # Forming the "mock" ground-truth batch.
        img, seg_mask = load_images(folder_path=images_folder_path)
        img = img.astype(np.float64, copy=False)

        # up to this point the mask image should contain one label for each distinct element.
        seg_mask = np.array(seg_mask, dtype=bool)
        seg_mask = seg_mask.astype(np.float64, copy=False)
        gt_float_img = get_mock_float_batch(img)
        gt_categorical_img = get_mock_categorical_batch(seg_mask)
        gt_batch = {"cell": gt_float_img, "mask": gt_categorical_img}
            
        test_arguments = [
            {"regression_loss": "l1", "classification_loss": "cross-entropy",  "mean": 40, "var": 20},
            {"regression_loss": "smooth_l1", "classification_loss": "weighted-cross-entropy", "mean": 60, "var": 10}
        ]
        for test_args in test_arguments:

            # Add artifacts on the images for the loss computation.
            noised_img = add_gaussian_noise(img, test_args["mean"], test_args["var"])
            noised_seg_mask = add_class_label(seg_mask)
            pred_float_img = get_mock_float_batch(noised_img)
            pred_categorical_img = get_mock_categorical_batch(noised_seg_mask, gt=False)
            pred_batch = {"cell": pred_float_img, "mask": pred_categorical_img}
            computator = LossComputator(test_args["regression_loss"], test_args["classification_loss"], device)
            _ = computator.compute_loss(pred_batch, gt_batch)
            
