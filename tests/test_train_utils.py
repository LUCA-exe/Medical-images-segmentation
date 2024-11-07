"""This module will test the functionality of the trainining util functions.

It will call the sparse configuration functions as the optimizer or the max_epochs
setter.
"""
from typing import List
import numpy as np
from pathlib import Path

from training.training import get_max_epochs, update_running_losses
from net_utils.utils import save_training_loss


class TestTrainUtils:
    """This class contains functions to test the configuration settings (epoch, optimizer etc...)
    and additional util functions.
    """

    def test_max_epochs_retrieval(self):
        """This function will try both the base number of epochs and the overriding option.
        """
        test_arguments = [
            {"model_pipeline": "dual-unet", "n_sample": 34, "expected_epochs": 200},
            {"model_pipeline": "original-dual-unet", "n_sample": 34, "expected_epochs": 40},
            {"model_pipeline": None, "n_sample": 51, "expected_epochs": 480},
            {"model_pipeline": None, "n_sample": 101, "expected_epochs": 400}
        ]

        for test_args in test_arguments:
            assert get_max_epochs(test_args["n_sample"], test_args["model_pipeline"]) == test_args["expected_epochs"]

    def test_update_losses(self):
        """Simulate the cumulative updates of losses.

        The tested util function will be used during both val and train
        phase.
        """
        mock_batch_size = 1
        test_arguments = [
            {"losses_to_sum": [1.0, 2.0], "expected_losses": [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]},
            {"losses_to_sum": [1.0, 3.0], "expected_losses": [[1.0, 3.0], [2.0, 6.0]]},
        ]
        for test_args in test_arguments:
            curr_losses = [0.0, 0.0]
            for losses in test_args["expected_losses"]:
                print(losses)
                curr_losses = update_running_losses(curr_losses, test_args["losses_to_sum"], batch_size=mock_batch_size)
                assert isinstance(curr_losses, List) == True
                assert curr_losses == losses

    # TODO: Finish automated parsing of the file to check for the integrity. 
    def test_losses_saving(self):
        """It testes the *.txt generation with mock losses.

        This function will generate a single *.txt file (inside the ./tests/ folder) 
        that will be overwrited every time a use-case is tested with different 
        loss labels and values.
        """
        test_arguments = [
            {"loss_values": [[1.0, 3.4, 4.23]], "loss_labels": ["cell_label", "mask_label"]},
            {"loss_values": [[1.0, 3.4, 4.23], [2.0, 4.4, 5.23]], "loss_labels": ["cell_label", "binary_border_label", "mask_label"]},
        ]
        mock_path_model = Path('./')
        second_run = False
        tot_time = 0.001
        tot_epochs = 1
        mock_config = {"run_name": "Mock-E2DV-train_GT_01_640_kit-ge_dual-unet_00"}
        for test_args in test_arguments:
            save_training_loss(test_args["loss_labels"], test_args["loss_values"], test_args["loss_values"],
                               second_run, mock_path_model, mock_config, tot_time, tot_epochs)