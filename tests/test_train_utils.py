"""This module will test the functionality of the trainining util functions.

It will call the sparse configuration functions as the optimizer or the max_epochs
setter.
"""
from typing import List
import numpy as np

from training.training import get_max_epochs, update_running_losses


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
                
    # TODO: Test the txt savings of both val and train losses.
    