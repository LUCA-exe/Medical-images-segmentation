"""This module will test the functionality of the trainining util functions.

It will call the sparse configuration functions as the optimizer or the max_epochs
setter.
"""

from training.training import get_max_epochs


class TestTrainUtils:
    """This class contains functions to test the configuration settings (epoch, optimizer etc...).
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