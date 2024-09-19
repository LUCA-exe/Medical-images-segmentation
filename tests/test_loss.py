"""This module tests the losses configurations and computation only.
"""

from training.losses import get_loss
from typing import Dict
import pytest
import torch

def mock_set_criterion(args: dict) -> Dict[str, torch.nn.Module]:
    """Function to retrieve the set of losses corresponding
    to the architecture.
    """
    criterion = get_loss(args["config"], args["device"])
    return criterion


class TestMockCriterionComputation:
    """
    This class contains functions to simulate the overall losses
    intantiation and retrieval with multiple configuration.
    """

    @pytest.mark.sub
    def test_criterion_retrieval(self):
        """Load the loss classes based on the architecture configurations.
        """

        
        test_arguments = [
            {"config": {"loss": "l1"}, "device": torch.device("cpu")}
        ]

        for test_args in test_arguments:
            raise NotImplementedError
