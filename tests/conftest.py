"""
Module that provides an additional configurations for the test_* files
in case the 'test_*' functions gathered are containing specific markers.

The markers are created in order to lunch just sub-functions present in the 
pipelines tested: e.g. testing just the dataset creation instead of the dataset
creation and traininig loop.

Implementing the pattern in https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
"""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-pipeline", action="store_true", default=False, help="Run the functions that call the entire train/test pipeline"
    )
    parser.addoption(
        "--run-sub", action="store_true", default=False, help="Run the functions that are sub-features of a pipeline"
    )
    
def pytest_configure(config):
    """
    Provide a double-marker definitions to segment the testing classes
    between sub-functions and pipelines.
    """

    config.addinivalue_line("markers", "pipeline: mark test as general pipelines")
    config.addinivalue_line("markers", "sub: mark test as general pipelines")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-pipeline"):

        # Skip all the functions containig 'sub'.
        skip_sub = pytest.mark.skip(reason="Skip sub-functions during the pipeline testing")
        for item in items:
            if "sub" in item.keywords:
                item.add_marker(skip_sub)
        return
    if config.getoption("--run-sub"):

        # Skip all the functions containig 'pipeline'.
        skip_pipeline = pytest.mark.skip(reason="Skip the pipeline during the sub-functions testing")
        for item in items:
            if "pipeline" in item.keywords:
                item.add_marker(skip_pipeline)