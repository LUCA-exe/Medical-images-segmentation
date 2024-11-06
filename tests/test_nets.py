"""This module will test the set up of the neural networks for both validation and traning phases.

It will perform system tests for the architectures consistency during train/val inference (structure implemented from papers) and 
unit tests for the custom inner Modules as well. 
Functions in this module will read the arguments from a file (./tests/*.json).

To test just the integrity of the custom Module of the nets:
...> python -m pytest -v --run-sub tests/test_nets.py

To test just the entire architecture:
...> python -m pytest -v --run-pipeline tests/test_nets.py

Launching pytest with -s will enable the stdout (on console for example).
"""
import pytest
from torch.nn import ModuleList
import numpy as np
import torch

from tests.test_train_pipelines import update_default_args, load_images
from net_utils.unets import build_unet, DUNet, ODUNet, ConvBlock
from utils import load_environment_variables, \
    create_logging, read_json_file, set_device, train_factory

def get_unet(args):
    """Util function to set up the parameters for the U-net retrieval.
    """
    load_environment_variables()
    log = create_logging()
    device, num_gpus = set_device()
    args["crop_size"] = [args["crop_size"]]
    log.info(f"**** THIS LOG FILE IS CREATED USING A MOCK PIPELINE: get_unet(args) method in the ./tests/test_nets.py module ****")
    train_args_cls = train_factory.create_arg_class(args["model_pipeline"],
                                                         args["act_fun"],
                                                         args["batch_size"],
                                                         args["filters"],
                                                         args["detach_fusion_layers"],
                                                         args["iterations"],
                                                         args["loss"],
                                                         args["norm_method"],
                                                         args["optimizer"],
                                                         args["pool_method"],
                                                         args["pre_train"],
                                                         args["retrain"],
                                                         args["split"],
                                                         args["crop_size"],
                                                         args["mode"],
                                                         args["pre_processing_pipeline"],
                                                         args["softmax_layer"],
                                                         args["classification_loss"])

    model_config = {'architecture': train_args_cls.get_arch_args(),
                    'batch_size': train_args_cls.batch_size,
                    'batch_size_auto': 2,
                    'label_type': "distance", # NOTE: Fixed param.
                    'loss': train_args_cls.loss,
                    'classification_loss': train_args_cls.classification_loss,
                    'num_gpus': num_gpus,
                    'optimizer': train_args_cls.optimizer,
                    'max_epochs': 1  # NOTE: Set to 1 for testing purposes 
                    }  # TODO: Add the device obj. directly inside the 'model_config' hashmap. 
    
    net = build_unet(log, 
                           unet_type = model_config['architecture'][0],
                           act_fun = model_config['architecture'][2],
                           pool_method = model_config['architecture'][1],
                           normalization = model_config['architecture'][3],
                           device = device,
                           num_gpus = model_config['num_gpus'],
                           ch_in = 1,
                           ch_out = 1,
                           filters = model_config['architecture'][4],
                           detach_fusion_layers = model_config['architecture'][5],
                           softmax_layer = model_config['architecture'][6])
    return net
    
def prepare_img(img: np.ndarray) -> torch.Tensor:
    """Prepare the image for the net inference.

    It will deacrease dimensions (even decreased, the dims for the NN has 
    to be adjusted for the concatenation in the decoder paths) and format 
    to be suitable for a quick inference on the chosen network.

    Returns:
        A torch tensor, moved on the correct device, for the network
        inference.
    """
    img = np.sum(img, axis=2)
    img = img[:320, :320].astype(float)
    img = img[np.newaxis, np.newaxis, :, :]
    return torch.from_numpy(img).to(torch.float)

class TestNets:
    """This class contains functions to test the arch. settings of both
    composed networks and custom Module class.
    """

    @pytest.mark.pipeline
    def test_unet_creation_pipeline(self):
        """This method will test the U-net architecture composing and retrival.

        For every architecture, there are a specifics number of modules (ModuleList classes) 
        and sub-modules (Built-in and custom classes - e.g. ConvBlock).
        """

        default_args = read_json_file("./tests/mock_train_args.json")
        test_arguments = [
            {"model_pipeline": "dual-unet", "expected_net_class": DUNet, "expected_modulelist": 6, "expected_convblocks": 13},
            {"model_pipeline": "dual-unet", "filters": [64, 2048], "expected_net_class": DUNet, "expected_modulelist": 6, "expected_convblocks": 16},
            {"model_pipeline": "original-dual-unet", "expected_net_class": ODUNet, "expected_modulelist": 7, "expected_convblocks": 13},
            {"model_pipeline": "original-dual-unet", "pool_method": "max", "expected_net_class": ODUNet, "expected_modulelist": 6, "expected_convblocks": 13}
        ]

        for test_args in test_arguments:
            run_parameters = update_default_args(default_args, test_args)
            net = get_unet(run_parameters)
            assert isinstance(net, run_parameters["expected_net_class"]) == True
            modules = [module for module in net.named_modules() if isinstance(module[1], ModuleList)]
            conv_sub_modules = [module for module in net.named_modules() if isinstance(module[1], ConvBlock)]
            # Asserting both ModuleList and custom Module classes.
            assert run_parameters["expected_modulelist"] == len(modules)
            assert run_parameters["expected_convblocks"] == len(conv_sub_modules)

    @pytest.mark.pipeline
    def test_unet_inference_pipeline(self):
        """This method will test the U-net architecture's inference.

        For every architecture, there are different number/types of output depending
        on the output paths.
        """
        images_folder_path = "tests"
        default_args = read_json_file("./tests/mock_train_args.json")
        test_arguments = [
            {"model_pipeline": "dual-unet", "expected_net_class": DUNet, "expected_modulelist": 6, "expected_convblocks": 13, "expected_output": tuple, "expected_ch_out": (1, 1), "expected_dim": 320},
            {"model_pipeline": "dual-unet", "filters": [64, 2048], "expected_net_class": DUNet, "expected_modulelist": 6, "expected_convblocks": 16, "expected_output": tuple, "expected_ch_out": (1, 1), "expected_dim": 320},
            {"model_pipeline": "original-dual-unet", "expected_net_class": ODUNet, "expected_modulelist": 7, "expected_convblocks": 13, "expected_output": tuple, "expected_ch_out": (2, 1, 2), "expected_dim": 320},
            {"model_pipeline": "original-dual-unet", "pool_method": "max", "expected_net_class": ODUNet, "expected_modulelist": 6, "expected_convblocks": 13, "expected_output": tuple, "expected_ch_out": (2, 1, 2), "expected_dim": 320}
        ]
        images = load_images(images_folder_path)
        train_img = prepare_img(images[0])
        for test_args in test_arguments:
            run_parameters = update_default_args(default_args, test_args)
            net = get_unet(run_parameters)
            output = net(train_img)
            if test_args["expected_output"] == tuple:
                assert isinstance(output, tuple) == True

                # Assert singularly the output channels and dimensions for every output batch of the tuples.  
                for idx in range(len(output)):
                    assert output[idx].shape[1] == test_args["expected_ch_out"][idx]
                    assert output[idx].shape[2] == test_args["expected_dim"]
                    assert output[idx].shape[3] == test_args["expected_dim"]
                   