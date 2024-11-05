"""This module will test the set up of the neural networks for both validation and traning phases.

It will perform system tests for the architectures consistency during train/val inference (structure implemented from papers) and 
unit tests for the custom inner Modules as well.

To test just the integrity of the custom Module of the nets:
...> python -m pytest -v --run-sub tests/test_nets.py

To test just the entire architecture:
...> python -m pytest -v --run-pipeline tests/test_nets.py
"""
import pytest
from torch.nn import ModuleList

from tests.test_train_pipelines import update_default_args
from training.training import get_max_epochs
from net_utils.unets import build_unet, DUNet, ODUNet
from utils import load_environment_variables, set_current_run_folders, \
    create_logging, read_json_file, set_device, check_path, \
        train_factory

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
    

class TestNets:
    """This class contains functions to test the arch. settings of both
    composed networks and custom Module class.
    """

    @pytest.mark.pipeline
    def test_unet_creation_pipeline(self):
        """This method will test the U-net architecture composing and retrival.

        This function will read the arguments from a file (./tests/*.json).
        For every architecture, there are a specifics number of modules and sub-modules 
        in a expected orders.
        """

        default_args = read_json_file("./tests/mock_train_args.json")
        test_arguments = [
            {"model_pipeline": "dual-unet", "expected_net_class": DUNet},
            {"model_pipeline": "original-dual-unet", "expected_net_class": ODUNet}
        ]

        for test_args in test_arguments:
            run_parameters = update_default_args(default_args, test_args)
            net = get_unet(run_parameters)
            assert isinstance(net, run_parameters["expected_net_class"]) == True

            # TODO: set up testing of just the higher ModuleList blocks and the custom Blocks inside.
            modules = [module for module in net.named_modules() if isinstance(module[1], ModuleList)]
            print(len(modules))
           
        
        
