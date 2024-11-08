"""This module is the high-level funcitonality of executing the training loop.

Specifically, It will set up the optimizer and loss manager following the current
configurations and handling the losses and model state savings.
"""
import gc
import numpy as np
import random
import time
import torch
import torch.optim as optim
from torch.optim import Optimizer
from multiprocessing import cpu_count
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch.nn as nn
from copy import deepcopy, copy
from typing import Dict, Type, Any, Optional, List, Tuple, Union

from training.ranger2020 import Ranger
from training.losses import LossComputator
from net_utils.utils import get_num_workers, save_current_model_state, save_current_branch_state, save_training_loss, show_training_dataset_samples, save_image

def train(log, net: Type[nn.Module], datasets, config: Dict[str, Any], device, path_models, best_loss=1e4):
    """Train the model using the .

    :param net: Model/Network to train.
        :type net:
    :param datasets: Dictionary containing the training and the validation data set.
        :type datasets: dict
    :param configs: Dictionary with configurations of the training process.
        :type configs: dict
    :param device: Use (multiple) GPUs or CPU.
        :type device: torch device
    :param path_models: Path to the directory to save the models.
        :type path_models: pathlib Path object
    :param best_loss: Best loss (only needed for second run to see if val loss further improves).
        :type best_loss: float

    :return: None
    """
    show_training_dataset_samples(log, datasets["train"])

    # If it is not passed as arguments, set in this method.
    if not "max_epochs" in config:
        log.info(f"The 'max epochs' param is not set yet, it will be inferred now!")
        # Get number of training epochs depending on dataset size (just roughly to decrease training time):
        config['max_epochs'] = get_max_epochs(len(datasets['train']), arch=config['architecture'][0])

    # NOTE: Make the training.py more clean - all computation like the one belowe are passed from the calling function
    print(f"Number of epochs without improvement allowed {2 * config['max_epochs'] // 20 + 5}")
    print('-' * 20)
    print('Train {0} on {1} images, validate on {2} images'.format(config['run_name'],
                                                                   len(datasets['train']),
                                                                   len(datasets['val'])))
    # Added info on the log file (preferred debug for now).
    log.debug('Train {0} on {1} images, validate on {2} images'.format(config['run_name'],
                                                                   len(datasets['train']),
                                                                   len(datasets['val'])))
    # Data loader for training and validation set
    apply_shuffling = {'train': True, 'val': False}
    num_workers = get_num_workers(device)
    num_workers = np.minimum(num_workers, 16)
    dataloader = {x: torch.utils.data.DataLoader(datasets[x],
                                                 batch_size=config['batch_size'],
                                                 shuffle=apply_shuffling,
                                                 pin_memory=True,
                                                 worker_init_fn=seed_worker,
                                                 num_workers=num_workers)
                  for x in ['train', 'val']}
    loss_computator = LossComputator(config["loss"], 
                                     config["classification_loss"],
                                     device = device)
    gt_labels = datasets["val"].get_sample_keys()  # Image labels used for the loss computation
    log.info(f"Loss that will be used (one for each labels) are: {loss_computator.get_loss_criterions(gt_labels)}")

    second_run = False # WARNING: Fixed arg - to change.
    optimizer, scheduler, break_condition = set_up_optimizer_and_scheduler(config, net, best_loss)

    # Auxiliary variables for training process
    epochs_wo_improvement, train_loss, val_loss,  = 0, [], []
    since = time.time()
    arch_name = config['architecture'][0]
    # Validation sample counter - index of what to plot in training phase images
    val_phase_counter = 0
    # Added single regressive branch loss in case using the 'triple u-net'
    regressive_best_loss = 0

    # Starting training phase.
    for epoch in range(config['max_epochs']):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, config['max_epochs']))
        print('-' * 10)
        start = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # Set model to training mode
            else:
                val_phase_counter += 1
                net.eval()  # Set model to evaluation mode

            # Keep track of aggregated/single running losses.
            running_loss = 0.0
            running_losses = [0.0] * len(gt_labels)
            loss_labels = copy(gt_labels)
            
            # As convention, the first position is the total epoch loss for the current phase. 
            loss_labels.insert(0, "total loss")
            for samples in dataloader[phase]:
                samples_dict = move_batches_to_device(samples, device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # NOTE: The padding in case of different input image format?

                    # NOTE: important the orders of the true_label_batch
                    loss, losses_list = get_losses_from_model(samples_dict, arch_name, net, loss_computator, config, phase, val_phase_counter)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # NOTE: loss.item() as default contains already the average of the mini_batch loss.
                running_loss += loss.item() * config['batch_size']
                running_losses = update_running_losses(running_losses, losses_list, config['batch_size'])
    
            # Compute average epoch loss and losses.
            epoch_loss = running_loss / len(datasets[phase])
            epoch_running_losses = [loss/len(datasets[phase]) for loss in running_losses]

            # Merge in a single list all the losses.
            concat_losses = [epoch_loss] + epoch_running_losses 
            if phase == 'train': 
                train_loss.append(concat_losses)
                current_epoch_string = "Training -"
                for label, val in zip(loss_labels, concat_losses):
                    current_epoch_string += f" {label}: {str(round(val, 5))} -"
                current_epoch_string = current_epoch_string[:-2]
                print(current_epoch_string)
                
            else:
                
                val_loss.append(concat_losses)
                current_epoch_string = "Validation -"
                for label, val in zip(loss_labels, concat_losses):
                    current_epoch_string += f" {label}: {str(round(val, 5))} -"
                current_epoch_string = current_epoch_string[:-2]
                print(current_epoch_string)
                
                # NOTE: The update control just the total loss decrement, not the single ones.
                if epoch_loss < best_loss:
                    print('Validation loss improved from {:.5f} to {:.5f}. Save model.'.format(best_loss, epoch_loss))
                    best_loss = epoch_loss

                    # Special case for 'triple u-net' - update the 'best loss' of regression branch in every case
                    if config['architecture'][0] == 'triple-unet':
                        regressive_best_loss = epoch_loss_cell
                    save_current_model_state(config, net, path_models)
                    epochs_wo_improvement = 0

                # NOTE: Added 'hard-coded' condition just for my 'triple u-net' - not counted as 'improving' the loss but updated the weights of 'part' of the neural networks.
                elif (config['architecture'][0] == 'triple-unet') and (epoch_loss_cell < regressive_best_loss):
                    print('Cell validation loss improved from {:.5f} to {:.5f}. Save just the regression branch.'.format(regressive_best_loss, epoch_loss_cell))
                    regressive_best_loss = epoch_loss_cell
                    save_current_branch_state(config, net, path_models, device)
                else:
                    print('Validation loss did not improve.')
                    epochs_wo_improvement += 1

                # NOTE: Deprecated - Update learning rate differently.
                if config['optimizer'] == 'ranger' and second_run:
                    scheduler.step()
                else:
                    # NOTE: To affect the learning rate the step is performed at the val phase (at the end of each current epoch).
                    scheduler.step(epoch_loss)
        print('Epoch training time: {:.1f}s'.format(time.time() - start))

        # Break training if plateau is reached.
        if epochs_wo_improvement == break_condition:
            log.info(str(epochs_wo_improvement) + ' epochs without validation loss improvement --> break')
            print(str(epochs_wo_improvement) + ' epochs without validation loss improvement --> break')
            break

    # Total training time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}min {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('-' * 20)
    save_training_loss(loss_labels, train_loss, val_loss, second_run, path_models, config, time_elapsed, epoch)

    # Clear memory
    del net
    gc.collect()
    return best_loss

def sample_plot_during_validation(sample_batch: Dict[str, torch.Tensor],
                                  val_phase_counter: int,
                                  binary_channel : int = 1) -> None:
    """Plot the current validation batch.

    The sample batch in input will contain the original image plus the predicted
    tensors by the current NN.
    """
    idx = 0
    folder_to_save = "./tmp"
    # FIXME: Labels coupled with the LossComputator class internal labels and CellSegDataset class (two dependencies in different modules).
    float_labels = ["image", "border_label", "cell_label"]
    categorical_labels = ["binary_border_label", "mask_label"]
    for key, batch in sample_batch.items():

        # Remove one image from the current batch.
        image = batch[idx]
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"The following {key} is not the {torch.Tensor} expected type!")

        if key in float_labels:
            save_image(np.squeeze(image.cpu().detach().numpy()), folder_to_save, f"{key} (Batch position {idx}) (Validation {val_phase_counter})")
        elif key in categorical_labels:
            save_image(np.squeeze(image.cpu().detach().numpy()[binary_channel, :, :]), folder_to_save, f"{key} (Batch position {idx})(Validation {val_phase_counter})")
        else:
            raise ValueError(f"The key {key} is not supported!")    

def get_losses_from_model(gt_batch: Dict[str, torch.Tensor], 
                          arch_name: str, 
                          net: nn.Module, 
                          criterions: LossComputator, 
                          config, 
                          phase: str, 
                          val_phase_counter: int) -> Tuple[float, List[float]]:
    """Execute the inference and compute the losses.

    This function will compose the prediction batch, compute the total and single
    losses. 
    As utility during the training phase, this function will calls public method to plot
    the images in the ./tmp/ folder used in the valdation phases.
    """
    p_batch = {}
    if arch_name == 'dual-unet':
        p_batch["border_label"], p_batch["cell_label"] = net(gt_batch["image"])
    elif arch_name == 'original-dual-unet':
        p_batch["binary_border_label"], p_batch["cell_label"], p_batch["mask_label"] = net(gt_batch["image"])
    else:
        raise ValueError(f"The architecture {arch_name} is not supported!")
    
    if phase == "val":
            # Compose the dict. for plotting the current predicted batch.
            sample_batch = deepcopy(p_batch)
            sample_batch["image"] = gt_batch["image"]
            sample_plot_during_validation(sample_batch, val_phase_counter)
    
    if "image" in gt_batch: 
        gt_batch = deepcopy(gt_batch)
        del gt_batch["image"]
    loss, losses_list = criterions.compute_loss(p_batch, gt_batch)
    return loss, losses_list

def set_up_optimizer_and_scheduler(config, net, best_loss) -> Tuple[type[Optimizer], 
                                                                    Union[ReduceLROnPlateau,  CosineAnnealingLR],
                                                                    int]:
    """Set up the optimizer and scheduler configurations.

    For all the architecture the optimal optimizer and scheduler
    are the Adam optimizer and ReduceLROnPlateau scheduler.

    :param n_samples: number of training samples.
        :type n_samples: int
    :return: maximum amount of training epochs
    """
    if config["architecture"][0] == "dual-unet":

        if config['optimizer'] == 'adam':

            optimizer = optim.Adam(net.parameters(),
                                lr=8e-4,
                                betas=(0.9, 0.999),
                                eps=1e-08,
                                weight_decay=0,
                                amsgrad=True)

            scheduler = ReduceLROnPlateau(optimizer,
                                        mode='min',
                                        factor=0.25,
                                        patience=config['max_epochs'] // 20,
                                        verbose=True,
                                        min_lr=6e-5) 
            break_condition = 2 * config['max_epochs'] // 20 + 8

        elif config['optimizer'] == 'ranger':

            lr = 6e-3
            if best_loss < 1e3:  # probably second run

                second_run = True

                optimizer = Ranger(net.parameters(),
                                lr=0.09 * lr,
                                alpha=0.5, k=6, N_sma_threshhold=5,  # Ranger options
                                betas=(.95, 0.999), eps=1e-6, weight_decay=0,  # Adam options
                                # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                                use_gc=True, gc_conv_only=False, gc_loc=True)

                scheduler = CosineAnnealingLR(optimizer,
                                            T_max=config['max_epochs'] // 10,
                                            eta_min=3e-5,
                                            last_epoch=-1,
                                            verbose=True)
                break_condition = config['max_epochs'] // 10 + 1
                max_epochs = config['max_epochs'] // 10
            else:
                optimizer = Ranger(net.parameters(),
                                lr=lr,
                                alpha=0.5, k=6, N_sma_threshhold=5,  # Ranger options
                                betas=(.95, 0.999), eps=1e-6, weight_decay=0,  # Adam options
                                # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                                use_gc=True, gc_conv_only=False, gc_loc=True)
                scheduler = ReduceLROnPlateau(optimizer,
                                            mode='min',
                                            factor=0.25,
                                            patience=config['max_epochs'] // 10,
                                            verbose=True,
                                            min_lr=0.075*lr)
                break_condition = 2 * config['max_epochs'] // 10 + 5
        else:
            raise Exception('Optimizer not known')

    # Config from thei original paper
    elif config["architecture"][0] == "original-dual-unet" or config["architecture"][0] == "triple-unet":

        optimizer = optim.Adam(net.parameters(),
                                betas=(0.9, 0.99),
                                weight_decay=1e-4,
                                amsgrad=True)

        scheduler = ReduceLROnPlateau(optimizer,
                                        mode='min',
                                        factor=0.1,
                                        patience=config['max_epochs'] // 4,
                                        verbose=True) 
        # NOTE: No break condition mentioned on the paper
        break_condition = config['max_epochs'] // 4

    else:
        raise Exception('Architecture not known')
    return optimizer, scheduler, break_condition

def get_max_epochs(n_samples: int, arch: Optional[str] = None) -> int:
    """Get maximum amount of training epochs based on the number of sample.

    If the config hasmap is provided override the returned epochs with the custom
    number contained in it.

    Args:
        n_samples: Number of training samples.
        arch: The current model architecture if provided.
    """
    if n_samples >= 1000:
        max_epochs = 200
    elif n_samples >= 500:
        max_epochs = 240
    elif n_samples >= 200:
        max_epochs = 320
    elif n_samples >= 100:
        max_epochs = 400
    elif n_samples >= 50:
        max_epochs = 480
    else:
        max_epochs = 560

    # Override the numbers with custom configurations.
    if arch is not None:
        if arch == "dual-unet":
            max_epochs = 200
        elif arch == "original-dual-unet":
            max_epochs = 40
    return max_epochs

def get_weights(net, weights, device, num_gpus):
    """Load weights into model.

    :param net: Model to load the weights into.
        :type net:
    :param weights: Path to the weights.
        :type weights: pathlib Path object
    :param device: Device to use ('cpu' or 'cuda')
        :type device:
    :param num_gpus: Amount of GPUs to use.
        :type num_gpus: int
    :return: model with loaded weights.

    """
    if num_gpus > 1:
        net.module.load_state_dict(torch.load(weights, map_location=device))
    else:
        net.load_state_dict(torch.load(weights, map_location=device))
    return net

def update_running_losses(running_losses_list: List[float], 
                          losses_list: List[float], 
                          batch_size: int) -> List[float]:
    """Updates the running losses during the training phase.

    Given a list of cumulative and single loop losses, multiply the single
    loop losses (averaged by batch) and add to the cumulative mini-batch correspondent.
    """
    updated_losses = [loss * batch_size for loss in losses_list]
    updated_running_losses = []
    for runn_loss, loss in zip(running_losses_list, updated_losses):
        updated_running_losses.append(runn_loss + loss)
    return updated_running_losses

def move_batches_to_device(samples_dict: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """It gets the dict. sample batch and move to the desired device.

    Returns:
        The updated tensored in the original key-value pairs.
    """
    samples_dict = deepcopy(samples_dict)
    # Remove not-tensors values.
    if "id" in samples_dict:
        del samples_dict["id"]

    for key, _ in samples_dict.items():
        samples_dict[key] = samples_dict[key].to(device)
    return samples_dict

def seed_worker(worker_id):
    """ Fix pytorch seeds on linux

    https://pytorch.org/docs/stable/notes/randomness.html
    https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/

    :param worker_id:
    :return:
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
