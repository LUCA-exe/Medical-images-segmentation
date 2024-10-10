import gc
import numpy as np
import random
import time
import torch
import torch.optim as optim
from multiprocessing import cpu_count
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch.nn as nn
from copy import deepcopy
from typing import Dict, Type, Any

from training.ranger2020 import Ranger
from training.losses import get_loss, get_weights_tensor, WeightedCELoss, compute_cross_entropy, compute_weighted_cross_entropy, compute_j_cross_entropy
from net_utils.utils import get_num_workers, save_current_model_state, save_current_branch_state, save_training_loss, show_training_dataset_samples, save_image


def sample_plot_during_validation(batches_list, val_phase_counter, binary_pred = True, binary_channel = 1):
    # Wrapper/util function to plot sample batches during the validation phase to monitor visually the training
    
    # NOTE: For now take the first image
    sample = 0
    # Take always the first element of the batch for every batch provided -  the first batch is alway the true image and the remaining all the prediction
    for idx, batch in enumerate(batches_list):
        
        # If the batches of the true image
        if idx == 0:
            save_image(np.squeeze(batch[sample].cpu().detach().numpy()), "./tmp", f"Original image batch sample {sample} (Validation {val_phase_counter})")

        else:
            # Saving the batch image considering two channel (prediceted segmentation mask)
            if binary_pred == True:
                save_image(np.squeeze(batch[sample].cpu().detach().numpy()[binary_channel, :, :]), "./tmp", f"Binary batch {idx} sample {sample} (Channel {binary_channel}) (Validation {val_phase_counter})")
            else:
                save_image(np.squeeze(batch[sample].cpu().detach().numpy()), "./tmp", f"Floating batch {idx} sample {sample} (Validation {val_phase_counter})")
            

def get_losses_from_model(batches_dict, arch_name, net, criterion, config, phase, val_phase_counter):
    # Extendible function for the deepen of the sstudy on different architecture

    if arch_name == 'dual-unet':
        border_pred_batch, cell_pred_batch = net(batches_dict["image"])
        loss_border = criterion['border'](border_pred_batch, batches_dict["border_label"])
        loss_cell = criterion['cell'](cell_pred_batch, batches_dict["cell_label"])

        loss = loss_border + loss_cell
        losses_list = [loss_border.item(), loss_cell.item()]

        # Qualitative plotting in the validation phase
        if phase == "val":
            sample_plot_during_validation([batches_dict["image"], border_pred_batch, cell_pred_batch], val_phase_counter, binary_pred = False)


    if arch_name == 'triple-unet':
        binary_border_pred_batch, cell_pred_batch, mask_pred_batch = net(batches_dict["image"])

        # Prepare dict. of predicted batches
        pred_batches = {"binary_border_pred": binary_border_pred_batch, "cell_pred": cell_pred_batch,  "mask_pred":  mask_pred_batch}

        if config["classification_loss"] == "cross-entropy":
            loss, losses_list = compute_cross_entropy(pred_batches, batches_dict, criterion)

        elif config["classification_loss"] == "cross-entropy-dice":
            loss, losses_list = compute_weighted_cross_entropy(pred_batches, batches_dict, criterion)

        elif config["classification_loss"] == "weighted-cross-entropy":
            loss, losses_list = compute_weighted_cross_entropy(pred_batches, batches_dict, criterion)

        # Qualitative plotting in the validation phase
        if phase == "val":

            # NOTE: Used both inspecting validation prediction for this network
            sample_plot_during_validation([batches_dict["image"], pred_batches["binary_border_pred"],  pred_batches["mask_pred"]], val_phase_counter)
            sample_plot_during_validation([batches_dict["image"], pred_batches["cell_pred"]], val_phase_counter,  binary_pred = False)

    if arch_name == 'original-dual-unet':

        binary_border_pred_batch, cell_pred_batch, mask_pred_batch = net(batches_dict["image"])
        # Prepare dict. of predicted batches
        pred_batches = {"binary_border_pred": binary_border_pred_batch, "cell_pred": cell_pred_batch,  "mask_pred":  mask_pred_batch}
        
        # Added 'interface' function for readibility
        if config["classification_loss"] == "cross-entropy":
            loss, losses_list = compute_cross_entropy(pred_batches, batches_dict, criterion)
        
        elif config["classification_loss"] == "cross-entropy-dice":
            loss, losses_list = compute_weighted_cross_entropy(pred_batches, batches_dict, criterion)
        
        elif config["classification_loss"] == "weighted-cross-entropy":
            loss, losses_list = compute_weighted_cross_entropy(pred_batches, batches_dict, criterion)

        elif config["classification_loss"] == "j-cross-entropy":
            loss, losses_list = compute_j_cross_entropy(pred_batches, batches_dict, criterion)
        
        # Qualitative plotting in the validation phase
        if phase == "val":
            # NOTE: Plot all binary prediction or all floating prediction
            sample_plot_during_validation([batches_dict["image"], pred_batches["binary_border_pred"],  pred_batches["mask_pred"]], val_phase_counter)

    return loss, losses_list


def set_up_optimizer_and_scheduler(config, net, best_loss):
    """ Set up the optimizer and scheduler configurations adn return them to the main function.

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


def get_max_epochs(n_samples, config):
    """ Get maximum amount of training epochs.

    :param n_samples: number of training samples.
        :type n_samples: int
    :return: maximum amount of training epochs
    """

    # From original repository
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

    # NOTE: Temporary fix - following the original paper
    if config["architecture"][0] == "dual-unet":
        max_epochs = 200
    elif config["architecture"][0] == "original-dual-unet" or config["architecture"][0] == "triple-unet":
         max_epochs = 40
    return max_epochs


def get_weights(net, weights, device, num_gpus):
    """ Load weights into model.

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


def update_running_losses(running_losses_list, losses_list, batch_size):
    # In input a list of losses computed during a mini_batch images, for every item of the list just update the values and returns it

    updated_losses = [loss * batch_size for loss in losses_list]
    updated_running_losses = []
    for runn_loss, loss in zip(running_losses_list, updated_losses): # Update every running loss by the corrispondent current mini_batch loss

        updated_running_losses.append(runn_loss + loss)
    return updated_running_losses


def move_batches_to_device(samples_dict, device):
    # util function to get the dict. batch from the dataloader and move to the correct device

    # Un-pack the dict.
    for key, tensor in samples_dict.items():
        samples_dict[key] = samples_dict[key].to(device)

    # Return updated dict.
    return samples_dict


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
    # Assert that the datasets has been created correctly before the loop over the images
    show_training_dataset_samples(log, datasets["train"])

    # NOTE: If it is not passed as arguments, set in this method.
    if not "max_epochs" in config:
        log.info(f"The 'max epochs' param is not set yet, it will be inferred now!")
        # Get number of training epochs depending on dataset size (just roughly to decrease training time):
        config['max_epochs'] = get_max_epochs(len(datasets['train']) + len(datasets['val']), config)

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

    # Set-up the Loss function.
    criterion = get_loss(config, device)
    log.info(f"Loss that will be used are: {criterion}")

    second_run = False # WARNING: Fixed arg - to change.
    max_epochs = config['max_epochs']
    # Set-up the optimizer.
    optimizer, scheduler, break_condition = set_up_optimizer_and_scheduler(config, net, best_loss)

    # Auxiliary variables for training process
    epochs_wo_improvement, train_loss, val_loss,  = 0, [], []
    since = time.time()
    arch_name = config['architecture'][0]
    # Validation sample counter - index of what to plot in training phase images
    val_phase_counter = 0
    # Added single regressive branch loss in case using the 'triple u-net'
    regressive_best_loss = 0

    # Training process
    for epoch in range(max_epochs):

        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, max_epochs))
        print('-' * 10)
        start = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # Set model to training mode
            else:
                val_phase_counter += 1
                net.eval()  # Set model to evaluation mode

            # keep track of running losses
            running_loss = 0.0
            running_loss_border, running_loss_cell, running_loss_mask, running_loss_binary_border = 0.0, 0.0, 0.0, 0.0
            loss_labels = ["Total loss", "Border loss", "Cell loss", "Mask loss", "Binary border loss"]

            # Iterate over data
            for samples in dataloader[phase]:

                # Load always all 'labels'
                samples_dict = move_batches_to_device(samples, device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass (track history if only in train)
                with torch.set_grad_enabled(phase == 'train'):

                    # NOTE: important the orders of the true_label_batch
                    loss, losses_list = get_losses_from_model(samples_dict, arch_name, net, criterion, config, phase, val_phase_counter)

                    # Backward (optimize only if in training phase)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # Statistics - both general and single losses
                running_loss += loss.item() * samples_dict["image"].size(0) # NOTE: loss.item() as default contains already the average of the mini_batch loss.

                if config['architecture'][0] == 'dual-unet':
                    running_loss_border, running_loss_cell = update_running_losses([running_loss_border, running_loss_cell], losses_list, samples_dict["image"].size(0))

                if config['architecture'][0] == 'original-dual-unet':
                    running_loss_binary_border, running_loss_cell, running_loss_mask = update_running_losses([running_loss_binary_border, running_loss_cell, running_loss_mask], losses_list, samples_dict["image"].size(0))

                if config['architecture'][0] == 'triple-unet':
                    running_loss_binary_border, running_loss_cell, running_loss_mask = update_running_losses([running_loss_binary_border, running_loss_cell, running_loss_mask], losses_list, samples_dict["image"].size(0))

            # Compute average epoch losses
            epoch_loss = running_loss / len(datasets[phase])
            epoch_loss_border =  running_loss_border / len(datasets[phase])
            epoch_loss_cell =  running_loss_cell / len(datasets[phase])
            epoch_loss_mask =  running_loss_mask / len(datasets[phase])
            epoch_loss_binary_border = running_loss_binary_border / len(datasets[phase])

            if phase == 'train': 

                train_loss.append([epoch_loss, epoch_loss_border, epoch_loss_cell, epoch_loss_mask, epoch_loss_binary_border])
                print('Training - total loss: {:.5f} - border loss: {:.5f} - cell loss: {:.5f} - mask loss:  {:.5f} - binary border loss: {:.5f}'.format(epoch_loss, epoch_loss_border, epoch_loss_cell, epoch_loss_mask, epoch_loss_binary_border))
            else:

                val_loss.append([epoch_loss, epoch_loss_border, epoch_loss_cell, epoch_loss_mask, epoch_loss_binary_border])
                print('Validation - total loss: {:.5f} - border loss: {:.5f} - cell loss: {:.5f} - mask loss:  {:.5f} - binary border loss: {:.5f}'.format(epoch_loss, epoch_loss_border, epoch_loss_cell, epoch_loss_mask, epoch_loss_binary_border))

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

                # Update learning rate differently
                if config['optimizer'] == 'ranger' and second_run:
                    scheduler.step()
                else:
                    scheduler.step(epoch_loss)

        # Epoch training time
        print('Epoch training time: {:.1f}s'.format(time.time() - start))

        # Break training if plateau is reached
        if epochs_wo_improvement == break_condition:
            print(str(epochs_wo_improvement) + ' epochs without validation loss improvement --> break')
            break

    # Total training time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}min {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('-' * 20)

    # Save loss
    save_training_loss(loss_labels, train_loss, val_loss, second_run, path_models, config, time_elapsed, epoch)

    # Clear memory
    del net
    gc.collect()
    return best_loss

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
