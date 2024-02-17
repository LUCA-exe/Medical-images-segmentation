import gc
import numpy as np
import random
import time
import torch
import torch.optim as optim
from multiprocessing import cpu_count
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from training.ranger2020 import Ranger
from training.losses import get_loss
from net_utils.utils import get_num_workers, save_current_model_state, save_training_loss


def set_up_optimizer_and_scheduler(config, net, best_loss):
    """ Set up the optimizer and scheduler configurations adn return them to the main function.

    :param n_samples: number of training samples.
        :type n_samples: int
    :return: maximum amount of training epochs
    """

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
                                      min_lr=3e-6) 
        break_condition = 2 * config['max_epochs'] // 20 + 5

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
    return optimizer, scheduler, break_condition


def get_max_epochs(n_samples):
    """ Get maximum amount of training epochs.

    :param n_samples: number of training samples.
        :type n_samples: int
    :return: maximum amount of training epochs
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


def train(log, net, datasets, config, device, path_models, best_loss=1e4):
    """ Train the model.

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

    # Get number of training epochs depending on dataset size (just roughly to decrease training time):
    config['max_epochs'] = get_max_epochs(len(datasets['train']) + len(datasets['val']))
    # NOTE: Make the training.py more clean - all computation like the one belowe are passed from the calling function
    print(f"Number of epochs without improvement allowed {2 * config['max_epochs'] // 20 + 5}")

    print('-' * 20)
    print('Train {0} on {1} images, validate on {2} images'.format(config['run_name'],
                                                                   len(datasets['train']),
                                                                   len(datasets['val'])))
    # Added info on the log file (preferred debug for now)
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

    # Loss function and optimizer
    criterion = get_loss(config)

    second_run = False # WARNING: Fixed arg - to change
    max_epochs = config['max_epochs']
    optimizer, scheduler, break_condition = set_up_optimizer_and_scheduler(config, net, best_loss)

    # Auxiliary variables for training process
    epochs_wo_improvement, train_loss, val_loss,  = 0, [], []
    since = time.time()

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
                net.eval()  # Set model to evaluation mode

            running_loss = 0.0
            # Iterate over data
            for samples in dataloader[phase]:

                # Get img_batch and label_batch and put them on GPU if available
                img_batch, border_label_batch, cell_label_batch, mask_label_batch = samples # Unpack always all 'labels'
                img_batch = img_batch.to(device)
                cell_label_batch, border_label_batch, mask_label_batch = cell_label_batch.to(device), border_label_batch.to(device), mask_label_batch.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass (track history if only in train)
                with torch.set_grad_enabled(phase == 'train'):
                    
                    # NOTE: Depending on the architecture, you can have different number of outputs
                    if config['architecture'] == 'DU':
                        border_pred_batch, cell_pred_batch = net(img_batch)
                        loss_border = criterion['border'](border_pred_batch, border_label_batch)
                        loss_cell = criterion['cell'](cell_pred_batch, cell_label_batch)
                        loss = loss_border + loss_cell

                    if config['architecture'] == 'TU':
                        border_pred_batch, cell_pred_batch, mask_pred_batch = net(img_batch)
                        loss_border = criterion['border'](border_pred_batch, border_label_batch)
                        loss_cell = criterion['cell'](cell_pred_batch, cell_label_batch)
                        loss_mask = criterion['mask'](mask_pred_batch, mask_label_batch)
                        loss = loss_border + loss_cell + loss_mask
                        
                    # Backward (optimize only if in training phase)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # Statistics
                running_loss += loss.item() * img_batch.size(0)

            epoch_loss = running_loss / len(datasets[phase])

            if phase == 'train':
                train_loss.append(epoch_loss)
                print('Training loss: {:.5f}'.format(epoch_loss))
            else:
                val_loss.append(epoch_loss)
                print('Validation loss: {:.5f}'.format(epoch_loss))

                if epoch_loss < best_loss:
                    print('Validation loss improved from {:.5f} to {:.5f}. Save model.'.format(best_loss, epoch_loss))
                    best_loss = epoch_loss
                    save_current_model_state(config, net, path_models)
                    epochs_wo_improvement = 0

                else:
                    print('Validation loss did not improve.')
                    epochs_wo_improvement += 1

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
    save_training_loss(train_loss, val_loss, second_run, path_models, config, time_elapsed, epoch)

    # Clear memory
    del net
    gc.collect()
    return best_loss


def train_auto(net, dataset, configs, device, path_models):
    """ Train the model.

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
    :return: None
    """

    max_epochs = 60

    print('-' * 20)
    print('Train {0} on {1} images'.format(configs['run_name'], len(dataset)))

    # Data loader: only training set
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=configs['batch_size_auto'],
                                             shuffle=True,
                                             pin_memory=True,
                                             worker_init_fn=seed_worker,
                                             num_workers=8)

    # Loss function and optimizer
    criterion = get_loss(configs['loss'])

    optimizer = Ranger(net.parameters(),
                       lr=6e-3,
                       alpha=0.5, k=6, N_sma_threshhold=5,  # Ranger options
                       betas=(.95, 0.999), eps=1e-6, weight_decay=0,  # Adam options
                       # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                       use_gc=True, gc_conv_only=False, gc_loc=True)

    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=2e-4, last_epoch=-1, verbose=True)

    # Auxiliary variables for training process
    train_loss = []

    # Training process
    for epoch in range(max_epochs):

        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, max_epochs))
        print('-' * 10)

        start = time.time()

        net.train()  # Set model to training mode
        running_loss = 0.0

        # Iterate over data
        for samples in dataloader:

            # Get img_batch and label_batch and put them on GPU if available
            img_batch, label_batch = samples
            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass (track history if only in train)
            with torch.set_grad_enabled(True):

                pred_batch = net(img_batch)
                loss = criterion['cell'](pred_batch, label_batch)

                # Backward (optimize only if in training phase)
                loss.backward()
                optimizer.step()

            # Statistics
            running_loss += loss.item() * img_batch.size(0)

        epoch_loss = running_loss / len(dataset)
        train_loss.append(epoch_loss)
        print('Training loss: {:.5f}'.format(epoch_loss))

        # The state dict of data parallel (multi GPU) models need to get saved in a way that allows to
        # load them also on single GPU or CPU
        if configs['num_gpus'] > 1:
            torch.save(net.module.state_dict(), str(path_models / (configs['run_name'] + '.pth')))
        else:
            torch.save(net.state_dict(), str(path_models / (configs['run_name'] + '.pth')))

        scheduler.step()

        # Epoch training time
        print('Epoch training time: {:.1f}s'.format(time.time() - start))

    # Clear memory
    del net
    gc.collect()

    return None


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
