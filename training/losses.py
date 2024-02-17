import torch.nn as nn


def get_loss(config):
    """ Get loss function(s) for the training process based on the current architecture.

    :param loss_function: Loss function to use.
        :type loss_function: str
    :return: Loss function / dict of loss functions.
    """

    # NOTE: To module to different architectures.

    if config['loss'] == 'l1':
        border_criterion = nn.L1Loss()
        cell_criterion = nn.L1Loss()
    elif config['loss']  == 'l2':
        border_criterion = nn.MSELoss()
        cell_criterion = nn.MSELoss()
    elif config['loss']  == 'smooth_l1':
        border_criterion = nn.SmoothL1Loss()
        cell_criterion = nn.SmoothL1Loss()

    criterion = {'border': border_criterion, 'cell': cell_criterion}
    return criterion
