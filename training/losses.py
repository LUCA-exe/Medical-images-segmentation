import torch.nn as nn
import torch


def calculate_class_weights(num_bg_pixels: int, num_cell_pixels: int) -> torch.Tensor:
    """
    Calculates class weights for addressing class imbalance in binary segmentation tasks.

    Args:
        num_bg_pixels (int): Total number of background pixels in the dataset.
        num_cell_pixels (int): Total number of cell pixels in the dataset.

    Returns:
        torch.Tensor: A tensor of shape (2,) containing class weights for background and cells.

    Raises:
        ValueError: If either `num_bg_pixels` or `num_cell_pixels` is not a positive integer.
    """

    if not isinstance(num_bg_pixels, int) or num_bg_pixels <= 0:
        raise ValueError("`num_bg_pixels` must be a positive integer.")

    if not isinstance(num_cell_pixels, int) or num_cell_pixels <= 0:
        raise ValueError("`num_cell_pixels` must be a positive integer.")

    total_pixels = num_bg_pixels + num_cell_pixels

    # Calculate inverse frequency weights to compensate for class imbalance
    class_weights = torch.tensor([num_cell_pixels / total_pixels, num_bg_pixels / total_pixels])

    return class_weights


# TODO: work in progress
class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss function for addressing class imbalance in binary segmentation tasks.

    Args:
        alpha (float, optional): Hyperparameter to down-weight well-classified examples.
                                 Defaults to 0.25.
        gamma (float, optional): Hyperparameter for focusing on hard-to-classify examples.
                                 Defaults to 2.
    """

    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target, weights):
        """
        Calculates the Weighted Focal Loss.

        Args:
            input (torch.Tensor): Predicted logits (before sigmoid) with shape (N, C, H, W).
            target (torch.Tensor): Ground truth labels with shape (N, H, W).
            weights (torch.Tensor): Per-class weights with shape (C,).

        Returns:
            torch.Tensor: Weighted Focal Loss with shape (1,).
        """

        # Calculate base binary cross-entropy loss for each pixel
        ce_loss = nn.functional.binary_cross_entropy_with_logits(input, target, reduction="none")

        # Calculate probability of being the true class
        pt = torch.exp(-ce_loss)

        # Modulate loss for hard-to-classify examples and down-weight easy examples
        focal_loss = (self.alpha * pt ** self.gamma) * ce_loss

        # Apply per-class weights
        weighted_loss = weights * focal_loss

        # Return the average weighted loss
        return weighted_loss.mean()


def get_loss(config):
    """ Get loss function(s) for the training process based on the current architecture.

    :param loss_function: Loss function to use.
        :type loss_function: str
    :return: Loss function / dict of loss functions.
    """

    if config['loss'] == 'l1':
        border_criterion = nn.L1Loss()
        cell_criterion = nn.L1Loss()
    elif config['loss']  == 'l2':
        border_criterion = nn.MSELoss()
        cell_criterion = nn.MSELoss()
    elif config['loss']  == 'smooth_l1':
        border_criterion = nn.SmoothL1Loss()
        cell_criterion = nn.SmoothL1Loss()
    
    # NOTE: Cross entropy for the segmentation mask should be fixed.
    if config['architecture'][0] == 'dual-unet':
        criterion = {'border': border_criterion, 'cell': cell_criterion}
        
    elif config['architecture'][0] == 'triple-unet':
        mask_criterion = nn.CrossEntropyLoss()
        criterion = {'border': border_criterion, 'cell': cell_criterion, 'mask': mask_criterion}
    return criterion
