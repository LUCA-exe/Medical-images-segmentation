import torch.nn as nn
import torch

def get_weights_tensor(target_mask_batch: torch.Tensor, device: torch.device = None) -> torch.Tensor:
    """
    Calculates and returns a weight tensor for balanced cross-entropy loss in binary segmentation.

    Args:
        target_mask_batch (torch.Tensor): A 4D tensor of shape [N, 1, H, W],
                                            where N is the batch size, H is the height,
                                            and W is the width of the masks,
                                            and each element is 0 or 1.
        device (torch.device, optional): Device to move the calculated class weights to.
                                            Defaults to None (CPU).

    Returns:
        torch.Tensor: A tensor of shape [N, 2] containing weights for background (class 0)
                      and foreground (class 1) for each image in the batch.
    """

    if not isinstance(target_mask_batch, torch.Tensor):
        raise ValueError("`target_mask_batch` must be a torch.Tensor.")

    if target_mask_batch.dim() != 4:
        raise ValueError("`target_mask_batch` must have a dimension of 4.")

    if target_mask_batch.size(1) != 1:
        raise ValueError("`target_mask_batch` must have a channel dimension of 1.")

    # Calculate pixel counts for each class (assuming 0 is background, 1 is foreground)
    background_pixels, cells_pixels = count_pixels(target_mask_batch)

    # Calculate class weights using a separate function for clarity and modularity
    class_weights = calculate_class_weights(background_pixels, cells_pixels)

    # Move class weights to the specified device if provided
    if device is not None:
        class_weights = class_weights.to(device)
    return class_weights


def count_pixels(target_mask_batch):
    """
    Counts the number of pixels equal to 1 and 0 in a target mask tensor, usefull for computing the weights of the classes.

    Args:
        target_mask (torch.Tensor): A 4D tensor of shape [N, 1, 320, 320],
                                    where N is the batch size and
                                    each element is 0 or 1.

    Returns:
        tuple: A tuple of two torch.Tensors, where:
            - The first element is the count of pixels equal to 1, with shape (N,).
            - The second element is the count of pixels equal to 0, with shape (N,).
    """

    # Reshape the target mask to remove the channel dimension (assuming single channel for the mask)
    target_mask_flat = target_mask_batch.view(target_mask_batch.size(0), -1)  # Reshape to [N, 320 * 320]

    # Count the number of pixels equal to 1 and 0 using torch.sum
    num_ones = torch.sum(target_mask_flat == 1, dim=1).sum().item()  # Counts across columns for each image
    num_zeros = torch.sum(target_mask_flat == 0, dim=1).sum().item()   # Counts across columns for each image
    return num_ones, num_zeros


def calculate_class_weights(num_bg_pixels: int, num_cell_pixels: int) -> torch.Tensor:
    """
    Calculates class weights for addressing class imbalance in binary segmentation tasks during the training phase.

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


# TODO: Custom class to dinamically use a crtierion woth the current class imbalancness - It overrides the forward pass.
class WeightedCELoss(nn.Module):
    """
    Wrapper function for dynamically weighted cross-entropy loss.

    Args:
        weight_func (callable, optional): Function that calculates class weights for each batch.
                                            Defaults to None.
    """

    def __init__(self, weight_func=None):
        super().__init__()
        self.weight_func = weight_func
        #self.ce = nn.CrossEntropyLoss()  # Standard cross-entropy loss

    def forward(self, input, target):
        """
        Calculates forward pass with dynamic weight calculation.

        Args:
            input (torch.Tensor): Model output, shape (N, C, H, W).
            target (torch.Tensor): Ground truth labels, shape (N, H, W).

        Returns:
            torch.Tensor: Weighted cross-entropy loss.
        """
  
        if self.weight_func is not None:
            # Calculate class weights based on current batch
            class_weights = self.weight_func(target)

            # Ensure class weights have the correct shape
            '''if target.dim() > 2 and class_weights.dim() == 1:
                class_weights = class_weights.unsqueeze(1).expand_as(target)  # Expand for consistency'''

            # Apply weights in forward pass
            loss = nn.CrossEntropyLoss(weight=class_weights)
            
            # TODO: Modify here the shape of the target if necessary - Attention to the shape of the "target" of the cross entropy loss.
            target = target[:, 0, :, :]

            #print(target.size())
            #print(input.size())

            return loss(input, target)

        else:
            raise ValueError(f"The provided weight function is not valid!")


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
    
    if config['architecture'][0] == 'dual-unet':
        criterion = {'border': border_criterion, 'cell': cell_criterion}
        
    elif config['architecture'][0] == 'triple-unet':

        if config["classification_loss"] == "weigthed-cross-entropy":
            mask_criterion = WeightedCELoss(weight_func=get_weights_tensor)
        elif config["classification_loss"] == "cross-entropy":
            mask_criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"The {config['classification_loss']} is not supported among the classificaiton losses!")


        criterion = {'border': border_criterion, 'cell': cell_criterion, 'mask': mask_criterion}
    return criterion



