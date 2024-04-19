import torch.nn as nn
import torch.nn.functional as F
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


def calculate_class_weights(num_bg_pixels: int, num_cell_pixels: int, emergency_coefficent = 10) -> torch.Tensor:
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
        # NOTE: In Experiment 02 can be accepted patch of just cells pixels...
        
        print(f"Current batch is found with 0 pixel for the background: using {emergency_coefficent} number of 'fake' background pixels")
        num_bg_pixels = emergency_coefficent
        # Console debug for easy visualization
        print("`num_bg_pixels` must be a positive integer.")

    if not isinstance(num_cell_pixels, int) or num_cell_pixels <= 0:
        raise ValueError("`num_cell_pixels` must be a positive integer.")

    total_pixels = num_bg_pixels + num_cell_pixels

    # Calculate inverse frequency weights to compensate for class imbalance
    class_weights = torch.tensor([num_cell_pixels / total_pixels, num_bg_pixels / total_pixels])
    return class_weights


# NOTE: Custom class to dinamically use a criterion with the current class imbalanceness
class WeightedCELoss(nn.Module):
    """
    Wrapper function for dynamically weighted cross-entropy loss.

    Args:
        weight_func (callable, optional): Function that calculates class weights for each batch.
                                            Defaults to None.
    """

    def __init__(self, weight_func=None, device = None):
        super().__init__()
        self.weight_func = weight_func
        self.device = device

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
            class_weights = self.weight_func(target, device = self.device)

            # Apply weights in forward pass
            loss = nn.CrossEntropyLoss(weight=class_weights)
            
            # TODO: Modify here the shape of the target if necessary - Attention to the shape of the "target" of the cross entropy loss.
            target = target[:, 0, :, :]
            return loss(input, target)

        else:
            raise ValueError(f"The provided weight function is not valid!")


# NOTE: Work in progress - finish to implement
class CrossEntropyDiceLoss(nn.Module):
    """
    Wrapper function for dynamically weighted cross-entropy loss plus the Dice loss for refining the borders.

    Args:
        weight_func (callable, optional): Function that calculates class weights for each batch.
                                            Defaults to None.
    """

    def __init__(self, weight_func = None, device = None):
        super().__init__()
        self.weight_func = weight_func
        self.device = device

    def forward(self, inputs, targets):
        
        if self.weight_func is not None:

            # Calculate class weights based on the current batch
            class_weights = self.weight_func(targets, device = self.device)
            # NOTE: Convert the targets to eliminate the single channel
            targets = targets[:, 0, :, :]

            # Compute weighted cross entropy loss for all the batch
            cross_entropy_loss = F.cross_entropy(inputs, targets, weight=class_weights)

            # Compute Dice loss
            dice_loss = self._dice_loss(inputs, targets)

            # Combine the two losses
            combined_loss = cross_entropy_loss + dice_loss
            return combined_loss

        elif self.weight_func is None:
            raise ValueError(f"The provided weight function for the cross-entropy loss is not valid!")

        
    def _dice_loss(self, inputs, targets):
        # Not parallelized - simple loop along the batch of predicted images and targets

        # Aggregated dice coefficent for all the classes
        batch_size = inputs.size(0)
        num_classes = inputs.size(1)
        dice_loss = 0

        # Cycle over the batch - not efficent for now but can allow the analysis of the experiment
        for i in range(batch_size):

            # Single coeff. for the multilabel targets
            dice_coeff = 0
            # Fetch the 'i' target to convert to one_hot encoding
            current_one_hot_target = F.one_hot(targets[i, :, :], num_classes = num_classes)
            current_inputs = inputs[i, :, :, :]

            for channel in range(num_classes):

                # Select correpsonding channel with the logits
                dice_coeff -= self._single_class_dice_coeff(current_inputs[channel, :, :], current_one_hot_target[:, :, channel])
 
            # Convert Dice coefficient to Dice loss for the current image depending on the number of classes
            dice_loss += (1 * num_classes) + dice_coeff
        
        # Average the loss along the batch
        dice_loss = dice_loss/batch_size
        return dice_loss


    def _single_class_dice_coeff(self, input, target, smooth = 1):
        # Single binary class dice coefficent computation
        
        intersection = torch.sum(input * target)
        union = torch.sum(input) + torch.sum(target)

        # Return the final coeff. for the current binary class
        return (2 * intersection + smooth) / (union + smooth)


# NOTE: Work in progress - to test the values and the efficacy in training
class MultiClassJLoss(nn.Module):
    """
    Combined loss function with multi-class J-regularization and binary cross-entropy.
    """

    def __init__(self, lambda_=1.0, class_weights=None, device=None):
        """
        Args:
            lambda_: Weighting factor for J-regularization term (default: 1.0).
            class_weights: Optional tensor of size (C, C) for pairwise class weights.
        """
        super(MultiClassJLoss, self).__init__()
        self.lambda_ = lambda_
        self.class_weights = class_weights

        # Temporary code
        if device != None:
            self.device = device

    def forward(self, predictions, targets):
        """
        Calculates the combined loss.

        Args:
            predictions: Predicted probability tensor (shape: [B, C, H, W]).
            targets: Ground truth label tensor (shape: [B, H, W]).

        Returns:
            Combined loss (tensor).
        """
        B, C, H, W = predictions.shape
    
        # Ensure shape compatibility for BCE loss
        if C != 1:

            # If predictions are multi-class, reduce them to binary for each class
            targets = targets.long()  # Ensure targets are long type (integers)
            bce_loss = torch.zeros((1)).to(self.device)

            for i in range(C):
                bool_mask = targets == i
                byte_mask = bool_mask.byte()  # Convert boolean mask to byte (int8)

                # Directly use byte_mask for BCE loss (no need to add to bce_loss) - both input and target have to be float!
                bce_loss += F.binary_cross_entropy_with_logits(
                    predictions[:, i, :, :], byte_mask.float(), reduction='mean'
                )
        else:
            # Otherwise, directly calculate binary cross-entropy with float target
            targets = targets.float()  # Ensure targets are float for BCE loss
            bce_loss = F.binary_cross_entropy_with_logits(
                predictions, targets, reduction='mean'
            )

        # One-hot encode targets for calculating y_i(p) - calculate normalization factors (n_i) for each class in the batch
        one_hot_targets = F.one_hot(targets, num_classes=C)  # One-hot encoding

        # Sum pixel counts across spatial dimensions (H and W)
        n_i = torch.sum(one_hot_targets, dim=(1, 2))

        # In case of zero division
        eps =  1e-6
        # Reshape
        reshaped_n_i = n_i.unsqueeze(1).unsqueeze(2)  # Shape: [2, 1, 1, 2]
        # Calculate normalized targets (varphi_i)
        normalized_targets = one_hot_targets / (reshaped_n_i+ eps).float() 
        
        # Initialize J_reg_term for batch accumulation - TO TEST
        j_reg_term = torch.zeros(B).to(self.device)

        if self.class_weights is None:
            self.class_weights = torch.ones((C, C))

        # Iterate through class pairs considering the all batch
        for i in range(C):
            for k in range(C):
                if i != k:  # Avoid self-comparison
                    # Calculate difference of normalized activations (Delta_ik)
                    delta_ik = (normalized_targets[:, :, :, i] - normalized_targets[:, :, :, k]) / 2

                    reshaped_delta_ik = delta_ik.unsqueeze(1)
                    # Multiply the correpsondent i-channel with the computed delta_ik
                    sum_multiplied_z_i = torch.sum(predictions[:, i, :, :] * reshaped_delta_ik)

                    # J_reg term for each class pair in the batch
                    j_reg_term += self.class_weights[i, k] * torch.log(0.5 + sum_multiplied_z_i)

        # Combine loss terms after taking the mean of the j_reg_term batch (what should i do with the bce_loss? mean or keeping the sum?)
        loss = bce_loss + self.lambda_ * j_reg_term.mean()
        return loss


def get_loss(config, device):
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

        if config["classification_loss"] == "weighted-cross-entropy":
            mask_criterion = WeightedCELoss(weight_func=get_weights_tensor, device = device)

        elif config["classification_loss"] == "cross-entropy":
            mask_criterion = nn.CrossEntropyLoss()

        else:
            raise ValueError(f"The {config['classification_loss']} is not supported among the classificaiton losses!")

        criterion = {'border': border_criterion, 'cell': cell_criterion, 'mask': mask_criterion}

    elif config['architecture'][0] == 'original-dual-unet':
        
        if config["classification_loss"] == "weighted-cross-entropy":
            mask_criterion = WeightedCELoss(weight_func=get_weights_tensor, device = device)

        elif config["classification_loss"] == "cross-entropy-dice":
            mask_criterion = CrossEntropyDiceLoss(weight_func=get_weights_tensor, device = device)
        
        elif config["classification_loss"] == "cross-entropy":
            mask_criterion = nn.CrossEntropyLoss()

        # NOTE: Finish test
        elif config["classification_loss"] == "j-cross-entropy":
            mask_criterion = MultiClassJLoss(lambda_=0.5,  # Adjust weighting factor as needed
                                            class_weights=torch.tensor([[1.0, 1.0, 1.0],  # Example class weights - try with 2 and 3 classes.
                                                                        [1.0, 1.0, 1.0],
                                                                        [1.0, 1.0, 1.0]]),
                                                                        device = device)
        else:
            raise ValueError(f"The {config['classification_loss']} is not supported among the classificaiton losses!")

        criterion = {'binary_border': mask_criterion, 'cell': cell_criterion, 'mask': mask_criterion}
    return criterion


def compute_cross_entropy(pred_batches, batches_dict, criterion):
    # Pred. batches as dict. to decrease the number of args - batches dict. is the correspondent GT images batches

    # Adapt the dim. of the target batches
    target_binary_border = batches_dict["binary_border_label"][:, 0, :, :]
    target_mask =  batches_dict["mask_label"][:, 0, :, :]    

    loss_cell = criterion['cell'](pred_batches["cell_pred"], batches_dict["cell_label"])
    loss_binary_border = criterion['binary_border'](pred_batches["binary_border_pred"], target_binary_border)
    loss_mask = criterion['mask'](pred_batches["mask_pred"], target_mask)

    loss = loss_binary_border + loss_cell + loss_mask
    losses_list = [loss_binary_border.item(), loss_cell.item(), loss_mask.item()]
    return loss, losses_list


def compute_weighted_cross_entropy_dice(pred_batches, batches_dict, criterion):
    # Wrappper function to encapsulate the loss computation (in this case of the cross-entropy plus the dice contribution)

    loss_cell = criterion['cell'](pred_batches["cell_pred"], batches_dict["cell_label"])
    loss_binary_border = criterion['binary_border'](pred_batches["binary_border_pred"], batches_dict["binary_border_label"])
    loss_mask = criterion['mask'](pred_batches["mask_pred"], batches_dict["mask_label"])

    loss = loss_binary_border + loss_cell + loss_mask
    losses_list = [loss_binary_border.item(), loss_cell.item(), loss_mask.item()]
    return loss, losses_list


def compute_weighted_cross_entropy(pred_batches, batches_dict, criterion):
    # Wrappper function to encapsulate the loss computation

    loss_cell = criterion['cell'](pred_batches["cell_pred"], batches_dict["cell_label"])
    loss_binary_border = criterion['binary_border'](pred_batches["binary_border_pred"], batches_dict["binary_border_label"])
    loss_mask = criterion['mask'](pred_batches["mask_pred"], batches_dict["mask_label"])

    loss = loss_binary_border + loss_cell + loss_mask
    losses_list = [loss_binary_border.item(), loss_cell.item(), loss_mask.item()]
    return loss, losses_list


def compute_j_cross_entropy(pred_batches, batches_dict, criterion):

    # Adapt the dim. of the target batches
    target_binary_border = batches_dict["binary_border_label"][:, 0, :, :]
    target_mask =  batches_dict["mask_label"][:, 0, :, :]

    loss_cell = criterion['cell'](pred_batches["cell_pred"], batches_dict["cell_label"])
    loss_binary_border = criterion['binary_border'](pred_batches["binary_border_pred"], target_binary_border)
    loss_mask = criterion['mask'](pred_batches["mask_pred"], target_mask)

    loss = loss_binary_border + loss_cell + loss_mask
    # The custom j regularized cross entropy is a value from functional package - not an object
    losses_list = [loss_binary_border.item(), loss_cell.item(), loss_mask.item()]

    # DEBUG
    print(loss)
    print(losses_list)
    return loss, losses_list




