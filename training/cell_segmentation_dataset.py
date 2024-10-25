import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset
import scipy.ndimage as ndimage
import copy

from net_utils.utils import save_image


# NOTE: TDD and refactoring in progress.
class CellSegDataset(Dataset):
    """Custom data set for instance cell nuclei segmentation.
    """

    def __init__(self, root_dir, mode='train', transform=lambda x: x):
        """

        :param root_dir: Directory containing all created training/validation data sets.
            :type root_dir: pathlib Path object.
        :param mode: 'train' or 'val'.
            :type mode: str
        :param transform: transforms.
            :type transform:
        :return: Dict (image, cell_label, border_label, id).
        """
        self.img_ids = sorted((root_dir / mode).glob('img*.tif'))
        if len(self.img_ids) == 0:
            raise ValueError(f"The {(root_dir / mode)} doens't contains any image!")
        
        self.mode = mode
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __prepare_seg_mask(self, mask_picture):
        # Prepare the segmentation mask for the binary cross entropy.

        mask_picture = copy.deepcopy(mask_picture) # Work on the copy of the object - no reference
        mask_picture[mask_picture > 0] = 1 # NOTE: Treating all the cells as the same class (not distinguish between the EVs and Cells)
        return mask_picture

    
    # TODO: Prepare the "cell borders" from the original mask - work in progress!!!
    def __extract_cell_borders(self, mask, border_width = 4):

        """
        Extracts borders of cells from a binary segmentation mask adn return it for the ground truth batches.

        Args:
            mask (np.ndarray): Binary mask with cells represented as 1 and background as 0.
            border_width (int, optional): Width of the border to extract. Defaults to 10.

        Returns:
            np.ndarray: Mask with only cell borders remaining.
        """

        # Check the shape of the mask - should be one from the creation of the trianig set of KIT-GE pipeline
        #mask = self.__prepare_seg_mask(mask)
        mask = copy.deepcopy(mask) # Work on the copy of the object - no reference
        mask[mask > 0] = 1 # NOTE: Treating all the cells as the same class (not distinguish between the EVs and Cells)

        # Work with the boolean array to invert the original mask
        inverted_mask = ~mask.astype(bool)
        inverted_mask = inverted_mask.astype(int)

        # Dilate the mask to slightly enlarge the borders (handling very thin borders)
        dil_inverted_mask = ndimage.binary_dilation(inverted_mask, iterations = border_width)
        
        # Obtain the borders directly by difference
        cell_border = dil_inverted_mask ^ inverted_mask
        # NOTE: This now gives an error, the training executed before not - attention on the log file in case of anomaly, maybe updates of packages in colab
        #cell_border = np.expand_dims(cell_border, axis=2)
        return cell_border.astype(mask.dtype)


    def __getitem__(self, idx):

        img_id = self.img_ids[idx]
        img = tiff.imread(str(img_id))

        dist_label_id = img_id.parent / ('dist_cell' + img_id.name.split('img')[-1])
        mask_label_id = img_id.parent / ('mask' + img_id.name.split('img')[-1])
        dist_neighbor_label_id = img_id.parent / ('dist_neighbor' + img_id.name.split('img')[-1])

        dist_label = tiff.imread(str(dist_label_id)).astype(np.float32)
        # NOTE: Mask need further processing - kept the processing in this class.
        mask_label = tiff.imread(str(mask_label_id)).astype(np.uint8)
        binary_border_label = self.__extract_cell_borders(mask_label)
        processed_mask_label = self.__prepare_seg_mask(mask_label)
        dist_neighbor_label = tiff.imread(str(dist_neighbor_label_id)).astype(np.float32)

        sample = {'image': img,
                  'cell_label': dist_label,
                  'border_label': dist_neighbor_label,
                  'mask_label': processed_mask_label,
                  'binary_border_label': binary_border_label,
                  'id': img_id.stem}
                  
        # Apply tranformation for the training.
        sample = self.transform(sample)
        return sample