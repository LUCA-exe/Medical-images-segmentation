import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset


class CellSegDataset(Dataset):
    """ Pytorch data set for instance cell nuclei segmentation """

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
        self.mode = mode
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __prepare_seg_mask(self, mask_picture):
        # Prepare the segmentation mask for the binary cross entropy.

        mask_picture[mask_picture > 0] = 1 # NOTE: Treating all the cells as the same class (not distinguish between the EVs and Cells)
        return mask_picture

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]
        img = tiff.imread(str(img_id))

        dist_label_id = img_id.parent / ('dist_cell' + img_id.name.split('img')[-1])
        mask_label_id = img_id.parent / ('mask' + img_id.name.split('img')[-1])
        dist_neighbor_label_id = img_id.parent / ('dist_neighbor' + img_id.name.split('img')[-1])

        dist_label = tiff.imread(str(dist_label_id)).astype(np.float32)
        # NOTE: Mask need further processing - kept the processing in this class.
        mask_label = tiff.imread(str(mask_label_id)).astype(np.uint8)
        mask_label = self.__prepare_seg_mask(mask_label)
        dist_neighbor_label = tiff.imread(str(dist_neighbor_label_id)).astype(np.float32)

        sample = {'image': img,
                  'cell_label': dist_label,
                  'border_label': dist_neighbor_label,
                  'mask_label': mask_label,
                  'id': img_id.stem}
                  
        # Apply tranformation for the training.
        sample = self.transform(sample)
        return sample