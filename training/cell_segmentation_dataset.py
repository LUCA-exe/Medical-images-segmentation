import numpy as np
import tifffile as tiff
import torch
from typing import Union, Dict, List
from torch.utils.data import Dataset
import scipy.ndimage as ndimage
import copy
from pathlib import Path
from torchvision import transforms

from training.mytransforms import ToTensor
from net_utils.utils import save_image


class CellSegDataset(Dataset):
    """Custom data set for instance cell nuclei segmentation.
    """

    def __init__(self, root_dir: Path, 
                 labels: tuple[str],
                 mode: str ='train', 
                 transform: transforms.Compose = lambda x: x):
        """
        Args:
            root_dir: Directory containing all created training/validation data sets.
            labels: Ground-truth labels of the images to fetch for the train or val phase. 
            mode: 'train' or 'val'.
            transform: transforms.
            :type transform:
        """
        if not isinstance(root_dir, Path):
            raise TypeError(f'The parameter root_dir passed is a type {type(root_dir)} instead of {Path}')
        self.img_ids = sorted((root_dir / mode).glob('img*.tif'))
        if len(self.img_ids) == 0:
            raise ValueError(f"The {(root_dir / mode)} doens't contains any image!")
        self.labels = labels
        self.mode = mode
        self.root_dir = root_dir
        if not isinstance(transform, transforms.Compose) and mode == 'train':
            raise TypeError(f'The parameter tranform passed is a type {type(transform)} instead of {transforms.Compose}')
        if not isinstance(transform, ToTensor) and mode == 'val':
            raise TypeError(f'The parameter tranform passed is a type {type(transform)} instead of {transforms.Compose}')
        self.transform = transform

    def get_sample_keys(self, idx: int = 0) -> List[str]:
        """Util function to get the expected filtered keys.

        Specifically, It will return the 'example' of ground truth
        labels expected during the loss computation.
        """
        keys = list(self.__getitem__(idx).keys())
        keys.remove("id")
        keys.remove("image")
        return keys

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Get the item by loading the correct images and applying tranformation functions.

        Args:
            idx: Integer corresponding to the idx-th images in the stored 
            file names.
        """
        img_id = self.img_ids[idx]
        img = tiff.imread(str(img_id))
        sample = {}

        # Add the fixed key-item pairs.
        sample['image'] = img
        sample['id'] = img_id.stem

        # Composing the custom sample for the current neural networks.
        for label in self.labels:
            if label == "dist_neighbor":
                dist_neighbor_label_id = img_id.parent / ('dist_neighbor' + img_id.name.split('img')[-1])
                sample['border_label'] = tiff.imread(str(dist_neighbor_label_id)).astype(np.float32)

            # FIXME: Temporary coupling of creating just the cell_label key in the hasmap.
            if label == "dist_cell":
                dist_label_id = img_id.parent / ('dist_cell' + img_id.name.split('img')[-1])
                sample['cell_label'] = tiff.imread(str(dist_label_id)).astype(np.float32)

            if label == "mask_label":
                mask_label_id = img_id.parent / ('mask_label' + img_id.name.split('img')[-1])
                sample['mask_label'] = tiff.imread(str(mask_label_id)).astype(np.uint8)
            
            if label == "binary_border_label":
                binary_border_label_id = img_id.parent / ('binary_border_label' + img_id.name.split('img')[-1])
                sample["binary_border_label"] = tiff.imread(str(binary_border_label_id)).astype(np.uint8)
          
        # Apply custom Compose function.
        sample = self.transform(sample)
        return sample