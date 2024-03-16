'''
This script helps the creation of the inference (evaluation/test) dataset.
'''
import numpy as np
import tifffile as tiff
import torch
from pathlib import Path

from skimage.exposure import equalize_adapthist
from skimage.transform import rescale
from torch.utils.data import Dataset
from torchvision import transforms

from net_utils.utils import zero_pad_model_input


class CTCDataSet(Dataset):
    """ Pytorch data set for Cell Tracking Challenge format data. """

    def __init__(self, data_dir, transform=lambda x: x):
        """

        :param data_dir: Directory with the Cell Tracking Challenge images to predict (e.g., t001.tif)
        :param transform:
        """

        self.img_ids = sorted(Path(data_dir).glob('t*.tif'))
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx): # Here adapat my 3 channel images to the single channel expected input.

        img_id = self.img_ids[idx]
        image = tiff.imread(str(img_id))  

        # Processing in case of RGB images as my dataset
        if len(image.shape) > 2:

            single_channel_img = image[:, :, 0] # 0 is the EVs channel (Red)
            img = np.sum(image, axis=2) # Keep all object
        else:
            # Flag for the other dataset
            single_channel_img = None

        sample = {'image': img,

                # AD-HOC for my dataset
                'single_channel_image': single_channel_img,
                'id': img_id.stem}
        sample = self.transform(sample)
        return sample


def pre_processing_transforms(apply_clahe, scale_factor):
    """ Get transforms for the CTC data set.

    :param apply_clahe: apply CLAHE.
        :type apply_clahe: bool
    :param scale_factor: Downscaling factor <= 1.
        :type scale_factor: float

    :return: transforms
    """

    data_transforms = transforms.Compose([ContrastEnhancement(apply_clahe),
                                          Normalization(),
                                          Scaling(scale_factor),
                                          Padding(),
                                          ToTensor()])

    return data_transforms


class ContrastEnhancement(object):

    def __init__(self, apply_clahe):
        self.apply_clahe = apply_clahe

    def __call__(self, sample):

        if self.apply_clahe:
            img = sample['image']
            img = equalize_adapthist(np.squeeze(img), clip_limit=0.01)
            img = (65535 * img).astype(np.uint16)
            sample['image'] = img

            # TODO: refactor the tranformation in a private function of this objetc
            if not sample["single_channel_image"] is None:
                img = sample["single_channel_image"]
                img = equalize_adapthist(np.squeeze(img), clip_limit=0.01)
                img = (65535 * img).astype(np.uint16)
                sample["single_channel_image"] = img
        return sample


class Normalization(object):

    def __call__(self, sample):

        img = sample['image']
        img = 2 * (img.astype(np.float32) - img.min()) / (img.max() - img.min()) - 1
        sample['image'] = img

        # TODO: refactor the tranformation in a private function of this objetc
        if not sample["single_channel_image"] is None:
            img = sample["single_channel_image"]
            img = 2 * (img.astype(np.float32) - img.min()) / (img.max() - img.min()) - 1
            sample["single_channel_image"] = img
        return sample


class Padding(object):

    def __call__(self, sample):

        img = sample['image']
        img, pads = zero_pad_model_input(img=img, pad_val=np.min(img))
        sample['image'] = img
        sample['pads'] = pads

        # TODO: refactor the tranformation in a private function of this objetc
        if not sample["single_channel_image"] is None:
            img = sample["single_channel_image"]
            img, _ = zero_pad_model_input(img=img, pad_val=np.min(img))
            sample["single_channel_image"] = img
        return sample


class Scaling(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):

        img = sample['image']
        sample['original_size'] = img.shape

        if self.scale < 1:

            if len(img.shape) == 3:
                img = rescale(img, (1, self.scale, self.scale), order=2, preserve_range=True).astype(img.dtype)
            else:

                img = rescale(img, (self.scale, self.scale), order=2, preserve_range=True).astype(img.dtype)
                
                # Additional processing on the single channel if not None
                if not sample['single_channel_image'] is None:
                    single_channel_img = sample['single_channel_image']
                    single_channel_img = rescale(single_channel_img, (self.scale, self.scale), order=2, preserve_range=True).astype(single_channel_img.dtype)
                    sample['single_channel_image'] = single_channel_img

            sample['image'] = img
        return sample


class ToTensor(object):
    """ Convert image and label image to Torch tensors """

    def __call__(self, sample):

        img = sample['image']
        if len(img.shape) == 2:

            img = img[None, :, :]
            if not sample["single_channel_image"] is None:

                single_channel_image = sample["single_channel_image"]
                single_channel_image = single_channel_image[None, :, :]
                single_channel_image = torch.from_numpy(single_channel_image).to(torch.float)
                sample["single_channel_image"] = single_channel_image

        img = torch.from_numpy(img).to(torch.float)
        sample["image"] = img
        #return img, sample['id'], sample['pads'], sample['original_size']
        return sample # Directly return the dict. of the batches