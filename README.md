# Medical images segmentation
This repository implements and evaluates various [techniques](#projects-considered-for-this-implementation) for a specialized use case: the identification and highlighting of extracellular vesicles in the presence of different cell types (e.g., muscle cells), utilizing validated Deep Learning methods trained on our proprietary datasets.

## Training

### Parameters
The default values for the parameters will be highlighted in bold:
* ```--dataset```: Name of the dataset to use for generate the training images. The default position for the training dataset folder is *./training_data*.
* ```--batch-size```: Batch size used for training (**2**).
* ```--crop_size```: Crop to perform over the original images (**320**). It will be used for generating the training dataset.
* ```--filters```: Number of kernels (**64 1024**). After each pooling, the number is doubled in the encoder till the maximum is reached.
* ```--model-pipeline```: Architecture of the neural networks to select (**"dual-unet"**, "original-dual-unet"). The architecture differs both in terms of structure and losses available.
* ```--loss```: Loss to implement with the chosen architecture (**l1_smooth**, "weightd-cross-entropy", "dice-cross-entroy").
  * **NOTE**: The loss chosen through this parameter has to be available for the chosen architecture.
### Examples
To train new model end-to-end with a specific dataset you can execute:
```
python train.py --crop_size 480 --dataset Fluo-M2DL-train --model_pipeline dual-unet
```

## Contributions
This project was carried out in collaboration with the [TNH Lab](https://tnhlab.polito.it/) at the Polytechnic University of Turin and [U-Care Medical S.r.l.](https://u-caremedical.com/), contributing to both the selection of models and the implementation of segmentation techniques.

## Projects considered for this implementation
* [Cell Segmentation and Tracking using CNN-Based Distance Predictions and a Graph-Based Matching Strategy](https://github.com/TimScherr/KIT-GE-3-Cell-Segmentation-for-CTC)
* [Dual U-Net for the Segmentation of Overlapping Glioma Nuclei](https://ieeexplore.ieee.org/document/8744511)
* [The Importance of Skip Connections in Biomedical Image Segmentation](https://arxiv.org/abs/1608.04117)
