# Datasets folder path

## 1. Overview
This datasets are intended for image segmentation and follow a specific format.
In this folder will contain all the dataset folders used for model training and evaluation
with ground-truth label.
Those images will be used for successive processing after the segmentaion process (e.g EVs counting).
(The format specified by the "Cell tracking challenge" - https://celltrackingchallenge.net/.)

## 2. Directory Structure
Fluo-E2DV-train/
├── 01/                        # Contains all training images in TIFF format
│   ├── image1.tiff
│   ├── image2.tiff
│   └── ...                    # Additional images
│
└── 01_GT/                     # Contains ground truth annotations for training images
    ├── SEG/                   # SEG folder for segmentation annotations (masks)
    │   ├── annotations.json   # JSON file with image annotations (used to create the segmentation masks)
    │   ├── mask1.tiff         # Segmentation mask in TIFF format
    │   └── ...                # Additional masks
    │
    └── TRA/                   # TRA folder for tracking annotations
        ├── man_track.txt      # Manual cell tracking annotations (in TXT format)
        ├── track_mask1.tiff   # Tracking segmentation mask in TIFF format
        └── ...                # Additional tracking masks

Fluo-E2DV-test/                # (Structure identical to Fluo-E2DV-train)
├── 01/                        # Testing images in TIFF format
│   ├── image1.tiff
│   ├── image2.tiff
│   └── ...                    # Additional images
│
└── 01_GT/                     # Contains ground truth annotations for testing images
    ├── SEG/                   # SEG folder for segmentation annotations (masks)
    │   ├── annotations.json   # JSON file with image annotations (used to create the segmentation masks)
    │   ├── mask1.tiff         # Segmentation mask in TIFF format
    │   └── ...                # Additional masks
    │
    └── TRA/                   # TRA folder for tracking annotations
        ├── man_track.txt      # Manual cell tracking annotations (in TXT format)
        ├── track_mask1.tiff   # Tracking segmentation mask in TIFF format
        └── ...                # Additional tracking masks


## 3. Dataset Creation
- **Image Collection**: Images were sourced in laboratory and choosen based on performance on traning the models.
- **Annotation**: Annotations were created using VGG Annotation tool in png format.
- **Additional specifics**: Can be found in the 'Annotaion pipeline.docx'