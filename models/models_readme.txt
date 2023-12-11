The folder 'models' contains respectively 3 folders:
- 'Trained': Models that are end_to_end trained on a dataset.
- 'Pre-trained': Models that are trained on a dataset and waiting to be fine-tuned.
- 'Fine-tuned': Models that are fine-tuned after being pretrained.
- 'all': model to train/fine-tune. Default folder to fetch for training/inference.

The name format of the model files (both for '*.pth' and '*.json') is:
1. <Dataset used for training>_<GT or GT_ST>_model.pth
2. <Dataset used for training>_<GT or GT_ST>_<Dataset used for fine-tuning>_<GT or GT_ST>_model.pth

NOTE: In case of both training/fine-tuning are used all the GT_<id> experiments of the dataset.