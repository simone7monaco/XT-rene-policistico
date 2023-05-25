# rene-policistico

## Prerequisites

```
numpy
Pillow
tqdm
matplotlib
easydict
PyYAML
torch
torchvision
pandas
sklearn
opencv-python
IPython
seaborn
histolab
addict
pytorch_toolbelt
albumentations
segmentation_models_pytorch
thop
pytorch_lightning
wandb
```

## Data preprocessing
* `annotation_original_preprocessing.py`: Clean json files created with labelme.
* `create_mask.ipynb`: Create and visualize binary masks from annotations.

## Training
The training scheme as an run of the cross validation pipeline is available in `crossval_perexp.py`, see
```crossval_perexp.py --help```
for more information.
