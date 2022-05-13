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

Example: `python train.py -c configs/baseline.yaml`

## Data preprocessing
* `annotation_original_preprocessing.py`: Clean json files created with labelme.
* `create_mask.ipynb`: Create and visualize binary masks from annotations.