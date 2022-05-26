# rene-policistico

## Project structure
Pre-train
- *crossval_perexp.py*: for k-fold cross-validation
- *./loto.sh*: runs multiple times *crossval_perexp.py*

Post-train
- *annotation_original_preprocessing.py*: clean json files created with labelme
- *create_mask.ipynb*: create and visualize binary masks from annotations

Lib
- *train.py*: used by *crossval_perexp.py*
- *eval.py*: used by *crossval_perexp.py*
- *experiment.py*: LightningModule
- *c_hist_utils.py*
- *dataloaders.py*

To undestand
- *pr_bar_charts.py*
- *simplify_names.py*
- *sweep_params.py*
- *Tuning_params.py*
- *utils.py*
- *write_results.py*

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
pandas==1.3.5
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
