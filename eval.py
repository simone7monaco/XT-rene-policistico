import pickle
import torch
from PIL import Image
from pathlib import Path
import argparse
from dataloaders import SegmentationDataset
from torch.utils.data import DataLoader
import albumentations as albu
from tqdm import tqdm
from scipy.stats import logistic
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils import (
    get_samples,
    object_from_dict,
    pad_to_size,
    state_dict_from_disk,
    tensor_from_rgb_image,
)


def get_args():
    parser = argparse.ArgumentParser(description=f'Test network for selected experiment or for teh full dataset.')
    parser.add_argument('exp', nargs='*', type=int, default=None, help="Exp between 1~4")
    parser.add_argument('-i', '--inpath', type=Path, default=None, help="Path in which to find the model weights. Alternative to 'exp'.")
    parser.add_argument('--valid', nargs='?', default=False, const=True, help="If used, saples as taken from file 'samples_*' inside INPATH (only validation)")
    parser.add_argument('-t', '--thresh', type=float, default=.5, help="threshold for discretization (None for heatmaps of the predictions). Default is 0.5.")
    parser.add_argument('-o', '--outpath', type=str, default='result', help="Name of the result folder. Default is 'result'.")
    args = parser.parse_args()
    return args


args = get_args()
if args.exp:
    exp = args.exp[-1]
    in_PATH = Path(f"cv_perexp/exp{exp}")
else:
    in_PATH = args.inpath

res_PATH = in_PATH / args.outpath
res_PATH.mkdir(exist_ok=True, parents=True)

device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
model = torch.load(next(in_PATH.glob("*.ckpt")))['hyper_parameters']['model']

# {
#     "type": "segmentation_models_pytorch.UnetPlusPlus",
#     "encoder_name": "resnet34",
#     "classes": 1,
#     "encoder_weights": "imagenet",
# }
model = object_from_dict(model)
model = model.to(device)

corrections = {"model.": ""}
state_dict = state_dict_from_disk(
    file_path=next(in_PATH.glob("*.ckpt")),
    rename_in_layers=corrections,
)


model.load_state_dict(state_dict)

# transform = albu.augmentations.transforms.Normalize()
transform = albu.augmentations.transforms.Normalize(
    always_apply=False, max_pixel_value=255.0,
    mean=[0.485,0.456,0.406], p=1,std=[0.229,0.224,0.225]
)
        


if args.exp:
    res_PATH = res_PATH / "test"
    res_PATH.mkdir(exist_ok=True, parents=True)
    with Path(f"cv_perexp/exp{exp}/test_samples_oldnames.pickle").open("rb") as file:
        samples = pickle.load(file)
if args.valid:
    res_PATH = res_PATH / "valid"
    res_PATH.mkdir(exist_ok=True, parents=True)
    with open(in_PATH / 'split_samples.pickle', 'rb') as file:
        samples = pickle.load(file)['valid']
else:
    samples = get_samples('artifacts/dataset:v6/images', 'artifacts/dataset:v6/masks')

dataset = SegmentationDataset(samples, transform, length=None)
    
dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

try:
    assert not any(res_PATH.iterdir())

    desc = f" Test model exp {exp}" if args.exp else f" Test model ({args.inpath})"
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc=desc):
            x = data["features"].to(device)
            batch_result = model(x)
            for i in range(batch_result.shape[0]):
                name = data["image_id"][i]

                result = batch_result[i][0]
                result = logistic.cdf(result.cpu().numpy())

                if args.thresh:
                    result = (result > args.thresh).astype(np.uint8)
                    Image.fromarray(result*255).save(res_PATH / f"{name}.png")
                else:
                    fig, ax = plt.subplots(figsize=(8,8))
                    sns.heatmap(result, ax=ax, xticklabels=False, yticklabels=False, cmap='jet', cbar=False)
                    plt.savefig(res_PATH / f"{name}.png")

except:
    print(f" Test for {args.inpath} already done")