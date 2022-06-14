import pickle
import torch
import pandas as pd
from time import time
from PIL import Image
from pathlib import Path
import argparse
from dataloaders import SegmentationDataset
from torch.utils.data import DataLoader
import albumentations as albu
from tqdm import tqdm
from scipy.stats import logistic

from HarDNetMSEG.lib.HarDMSEG import HarDMSEG
import segmentation_models_pytorch as smp
from utils import get_tubules_from_json, get_dataloaders

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision import transforms
from MyDataset.MyDataset import MyDataset

from utils import (
    get_samples,
    object_from_dict,
    pad_to_size,
    state_dict_from_disk,
    tensor_from_rgb_image,
)
from argparse import Namespace
import json


def load_split_tubules():
    with open(args.inpath / "split_tubules.json") as f:
        tubules = json.load(f)
    assert tubules["train"] and tubules["validation"] and tubules["test"]
    return tubules


def eval_model(args: Namespace, model, save_fps=False):
    res_PATH = args.inpath / "inference_3d"
    res_PATH.mkdir(exist_ok=True, parents=True)

    test_aug = transforms.Compose([])

    tubules = load_split_tubules()
    tubs_test = tubules["test"]

    dataset = MyDataset(512, test_aug, tubs_test, False)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    if any(res_PATH.glob("*.png")):
        print(f" Test for {args.inpath} already done, overwriting images")
    #     return

    desc = f"Test model ({'/'.join(str(args.inpath).split('/')[-2:])})"
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc=desc):
            x = data["features"].to(device)
            batch_result = model(x)
            if type(batch_result) == dict:
                batch_result = batch_result["pred"]
            for i in range(batch_result.shape[0]):
                name = data["image_id"][i]
                result = batch_result[i][0]
                result = logistic.cdf(result.cpu().numpy())
                result = (result > args.thresh).astype(np.uint8)
                Image.fromarray(result * 255).save(res_PATH / f"{name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Test network for selected experiment or for teh full dataset."
    )
    # parser.add_argument(
    #     "exp", nargs="*", type=int, default=None, help="Exp between 1~4"
    # )
    parser.add_argument(
        "-i",
        "--inpath",
        type=Path,
        required=True,
        help="Path in which to find the model weights. Example: 'cyst_checkpoints/Nature_v3/2022-06-12 17:03:49'",
    )
    parser.add_argument(
        "-t",
        "--thresh",
        type=float,
        default=0.5,
        help="threshold for discretization (None for heatmaps of the predictions). Default is 0.5.",
    )
    args = parser.parse_args()

    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )

    checkpoint = torch.load(
        next(args.inpath.glob("*.ckpt")), map_location=lambda storage, loc: storage
    )
    model = object_from_dict(checkpoint["hyper_parameters"]["model"])
    model = model.to(device)

    state_dict = {k.split("model.")[-1]: v for k, v in checkpoint["state_dict"].items()}

    model.load_state_dict(state_dict)

    eval_model(args, model)
