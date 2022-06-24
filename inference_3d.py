import torch
from PIL import Image
from pathlib import Path
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import logistic
import numpy as np
from torchvision import transforms
from MyDataset.MyDataset import MyDataset
from utils import object_from_dict
from argparse import Namespace
import json


DEVICE = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


def load_split_tubules(ckpt_path: Path):
    with open(ckpt_path / "split_tubules.json") as f:
        tubules = json.load(f)
    assert tubules["train"] and tubules["validation"] and tubules["test"]
    return tubules


def eval_model(args: Namespace, model, ckpt_path: Path, arch: str):
    test_aug = transforms.Compose([])
    tubules = load_split_tubules(ckpt_path)
    tubs_test = tubules["test"]
    dataset = MyDataset(512, test_aug, tubs_test, False, arch)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    result_dir = ckpt_path / "inference_3d"
    result_dir.mkdir(exist_ok=True)
    if any(result_dir.glob("*.png")):
        # print(f"Overwriting test images for {ckpt_path}")
        print(f"Skipping {ckpt_path}")
        return

    desc = f"Test model ({'/'.join(str(ckpt_path).split('/')[-2:])})"
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc=desc):
            x = data["features"].to(DEVICE)
            batch_result = model(x)
            if type(batch_result) == dict:
                batch_result = batch_result["pred"]
            for i in range(batch_result.shape[0]):
                name = data["image_id"][i]
                result = batch_result[i][0]
                result = logistic.cdf(result.cpu().numpy())
                result = (result > args.thresh).astype(np.uint8)
                Image.fromarray(result * 255).save(result_dir / f"{name}.png")


def load_model(ckpt_path: Path):
    print("Loading model:", ckpt_path)
    # Load model and weights
    checkpoint = torch.load(
        next(ckpt_path.glob("*.ckpt")), map_location=lambda storage, loc: storage
    )
    model = object_from_dict(checkpoint["hyper_parameters"]["model"])
    model = model.to(DEVICE)
    state_dict = {k.split("model.")[-1]: v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    return model
    # Compute metrics with write_results on generated images: Moved to notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Test network for selected experiment or for teh full dataset."
    )
    # parser.add_argument(
    #     "-i",
    #     "--inpath",
    #     type=Path,
    #     required=True,
    #     help="Path in which to find the model weights. Example: 'cyst_checkpoints/Nature_v3/2022-06-12 17:03:49'",
    # )
    parser.add_argument(
        "-t",
        "--thresh",
        type=float,
        default=0.5,
        help="threshold for discretization (None for heatmaps of the predictions). Default is 0.5.",
    )
    args = parser.parse_args()

    # runs_dir = Path("cyst_checkpoints") / "Nature_v3"
    runs_dir = Path("cyst_checkpoints") / "3d_lug21_256"
    dirs = sorted(runs_dir.glob("crossval_tube_*"))
    assert dirs, "No tubules to scan"

    for d in dirs:
        model = load_model(d)
        arch = d.stem[-2:]
        assert arch in ["2d", "3d"]
        eval_model(args, model, d, arch)
