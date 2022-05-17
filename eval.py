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
    parser.add_argument('-s', '--subset', default=None, help="If used, saples are taken from file 'samples_*' inside INPATH, using one of {test, valid,train} set.")
    parser.add_argument('-t', '--thresh', type=float, default=.5, help="threshold for discretization (None for heatmaps of the predictions). Default is 0.5.")
    parser.add_argument('-o', '--outpath', type=str, default='result', help="Name of the result folder. Default is 'result'.")
    args = parser.parse_args()
    return args

device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

def eval_model(args, model, save_fps=False):
    if args.exp:
        exp = args.exp[-1]
        args.inpath = Path(f"cv_perexp/exp{exp}")

    res_PATH = args.inpath / args.outpath
    res_PATH.mkdir(exist_ok=True, parents=True)

    transform = albu.augmentations.transforms.Normalize(
        always_apply=False, max_pixel_value=255.0,
        mean=[0.485,0.456,0.406], p=1,std=[0.229,0.224,0.225]
    )
    
    if args.subset:
        res_PATH = res_PATH / args.subset
        res_PATH.mkdir(exist_ok=True, parents=True)
        with open(args.inpath / 'split_samples.pickle', 'rb') as file:
            p = pickle.load(file)
            samples = p[args.subset]
    else:
        d_fold = sorted(Path('artifacts').iterdir(), key=lambda n: int(n.stem.split(':v')[-1]))[-1]
        samples = get_samples(d_fold / 'images', d_fold / 'masks')

    assert samples is not None, "test set is empty, select a tube with '--tube'"

    dataset = SegmentationDataset(samples, transform, length=None)

    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    if not any(res_PATH.glob('*.png')):

        desc = f" Test model exp {exp}" if args.exp else f" Test model ({'/'.join(str(args.inpath).split('/')[-2:])})"
        model.eval()
        
        timing = []
        with torch.no_grad():
            for data in tqdm(dataloader, desc=desc):
                x = data["features"].to(device)
                batch_result = model(x)
                
                if type(batch_result) == dict:
                    batch_result = batch_result['pred']
                for i in range(batch_result.shape[0]):
                    t0 = time()
                    name = data["image_id"][i]

                    result = batch_result[i][0]
                    result = logistic.cdf(result.cpu().numpy())

                    if args.thresh:
                        result = (result > args.thresh).astype(np.uint8)
                        if save_fps:
                            timing.append([name, time()-t0])
                        Image.fromarray(result*255).save(res_PATH / f"{name}.png")
                    else:
                        fig, ax = plt.subplots(figsize=(8,8))
                        sns.heatmap(result, ax=ax, xticklabels=False, yticklabels=False, cmap='jet', cbar=False)
                        plt.savefig(res_PATH / f"{name}.png")
                        
        if save_fps:
            pd.DataFrame(timing, columns=['name', 'time']).to_csv(res_PATH / "timing.csv")

    else:
        print(f" Test for {args.inpath} already done")

    
if __name__ == '__main__':
    args = get_args()
    
    checkpoint = torch.load(next(args.inpath.glob("*.ckpt")), map_location=lambda storage, loc: storage)
    model = object_from_dict(checkpoint['hyper_parameters']['model'])
    model = model.to(device)
    
    state_dict = {k.split('model.')[-1]: v for k, v in checkpoint["state_dict"].items()}

    model.load_state_dict(state_dict)

    eval_model(args, model)