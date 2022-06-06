import argparse
from easydict import EasyDict as ed

import os
from pathlib import Path
from zipfile import ZipFile
import pandas as pd
import numpy as np
import ast
import torch
import yaml
import pickle
from utils import object_from_dict, get_samples, str2bool, split_dataset
from experiment import SegmentCyst

from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from eval import eval_model
from train import train
from argparse import Namespace
from write_results import *
from albumentations.core.serialization import from_dict

import wandb

def get_args():
    parser = argparse.ArgumentParser(description='CV with selected experiment as test set and train/val (+test) stratified from the others')
    parser.add_argument("-c", "--config_path", type=Path, help="Path to the config.", default="configs/baseline.yaml")
    parser.add_argument("-d", "--dataset", type=str, help="Select dataset version from wandb Artifact (v1, v2...), set to 'nw' (no WB) to use paths from the config file. Default is 'latest'.", default='v1')
    parser.add_argument("--tag", type=str, help="Add custom tag on the wandb run (only one tag is supported).", default=None)#'loto_cv_with5')
    
    parser.add_argument("-e", "--exp", default=None, type=int, help="Experiment to put in test set")
    parser.add_argument("--single_exp", default=None, type=int, help="Perform CV only on a single experiment.")
    parser.add_argument("-t", "--tube", default=None, type=int, help="If present, select a single tube as test set (integer index between 0 and 31).")
    
    parser.add_argument("-m", "--alternative_model", type=str, default=None, help="Select model different from U++.")
    parser.add_argument("-k", "--k", type=int, default=0, help="Number of the fold to consider between 0 and 4.")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Change the seed to the desired one.")
    
    parser.add_argument("--stratify_fold", nargs='?', type=str2bool, default=False, const=True, help="Split dataset with StratifiedKFold instead of GroupKFold.")
    
    parser.add_argument("-f", "--focus_size", default=None, help="Select 'small_cysts' ('s') or 'big_cysts' ('b') labels (only avaiable from 'v6' dataset).")
    parser.add_argument("--tiling", nargs='?', type=str2bool, default=False, const=True, help="If applied, uses the latest tiled-dataset available in WB.")
    
#     parser.add_argument("--eval_network", nargs='?', type=str2bool, default=False, const=True, help="Performs evaluation on the defined test set.")
    parser.add_argument('--discard_results', nargs='?', type=str2bool, default=False, const=True, help = "Prevent Wandb to save validation result for each step.")
    
    parser.add_argument('--debug', type=str2bool, default=False, help = "If enabled skip checks and logging")
    
    return parser.parse_args()


def main(args):
    print("Running cross validation")
    os.environ["WANDB_START_METHOD"] = "fork"
#     os.environ["WANDB_RUN_GROUP"] = "loto_cv_newdf"
#     torch.cuda.empty_cache()

    name = f"crossval"
    for kind in ["tube", "exp", "k", "seed", "alternative_model", "tag"]:
        if getattr(args, kind) is not None:
            n = f"_{kind}" if not "model" in kind else "_model"
            if kind == "tag": n = ""
            name += f"{n}_{getattr(args, kind)}"
    
    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)
            
    hparams['seed'] = args.seed
    hparams['tube'] = args.tube
    
    print("---------------------------------------")
    print("        Running Crossvalidation        ")
    if args.tiling:
        print("         with tiled dataset        ")
    if args.alternative_model is not None:
        print(f"         model: {args.alternative_model}  ")
    if args.exp is not None:
        print(f"           exp: {args.exp}  ")
    if args.tube is not None:
        print(f"         tube: {args.tube}  ")
    print(f"          seed: {args.seed}           ")
    print(f"          fold: {args.k}       ")
    print("---------------------------------------\n")
    

    # Download datast if not existing
    if args.dataset != 'nw':
        if not args.debug:
            run = wandb.init(project="3d",
                            entity="rene-policistico", config=hparams,
                            settings=wandb.Settings(start_method='fork'),
                            tags=[args.tag] if args.tag else None, reinit=True,
                            name=name
                            )
            # Directory: 'upp' for 2d, or '3d'
            dataset = run.use_artifact(f'rene-policistico/3d/dataset:{args.dataset}', type='dataset')
            dataset.download()
            print("DATASET DOWNLOADED")

    # splits = _get_splits(hparams, args)
    # model = train(args, splits, hparams, name)
    print("INITIALIZING TRAIN")
    model = train(args, None, hparams, name)
    if model is None:
        print("MODEL IS NONE")
        wandb.finish 
        return

    
#     if args.eval_network:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.model = model.model.to(device)
    print("EVALUATING")
    eval_model(args=ed(inpath=hparams["checkpoint_callback"]["dirpath"],
                       batch_size=hparams["val_parameters"]["batch_size"],
                       tube=hparams['tube'],
                       subset='test',
                       exp=None,
                       thresh=.5,
                       outpath='result',
                       debug=args.debug
                      ),
               model=model,
               save_fps=True
              )
        
    
    # real_mask_PATH = hparams["mask_path"]
    # real_img_PATH = hparams["image_path"]
    write_results(hparams["checkpoint_callback"]["dirpath"]/'result'/'test',
                  maskp=hparams["mask_path"], imgp=hparams["image_path"])
        
    wandb.finish()
    return


if __name__ == '__main__':
    args = get_args()
    main(args)


