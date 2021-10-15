import argparse
from easydict import EasyDict as ed

import os
from pathlib import Path
from zipfile import ZipFile
import pandas as pd
import numpy as np
import ast
import torch
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, GroupKFold
import yaml
import pickle
from utils import object_from_dict, get_samples, str2bool
from experiment import SegmentCyst

from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from simplify_names import get_packs
from eval import eval_model
from UACANet.run.Train import train

from write_results import *
from albumentations.core.serialization import from_dict

import wandb

def get_args():
    parser = argparse.ArgumentParser(description='CV with selected experiment as test set and train/val (+test) stratified from the others')
    parser.add_argument("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    parser.add_argument("-d", "--dataset", type=str, help="Select dataset version from wandb Artifact (v1, v2...), set to 'nw' (no WB) to use paths from the config file. Default is 'latest'.", default='latest')
    parser.add_argument("--tag", type=str, help="Add custom tag on the wandb run (only one tag is supported).", default=None)#'loto_cv_with5')
    
    parser.add_argument("-e", "--exp", default=None, type=int, help="Experiment to put in test set")
    parser.add_argument("--single_exp", default=None, type=int, help="Perform CV only on a single experiment.")
    parser.add_argument("-t", "--tube", default=None, type=int, help="If present, select a single tube as test set (integer index between 0 and 31).")
    
    parser.add_argument("-m", "--alternative_model", type=str, default=None, help="Select model different from U++.")
    parser.add_argument("-k", "--kth_fold", type=int, default=0, help="Number of the fold to consider between 0 and 4.")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Change the seed to the desired one.")
    
    parser.add_argument("--stratify_fold", nargs='?', type=str2bool, default=False, const=True, help="Split dataset with StratifiedKFold instead of GroupKFold.")
    
    parser.add_argument("-f", "--focus_size", default=None, help="Select 'small_cysts' ('s') or 'big_cysts' ('b') labels (only avaiable from 'v6' dataset).")
    parser.add_argument("--tiling", nargs='?', type=str2bool, default=False, const=True, help="If applied, uses the latest tiled-dataset available in WB.")
    
#     parser.add_argument("--eval_network", nargs='?', type=str2bool, default=False, const=True, help="Performs evaluation on the defined test set.")
    parser.add_argument('--discard_results', nargs='?', type=str2bool, default=False, const=True, help = "Prevent Wandb to save validation result for each step.")
    
    
    return parser.parse_args()


def split_dataset(hparams, k=0, test_exp=None, leave_one_out=None, strat_nogroups=False, single_exp=None):
    samples = get_samples(hparams["image_path"], hparams["mask_path"])
    
    ##########################################################
    if single_exp == 1:
        samples = [u for u in samples if "09.19" in u[0].stem]
    if single_exp == 2:
        samples = [u for u in samples if "10.19" in u[0].stem]
    if single_exp == 3:
        samples = [u for u in samples if "07.2020" in u[0].stem or "09.2020" in u[0].stem]
    if single_exp == 4:
        samples = [u for u in samples if "12.2020" in u[0].stem]
        samples = [u for u in samples if "ctrl 11" in u[0].stem.lower() or "t4" in u[0].stem.lower()]
    if single_exp == 5:
        samples = [u for u in samples if "07.21" in u[0].stem]
    ##########################################################
    
    names = [file[0].stem for file in samples]

#     date, treatment, tube, zstack, side =
    unpack = [get_packs(name) for name in names]
    df = pd.DataFrame([])
    df["filename"] = names
    df["treatment"] = [u[1] for u in unpack]
    df["exp"] = [date_to_exp(u[0]) for u in unpack]
    df["tube"] = [u[2] for u in unpack]
#     df["te"] = df.treatment + '_' + df.exp.astype(str)
    df["te"] = df.treatment + '_' + df.exp.astype(str) + '_' + df.tube.astype(str)
    df.te = df.te.astype('category')
    
    if test_exp is not None or leave_one_out is not None:
        if leave_one_out is not None:
            tubes = df[['exp','tube']].astype(int).sort_values(by=['exp', 'tube']).drop_duplicates().reset_index(drop=True).xs(leave_one_out)
            test_idx = df[(df.exp == tubes.exp)&(df.tube == str(tubes.tube))].index
            
        else:
            test_idx = df[df.exp == test_exp].index
    
        test_samp = [x for i, x in enumerate(samples) if i in test_idx]
        samples = [x for i, x in enumerate(samples) if i not in test_idx]
        df = df.drop(test_idx)
    else:
        test_samp = None
        
    if strat_nogroups:
        skf = StratifiedKFold(n_splits=5, random_state=hparams["seed"], shuffle=True)
        train_idx, val_idx = list(skf.split(df.filename, df.te))[k]
    else:
        df, samples = shuffle(df, samples, random_state=hparams["seed"])
        gkf = GroupKFold(n_splits=5)# =5)
        train_idx, val_idx = list(gkf.split(df.filename, groups=df.te))[k]
    
    train_samp = [tuple(x) for x in np.array(samples)[train_idx]]
    val_samp = [tuple(x) for x in np.array(samples)[val_idx]]
    
    return {
        "train": train_samp,
        "valid": val_samp,
        "test": test_samp
    }

def main(args):
    os.environ["WANDB_START_METHOD"] = "fork"
#     os.environ["WANDB_RUN_GROUP"] = "loto_cv_newdf"
    torch.cuda.empty_cache()
    
    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)
    
    if args.alternative_model:
        with open("configs/searched_params.yaml") as f:
            bs_results = yaml.load(f, Loader=yaml.SafeLoader)
            if args.alternative_model in bs_results.keys():
                hparams["train_parameters"]["batch_size"] = bs_results[args.alternative_model]["train_parameters"]["batch_size"]
                hparams["optimizer"]["lr"] = bs_results[args.alternative_model]["optimizer"]["lr"]
    
            
    hparams['seed'] = args.seed
    hparams['tube'] = args.tube
    
    if "SLURM_JOB_ID" in os.environ:
        hparams["num_workers"] = 4
        hparams["train_parameters"]["batch_size"] = 16
        print('HI LEGION!')

#     name = f"crossval_exp{exp}"
    name = "test"
    for kind in ["tube", "exp", "seed", "alternative_model", "tag"]:
        if getattr(args, kind) is not None:
            n = f"_{kind}" if not "model" in kind else "_model"
            if kind == "tag": n = ""
            name += f"{n}_{getattr(args, kind)}"
            
#     if args.tube is not None:
#         name = f"tube_{args.tube}_seed_{args.seed}_model_{args.alternative_model}"
#     elif args.exp is not None:
#         name = f"test_exp_{args.exp}_seed_{args.seed}_model_{args.alternative_model}"
#     else:
#         name = f"fold_{args.kth_fold}_seed_{args.seed}"
    
#     if args.tag is not None:
#         name += f"_{args.tag}"
    wandb.login()

#     name = None
    run = wandb.init(project="comparison", entity="smonaco", name=name, tags=[args.tag], reinit=True)
    
    
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
    print(f"          fold: {args.kth_fold}       ")
    print("---------------------------------------\n")
    
    
    if args.dataset != 'nw' and not args.tiling:
        dataset = run.use_artifact(f'rene-policistico/upp/dataset:{args.dataset}', type='dataset')
        data_dir = dataset.download()
    #     data_dir = f"artifacts/dataset:{args.dataset}"

        if not (Path(data_dir) / "images").exists():
            zippath = next(Path(data_dir).iterdir())
            with ZipFile(zippath, 'r') as zip_ref:
                zip_ref.extractall(data_dir)

        hparams["image_path"] = Path(data_dir) / "images"
        hparams["mask_path"] = Path(data_dir) / "masks"
    elif args.tiling:
        data_dir = "artifacts/tiled-dataset:v0"
        hparams["image_path"] = Path(data_dir) / "images"
        hparams["mask_path"] = Path(data_dir) / "masks"
    else:
        hparams["image_path"] = Path(hparams["image_path"])
        hparams["mask_path"] = Path(hparams["mask_path"])

    
#     if args.alternative_model is not None:
#         hparams["checkpoint_callback"]["dirpath"] = Path(hparams["checkpoint_callback"]["dirpath"]) / f"{args.alternative_model}"
#     if args.exp is not None:
#         hparams["checkpoint_callback"]["dirpath"] = Path(hparams["checkpoint_callback"]["dirpath"]) / f"exp_{args.exp}"
#     if args.tube is not None:
#         hparams["checkpoint_callback"]["dirpath"] = Path(hparams["checkpoint_callback"]["dirpath"]) / f"exp_{args.tube}"
    
    hparams["checkpoint_callback"]["dirpath"] = Path(hparams["checkpoint_callback"]["dirpath"]) / wandb.run.name
    hparams["checkpoint_callback"]["dirpath"].mkdir(exist_ok=True, parents=True)

    checkpoint_callback = object_from_dict(hparams["checkpoint_callback"])

    earlystopping_callback = object_from_dict(hparams["earlystopping_callback"])
    
    splits = split_dataset(hparams, k=args.kth_fold, test_exp=args.exp, leave_one_out=args.tube, strat_nogroups=args.stratify_fold, single_exp=args.single_exp)
    
    with (hparams["checkpoint_callback"]["dirpath"] / "split_samples.pickle").open('wb') as file:
        pickle.dump(splits, file)
    print(f'\nSaving in {hparams["checkpoint_callback"]["dirpath"]}\n')
    
    model = SegmentCyst(hparams, splits, discard_res=args.discard_results, alternative_model=args.alternative_model)

    logger = WandbLogger(name=name)
    logger.log_hyperparams(hparams)
    logger.watch(model, log='all', log_freq=1)

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
    #     accumulate_grad_batches=4,
        max_epochs=100,
    #     distributed_backend="ddp",  # DistributedDataParallel
        progress_bar_refresh_rate=1,
        benchmark=True,
        callbacks=[checkpoint_callback,
                   earlystopping_callback
                  ],
        precision=16 if torch.cuda.is_available() else 32,
        gradient_clip_val=5.0,
        num_sanity_val_steps=5,
        sync_batchnorm=True,
        logger=logger,
    )

    trainer.fit(model)
    
#     if args.eval_network:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.model = model.model.to(device)
    eval_model(args=ed(inpath=hparams["checkpoint_callback"]["dirpath"],
                       subset='test',
                       exp=None,
                       thresh=.5,
                       outpath='result'
                      ),
               model=model,
               save_fps=True
              )
        
    
    real_mask_PATH = hparams["mask_path"]
    real_img_PATH = hparams["image_path"]
    write_results(hparams["checkpoint_callback"]["dirpath"]/'result'/'test',
                  maskp=hparams["mask_path"], imgp=hparams["image_path"])
        
    wandb.finish()
    return


if __name__ == '__main__':
    args = get_args()
    main(args)
