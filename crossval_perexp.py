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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from simplify_names import get_packs
from eval import eval_model
from UACANet.run.Train import train

from write_results import *
from albumentations.core.serialization import from_dict

import wandb

def get_args():
    parser = argparse.ArgumentParser(description='CV with selected experiment as test set and train/val (+test) stratified from the others')
    parser.add_argument("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    parser.add_argument("-d", "--dataset", type=Path, help="Select dataset version from wandb Artifact (v1, v2...), set to 'nw' (no WB) to use paths from the config file. Default is 'latest'.", default='latest')
    parser.add_argument("-e", "--exp_tested", default=None, type=int, help="Experiment to put in test set")
    parser.add_argument("-t", "--test_tube", default=None, type=int, help="If present, select a single tube as test set (integer index between 0 and 31).")
    parser.add_argument("-f", "--focus_size", default=None, help="Select 'small_cysts' ('s') or 'big_cysts' ('b') labels (only avaiable from 'v6' dataset).")
    
    parser.add_argument("--alternative_model", type=str, default=None, help="Select model different from U++.")

    parser.add_argument("-k", "--kth_fold", type=int, default=0, help="Number of the fold to consider between 0 and 4.")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Change the seed to the desired one.")
    parser.add_argument("--stratify_fold", nargs='?', type=str2bool, default=False, const=True, help="Split dataset with StratifiedKFold instead of GroupKFold.")
    
    parser.add_argument("--eval_network", nargs='?', type=str2bool, default=False, const=True, help="Performs evaluation on the defined test set.")
    parser.add_argument("--tiling", nargs='?', type=str2bool, default=False, const=True, help="If applied, uses the latest tiled-dataset available in WB.")
    parser.add_argument('--discard_results', nargs='?', type=str2bool, default=False, const=True, help = "Prevent Wandb to save validation result for each step.")
    
    
    return parser.parse_args()


def date_to_exp(date):
    date_exps = {'0919': 1, '1019': 2, '072020':3, '092020':3, '122020':4}
    date = ''.join((date).split('.')[1:])
    return date_exps[date]


def split_dataset(hparams, k=0, test_exp=None, leave_one_out=None, strat_nogroups=False):
    samples = get_samples(hparams["image_path"], hparams["mask_path"])
    
    names = [file[0].stem for file in samples]

#     date, treatment, tube, zstack, side =
    unpack = [get_packs(name) for name in names]
    df = pd.DataFrame([])
    df["filename"] = names
    df["treatment"] = [u[1] for u in unpack]
    df["exp"] = [date_to_exp(u[0]) for u in unpack]
    df["tube"] = [u[2] for u in unpack]
    df["te"] = df.treatment + '_' + df.exp.astype(str)
    df.te = df.te.astype('category')
    
    if test_exp is not None or leave_one_out is not None:
        if leave_one_out is not None:
            tubes = df[['exp','tube']].astype(int).sort_values(by=['exp', 'tube']).drop_duplicates().reset_index().values[leave_one_out]
            test_idx = df[(df.exp == tubes[1])&(df.tube == str(tubes[2]))].index
            
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
        gkf = GroupKFold(n_splits=5)
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
    os.environ["WANDB_RUN_GROUP"] = "loto_cv"
    torch.cuda.empty_cache()
    
    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)
    
    if args.alternative_model:
        with open("configs/searched_params.yaml") as f:
            bs_results = yaml.load(f, Loader=yaml.SafeLoader)
            if args.alternative_model in bs_results.keys():
                hparams["train_parameters"]["batch_size"] = bs_results[args.alternative_model]["train_parameters"]["batch_size"]
                hparams["optimizer"]["lr"] = bs_results[args.alternative_model]["optimizer"]["lr"]
    
            

    if torch.cuda.device_count() > 1:
        hparams["num_workers"] = 4
        hparams["train_parameters"]["batch_size"] = 16
        print('HI LEGION!')

#     name = f"crossval_exp{exp}"
    if args.test_tube is not None:
        name = f"test_tube_{args.test_tube}_seed_{args.seed}"
    else:
        name = f"fold_{args.kth_fold}_seed_{args.seed}"
    wandb.login()

    name = None
    run = wandb.init(project="upp", entity="smonaco", name=name)
    
    
    print("---------------------------------------")
    print("        Running Crossvalidation        ")
    if args.tiling:
        print("         with tiled dataset        ")
    if args.alternative_model is not None:
        print(f"         model: {args.alternative_model}  ")
    if args.exp_tested is not None:
        print(f"           exp: {args.exp_tested}  ")
    if args.test_tube is not None:
        print(f"     test_tube: {args.test_tube}  ")
    print(f"          seed: {args.seed}           ")
    print(f"          fold: {args.kth_fold}       ")
    print("---------------------------------------\n")
    
        
    if str(args.dataset) != 'nw' and not args.tiling:
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

    
    if args.alternative_model is not None:
        hparams["checkpoint_callback"]["filepath"] = Path(hparams["checkpoint_callback"]["filepath"]) / f"{args.alternative_model}"
    if args.exp_tested is not None:
        hparams["checkpoint_callback"]["filepath"] = Path(hparams["checkpoint_callback"]["filepath"]) / f"exp_{args.exp_tested}"
    if args.test_tube is not None:
        hparams["checkpoint_callback"]["filepath"] = Path(hparams["checkpoint_callback"]["filepath"]) / f"exp_{args.test_tube}"
    
    hparams["checkpoint_callback"]["filepath"] = Path(hparams["checkpoint_callback"]["filepath"]) / wandb.run.name
    hparams["checkpoint_callback"]["filepath"].mkdir(exist_ok=True, parents=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams["checkpoint_callback"]["filepath"],
        monitor="val_iou",
        verbose=True,
        mode="max",
        save_top_k=1,
    )

    earlystopping_callback = EarlyStopping(
        monitor='val_iou',
        min_delta=0.001,
        patience=10,
        verbose=True,
        mode='max',
    )
    
    splits = split_dataset(hparams, k=args.kth_fold, test_exp=args.exp_tested, leave_one_out=args.test_tube, strat_nogroups=args.stratify_fold)
    with (hparams["checkpoint_callback"]["filepath"] / "split_samples.pickle").open('wb') as file:
        pickle.dump(splits, file)
    print(f'\nSaving in {hparams["checkpoint_callback"]["filepath"]}\n')
    
    model = SegmentCyst(hparams, splits, discard_res=args.discard_results, alternative_model=args.alternative_model)

    logger = WandbLogger(name=name)
    logger.log_hyperparams(hparams)
    logger.watch(model, log='all', log_freq=1)

#     if args.alternative_model == 'uacanet':
#         opt = ed(yaml.load(open('UACANet/configs/UACANet-L.yaml'), yaml.FullLoader))
#         opt.Train.train_save = hparams["checkpoint_callback"]["filepath"]
#         train_aug = from_dict(hparams["train_aug"])
#         train(opt, splits['train'], splits['valid'], train_aug)
#     elif args.alternative_model == 'pranet':
#         opt = ed(yaml.load(open('UACANet/configs/PraNet.yaml'), yaml.FullLoader))
#         opt.Train.train_save = hparams["checkpoint_callback"]["filepath"]
#         train_aug = from_dict(hparams["train_aug"])
#         train(opt, splits['train'], splits['valid'], train_aug)
    if True:
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
    
    if args.eval_network:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        model.model = model.model.to(device)
        eval_model(args=ed(inpath=hparams["checkpoint_callback"]["filepath"],
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
        write_results(hparams["checkpoint_callback"]["filepath"]/'result'/'test')
    return


if __name__ == '__main__':
    args = get_args()
    main(args)
