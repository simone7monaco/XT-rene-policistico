import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import ast
import torch
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, GroupKFold
import yaml
import pickle
from utils import object_from_dict, get_samples
from experiment import SegmentCyst

from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from simplify_names import unpack_name

import wandb

def get_args():
    parser = argparse.ArgumentParser(description='CV with selected experiment as test set and train/val (+test) stratified from the others')
    parser.add_argument("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    parser.add_argument("-t", "--test_list", default=None, help="list of <treat_exp> to put in test set, ")
    parser.add_argument("-f", "--focus_size", default=None, help="Select 'small_cysts' ('s') or 'big_cysts' ('b') labels.")
    
    parser.add_argument("-k", "--kth_fold", type=int, default=0, help="Number of the fold to consider between 0 and 4.")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Change the seed to the desired one.")
    parser.add_argument("--stratify_fold", nargs='?', default=False, const=True, help="Split dataset with StratifiedKFold instead of GroupKFold.")
    
    
    return parser.parse_args()


def date_to_exp(date):
    date_exps = {'0919': 1, '1019': 2, '072020':3, '092020':3, '122020':4}
    date = ''.join((date).split('.')[1:])
    return date_exps[date]


def split_dataset(hparams, k=0, test_list=None, strat_nogroups=False):
    samples = get_samples(hparams["image_path"], hparams["mask_path"])
    
    names = [file[0].stem for file in samples]
    
#     date, treatment, tube, zstack, side =
    unpack = [unpack_name(name) for name in names]
    df = pd.DataFrame([])
    df["filename"] = names
    df["treatment"] = [u[1] for u in unpack]
    df["exp"] = [date_to_exp(u[0]) for u in unpack]
    df["te"] = df.treatment + '_' + df.exp.astype(str)
    df.te = df.te.astype('category')
    
    if test_list is not None:
        test_idx = df[df.te.isin(test_list)].index
        test_samp = [tuple(x) for x in np.array(samples)[test_idx]]
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
    
    return train_samp, val_samp, test_samp

def main(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["WANDB_START_METHOD"] = "fork"
    
    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)
    if args.seed: hparams["seed"] = args.seed

    if torch.cuda.device_count() > 1:
        hparams["num_workers"] = 4
        hparams["train_parameters"]["batch_size"] = 16
        print('HI LEGION!')

#     name = f"crossval_exp{exp}"
    name = f"fold_{args.kth_fold}_seed_{args.seed}"
    wandb.login()

    run = wandb.init(project="upp", entity="smonaco", name=name)

    dataset = run.use_artifact('rene-policistico/upp/dataset:latest', type='dataset')
    data_dir = dataset.download()
    
    print("---------------------------------------")
    print("        Running Crossvalidation        ")
    print(f"          seed: {args.seed}           ")
    print(f"          fold: {args.kth_fold}       ")
    print("---------------------------------------\n")
    
    if 's' in str(args.focus_size):
        msk = 'masks_small'
        print("Dataset labels have only SMALL cysts.\n")
    if 'b' in str(args.focus_size):
        msk = 'masks_big'
        print("Dataset labels have only BIG cysts.\n")
    else:
        msk = 'masks'
        
    hparams["image_path"] = Path(data_dir) / "images"
    hparams["mask_path"] = Path(data_dir) / msk
    
    
    hparams["checkpoint_callback"]["filepath"] = Path(hparams["checkpoint_callback"]["filepath"]) / "small" / wandb.run.name
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
    
    test_list = ast.literal_eval(args.test_list) if args.test_list else None
    splits = split_dataset(hparams, k=args.kth_fold, test_list=test_list, strat_nogroups=args.stratify_fold)
    
    if splits[-1] is not None:
        with (hparams["checkpoint_callback"]["filepath"] / "test_samples.pickle").open('wb') as file:
            pickle.dump(splits[-1], file)
        
    model = SegmentCyst(hparams, splits[:-1])

    logger = WandbLogger(name=name)
    logger.log_hyperparams(hparams)
    logger.watch(model, log='all', log_freq=1)

    trainer = pl.Trainer(
        gpus=1,
    #     accumulate_grad_batches=4,
        max_epochs=100,
    #     distributed_backend="ddp",  # DistributedDataParallel
        progress_bar_refresh_rate=1,
        benchmark=True,
        callbacks=[checkpoint_callback,
                   earlystopping_callback
                  ],
        precision=16,
        gradient_clip_val=5.0,
        num_sanity_val_steps=5,
        sync_batchnorm=True,
        logger=logger,
    )

    trainer.fit(model)
    return


if __name__ == '__main__':
    args = get_args()
    main(args)
