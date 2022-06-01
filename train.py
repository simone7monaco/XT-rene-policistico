from datetime import datetime
import argparse
import os
from pathlib import Path
from zipfile import ZipFile
import yaml
from easydict import EasyDict as ed
import ast
import pickle

from utils import object_from_dict, str2bool, print_usage, split_dataset
from experiment import SegmentCyst
from sweep_params import get_sweep
from Models.attention_unet import AttentionUnet

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import wandb
# from ray import tune

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-c", "--config_path", type=Path, help="Path to the config.", default="configs/baseline.yaml")
    parser.add_argument("-d", "--dataset", type=str, help="Select dataset version from wandb Artifact (v1, v2...), set to 'nw' (no wandb) to use paths from the config file. Default is 'latest'.", default='v7')
    parser.add_argument('--discard_results', nargs='?', type=str2bool, default=False, const=True, help = "Prevent Wandb to save validation result for each step.")
    
    parser.add_argument("-b", "--batch_size", type=int, default=None, help="Batch size for sweep.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate for sweep.")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer for sweep.")
    parser.add_argument("--scheduler", type=str, default=None, help="Scheduler for sweep.")
#     parser.add_argument("--transf_imsize", type=int, default=None, help="Rescale images for sweep.")
    
    parser.add_argument('-m', '--alternative_model', type=str, default=None, help="Select model different from U++.")
    parser.add_argument('--tag', type=str, default=None, help="Add a tag to Wandb.")
#     parser.add_argument('-bs', '--b_search', type=str2bool, help="Internal bs configuration", nargs='?', const=True, default=False)    

    parser.add_argument('--debug', type=str2bool, default=False, help = "If enabled skip checks and logging")

    return parser.parse_args()


def init_training(config):
    if type(config) == dict:
        config = ed(config)
    with open(config.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    hparams = get_sweep(hparams, config)

    print("-------------------------------------------------")
    print("        Running Training        ")

    for v in vars(config):
        if getattr(config, v):
            print(f"           {v}: {getattr(config, v)}  ")
    print("-------------------------------------------------\n")
    
    splits = split_dataset(hparams)
    
    return splits, hparams


def train(config, splits, hparams, name=None):
    if not config.debug:
        run = wandb.init(project="3d",
                        entity="rene-policistico", config=hparams,
                        settings=wandb.Settings(start_method='fork'),
                        tags=[config.tag] if config.tag else None, reinit=True,
                        name=name
                        )
    model = SegmentCyst(hparams, config.debug, splits,
                        discard_res=config.discard_results,
                        alternative_model=config.alternative_model,
                       )
    
    if config.alternative_model:
        with open("configs/searched_params.yaml") as f:
            bs_results = yaml.load(f, Loader=yaml.SafeLoader)
            if config.alternative_model in bs_results.keys():
                hparams["max_supported_bs"] = bs_results[config.alternative_model]["max_supported_bs"]
                hparams["optimizer"]["lr"] = bs_results[config.alternative_model]["lr"]
        
    if "SLURM_JOB_ID" in os.environ:
        print("Running in Slurm")
        hparams["job_id"] = os.environ["SLURM_JOB_ID"]
        hparams["num_workers"] = 4
        if hparams.get("max_supported_bs"): hparams["max_supported_bs"] *= 4
    
    if hparams["train_parameters"]["batch_size"] > hparams["max_supported_bs"]:
        accumulated_ratio = hparams["train_parameters"]["batch_size"] // hparams["max_supported_bs"]
        if hasattr(config, "acc_grad"):
            config.acc_grad *= accumulated_ratio
        else:
            setattr(config, "acc_grad", accumulated_ratio)
        hparams["train_parameters"]["batch_size"] = hparams["max_supported_bs"]    
        
    if config.dataset != 'nw':
        if not config.debug:
            # upp (is 2d) or 3d
            dataset = run.use_artifact(f'rene-policistico/3d/dataset:{config.dataset}', type='dataset')
            data_dir = dataset.download()
        else:
            data_dir = f"artifacts/dataset:{config.dataset}"
        if not (Path(data_dir) / "images").exists():
            print("Not existing")
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

    if not config.debug:
        hparams["checkpoint_callback"]["dirpath"] = Path(hparams["checkpoint_callback"]["dirpath"]) / wandb.run.name
    else:
        hparams["checkpoint_callback"]["dirpath"] = Path(hparams["checkpoint_callback"]["dirpath"]) / datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    hparams["checkpoint_callback"]["dirpath"].mkdir(exist_ok=True, parents=True)

    
    checkpoint_callback = object_from_dict(hparams["checkpoint_callback"])
    if any(hparams["checkpoint_callback"]["dirpath"].iterdir()): return

    earlystopping_callback = object_from_dict(hparams["earlystopping_callback"])
    
    with (hparams["checkpoint_callback"]["dirpath"] / "split_samples.pickle").open('wb') as file:
        pickle.dump(splits, file)
    print(f'\nSaving in {hparams["checkpoint_callback"]["dirpath"]}\n')
    if not config.debug:
        logger = WandbLogger()
        logger.log_hyperparams(hparams)
        logger.watch(model, log='all', log_freq=1)

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        accumulate_grad_batches=config.acc_grad if hasattr(config, 'acc_grad') else 1,
        max_epochs=100 if not config.debug else 1,
    #     distributed_backend="ddp",  # DistributedDataParallel
        # progress_bar_refresh_rate=1,
        benchmark=True,
        callbacks=[checkpoint_callback,
                   earlystopping_callback,
                  ],
        precision=16 if torch.cuda.is_available() else 32,
        gradient_clip_val=5.0,
        num_sanity_val_steps=3 if not config.debug else 0,
        sync_batchnorm=True,
        logger=logger if not config.debug else False,
    #     resume_from_checkpoint="cyst_checkpoints/prova1/epoch=20-step=8546.ckpt"
    )
    trainer.fit(model)
    if not config.debug:
        model.logger.experiment.log({"max_val_iou": model.max_val_iou})
    return model


if __name__ == "__main__":
    args = get_args()
    splits, hparams = init_training(args)
    
    train(args, splits, hparams)
