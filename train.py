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
    
    parser.add_argument("-c", "--config_path", type=Path, help="Path to the config.", required=True)
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
    
#     parser.add_argument('-a', '--active_attention_layers', help="List of Active Attention layer (between 1~4). Write [1,3] to set AAL on blocks 1 and 3.", default=None)
#     parser.add_argument('-a1', '--active_attention_layer1', type=str2bool, help="Activate Attention layer 1", nargs='?', const=True, default=False)
#     parser.add_argument('-a2', '--active_attention_layer2', type=str2bool, help="Activate Attention layer 2", nargs='?', const=True, default=False)
#     parser.add_argument('-a3', '--active_attention_layer3', type=str2bool, help="Activate Attention layer 3", nargs='?', const=True, default=False)
#     parser.add_argument('-a4', '--active_attention_layer4', type=str2bool, help="Activate Attention layer 4", nargs='?', const=True, default=False)
#     --batch_size=8 --lr=0.005 --model=unet --optimizer=adamp --scheduler=cosannealing
    
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
#     wandb.login()
    run = wandb.init(project="at_nature",
                     entity="rene-policistico", config=hparams,
                     settings=wandb.Settings(start_method='fork'),
                     tags=[config.tag] if config.tag else None, reinit=True
                    )
    model = SegmentCyst(hparams, splits,
                        discard_res=config.discard_results,
                        alternative_model=config.alternative_model
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
        dataset = run.use_artifact(f'rene-policistico/upp/dataset:{config.dataset}', type='dataset')
        data_dir = dataset.download()
    #     data_dir = f"artifacts/dataset:{config.dataset}"

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
    

#     if "activate_attention_layers" in dir(model.model):
#         if config.active_attention_layers:
#             active_attention_layers = ast.literal_eval(config.active_attention_layers)
#         else:
#             active_attention_layers = [
#                 i+1 for i, act in enumerate([
#                     config.active_attention_layer1,
#                     config.active_attention_layer2,
#                     config.active_attention_layer3,
#                     config.active_attention_layer4
#                 ]) if act]

#         print(f"> Activating attention layers {active_attention_layers}")
#         model.model.activate_attention_layers(active_attention_layers)

    hparams["checkpoint_callback"]["dirpath"] = Path(hparams["checkpoint_callback"]["dirpath"]) / wandb.run.name
    hparams["checkpoint_callback"]["dirpath"].mkdir(exist_ok=True, parents=True)

    
    checkpoint_callback = object_from_dict(hparams["checkpoint_callback"])
    if any(hparams["checkpoint_callback"]["dirpath"].iterdir()): return

    earlystopping_callback = object_from_dict(hparams["earlystopping_callback"])
    
    with (hparams["checkpoint_callback"]["dirpath"] / "split_samples.pickle").open('wb') as file:
        pickle.dump(splits, file)
    print(f'\nSaving in {hparams["checkpoint_callback"]["dirpath"]}\n')

    logger = WandbLogger()
    logger.log_hyperparams(hparams)
    logger.watch(model, log='all', log_freq=1)

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        accumulate_grad_batches=config.acc_grad if hasattr(config, 'acc_grad') else 1,
        max_epochs=100,
    #     distributed_backend="ddp",  # DistributedDataParallel
        progress_bar_refresh_rate=1,
        benchmark=True,
        callbacks=[checkpoint_callback,
                   earlystopping_callback,
                  ],
        precision=16 if torch.cuda.is_available() else 32,
        gradient_clip_val=5.0,
        num_sanity_val_steps=3,
        sync_batchnorm=True,
        logger=logger,
    #     resume_from_checkpoint="cyst_checkpoints/prova1/epoch=20-step=8546.ckpt"
    )
            
    trainer.fit(model)
    model.logger.experiment.log({"max_val_iou": model.max_val_iou})



if __name__ == "__main__":
    args = get_args()
    splits, hparams = init_training(args)
    
    train(args, splits, hparams)
