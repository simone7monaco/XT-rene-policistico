import argparse
import os
from pathlib import Path
from zipfile import ZipFile
import yaml
from easydict import EasyDict as ed
import ast

from utils import object_from_dict, str2bool, print_usage
from experiment import SegmentCyst
from sweep_params import get_sweep
from crossval_perexp import split_dataset
from Models.attention_unet import AttentionUnet

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import wandb
from ray import tune

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--debug', type=str2bool, help="Print memory usage in different steps", nargs='?', const=True, default=False)
    parser.add_argument("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    parser.add_argument("-d", "--dataset", type=Path, help="Select dataset version from wandb Artifact (v1, v2...), set to 'nw' (no wandb) to use paths from the config file. Default is 'latest'.", default='latest')
    
    parser.add_argument("-b", "--batch_size", type=int, default=None, help="Batch size for sweep.")
    parser.add_argument("--lr", type=float, default=None, help="Lr for sweep.")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer for sweep.")
    parser.add_argument("--scheduler", type=str, default=None, help="Scheduler for sweep.")
#     parser.add_argument("--transf_imsize", type=int, default=None, help="Rescale images for sweep.")
    
    parser.add_argument("--alternative_model", type=str, default=None, help="Select model different from U++.")
    parser.add_argument('-bs', '--b_search', type=str2bool, help="Internal bs configuration", nargs='?', const=True, default=False)
    
    parser.add_argument('-a', '--active_attention_layers', help="List of Active Attention layer (between 1~4). Write [1,3] to set AAL on blocks 1 and 3.", default=None)
    parser.add_argument('-a1', '--active_attention_layer1', type=str2bool, help="Activate Attention layer 1", nargs='?', const=True, default=False)
    parser.add_argument('-a2', '--active_attention_layer2', type=str2bool, help="Activate Attention layer 2", nargs='?', const=True, default=False)
    parser.add_argument('-a3', '--active_attention_layer3', type=str2bool, help="Activate Attention layer 3", nargs='?', const=True, default=False)
    parser.add_argument('-a4', '--active_attention_layer4', type=str2bool, help="Activate Attention layer 4", nargs='?', const=True, default=False)
#     --batch_size=8 --lr=0.005 --model=unet --optimizer=adamp --scheduler=cosannealing
    
    return parser.parse_args()


def train(config):
    if type(config) == dict:
        config = ed(config)
    with open(config.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    if torch.cuda.device_count() > 1:
        hparams["num_workers"] = 4
        hparams["train_parameters"]["batch_size"] = 16
        print('HI LEGION!')

    hparams = get_sweep(hparams, config)

    print("---------------------------------------")
    print("        Running Training        ")

    for v in vars(config):
        if getattr(config, v):
            print(f"           {v}: {getattr(config, v)}  ")
    print("---------------------------------------\n")
    
    if args.debug: print_usage()
    wandb.login()
    run = wandb.init(project="ca-net", entity="rene-policistico", config=hparams, settings=wandb.Settings(start_method='fork'))

    if str(config.dataset) != 'nw':
        dataset = run.use_artifact(f'rene-policistico/upp/dataset:{config.dataset}', type='dataset')
        data_dir = dataset.download()
    #     data_dir = f"artifacts/dataset:{config.dataset}"

        if not (Path(data_dir) / "images").exists():
            zippath = next(Path(data_dir).iterdir())
            with ZipFile(zippath, 'r') as zip_ref:
                zip_ref.extractall(data_dir)

        hparams["image_path"] = Path(data_dir) / "images"
        hparams["mask_path"] = Path(data_dir) / "masks"
    else:
        hparams["image_path"] = Path(hparams["image_path"])
        hparams["mask_path"] = Path(hparams["mask_path"])

    splits = split_dataset(hparams)
    model = SegmentCyst(hparams, splits, alternative_model=config.alternative_model, debug=args.debug)
    
    if args.debug: print_usage()

    if "activate_attention_layers" in dir(model.model):
        if config.active_attention_layers:
            active_attention_layers = ast.literal_eval(config.active_attention_layers)
        else:
            active_attention_layers = [
                i+1 for i, act in enumerate([
                    config.active_attention_layer1,
                    config.active_attention_layer2,
                    config.active_attention_layer3,
                    config.active_attention_layer4
                ]) if act]

        print(f"> Activating attention layers {active_attention_layers}")
        model.model.activate_attention_layers(active_attention_layers)

    # hparams["checkpoint_callback"]["filepath"] = Path(hparams["checkpoint_callback"]["filepath"]) / wandb.run.name
    hparams["checkpoint_callback"]["filepath"] = Path(hparams["checkpoint_callback"]["filepath"]) / "Search" / wandb.run.name
    hparams["checkpoint_callback"]["filepath"].mkdir(exist_ok=True, parents=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams["checkpoint_callback"]["filepath"],
        monitor="val_iou",
        verbose=True,
        mode="max",
        save_top_k=1,
    )
    print(f'\nSaving in {hparams["checkpoint_callback"]["filepath"]}\n')

    earlystopping_callback = EarlyStopping(
            monitor='val_iou',
            min_delta=0.001,
            patience=10,
            verbose=True,
            mode='max',
        )

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
        num_sanity_val_steps=5,
        sync_batchnorm=True,
        logger=logger,
    #     resume_from_checkpoint="cyst_checkpoints/prova1/epoch=20-step=8546.ckpt"
    )
    
    if args.debug: print_usage()
        
    trainer.fit(model)

    if config.b_search:
        tune.report(score=model.max_val_iou)
#             return model.max_val_iou
    else:
        model.logger.experiment.log({"max_val_iou": model.max_val_iou})
# wandb.save(checkpoint_callback.best_model_path)


if __name__ == "__main__":
    args = get_args()
    train(args)
