import argparse
import os
from pathlib import Path
import yaml
import ast
from utils import object_from_dict
from experiment import SegmentCyst
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sweep_params import get_sweep
from crossval_perexp import split_dataset
from Models.attention_unet import AttentionUnet

import wandb

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for sweep.")
    parser.add_argument("--lr", type=float, default=None, help="Lr for sweep.")
    parser.add_argument("--model", type=str, default=None, help="Model type for sweep.")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer for sweep.")
    parser.add_argument("--scheduler", type=str, default=None, help="Scheduler for sweep.")
    
#     parser.add_argument("--transf_imsize", type=int, default=None, help="Rescale images for sweep.")
    parser.add_argument("--transf_onlycysts", type=int, default=None, help="RandomCrop if false, CropNonEmptyMaskIfExists if true for sweep.")
    parser.add_argument("--transf_crop", type=int, default=None, help="Portion of the image of size 1024/n")
    parser.add_argument("--transf_enhancement", type=int, default=None, help="Add enhancement on images for sweep.")
    
    parser.add_argument('-a1', '--active_attention_layer1', type=bool, help="Activate Attention layer 1", default=False)
    parser.add_argument('-a2', '--active_attention_layer2', type=bool, help="Activate Attention layer 2", default=False)
    parser.add_argument('-a3', '--active_attention_layer3', type=bool, help="Activate Attention layer 3", default=False)
    parser.add_argument('-a4', '--active_attention_layer4', type=bool, help="Activate Attention layer 4", default=False)
#     --batch_size=8 --lr=0.005 --model=unet --optimizer=adamp --scheduler=cosannealing
    
    return parser.parse_args()


args = get_args()

with open(args.config_path) as f:
    hparams = yaml.load(f, Loader=yaml.SafeLoader)

hparams = get_sweep(hparams, args)

wandb.login()
run = wandb.init(project="ca-net", entity="rene-policistico", config=hparams, settings=wandb.Settings(start_method='fork'))

dataset = run.use_artifact('rene-policistico/upp/dataset:latest', type='dataset')
data_dir = dataset.download()


hparams["image_path"] = Path(data_dir) / "images"
hparams["mask_path"] = Path(data_dir) / "masks"

splits = split_dataset(hparams)
model = SegmentCyst(hparams, splits[:-1])

if "activate_attention_layers" in model.model.__dict__:
    active_attention_layers = [
            i+1 for i, act in enumerate([
                args.active_attention_layer1, 
                args.active_attention_layer2, 
                args.active_attention_layer3, 
                args.active_attention_layer4
        ]) if act]
    print(f"> Activating attention layers {active_attention_layers}")
    model.model.activate_attention_layers(active_attention_layers)

hparams["checkpoint_callback"]["filepath"] = Path(hparams["checkpoint_callback"]["filepath"]) / wandb.run.name
hparams["checkpoint_callback"]["filepath"].mkdir(exist_ok=True, parents=True)

checkpoint_callback = ModelCheckpoint(
    filepath=hparams["checkpoint_callback"]["filepath"],
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

logger = WandbLogger()
# logger.log_hyperparams(hparams)
logger.watch(model, log='all', log_freq=1)

trainer = pl.Trainer(
    gpus=1,
#     accumulate_grad_batches=4,
    max_epochs=100,
#     distributed_backend="ddp",  # DistributedDataParallel
    progress_bar_refresh_rate=1,
    benchmark=True,
    callbacks=[checkpoint_callback,
               earlystopping_callback,
              ],
    precision=16,
    gradient_clip_val=5.0,
    num_sanity_val_steps=5,
    sync_batchnorm=True,
    logger=logger,
#     resume_from_checkpoint="cyst_checkpoints/prova1/epoch=20-step=8546.ckpt"
)

trainer.fit(model)

model.logger.experiment.log({"max_val_iou": model.max_val_iou})
wandb.save(checkpoint_callback.best_model_path)
