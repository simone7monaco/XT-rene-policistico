import yaml
from easydict import EasyDict as ed

from pathlib import Path
from pytorch_toolbelt.losses import JaccardLoss, BinaryFocalLoss, BinarySoftF1Loss
import torch
from torch import nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import object_from_dict, print_usage, find_average, state_dict_from_disk, get_samples
import torchvision.utils as vutils
# from Models.networks.network import Comprehensive_Atten_Unet


from dataloaders import SegmentationDataset
from metrics import binary_mean_iou

import segmentation_models_pytorch as smp
# from ColonSegNet import CompNet
# from Pranet_lib.PraNet_Res2Net import PraNet
from HarDNetMSEG.lib.HarDMSEG import HarDMSEG
from UACANet.lib.PraNet import PraNet
from UACANet.lib.UACANet import UACANet
from UACANet.utils import *
from UACANet.lib import *
from UACANet.lib.losses import *

from albumentations.core.serialization import from_dict
from typing import Dict
import torchvision.transforms.functional as TF
from PIL import Image
import pytorch_lightning as pl
import wandb
import numpy as np
import os


def get_model(alternative_model, hparams):
    if alternative_model == 'colonsegnet':
        model = CompNet()
    elif alternative_model == 'pranet':
        conf = ed(yaml.load(open(Path.home() / "my_rene-policistico/UACANet/configs/PraNet.yaml"), yaml.FullLoader))
        model = eval(conf.Model.name)(conf.Model)
        hparams["model"] = {
            'type': model.__module__,
            'opt': conf.Model
        }
    elif alternative_model == 'hardnet':
        model = HarDMSEG()

    elif alternative_model == 'pspnet':
        model = smp.PSPNet(encoder_name='resnet50', encoder_weights='imagenet')

    elif alternative_model == 'uacanet':
        conf = ed(yaml.load(open(Path.home() / "my_rene-policistico/UACANet/configs/UACANet-L.yaml"), yaml.FullLoader))
        model = eval(conf.Model.name)(conf.Model)
        hparams["model"] = {
            'type': model.__module__,
            'opt': conf.Model
        }
    elif alternative_model == 'unet':
        hparams["model"]["type"] = "segmentation_models_pytorch.Unet"
        model = object_from_dict(hparams["model"])

    else:
        model = object_from_dict(hparams["model"])

    hparams["model"]["type"] = str(model.__class__).split("'")[1]
    return model
    

class SegmentCyst(pl.LightningModule):
    def __init__(self, hparams, splits=[None, None], discard_res=False, alternative_model=None):
        super().__init__()
        
        self.model_name = alternative_model
        self.model = get_model(alternative_model, hparams)
        
        self.discard_res = discard_res
        self.hparams = hparams
        self.train_images = Path(self.hparams["checkpoint_callback"]["dirpath"]) / "images/train_predictions"
        self.train_images.mkdir(exist_ok=True, parents=True)
        self.val_images =  Path(self.hparams["checkpoint_callback"]["dirpath"]) / "images/valid_predictions"
        self.val_images.mkdir(exist_ok=True, parents=True)
        
            
            
        if "resume_from_checkpoint" in self.hparams:
            corrections: Dict[str, str] = {"model.": ""}

            state_dict = state_dict_from_disk(
                file_path=self.hparams["resume_from_checkpoint"],
                rename_in_layers=corrections,
            )
            self.model.load_state_dict(state_dict)

        self.bce = False
        if self.bce:
            self.loss = nn.BCEWithLogitsLoss()
        self.losses = [
            ("jaccard", 0.1, JaccardLoss(mode="binary", from_logits=True)),
            ("f1", 0.9, BinarySoftF1Loss()),
        ]
        self.train_samples=splits['train']
        self.val_samples=splits['valid']
        self.max_val_iou = 0

    def forward(self, batch: torch.Tensor, masks: torch.Tensor=None) -> torch.Tensor:
        if masks is not None:
            return self.model(batch, masks)
        else:
            return self.model(batch)

    def setup(self, stage=0):
        if self.train_samples is None:
            samples = get_samples(self.hparams["image_path"], self.hparams["mask_path"])
            num_train = int((1 - self.hparams["val_split"]) * len(samples))
            self.train_samples = samples[:num_train]
            self.val_samples = samples[num_train:]

        print("Len train samples = ", len(self.train_samples))
        print("Len val samples = ", len(self.val_samples))

    def train_dataloader(self):
        train_aug = from_dict(self.hparams["train_aug"])

        if "epoch_length" not in self.hparams["train_parameters"]:
            epoch_length = None
        else:
            epoch_length = self.hparams["train_parameters"]["epoch_length"]

        result = DataLoader(
            SegmentationDataset(self.train_samples, train_aug, epoch_length),
            batch_size=self.hparams["train_parameters"]["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        
        print("Train dataloader = ", len(result))
        return result

    def val_dataloader(self):
        val_aug = from_dict(self.hparams["val_aug"])

        result = DataLoader(
            SegmentationDataset(self.val_samples, val_aug, length=None),
            batch_size=self.hparams["val_parameters"]["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        print("Val dataloader = ", len(result))

#         self.logger.experiment.log({"val_input_image": [wandb.Image(result["mask"].cpu(), caption="val_input_image")]})

        return result
    
    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.hparams["optimizer"],
            params=[x for x in self.model.parameters() if x.requires_grad],
        )
        self.optimizers = [optimizer]
        
        if self.hparams["scheduler"] is not None:
            scheduler = object_from_dict(self.hparams["scheduler"], optimizer=optimizer)

            if type(scheduler) == ReduceLROnPlateau:
                    return {
                       'optimizer': optimizer,
                       'lr_scheduler': scheduler,
                       'monitor': 'val_iou'
                   }
            return self.optimizers, [scheduler]
        return self.optimizers
    
    
    def training_step(self, batch, batch_idx):
        features = batch["features"]
        masks = batch["masks"]
                
        if self.model_name in ['uacanet', 'pranet']:
            logits = self.forward(features, masks)
            total_loss = logits['loss']
            logits = logits['pred']
        else:
            logits = self.forward(features)
            total_loss = 0
            for loss_name, weight, loss in self.losses:
                ls_mask = loss(logits, masks)
                total_loss += weight * ls_mask
                self.log(f"train_mask_{loss_name}", ls_mask)
        
        logits_ = (logits > 0.5).cpu().detach().numpy().astype("float")

        self.log("train_iou", binary_mean_iou(logits, masks))
#         batch_size = self.hparams["train_parameters"]["batch_size"]
        
        if not self.discard_res:
            if self.trainer.current_epoch % 5 == 0:
                class_labels = {0: "background", 1: "cyst"}
                for i in range(features.shape[0]):
                    mask_img = wandb.Image(
                        features[i, :, :, :],
                        masks={
                            "predictions": {
                                "mask_data": logits_[i, 0, :, :],
                                "class_labels": class_labels,
                            },
                            "groud_truth": {
                                "mask_data": masks.cpu().detach().numpy()[i, 0, :, :],
                                "class_labels": class_labels,
                            },
                        },
                    )
                    fname = batch["image_id"][i]

                    self.logger.experiment.log({"generated_images": [mask_img]}, commit=False)
            # self.log("images_train", mask_img)

        # print(logits.shape, features.shape)

        self.log("train_loss", total_loss)

        self.log("lr", self._get_current_lr())
        return {"loss": total_loss}

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore
        
        if torch.cuda.is_available(): return torch.Tensor([lr])[0].cuda()
        return torch.Tensor([lr])[0]

    def validation_step(self, batch, batch_id):
        features = batch["features"]
        masks = batch["masks"]
        result = {}
        
        if self.model_name in ['uacanet', 'pranet']:
            logits = self.forward(features, masks)
            result["valid_loss"] = logits['loss']
            logits = logits['pred']
        else:
            logits = self.forward(features)
            total_loss = 0
            for loss_name, weight, loss in self.losses:
                ls_mask = loss(logits, masks)
                total_loss += weight * ls_mask
            result["valid_loss"] = total_loss
            
        logits_ = (logits > 0.5).cpu().detach().numpy().astype("float")

        result["val_iou"] = binary_mean_iou(logits, masks)
        
        if not self.discard_res:
            if self.trainer.current_epoch % 5 == 0:
                class_labels = {0: "background", 1: "cyst"}
                mask_img = wandb.Image(
                    features[0, :, :, :],
                    masks={
                        "predictions": {
                            "mask_data": logits_[0, 0, :, :],
                            "class_labels": class_labels,
                        },
                        "groud_truth": {
                            "mask_data": masks.cpu().detach().numpy()[0, 0, :, :],
                            "class_labels": class_labels,
                        },
                    },
                )
                self.logger.experiment.log({"valid_images": [mask_img]}, commit=False)
            
        self.log("valid_loss", result["valid_loss"])
        self.log("val_iou", result["val_iou"])
        return result

    def validation_epoch_end(self, outputs):
        self.log("epoch", self.trainer.current_epoch)

        avg_val_iou = find_average(outputs, "val_iou")
        
        self.log("val_iou", avg_val_iou)
        
        self.max_val_iou = max(self.max_val_iou, avg_val_iou)
        return
