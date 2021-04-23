#!/usr/bin/env python
# coding: utf-8

# In[4]:


import argparse
import os
from pathlib import Path
import yaml
from utils import object_from_dict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_toolbelt.losses import JaccardLoss, BinaryFocalLoss
import torch

from torch.utils.data import DataLoader
from utils import object_from_dict
import torchvision.utils as vutils

from dataloaders import SegmentationDataset
from metrics import binary_mean_iou
from utils import get_samples
import segmentation_models_pytorch as smp
from utils import find_average, state_dict_from_disk
from albumentations.core.serialization import from_dict
from typing import Dict
import torchvision.transforms.functional as TF
from PIL import Image
import pytorch_lightning as pl
import numpy as np
import os


# ## Batch size

# ## Learning rate

# In[10]:


class SegmentCyst(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = object_from_dict(self.hparams["model"])
        if "resume_from_checkpoint" in self.hparams:
            corrections: Dict[str, str] = {"model.": ""}

            state_dict = state_dict_from_disk(
                file_path=self.hparams["resume_from_checkpoint"],
                rename_in_layers=corrections,
            )
            self.model.load_state_dict(state_dict)

        self.losses = [
            ("jaccard", 0.1, JaccardLoss(mode="binary", from_logits=True)),
            ("focal", 0.9, BinaryFocalLoss()),
        ]

    def forward(self, batch: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(batch)

    def setup(self, stage=0):
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

        return result

    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.hparams["optimizer"],
            params=[x for x in self.model.parameters() if x.requires_grad],
        )
        scheduler = object_from_dict(self.hparams["scheduler"], optimizer=optimizer)
        self.optimizers = [optimizer]

        return self.optimizers, [scheduler]

    def training_step(self, batch, batch_idx):
        features = batch["features"]
        masks = batch["masks"]

        logits = self.forward(features)

        logits_ = (logits > 0.5).cpu().detach().numpy().astype("float")

        class_labels = {0: "background", 1: "cyst"}

        batch_size = self.hparams["train_parameters"]["batch_size"]

        
        total_loss = 0
        logs = {}
        for loss_name, weight, loss in self.losses:
            ls_mask = loss(logits, masks)
            total_loss += weight * ls_mask
            logs[f"train_mask_{loss_name}"] = ls_mask

        logs["train_loss"] = total_loss

        logs["lr"] = self._get_current_lr()
        # self.log("train_loss", loss)
        self.log("LR", self._get_current_lr())
        return {"loss": total_loss, "log": logs}

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore
        
        if torch.cuda.is_available(): return torch.Tensor([lr])[0].cuda()
        return torch.Tensor([lr])[0]

    def validation_step(self, batch, batch_id):
        features = batch["features"]
        masks = batch["masks"]

        logits = self.forward(features)
        logits_ = (logits > 0.5).cpu().detach().numpy().astype("float")

        result = {}
        for loss_name, _, loss in self.losses:
            result[f"val_mask_{loss_name}"] = loss(logits, masks)

        result["val_iou"] = binary_mean_iou(logits, masks)
        class_labels = {0: "background", 1: "cyst"}

        self.log("valid_iou", result["val_iou"])
        return result

    def validation_epoch_end(self, outputs):
        logs = {"epoch": self.trainer.current_epoch}

        avg_val_iou = find_average(outputs, "val_iou")

        logs["val_iou"] = avg_val_iou

        return {"val_iou": avg_val_iou, "log": logs}


# In[11]:


file = "configs/cyst_prova.yaml"

with open(file) as f:
    hparams = yaml.load(f, Loader=yaml.SafeLoader)
    
image_path = hparams["image_path"]
mask_path = hparams["mask_path"]
model = SegmentCyst(hparams)


# In[12]:


trainer = pl.Trainer(auto_lr_find=True)


# In[ ]:


lr_finder = trainer.tuner.lr_find(model)


# In[ ]:


fig = lr_finder.plot(suggest=True)
fig.savefig('lr.png')

