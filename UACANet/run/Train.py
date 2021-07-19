import os
import torch
import argparse
import yaml
import pandas as pd
import tqdm
import sys

from torch.autograd import Variable
from easydict import EasyDict as ed

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from dataloaders import SegmentationDataset
from torch.utils.data import DataLoader

from UACANet.lib import *
from UACANet.utils.dataloader import *
from UACANet.utils.utils import *
from metrics import binary_mean_iou

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/xomcnet.yaml')
    return parser.parse_args()

def train(opt, train_dataset=None, val_dataset=None, train_aug=None):
    model = eval(opt.Model.name)(opt.Model).cuda()
    
    if train_dataset is None:
        image_root = os.path.join(opt.Train.train_path, 'images')
        gt_root = os.path.join(opt.Train.train_path, 'masks')

        train_dataset = PolypDataset(image_root, gt_root, opt.Train)
        
        train_loader = data.DataLoader(dataset=train_dataset,
                                  batch_size=opt.Train.batchsize,
                                  shuffle=opt.Train.shuffle,
                                  num_workers=opt.Train.num_workers,
                                  pin_memory=opt.Train.pin_memory)
        
    else:
        train_loader = DataLoader(
            SegmentationDataset(train_dataset, train_aug, None),
            batch_size=opt.Train.batchsize,
            num_workers=opt.Train.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        ) 
        
        val_loader = DataLoader(
            SegmentationDataset(val_dataset, train_aug, None),
            batch_size=opt.Train.batchsize,
            num_workers=opt.Train.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        ) 

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.Train.lr)
    scheduler = PolyLr(optimizer, gamma=opt.Train.gamma,
                        minimum_lr=opt.Train.min_learning_rate,
                        max_iteration=len(train_loader) * opt.Train.epoch,
                        warmup_iteration=opt.Train.warmup_iteration)
    model.train()

    print('#' * 20, 'Train prep done, start training', '#' * 20)
    
    best_iou = 0
    IOUS = []
    for epoch in tqdm.tqdm(range(1, opt.Train.epoch + 1), desc='Epoch', total=opt.Train.epoch, position=0, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}'):
        pbar = tqdm.tqdm(enumerate(train_loader, start=1), desc='Iter', total=len(train_loader), position=1, leave=False, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')
        
        vbar = tqdm.tqdm(enumerate(train_loader, start=1), desc='Valid', total=len(val_loader), position=1, leave=False, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')
        for i, sample in pbar:
            optimizer.zero_grad()
            images, gts = sample["features"], sample["masks"]
            images = images.cuda()
            gts = gts.cuda()
            out = model(images, gts)
            out['loss'].backward()
            clip_gradient(optimizer, opt.Train.clip)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({'loss': out['loss'].item()})
            
        val_iou = []
        for i, sample in vbar:
            with torch.no_grad():
                images, gts = sample["features"], sample["masks"]
                images = images.cuda()
                gts = gts.cuda()
                out = model(images, gts)
                val_iou.append(binary_mean_iou(out['pred'], gts))
                
                pbar.set_postfix({'loss': out['loss'].item()})
        
        val_iou = torch.Tensor(val_iou)
        if torch.mean(val_iou) > best_iou:
            best_iou = torch.mean(val_iou)
            os.makedirs(opt.Train.train_save, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(opt.Train.train_save, f'best_val.pth'))
        IOUS.append([epoch, torch.mean(val_iou).detach().numpy()])
#         if epoch % opt.Train.checkpoint_epoch == 0:
            

    print('#' * 20, 'Train done', '#' * 20)
    pd.DataFrame(IOUS).to_csv(opt.Train.train_save / "perf.csv")
    
if __name__ == '__main__':
    args = _args()
    opt = ed(yaml.load(open(args.config), yaml.FullLoader))
    train(opt)
