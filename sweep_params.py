optimizer_types = {
    "adam" : "torch.optim.Adam",
    "adamp" : "adamp.AdamP",
}

model_types = {
    "unet" : "segmentation_models_pytorch.Unet",
    "upp" : "segmentation_models_pytorch.UnetPlusPlus",
}

scheduler_types = {
    "reducedonplateau" : {
        'type' : "torch.optim.lr_scheduler.ReduceLROnPlateau",
        'factor': 0.25,
        'patience': 5
    },
    "cosannealing" : {
        'type': "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
        'T_0': 10,
        'T_mult': 2,
    },
    "none" : None
}


def get_sweep(hparams, args):
    if hasattr(args, "batch_size") and args.batch_size:
        hparams["train_parameters"]["batch_size"] = args.batch_size
    
    if hasattr(args, "optimizer") and args.optimizer:
        hparams["optimizer"]["type"] = optimizer_types[args.optimizer]
        
    if hasattr(args, "lr") and args.lr:
        hparams["optimizer"]["lr"] = args.lr
        
    if hasattr(args, "scheduler") and args.scheduler:
        hparams["scheduler"] = scheduler_types[args.scheduler]
        
#     if args.model:
#         hparams["model"]["type"] = model_types[args.model]
        
        
#         if args.transf_imsize:
#     if any((args.transf_crop, args.transf_onlycysts, args.transf_enhancement)):
#         crop = 1024 // int(args.transf_crop)
#         if args.transf_crop == 1:
#             transf_t = [{"__class_fullname__": "albumentations.augmentations.transforms.LongestMaxSize",
#                          "always_apply": False, "max_size": 800, "p": 1}]
#             transf_v = [{"__class_fullname__": "albumentations.augmentations.transforms.LongestMaxSize",
#                          "always_apply": False, "max_size": 800, "p": 1}]
#         else:
#             if args.transf_onlycysts:
#                 transf_t = [{"__class_fullname__": "albumentations.augmentations.transforms.CropNonEmptyMaskIfExists",
#                          "always_apply": False, "height": crop, "width": crop, "p": 1}]
#                 transf_v = [{"__class_fullname__": "albumentations.augmentations.transforms.LongestMaxSize",
#                          "always_apply": False, "max_size": 800, "p": 1}]
#             else:
#                 transf_t = [{"__class_fullname__": "albumentations.augmentations.transforms.RandomCrop",
#                          "always_apply": False, "height": crop, "width": crop, "p": 1}]
#                 transf_v = [{"__class_fullname__": "albumentations.augmentations.transforms.LongestMaxSize",
#                          "always_apply": False, "max_size": 800, "p": 1}]

#         transf_t += [
#             {"__class_fullname__": "albumentations.augmentations.transforms.HorizontalFlip", 
#              "always_apply": False, "p": 0.5},
#             {"__class_fullname__": "albumentations.augmentations.transforms.RandomRotate90", 
#              "always_apply": False, "p": 0.5}
#         ]
        
#         pad = min(crop, 800)
#         transf_t.append({"__class_fullname__": "albumentations.augmentations.transforms.PadIfNeeded",
#                          "always_apply": False, "min_height": pad, "min_width": pad, "border_mode": 0, "value": 0, "mask_value": 0, "p": 1})
#         transf_v.append({"__class_fullname__": "albumentations.augmentations.transforms.PadIfNeeded",
#                          "always_apply": False, "min_height": pad, "min_width": pad, "border_mode": 0, "value": 0, "mask_value": 0, "p": 1})
        
#         if args.transf_enhancement:
#             transf_t += [
#                 {"__class_fullname__": "albumentations.augmentations.transforms.RandomBrightnessContrast",
#                  "p": 1.0},
#                 {"__class_fullname__": "albumentations.augmentations.transforms.RandomGamma",
#                  "always_apply": False, "p": 0.5},
#                 {"__class_fullname__": "albumentations.augmentations.transforms.CLAHE",
#                  "always_apply": False, "p": 0.5},
#             ]
#             transf_v += [
#                 {"__class_fullname__": "albumentations.augmentations.transforms.RandomBrightnessContrast",
#                  "p": 1.0},
#                 {"__class_fullname__": "albumentations.augmentations.transforms.RandomGamma",
#                  "always_apply": False, "p": 0.5},
#                 {"__class_fullname__": "albumentations.augmentations.transforms.CLAHE",
#                  "always_apply": False, "p": 0.5},
#             ]

#         transf_t.append(
#             {"__class_fullname__": "albumentations.augmentations.transforms.Normalize",
#              "always_apply": False, "max_pixel_value": 255.0, "mean": [0.485, 0.456, 0.406],
#              "p": 1, "std": [0.229, 0.224, 0.225]}
#         )
#         transf_v.append(
#             {"__class_fullname__": "albumentations.augmentations.transforms.Normalize",
#              "always_apply": False, "max_pixel_value": 255.0, "mean": [0.485, 0.456, 0.406],
#              "p": 1, "std": [0.229, 0.224, 0.225]}
#         )
#         hparams["train_aug"]["transform"]["transforms"] = transf_t
#         hparams["val_aug"]["transform"]["transforms"] = transf_v

#         trfs = ", ".join([n["__class_fullname__"].split(".")[-1] for n in transf_t])
#         print(f"Used data augmentations: {trfs}\n")
    
    return hparams
