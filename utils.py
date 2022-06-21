import argparse
import torch
from pathlib import Path
from typing import Union, Dict, List, Tuple
from simplify_names import get_packs
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, GroupKFold
import cv2
from write_results import date_to_exp
from sklearn.model_selection import train_test_split


def get_id2_file_paths(path: Union[str, Path]) -> Dict[str, Path]:
    paths = {x.stem: x for x in Path(path).glob("*.*")}
    assert paths, "No images available, maybe you didn't download the dataset?"
    return paths


def mem_values(mem):
    keys = iter(["B", "kB", "MiB", "GiB"])
    
    mem = float(mem)
    while mem/1024 >= 1:
        mem /= 1024
        next(keys)
        
    return f"{mem:.3} {next(keys)}"


def split_dataset(hparams, k=0, test_exp=None, leave_one_out=None, strat_nogroups=False, single_exp=None):
    samples = get_samples(hparams["image_path"], hparams["mask_path"])
    
    ##########################################################
    if single_exp == 1:
        samples = [u for u in samples if "09.19" in u[0].stem]
    if single_exp == 2:
        samples = [u for u in samples if "10.19" in u[0].stem]
    if single_exp == 3:
        samples = [u for u in samples if "07.2020" in u[0].stem or "09.2020" in u[0].stem]
    if single_exp == 4:
        samples = [u for u in samples if "12.2020" in u[0].stem]
#         samples = [u for u in samples if "ctrl 11" in u[0].stem.lower() or "t4" in u[0].stem.lower()]
    if single_exp == 5:
        samples = [u for u in samples if "07.21" in u[0].stem]
    ##########################################################
    
    names = [file[0].stem for file in samples]

#     date, treatment, tube, zstack, side =
    unpack = [get_packs(name) for name in names]
    df = pd.DataFrame([])
    df["filename"] = names
    df["treatment"] = [u[1] for u in unpack]
    df["exp"] = [date_to_exp(u[0]) for u in unpack]
    df["tube"] = [u[2] for u in unpack]
#     df["te"] = df.treatment + '_' + df.exp.astype(str)
    df["te"] = df.treatment + '_' + df.exp.astype(str) + '_' + df.tube.astype(str)
    df.te = df.te.astype('category')
    
    if test_exp is not None or leave_one_out is not None:
        if leave_one_out is not None:
            tubes = df[['exp','tube']].astype(int).sort_values(by=['exp', 'tube']).drop_duplicates().reset_index(drop=True).xs(leave_one_out)
            test_idx = df[(df.exp == tubes.exp)&(df.tube == str(tubes.tube))].index
            
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
        gkf = GroupKFold(n_splits=5)# =5)
        train_idx, val_idx = list(gkf.split(df.filename, groups=df.te))[k]
    
    train_samp = [tuple(x) for x in np.array(samples)[train_idx]]
    val_samp = [tuple(x) for x in np.array(samples)[val_idx]]
    
    return {
        "train": train_samp,
        "valid": val_samp,
        "test": test_samp
    }


def print_usage():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f"\n[DEBUG]> Cuda free memory: {mem_values(f)} / {mem_values(r)} (out of a total of {mem_values(t)})\n")
    
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_samples(image_path: Path, mask_path: Path) -> List[Tuple[Path, Path]]:
    """Couple masks and images.

    Args:
        image_path:
        mask_path:

    Returns:
    """

    image2path = get_id2_file_paths(image_path)
    mask2path = get_id2_file_paths(mask_path)

    return [(image_file_path, mask2path[file_id]) for file_id, image_file_path in image2path.items()]


from typing import List

import torch


def find_average(outputs: List, name: str) -> torch.Tensor:
    if len(outputs[0][name].shape) == 0:
        return torch.stack([x[name] for x in outputs]).mean()
    return torch.cat([x[name] for x in outputs]).mean()


import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


def rename_layers(state_dict: Dict[str, Any], rename_in_layers: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in state_dict.items():
        for key_r, value_r in rename_in_layers.items():
            key = re.sub(key_r, value_r, key)

        result[key] = value

    return result


def state_dict_from_disk(
    file_path: Union[Path, str], rename_in_layers: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Loads PyTorch checkpoint from disk, optionally renaming layer names.
    Args:
        file_path: path to the torch checkpoint.
        rename_in_layers: {from_name: to_name}
            ex: {"model.0.": "",
                 "model.": ""}
    Returns:
    """
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if rename_in_layers is not None:
        state_dict = rename_layers(state_dict, rename_in_layers)

    return state_dict


def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    return torch.from_numpy(image)


import pydoc
import sys
from importlib import import_module
from pathlib import Path
from typing import Union

from addict import Dict


class ConfigDict(Dict):
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        else:
            return value
        raise ex


def py2dict(file_path: Union[str, Path]) -> dict:
    """Convert python file to dictionary.
    The main use - config parser.
    file:
    ```
    a = 1
    b = 3
    c = range(10)
    ```
    will be converted to
    {'a':1,
     'b':3,
     'c': range(10)
    }
    Args:
        file_path: path to the original python file.
    Returns: {key: value}, where key - all variables defined in the file and value is their value.
    """
    file_path = Path(file_path).absolute()

    if file_path.suffix != ".py":
        raise TypeError(f"Only Py file can be parsed, but got {file_path.name} instead.")

    if not file_path.exists():
        raise FileExistsError(f"There is no file at the path {file_path}")

    module_name = file_path.stem

    if "." in module_name:
        raise ValueError("Dots are not allowed in config file path.")

    config_dir = str(file_path.parent)

    sys.path.insert(0, config_dir)

    mod = import_module(module_name)
    sys.path.pop(0)
    cfg_dict = {name: value for name, value in mod.__dict__.items() if not name.startswith("__")}

    return cfg_dict


def py2cfg(file_path: Union[str, Path]) -> ConfigDict:
    cfg_dict = py2dict(file_path)

    return ConfigDict(cfg_dict)


def get_class( kls ):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def object_from_dict(d, parent=None, **default_kwargs):
    """https://stackoverflow.com/a/452981/7924557"""
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)  # skipcq PTC-W0034
    # return pydoc.locate(object_type)(**kwargs) if pydoc.locate(object_type) is not None else pydoc.locate(object_type.rsplit('.', 1)[0])(**kwargs)
    return get_class(object_type)(**kwargs)

def load_rgb(image_path: Union[Path, str], lib: str = "cv2") -> np.array:
    """Load RGB image from path.
    Args:
        image_path: path to image
        lib: library used to read an image.
            currently supported `cv2` and `jpeg4py`
    Returns: 3 channel array with RGB image
    """
    if Path(image_path).is_file():
        if lib == "cv2":
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif lib == "jpeg4py":
            image = jpeg4py.JPEG(str(image_path)).decode()
        else:
            raise NotImplementedError("Only cv2 and jpeg4py are supported.")
        return image

    raise FileNotFoundError(f"File not found {image_path}")


def pad_to_size(
    target_size: Tuple[int, int],
    image: np.ndarray,
    bboxes: Optional[np.ndarray] = None,
    keypoints: Optional[np.ndarray] = None,
) :
    """Pads the image on the sides to the target_size
    Args:
        target_size: (target_height, target_width)
        image:
        bboxes: np.array with shape (num_boxes, 4). Each row: [x_min, y_min, x_max, y_max]
        keypoints: np.array with shape (num_keypoints, 2), each row: [x, y]
    Returns:
        {
            "image": padded_image,
            "pads": (x_min_pad, y_min_pad, x_max_pad, y_max_pad),
            "bboxes": shifted_boxes,
            "keypoints": shifted_keypoints
        }
    """
    target_height, target_width = target_size

    image_height, image_width = image.shape[:2]

    if target_width < image_width:
        raise ValueError(f"Target width should bigger than image_width" f"We got {target_width} {image_width}")

    if target_height < image_height:
        raise ValueError(f"Target height should bigger than image_height" f"We got {target_height} {image_height}")

    if image_height == target_height:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = target_height - image_height
        y_min_pad = y_pad // 2
        y_max_pad = y_pad - y_min_pad

    if image_width == target_width:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = target_width - image_width
        x_min_pad = x_pad // 2
        x_max_pad = x_pad - x_min_pad

    result = {
        "pads": (x_min_pad, y_min_pad, x_max_pad, y_max_pad),
        "image": cv2.copyMakeBorder(image, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_CONSTANT),
    }

    if bboxes is not None:
        bboxes[:, 0] += x_min_pad
        bboxes[:, 1] += y_min_pad
        bboxes[:, 2] += x_min_pad
        bboxes[:, 3] += y_min_pad

        result["bboxes"] = bboxes

    if keypoints is not None:
        keypoints[:, 0] += x_min_pad
        keypoints[:, 1] += y_min_pad

        result["keypoints"] = keypoints

    return result

def unpad_from_size(
    pads: Tuple[int, int, int, int],
    image: Optional[np.array] = None,
    bboxes: Optional[np.ndarray] = None,
    keypoints: Optional[np.ndarray] = None,
) :
    """Crops patch from the center so that sides are equal to pads.
    Args:
        image:
        pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
        bboxes: np.array with shape (num_boxes, 4). Each row: [x_min, y_min, x_max, y_max]
        keypoints: np.array with shape (num_keypoints, 2), each row: [x, y]
    Returns: cropped image
    {
            "image": cropped_image,
            "bboxes": shifted_boxes,
            "keypoints": shifted_keypoints
        }
    """
    x_min_pad, y_min_pad, x_max_pad, y_max_pad = pads

    result = {}

    if image is not None:
        height, width = image.shape[:2]
        result["image"] = image[y_min_pad : height - y_max_pad, x_min_pad : width - x_max_pad]

    if bboxes is not None:
        bboxes[:, 0] -= x_min_pad
        bboxes[:, 1] -= y_min_pad
        bboxes[:, 2] -= x_min_pad
        bboxes[:, 3] -= y_min_pad

        result["bboxes"] = bboxes

    if keypoints is not None:
        keypoints[:, 0] -= x_min_pad
        keypoints[:, 1] -= y_min_pad

        result["keypoints"] = keypoints

    return result

import json
import os
from pathlib import Path


def get_tubules_from_json():
    """Split train / validation / test dataset"""
    # Generate json if doesn't exist
    json_dir = Path("tubules-list.json")
    if not json_dir.exists():
        raise FileNotFoundError
    #     _generate_json(json_dir)
    with open(json_dir, "r") as f:
        tubules = json.load(f)
    return tubules


def get_dataloaders(tubule_to_exclude: int, tubules: list) -> tuple:
    """Make tubules split in train, valid and test

    Args:
        tubule_to_exclude (int): number of tubule to exclude
        tubules (list): list of tubules

    Returns:
        tuple: 3 datasets
    """
    # Do not edit original list
    tubules = tubules.copy()
    tubule_test = tubules.pop(tubule_to_exclude)

    # Train / Validation tubules
    exps = _tubule_by_exp(tubules)
    tubs_tr, tubs_val = [], []
    for exp_name, tubules in exps.items():
        train, valid = train_test_split(tubules, test_size=0.2)
        print(
            f"All: {len(tubules)} - Train: {len(train)} - Val: {len(valid)} - {exp_name}"
        )
        tubs_tr += train
        tubs_val += valid

    return tubs_tr, tubs_val, [tubule_test]


def _tubule_by_exp(tubules: list) -> dict:
    """Args:
        tubules (list): [
            "Dicembre 2020/tubule 1",
            "Dicembre 2020/tubule 2",
            "Ott 2019/tubule 2",
            "Ott 2019/tubule 2"
        ]

    Returns:
        dict: {
            "Dicembre 2020": [
                "Dicembre 2020/tubule 1",
                "Dicembre 2020/tubule 2"
            ]
            "Ott 2019": [
                "Ott 2019/tubule 1",
                "Ott 2019/tubule 2"
            ],
        }
    """
    exps = set()
    for tubule_dir in tubules:
        exp_name, _ = tubule_dir.split("/")
        # exps: {'Ott 2019', 'Dicembre 2020', ...}
        exps.add(exp_name)

    # exps_ls: {'Ott 2019': [], 'Dicembre 2020': [], ...}
    exps_ls: dict = {exp: [] for exp in exps}

    for tubule_dir in tubules:
        exp_name, _ = tubule_dir.split("/")
        # exps_ls: {
        #     "Ott 2019": ["Ott 2019/tubule 1", "Ott 2019/tubule 2"],
        #     "Dicembre 2020": ["Dicembre 2020/tubule 1", "Dicembre 2020/tubule 2"]
        # }
        exps_ls[exp_name].append(tubule_dir)
    return exps_ls

class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        from pathlib import PosixPath

        if isinstance(obj, PosixPath):
            return str(obj)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
