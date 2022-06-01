from pathlib import Path
import albumentations as albu
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import get_stacks
from typing import List, Tuple


class MyDataset(Dataset):
    def __init__(
        self,
        tileSz: int,
        transform: albu.Compose,
        tubules: list,
        debug:bool,
    ):
        """
        Args:
            tileSz (int): Crop size (e.g. 128, 256, 512)
            tubules (list): ["Ott 2019/tubule 1", "Ott 2019/tubule 2"]
        """
        self.tileSz = tileSz
        # ROOT = Path("dataset")
        ROOT = Path("artifacts") / "dataset:v1"
        self.images_path = ROOT / "images"
        self.masks_path = ROOT / "masks-png"
        self.transform = transform
        self.stacks = get_stacks(self.images_path, self.masks_path, tubules)

        if debug:
            print("=======================")
            print(f"DEBUG: reducing dataset from: {len(self.stacks)} to 10")
            print("=======================")
            # Take 10 images instead of ~500
            self.stacks = self.stacks[:10]


        assert len(self.stacks) > 0, f"Dataset is empty ({tubules})"
        print(f"Using {len(self.stacks)} images")

    def __getitem__(self, index: int):
        (img1, img2, img3), mask_path, stack_name = self.stacks[index]
        # From paths to arrays
        arr_np = _get_array(img1, img2, img3, mask_path)
        # img_np = _get_array_wo_mask(img1, img2, img3)
        # mask_np = _get_mask(mask_path)
        """
        Augmentation applied to both train and validation (not test) due to pytorch
        random_split pointing to the same dataset instance, e.g.
        train_dl.dataset.type = "valid" affects also train_dl.dataset "train" instance
        This if differentiate train and test (not validation)
        """
        arr = torch.Tensor(arr_np)
        arr = self.transform(arr)
        # sample = self.transform(image=img_np, mask=mask_np)

        stack, mask = arr[0:9], arr[9]

        # (128, 128) to (1, 128, 128)
        mask = torch.unsqueeze(mask, 0)

        return {
            "image_id": stack_name,  # TODO: Check if it's correct
            "features": stack,  # e.g. [9, 512, 512]
            "masks": mask,  # e.g. [1, 512, 512]
        }

    def __len__(self):
        return len(self.stacks)


def _get_array(img1: Path, img2: Path, img3: Path, mask: Path) -> np.ndarray:
    """Concatenate 3 images and it's mask in one array

    Args:
        img1 (Path): Path of the 1st image, size (3, 1024, 1024)
        img2 (Path): Path of the 2nd image, size (3, 1024, 1024)
        img3 (Path): Path of the 3rd image, size (3, 1024, 1024)
        mask (Path): Path of the mask, size (1024, 1024)

    Returns:
        np.ndarray: Concatenated array of size (10, 1024, 1024)
    """
    arr_img1 = _get_img_array(img1)
    arr_img2 = _get_img_array(img2)
    # arr_img2 = np.zeros((3, 1024, 1024), dtype=np.uint8)
    arr_img3 = _get_img_array(img3)

    # 9 for the images, 1 for the mask
    arr_mask = np.asanyarray(Image.open(mask), dtype=np.uint8)
    # (1024, 1024) -> (1, 1024, 1024)
    arr_mask = np.expand_dims(arr_mask, axis=0)
    # All arrays must be int
    assert all(a.dtype == np.uint8 for a in [arr_img1, arr_img2, arr_img3, arr_mask])
    return np.concatenate((arr_img1, arr_img2, arr_img3, arr_mask))


def _get_array_wo_mask(img1: Path, img2: Path, img3: Path) -> np.ndarray:
    arr_img1 = _get_img_array(img1)
    arr_img2 = _get_img_array(img2)
    # arr_img2 = np.zeros((3, 1024, 1024), dtype=np.uint8)
    arr_img3 = _get_img_array(img3)

    # All arrays must be int
    assert all(a.dtype == np.uint8 for a in [arr_img1, arr_img2, arr_img3])
    return np.concatenate((arr_img1, arr_img2, arr_img3))

def _get_mask(mask: Path):
    arr_mask = np.asanyarray(Image.open(mask), dtype=np.uint8)
    # (1024, 1024) -> (1, 1024, 1024)
    # arr_mask = np.expand_dims(arr_mask, axis=0)
    return arr_mask

def _get_img_array(img_path: Path) -> np.ndarray:
    """Get image array or empty array

    Args:
        img_path (Path): Image path, if empty string return empty array

    Returns:
        np.ndarray: Output array
    """
    if img_path:
        img = Image.open(img_path)
        # Histogram equalization (removed)
        # img = ImageOps.equalize(img)
        arr = np.asanyarray(img).transpose((2, 0, 1))
    else:
        arr = np.zeros((3, 1024, 1024), dtype=np.uint8)
    assert arr.min() >= 0 and arr.max() <= 255
    return arr
