import torch
import numpy as np
import albumentations as A
import pandas as pd

from PIL import Image
from typing import List, Callable
from torch.utils.data import Dataset

from adversarial.config import Config

class CustomNormalization16Bit(A.ImageOnlyTransform):
    def _norm(self, img):
        return img / 65535.

    def apply(self, img, **params):
        return self._norm(img)
    
def get_transforms(config : Config, augment : bool = False) -> List[Callable]:
    """
    Get the transforms to apply yo the data.
    Args:
        augment (bool): Flag to add data augmentation
    Returns:
        List[Callable]: list containing the tranformations
    """
    transforms = [
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
    ]

    if augment:
        transforms.append(A.HorizontalFlip())
        transforms.append(A.ShiftScaleRotate(rotate_limit=20))
        transforms.append(A.Blur(blur_limit=(1, 3)))
    
    transforms.append(A.Normalize())
                      
    return transforms
    
class AdversarialDataset(Dataset):
    def __init__(self, df : List, transforms : Callable, config : Config):
        self.df = df
        self.transforms = transforms
        self.config = config
    
    def __getitem__(self, idx : int):
        info = self.data.iloc[idx, :]

        label = info['label']
        domain = info['domain']
        img = Image.open(info['filename'])
        
        if self.transforms:
            augs = A.Compose(self.transforms)
            transformed = augs(image=img)

            img = transformed['image']

        # Channels first
        img = torch.from_numpy(img.transpose(2, 0, 1)) 
        
        label = torch.as_tensor(label, dtype=torch.int64)
        domain = torch.as_tensor(domain, dtype=torch.int64)

        return img, domain, label

    def __len__(self):
        return len(self.df)