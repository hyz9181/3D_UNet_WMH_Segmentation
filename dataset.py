import os
import numpy as np
import nibabel as nib 
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image

class WMH_Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #print(f"self images {self.images}")
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        #print(f"mask path {mask_path}")
        #print(mask_path)

        '''image = np.array(Image.open(img_path).convert("RGB"), dtype = np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype = np.float32)
        mask = np.stack((mask, mask), 0)'''

        image = nib.load(img_path)
        #print(f"image {image.get_fdata()}")
        image = np.array(image.get_fdata())
        mask = nib.load(mask_path)
        mask = np.array(mask.get_fdata())

        if self.transform is not None:
            augmentations = self.transform(image = image, mask = mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        #image = np.swapaxes(image, 0, 2)
        print(f"image shape {img_path} {image.shape} mask {mask_path} {mask.shape}")

        #print(f"image shape {img_path} {image.shape} mask {mask_path} {mask.shape}")
        # target shape (320, 320, 104)

        if image.shape[0] == 256:
            image = np.swapaxes(image, 0, 1)
        if mask.shape[0] == 256:
            mask = np.swapaxes(mask, 0, 1)

        if image.shape[0] == 212:
            #print(f"image path {img_path} shape {image.shape}")
            image = np.pad(image, ((6, 6),(0, 0),(0, 0)), 'constant', constant_values = 0)
            #print(f"image path {img_path} shape {image.shape}")

        '''elif image.shape[0] == 224:
            #print(f"enter image shape 224")
            #print(f"image path {img_path} shape {image.shape}")
            image = np.pad(image, ((48, 48),(0, 0),(0, 0)), 'constant', constant_values = 0)'''

        if mask.shape[0] == 212:
            #print(f"image path {img_path} shape {image.shape}")
            mask = np.pad(mask, ((6, 6),(0, 0),(0, 0)), 'constant', constant_values = 0)

        '''elif mask.shape[0] == 224:
            #print(f"mask path 224 {mask_path} {mask.shape}")
            mask = np.pad(mask, ((48, 48),(32, 32),(28, 28)), 'constant', constant_values = 0)

        #print(f"image shape {img_path} {image.shape} mask {mask_path} {mask.shape}")'''
        #print(f"\n")

        #print(f"image {image}")

        mask = np.stack((mask, mask), 0)
        return image, mask
