import numpy as np
import os
import nibabel as nib 
import sys
import torch
import torchvision
from torch import nn
from torch import squeeze
import torch.optim as optim
import torch.nn.functional as F
from Unet import Unet
from unet2 import UNet
from UNET3 import UNET
from dataset import WMH_Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
#from IOULoss import IoULoss

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
#torch.cuda.empty_cache()
#torch.cuda.memory_summary(device=None, abbreviated=False)
LEARNING_RATE = 1e-3
Batch_size = 1
NUM_EPOCH = 10
DEVICE = device
Train_Img_Dir = "/home/yuanzhe/Desktop/unet_segmentation/wmh_data/new_data/T2_FLAIR/"
Train_Mask_Dir = "/home/yuanzhe/Desktop/unet_segmentation/wmh_data/new_data/Mask/FLAIR_mask/"
Test_Img_Dir = "/home/yuanzhe/Desktop/unet_segmentation/wmh_data/new_data/FLAIR_unseen/"
Test_Mask_Dir = "/home/yuanzhe/Desktop/unet_segmentation/wmh_data/new_data/Mask/unseen_mask/"
save_dir = "/home/yuanzhe/Desktop/unet_segmentation/3D_UNET/"


def get_loaders(train_dir, 
                train_mask_dir, 
                batch_size,
                train_transform,
                ):
    full_dataset = WMH_Dataset(image_dir = train_dir, mask_dir = train_mask_dir, transform = train_transform)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader

def get_test_loaders(test_dir, test_mask_dir, batch_size, test_transform):
    test_dataset = WMH_Dataset(image_dir = test_dir, mask_dir = test_mask_dir, transform = test_transform)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False,)
    return test_loader

def IoULoss(inputs, targets, smooth = 1e-6):
    #comment out if your model contains a sigmoid or equivalent activation layer\
    inputs = torch.sigmoid(inputs)       
        
    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
        
    #intersection is equivalent to True Positive count
    #union is the mutually inclusive area of all labels & predictions 
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 
        
    IoU = (intersection + smooth)/(union + smooth)
    #print(f"IoU Loss {IoU}")
    return 1 - IoU

def train(loader, model, optimizer, loss_fn, scaler, epochs):
    model.train()
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    for batch_idx, (data, targets) in enumerate(loader):
        print(f"1 data {data.shape} targets {targets.shape}")
        data = data.float().to(device = DEVICE)
        targets = targets.float().unsqueeze(1).to(device = DEVICE)
        targets = targets.squeeze(1)
        #data = data.unsqueeze(1)
        '''if(data.shape[0] == 1):
            data = np.swapaxes(data, 0, 1)'''
        pred = data
        print(f"data shape {data.shape} targets shape {targets.shape}")

        with torch.cuda.amp.autocast():
            predictions = model(data)
            pred = predictions
            loss = IoULoss(predictions, targets)
        
        print(f"test batch index {batch_idx}")
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        preds = torch.sigmoid(pred)
        #p = preds.cpu().detach().numpy()
        #print(p.shape)
        preds = (preds > 0.5).float()
        num_correct += (preds == targets).sum()
        num_pixels += torch.numel(preds)
        print(f"loss type {loss}")
        print(f"num correct {num_correct} num pixels {num_pixels} pred {preds.shape} targets {targets.shape} pred targets equal {(preds == targets).shape}")
        dice_score += (2 * (preds * targets).sum()) / ((preds + targets).sum() + 1e-8)
        print(
            f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}"
        )
        print(f"Dice score: {dice_score/len(loader)}")
        num_correct = 0
        num_pixels = 0
        dice_score = 0

        '''if(batch_idx == 41):
            p = preds.cpu().detach().numpy()
            p_img = nib.Nifti1Image(p[0, :, :, :, :], affine = np.eye(4))
            nib.save(p_img, 'prediction.nii')'''

        print()

def check_accuracy(loader, model, epochs, device = "cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.float().to(device)
            x = x.squeeze(1)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            print(f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}")
            
            '''if epochs == NUM_EPOCH - 1:
                x_numpy = (preds.cpu().detach().numpy())[0, 0, :, :, :]
                y_numpy = (preds.cpu().detach().numpy())[0, 0, :, :, :]
                print(f"x save {x_numpy.shape} y save {y_numpy.shape}")
                x_img = nib.Nifti1Image(x_numpy, affine = np.eye(4))
                y_img = nib.Nifti1Image(y_numpy, affine = np.eye(4))
                nib.save(x_img, 'val_pred.nii')
                nib.save(y_img, 'g_truth.nii')
                #np.savetxt('val_pred_text.txt', x_numpy)'''

    num_correct = 0
    num_pixels = 0
    dice_score = 0

def save_predictions_as_imgs(loader, model, folder, device="cuda"):
    model.eval()
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)
        with torch.no_grad():
            '''preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()'''
            preds = model(x)
            print(f"x shape {preds.shape}")
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        print(f"y shape {y.shape}")
        y = np.swapaxes(y, 2, 3)
        torchvision.utils.save_image(y, f"{folder}{idx}.png")
        preds = (preds > 0.5).float()
        num_correct += (preds == y).sum()
        num_pixels += torch.numel(preds)
        dice_score += (2 * (preds * y).sum()) / (
            (preds + y).sum() + 1e-8
        )

    print(
        f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}"
    )

    model.train()

def test(loader, model, device = "cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.float().to(device)
            x = x.squeeze(1)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            print(f"Test Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}")

            if batch_idx == 3:
                    x_numpy = (preds.cpu().detach().numpy())[0, 0, :, :, :]
                    y_numpy = (preds.cpu().detach().numpy())[0, 0, :, :, :]
                    print(f"x save {x_numpy.shape} y save {y_numpy.shape}")
                    x_img = nib.Nifti1Image(x_numpy, affine = np.eye(4))
                    y_img = nib.Nifti1Image(y_numpy, affine = np.eye(4))
                    nib.save(x_img, 'val_pred.nii')
                    nib.save(y_img, 'g_truth.nii')
    num_correct = 0
    num_pixels = 0
    dice_score = 0

#model = Unet(3, 2)
#model = UNet(3, 2)
model = UNET(3, 2)
model = model.to(DEVICE)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.2)),
])

train_loader, val_loader = get_loaders(
    Train_Img_Dir,
    Train_Mask_Dir,
    Batch_size,
    train_transform = None, 
)

test_loader = get_test_loaders(Test_Img_Dir, Test_Mask_Dir, Batch_size, test_transform = None,)

print(f"start training ...")

scaler = torch.cuda.amp.GradScaler()
for epochs in range(NUM_EPOCH):
    print(f"\nEpoch {epochs+1}\n -------------------------------")
    print("Training...")
    train(train_loader, model, optimizer, loss_fn, scaler, epochs)
    print()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_stats(device = DEVICE))
    check_accuracy(val_loader, model, epochs, device = DEVICE)
print("Training Done!\n")
test(test_loader, model, device = DEVICE)
#save_predictions_as_imgs(val_loader, model, folder = "/home/yuanzhe/Desktop/unet_segmentation/3D_UNET/")
#test(val_loader, model)
torch.save(model, "3d_unet.pth")
