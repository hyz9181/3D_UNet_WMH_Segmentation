import numpy as np
import sys
import torch
import torchvision
from torch import nn
from torch import squeeze
import torch.optim as optim
import torch.nn.functional as F
from Unet import Unet
from unet2 import UNet
from dataset import WMH_Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
#from IOULoss import IoULoss


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
LEARNING_RATE = 1e-4
Batch_size = 2
NUM_EPOCH = 2
DEVICE = device
Train_Img_Dir = "C:/Users/Yuanzhe Huang/Desktop/unet_segmentation/wmh_data/new_data/T2_FLAIR/"
Train_Mask_Dir = "C:/Users/Yuanzhe Huang/Desktop/unet_segmentation/wmh_data/new_data/Mask/"


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
        shuffle=True,
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader

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

def train(loader, model, optimizer, loss_fn, scaler):
    model.train()
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    print("model start training ...")
    for batch_idx, (data, targets) in enumerate(loader):
    	#print(f"data shape {data.shape} targets shape {targets.shape}")
    	data = data.float().to(device=DEVICE)
    	targets = targets.float().unsqueeze(1).to(device = DEVICE)
    	targets = targets.squeeze(1)
    	print(f"train targets {targets.shape}")
    	data = data.unsqueeze(1)
    	if(data.shape[0] == 1):
    		print(f"data shape need swap {data.shape}")
    		data = np.swapaxes(data, 0, 1)
    	#targets = squeeze(targets, 1)
    	#print(f"data bytes {type(data)} {data.element_size() * data.nelement()}")
    	#print(f"target bytes {type(targets)} {targets.element_size() * targets.nelement()}")
    	#print(f"before model size {data.size}")
    	pred = data

    	#forward
    	with torch.cuda.amp.autocast():
    		#print(f"before model size {data.size}")
    		print(f"batch idx {batch_idx} data {data.shape} target {targets.shape}")
    		predictions = model(data)
    		pred = predictions    		
    		loss = IoULoss(predictions, targets)

    	optimizer.zero_grad()
    	scaler.scale(loss).backward()
    	scaler.step(optimizer)
    	scaler.update()

    	preds = torch.sigmoid(pred)
    	preds = (preds > 0.5).float()
    	num_correct += (preds == targets).sum()
    	num_pixels += torch.numel(preds)
    	print(f"num correct {num_correct} num pixels {num_pixels} pred {preds.shape} targets {targets.shape} pred targets equal {(preds == targets).shape}")
    	dice_score += (2 * (preds * targets).sum()) / ((preds + targets).sum() + 1e-8)
    	print(
        	f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}"
    	)
    	print(f"Dice score: {dice_score/len(loader)}")
    	num_correct = 0
    	num_pixels = 0
    	dice_score = 0

    	'''if batch_idx == 83:
    		print("final iteration")
    		targets_save = targets[:, :, 20]
	    	torchvision.utils.save_image(targets_save, f"g_truth.png")
	    	img_save = preds[:,:,20]
	    	torchvision.utils.save_image(img_save, f"prediction.png")'''
        


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
    	for batch_idx, (x, y) in enumerate(loader):
        	print(f"batch idx {batch_idx} before x shape {x.shape} y shape {y.shape}")
        	x = x.float().to(device)
        	#data = data.float().to(device=DEVICE)
        	x = x.unsqueeze(1)
        	y = y.to(device)
        	#y = np.swapaxes(y, 2, 3)
        	print(f"after x shape {x.shape} y shape {y.shape}")
        	preds = torch.sigmoid(model(x))
        	preds = (preds > 0.5).float()
        	num_correct += (preds == y).sum()
        	num_pixels += torch.numel(preds)
        	dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")

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


model = Unet(1, 2)
#model = UNet(1, 2)
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

print(f"start training ...")

scaler = torch.cuda.amp.GradScaler()
for epochs in range(NUM_EPOCH):
    print(f"\nEpoch {epochs+1}\n -------------------------------")
    print("Training...")
    train(train_loader, model, optimizer, loss_fn, scaler)
    #check_accuracy(val_loader, model, device = DEVICE)
print("Training Done!\n")
save_predictions_as_imgs(val_loader, model, folder = "/home/yuanzhe/Desktop/unet_segmentation/3D_UNET/")
#test(val_loader, model)
torch.save(model, "3d_unet.pth")