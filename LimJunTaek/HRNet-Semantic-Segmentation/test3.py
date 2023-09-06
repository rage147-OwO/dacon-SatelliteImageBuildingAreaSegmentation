import os
import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask



transform = A.Compose(
    [
        A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2()
    ]
)
dataset = SatelliteDataset(csv_file='./train.csv', transform=transform)

class ResBlock(nn.Module):
    def __init__(self, features):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=False) # modified here
        self.conv2 = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(features)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)

        out = self.relu(x)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out) # and also here

        return out

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        ResBlock(out_channels),
        nn.ReLU(inplace=False)
    )



# U-Net model with more layers and complexity
class UNetPlus(nn.Module):
    def __init__(self):
        super(UNetPlus, self).__init__()
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.dconv_down5 = double_conv(512, 1024)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = double_conv(1024 + 512, 512)
        self.dconv_up3 = double_conv(512 + 256, 256)
        self.dconv_up2 = double_conv(256 + 128, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        x = self.dconv_down5(x)

        x = self.upsample(x)
        x = torch.cat([x, conv4], dim=1)

        x = self.dconv_up4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


import torch, gc
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torch.nn import functional as F
import torchvision
from torchvision import datasets,transforms
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


from sklearn.metrics import jaccard_score

def calculate_iou(pred, target, eps=1e-7):
    pred = torch.sigmoid(pred) > 0.5
    pred = pred.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    if intersection == 0 and union == 0:
        iou = 1.0
    else:
        iou = (intersection + eps) / (union + eps)
    return iou.item()







seed=2147483647
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)



#==========================================================================
#===Train Setting=======================================================================
LoadIndex=15
savePath=r'C:\aa'
batch_size=2





indices = list(range(len(dataset)))
#indices = list(range(1000))


train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=seed)


train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx) # index 생성
val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx) # index 생성


# sampler를 이용한 DataLoader 정의
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler) # 해당하는 index 추출
valloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

# model 초기화
model = UNetPlus().to(device)
#model = UNetLite().to(device)


# loss function과 optimizer 정의
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)




if not os.path.exists(savePath):
    os.makedirs(savePath)



start_epoch = 0
if os.path.isfile(f'{savePath}/model_checkpoint_{LoadIndex}.pth'):
    checkpoint = torch.load(f'{savePath}/model_checkpoint_{LoadIndex}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'Last Epoch :  {LoadIndex}')
    start_epoch = checkpoint['epoch']



for start_epoch in range(start_epoch,100):
    print(f'============================Epoch {start_epoch+1} ===================')
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(trainloader):
        images = images.float().to(device)
        masks = masks.float().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.unsqueeze(1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {start_epoch+1}, Training Loss: {epoch_loss/len(trainloader)}')

    torch.save({
        'epoch': start_epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }, f'{savePath}/model_checkpoint_{start_epoch+1}.pth')

    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    epoch_loss = 0
    total_iou = 0
    num_batches = 0

    with torch.no_grad():
        for images, masks in tqdm(valloader):
            images = images.float().to(device)
            masks = masks.float().to(device)

            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            epoch_loss += loss.item()

            # Apply sigmoid and threshold
            pred = torch.sigmoid(outputs) > 0.5
            # Convert to int type tensor
            pred = pred.int()

            # Flatten the tensors to calculate jaccard score
            pred_flat = pred.view(-1).cpu().numpy()
            masks_flat = masks.view(-1).cpu().numpy()

            # Calculate jaccard score
            iou = jaccard_score(masks_flat, pred_flat, average='binary')
            total_iou += iou
            num_batches += 1



    avg_iou = total_iou / num_batches
    avg_iou_str = "{:.15f}".format(avg_iou)
    print(f'Epoch {start_epoch+1}, Validation Loss: {epoch_loss/num_batches}')
    print(f'Epoch {start_epoch+1}, Validation IoU: {avg_iou_str}')

    # Clear memory
    del images, masks, outputs, loss
    gc.collect()
    torch.cuda.empty_cache()





