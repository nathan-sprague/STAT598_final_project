import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image
from Conv_DCFD import *

# Define the U-Net model

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_dcfd = Conv_DCFD(64, 1, kernel_size=3, padding=1)#, inter_kernel_size=5, padding=0, stride=2, bias=True),

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        x = self.conv_dcfd(x)

        out = self.conv_last(x)

        return out

# Define the dataset class
class SegmentationDataset(data.Dataset):
    def __init__(self, image_dir, gt_dir, transform=None):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.images = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        gt_path = os.path.join(self.gt_dir, self.images[index][:-4] + '_mask.png')
        image = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        if np.random.random()>0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.transform is not None:
            image = self.transform(image)
        gt = np.array(gt) / 255.
        # gt[gt > 0] = 1
        gt = torch.from_numpy(gt)
        return image, gt

# Define the hyperparameters and data loaders
batch_size = 8
learning_rate = 0.001
num_epochs = 10

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


train_set = SegmentationDataset('/home/nathan/Desktop/stat598/paper/ACDA/input', 
                                '/home/nathan/Desktop/stat598/paper/ACDA/target', transform=train_transform)
# train_set = SegmentationDataset('/home/nathan/Desktop/ear_stalk_detection/stalk_videos/stalk_rgb_training_videos/wholePlant_segmentation/input', 
#                                 '/home/nathan/Desktop/ear_stalk_detection/stalk_videos/stalk_rgb_training_videos/wholePlant_segmentation/target', transform=train_transform)


# cv2.imshow("a", img)
# cv2.waitKey(0)
# val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

def checkImg(img, img2):
    array = img.detach().numpy()
    print(array.min(), array.max())
    array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
    array = array[1, :, :]
    cv2.imshow("A", array)

    array = img2.detach().numpy()
  
    array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
    array = array[1, :, :]
    cv2.imshow("b", array)
    cv2.waitKey(20)

if __name__ == "__main__":
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        train_loss = 0
        model.train()
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            

            masks = torch.stack([1 - masks, masks], dim=1)
            # masks = (masks > 0).to(torch.float32)
            masks = masks.to(torch.float32)
            # checkImg(masks[0])
            checkImg(masks[0], outputs[0])
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)
        print('Epoch: {}, Training Loss: {:.4f}'.format(epoch+1, train_loss))