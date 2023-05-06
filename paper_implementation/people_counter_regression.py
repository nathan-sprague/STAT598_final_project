import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data as data
import os
import cv2
from PIL import Image
from torchsummary import summary
import json
import torch.nn.functional as F
from Conv_DCFD import *



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layers
        self.conv1 = Conv_DCFD(3, 16, kernel_size=3, inter_kernel_size=5, padding=1, stride=1, bias=True).cuda()
        # self.conv1 = nn.Conv2d(3, 16, 3, padding=1).cuda() # uncomment this for normal convolutional layer

        self.pool = nn.MaxPool2d(2, 2).cuda()

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1).cuda()
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1).cuda()
 
        self.fc1 = nn.Linear(64 * 32 * 32, 512).cuda() 
        self.fc2 = nn.Linear(512, 128).cuda()
        self.fc3 = nn.Linear(128, 1).cuda()
        # Dropout layer
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(self.pool(F.relu(self.conv1(x))).cuda()) 
        x = self.pool(F.relu(self.conv2(x))).cuda()
        x = self.pool(F.relu(self.conv3(x))).cuda()
    
        x = x.view(-1, 65536).cuda()
        
        x = self.dropout(x).cuda()
        x = F.relu(self.fc1(x)).cuda()
        x = self.dropout(x).cuda()
        x = F.relu(self.fc2(x)).cuda()
        x = self.dropout(x).cuda()
        x = self.fc3(x).cuda()
        return x

# Define the dataset class
class ImageDataset(data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.images = os.listdir(self.image_dir)
        self.transform = transform
        with open("shanghaitech_people_count.json") as json_file:
            self.labels = json.load(json_file)



    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        
        gt = torch.FloatTensor([self.labels[self.images[index]]])
        gt = (torch.log(gt)-3.25)/(7.8-3.25)

        return image.cuda(), gt.cuda()



train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

if __name__ == "__main__":
    # Define the training data
    # X_train = torch.randn(100, 3, 256, 256)
    # y_train = torch.randn(100, 1)

    # Define the dataset and data loader
    dataset = ImageDataset("ACDA/input", transform=train_transform)
    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True)


    model = Net()

    # Define loss function and optimizer
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # summary(model, (3, 80, 80))
    running_loss = 0
    losses = []
    print_amount = 5
    # Train the network
    if True:
        for epoch in range(15):
            train_loss = 0
            model.train()
            i=0
            for images, labels in dataloader:
                optimizer.zero_grad()

                outputs = model(images.cuda())

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # print(loss.item())
                if i % print_amount == print_amount-1:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / print_amount))
                    losses += [running_loss/print_amount]
                    running_loss = 0.0
                i+=1
        torch.save(model.state_dict(), "people_reg_acda.pt")
    else:
        model.load_state_dict(torch.load("people_reg_acda.pt"))


    print('Finished Training')

    model.eval()

    error_sum_squared = 0
    error_sum = 0
    count = 0

    for data in dataloader:
        images, labels = data
        for image, label, in zip(images, labels):
            image = torch.unsqueeze(image, 0) # adds a dimension at the beginning
            print(image.shape)

            outputs = model(image)
            print(outputs)
            array = image.cpu().detach().numpy()
            array = array[0, :, :, :]
            print("arr", array.shape)
            array = np.transpose(array, (1, 2, 0))
            array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)

            img = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

            print(img.shape)
            # outputs = 0.5
            pred = float(np.exp(outputs.cpu().detach().numpy() * (7.8-3.25) + 3.25))
            real = float(np.exp(label.cpu() * (7.8-3.25) + 3.25))
            error_sum_squared += (pred-real)**2
            error_sum += abs(pred-real)
            count += 1
            print("pred", pred, "real", real, "mse", error_sum_squared/count, "mae", error_sum/count) # 62709 trained, 73484 not trained
            cv2.imshow("im",img)
            if real < 45:
                cv2.waitKey(0)
            else:
                if cv2.waitKey(1) == 27:
                    exit()

"""

normal mse 129681.30199577715 mae 199.84501936594646
DCFD   mse 76113.4768927789 mae 159.08056578318278

"""