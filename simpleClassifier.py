import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from Conv_DCFD import *
import torch.nn.functional as F
import torchvision.datasets as datasets


class Conv_DCFD_Classifier(nn.Module):
    def __init__(self):
        super(Conv_DCFD_Classifier, self).__init__()
        self.conv_dcf = Conv_DCFD(3, 10, kernel_size=3, inter_kernel_size=5, padding=1, stride=2, bias=True)
        # self.fc = nn.Linear(640, 10)
        self.fc = nn.Linear(2560, 640)


    def forward(self, x):
        x = self.conv_dcf(x)
        x = x.view(x.size(0), -1)  # Reshape tensor to 2D
        x = self.fc(x)
        return x

# Define the transforms for the dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load the CIFAR10 dataset
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Define the dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Define the device (GPU or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the classifier and move it to the device
classifier = Conv_DCFD_Classifier().to(device)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
losses = []
# Train the classifier
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = classifier(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            losses += [running_loss/100]
            running_loss = 0.0
        if i > 500:
            break
print("losses=", losses)

print('Finished Training')

# Test the network
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = classifier(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

"""
[1,   100] loss: 2.258
[1,   200] loss: 1.813
[1,   300] loss: 1.700
[1,   400] loss: 1.636
[1,   500] loss: 1.619
[2,   100] loss: 1.533
[2,   200] loss: 1.518
[2,   300] loss: 1.489
[2,   400] loss: 1.469
[2,   500] loss: 1.456
[3,   100] loss: 1.412
[3,   200] loss: 1.412
[3,   300] loss: 1.416
[3,   400] loss: 1.379
[3,   500] loss: 1.370
[4,   100] loss: 1.380
[4,   200] loss: 1.364
[4,   300] loss: 1.338
[4,   400] loss: 1.326
[4,   500] loss: 1.337
[5,   100] loss: 1.345
[5,   200] loss: 1.324
[5,   300] loss: 1.323
[5,   400] loss: 1.303
[5,   500] loss: 1.334
[6,   100] loss: 1.286
[6,   200] loss: 1.277
[6,   300] loss: 1.306
[6,   400] loss: 1.278
[6,   500] loss: 1.284
[7,   100] loss: 1.278
[7,   200] loss: 1.225
[7,   300] loss: 1.259
[7,   400] loss: 1.278
[7,   500] loss: 1.264
[8,   100] loss: 1.256
[8,   200] loss: 1.275
[8,   300] loss: 1.251
[8,   400] loss: 1.264
[8,   500] loss: 1.199
[9,   100] loss: 1.230
[9,   200] loss: 1.231
[9,   300] loss: 1.209
[9,   400] loss: 1.235
[9,   500] loss: 1.201
[10,   100] loss: 1.212
[10,   200] loss: 1.209
[10,   300] loss: 1.213
[10,   400] loss: 1.209
[10,   500] loss: 1.206
losses= [2.258490161895752, 1.8128305423259734, 1.6996322751045227, 1.6364883267879486, 1.6185766053199768, 1.5334397268295288, 1.5176745879650115, 1.488903980255127, 1.4692371988296509, 1.4563805210590361, 1.4120669889450073, 1.4122405540943146, 1.4161523520946502, 1.3794747698307037, 1.370017147064209, 1.3802701669931412, 1.3640703654289246, 1.337688197493553, 1.3264178371429443, 1.3371516960859298, 1.3445356345176698, 1.3243757951259614, 1.3228235125541687, 1.3030272245407104, 1.334000129699707, 1.2864820462465287, 1.2766881150007248, 1.3055348098278046, 1.2780967181921006, 1.2835199564695359, 1.2778408831357957, 1.2254830026626586, 1.258962196111679, 1.2777179753780366, 1.2636359274387359, 1.2558143693208694, 1.274523718357086, 1.2506900590658188, 1.2638163268566132, 1.1988948851823806, 1.2297281754016876, 1.2311033254861832, 1.2087304478883742, 1.2349180340766908, 1.2012667542695998, 1.2123251646757125, 1.2086483120918274, 1.213387445807457, 1.2086624205112457, 1.2063175165653228]
Finished Training
Accuracy of the network on the 10000 test images: 56 %
[Finished in 265.3s]
"""
