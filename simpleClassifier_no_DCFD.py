import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define transforms for data augmentation
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# Define the network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
losses = []
# Train the network
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            losses += [running_loss/200]
            running_loss = 0.0

print("losses", losses--)


print('Finished Training')

# Test the network
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

"""


Files already downloaded and verified
Files already downloaded and verified
[1,   200] loss: 2.304
[1,   400] loss: 2.302
[1,   600] loss: 2.300
[1,   800] loss: 2.294
[1,  1000] loss: 2.276
[1,  1200] loss: 2.188
[1,  1400] loss: 2.069
[2,   200] loss: 2.011
[2,   400] loss: 1.960
[2,   600] loss: 1.944
[2,   800] loss: 1.915
[2,  1000] loss: 1.892
[2,  1200] loss: 1.845
[2,  1400] loss: 1.831
[3,   200] loss: 1.766
[3,   400] loss: 1.744
[3,   600] loss: 1.722
[3,   800] loss: 1.700
[3,  1000] loss: 1.707
[3,  1200] loss: 1.671
[3,  1400] loss: 1.653
[4,   200] loss: 1.631
[4,   400] loss: 1.636
[4,   600] loss: 1.612
[4,   800] loss: 1.622
[4,  1000] loss: 1.611
[4,  1200] loss: 1.605
[4,  1400] loss: 1.574
[5,   200] loss: 1.589
[5,   400] loss: 1.567
[5,   600] loss: 1.544
[5,   800] loss: 1.586
[5,  1000] loss: 1.552
[5,  1200] loss: 1.548
[5,  1400] loss: 1.522
[6,   200] loss: 1.520
[6,   400] loss: 1.513
[6,   600] loss: 1.500
[6,   800] loss: 1.491
[6,  1000] loss: 1.500
[6,  1200] loss: 1.463
[6,  1400] loss: 1.486
[7,   200] loss: 1.468
[7,   400] loss: 1.441
[7,   600] loss: 1.446
[7,   800] loss: 1.468
[7,  1000] loss: 1.451
[7,  1200] loss: 1.426
[7,  1400] loss: 1.433
[8,   200] loss: 1.427
[8,   400] loss: 1.418
[8,   600] loss: 1.397
[8,   800] loss: 1.425
[8,  1000] loss: 1.395
[8,  1200] loss: 1.365
[8,  1400] loss: 1.395
[9,   200] loss: 1.372
[9,   400] loss: 1.358
[9,   600] loss: 1.370
[9,   800] loss: 1.343
[9,  1000] loss: 1.359
[9,  1200] loss: 1.349
[9,  1400] loss: 1.342
[10,   200] loss: 1.320
[10,   400] loss: 1.321
[10,   600] loss: 1.312
[10,   800] loss: 1.333
[10,  1000] loss: 1.332
[10,  1200] loss: 1.292
[10,  1400] loss: 1.308
losses [2.3043534398078918, 2.3018687057495115, 2.2996612048149108, 2.2937218809127806, 2.2758375668525694, 2.1880788773298265, 2.069392196536064, 2.011144340634346, 1.9604032093286514, 1.9436882776021958, 1.9154189556837082, 1.8919906222820282, 1.845144767165184, 1.8310850262641907, 1.7657001382112503, 1.743609363436699, 1.722141278386116, 1.700224152803421, 1.7066365045309067, 1.6709559285640716, 1.652999386191368, 1.6305730676651, 1.6360050296783448, 1.6119918292760849, 1.621503573656082, 1.6114286595582963, 1.6052942609786987, 1.5737512826919555, 1.58920982837677, 1.5668519055843353, 1.5436309254169465, 1.5864128249883651, 1.551838490962982, 1.5484082853794099, 1.5218671256303786, 1.5196978718042373, 1.5126364600658417, 1.50030499458313, 1.4914659744501113, 1.500360347032547, 1.462885201573372, 1.4860657721757888, 1.4684358328580855, 1.4411540746688842, 1.4458755505084993, 1.4683339953422547, 1.4508549624681473, 1.4259628784656524, 1.4332011920213699, 1.427429468035698, 1.4178384083509445, 1.3974132877588272, 1.424931594133377, 1.3954978400468827, 1.365444105565548, 1.394809166789055, 1.3721601715683938, 1.3582448500394821, 1.3704725992679596, 1.3428930458426476, 1.358964849114418, 1.3490368291735648, 1.3422558623552323, 1.3196065717935561, 1.3214653432369232, 1.311895807981491, 1.33319500207901, 1.3319497227668762, 1.291986185014248, 1.3084266948699952]
Finished Training
Accuracy of the network on the 10000 test images: 56 %
[Finished in 91.4s]

"""
