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
        self.conv1 = Conv_DCFD(3, 10, kernel_size=3, inter_kernel_size=5, padding=1, stride=2, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2).cuda()
        self.conv2 = nn.Conv2d(10,128, kernel_size=5, stride=1).cuda()
        self.pool2 = nn.MaxPool2d(kernel_size=2).cuda()
        self.fc = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        # self.fc = nn.Linear(2560, 640)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(torch.relu(nn.functional.relu(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Reshape tensor to 2D
        x = torch.relu(self.fc(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
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
for epoch in range(30):  # loop over the dataset multiple times

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
        # if i > 500:
        #     break
print("losses=", losses)

print('Finished Training')

# Test the network
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = classifier(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

"""
losses= [2.103508552312851, 1.823105252981186, 1.691538039445877, 1.6347151339054107, 1.5688945436477661, 1.5167546010017394, 1.4651647365093232, 1.4084107339382173, 1.3716975116729737, 1.3527373242378236, 1.3053640961647033, 1.3009671294689178, 1.2626765394210815, 1.2734773391485215, 1.228956555724144, 1.2100338512659072, 1.2078957563638688, 1.1835688024759292, 1.2142452335357665, 1.1517521280050278, 1.1629655212163925, 1.1324471098184585, 1.1310155111551286, 1.1198723781108857, 1.0937688648700714, 1.1070447093248368, 1.0942641717195511, 1.08359055519104, 1.078652908205986, 1.0695739579200745, 1.0432748293876648, 1.0339130467176438, 1.045070013999939, 1.0112333470582962, 1.0248430198431016, 1.008266689181328, 1.0343079870939256, 0.984439092874527, 0.9970525640249253, 0.9862067872285842, 0.987713344693184, 0.9706955093145371, 0.962193700671196, 0.9460295683145523, 0.9524020850658417, 0.9674182856082916, 0.9449582242965698, 0.9631678330898285, 0.9246120566129684, 0.9558971238136291, 0.9287527543306351, 0.9405477756261825, 0.9033707195520401, 0.9326031571626663, 0.9256747102737427, 0.9227723294496536, 0.9120088225603104, 0.88485216319561, 0.9212801796197891, 0.8940077614784241, 0.8892598420381546, 0.8946597880125046, 0.8868854397535324, 0.8737297189235688, 0.8839633786678314, 0.9135974651575088, 0.8677436876296997, 0.859672235250473, 0.8983107268810272, 0.8733320784568787, 0.8498422992229462, 0.8338770681619644, 0.867216631770134, 0.8484627690911293, 0.8590463346242905, 0.8682875674962998, 0.8427671825885773, 0.8150481271743775, 0.8303388324379921, 0.830264949798584, 0.8357601082324981, 0.8471581989526749, 0.8659210458397866, 0.8301180928945542, 0.8248957633972168, 0.8360925590991974, 0.8236035686731339, 0.7997745975852013, 0.8216737937927246, 0.813561155796051, 0.8263967514038086, 0.8319144153594971, 0.788689267039299, 0.7945614883303642, 0.823298299908638, 0.7941078397631646, 0.8038291096687317, 0.8166309466958046, 0.7944809260964394, 0.7869323346018792, 0.7912712159752846, 0.8005137401819229, 0.8049639990925789, 0.786469560265541, 0.8167840588092804, 0.771291486620903, 0.8183078956604004, 0.8009283727407456, 0.783267265856266, 0.7739223635196686, 0.7784388440847397, 0.7961561340093612, 0.7559994596242905, 0.7701414334774017, 0.7661877447366714, 0.7768925166130066, 0.7613136303424836, 0.7726956841349601, 0.7879298752546311, 0.7715326189994812, 0.752855287194252, 0.7626454716920853, 0.77536362439394, 0.7876544672250748, 0.7767735195159912, 0.7718994653224945, 0.7533013397455215, 0.7368003278970718, 0.7634089758992195, 0.7524215281009674, 0.7512139776349067, 0.7568250489234924, 0.7479991421103478, 0.7426509609818459, 0.7621765393018722, 0.7546345970034599, 0.7302149015665055, 0.745671501159668, 0.7588602212071419, 0.7280772143602371, 0.726377263367176, 0.7408083456754685, 0.7285840737819672, 0.7487705579400062, 0.7099458622932434, 0.7279139682650566, 0.7687492829561233, 0.7247628027200699, 0.7245256599783897, 0.7331125950813293, 0.7350282579660415, 0.7204903241991997, 0.720649103820324, 0.7280466973781585, 0.6987605553865432, 0.7280476838350296, 0.7184262576699257, 0.7306922805309296, 0.7245160031318665, 0.7441892805695534, 0.6802761003375053, 0.70727146089077, 0.6884003260731697, 0.6821051490306854, 0.7404723706841468, 0.7066051110625267, 0.7155739733576775, 0.7171436086297035, 0.7108439138531685, 0.6920862999558449, 0.6913438868522644, 0.702475958764553, 0.6982348427176476, 0.7338246726989746, 0.7060403174161911, 0.6987045502662659, 0.6846787586808205, 0.7186379000544548, 0.6961745363473892, 0.7209706619381905, 0.6900668790936471, 0.7028237184882165, 0.6952234923839569, 0.6965420863032341, 0.681664220392704, 0.6835215145349502, 0.6789633965492249, 0.6975915938615799, 0.7055650296807289, 0.6774420082569123, 0.6800863885879517, 0.700551327764988, 0.672514987885952, 0.6936406469345093, 0.67767911195755, 0.6993646728992462, 0.6664519587159157, 0.6852059894800187, 0.6543941482901573, 0.6848903051018715, 0.6728683370351791, 0.6874794268608093, 0.6789552426338196, 0.6641021916270256, 0.676251662671566, 0.6697239035367966, 0.6602196481823921, 0.6877518850564956, 0.6850332194566726, 0.6674047103524208]
Finished Training
Accuracy of the network on the 10000 test images: 73 %
"""
