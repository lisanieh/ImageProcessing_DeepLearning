import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data 

import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchvision import datasets

from numpy import *
import matplotlib.pyplot as plt

#settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())
batch_size = 64 
num_epochs = 40
learning_rate = 0.005
# num_classes = 1000

# Data agumentation
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Download training data from open datasets.
trainset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform = transform,
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                          shuffle =True)

# Download test data from open datasets.
testset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
total_step = len(trainloader)

#VGG19 model
# class Conv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size = 1,
#                  stride = 1, padding = None, groups = 1, activation = True):
#         super(Conv, self).__init__()
#         padding = kernel_size
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
#                               padding, groups=groups, bias=True)
#         self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

# class VGG19(nn.Module):
#     def __init__(self, num_classes=10):
#         super(VGG19, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU())
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(), 
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU())
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU())
#         self.layer6 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU())
#         self.layer7 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU())
#         self.layer8 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer9 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer10 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer11 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer12 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer13 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer14 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer15 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer16 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.fc = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(7*7*512, 4096),
#             nn.ReLU())
#         self.fc1 = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU())
#         self.fc2= nn.Sequential(
#             nn.Linear(4096, num_classes))
        
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.layer5(out)
#         out = self.layer6(out)
#         out = self.layer7(out)
#         out = self.layer8(out)
#         out = self.layer9(out)
#         out = self.layer10(out)
#         out = self.layer11(out)
#         out = self.layer12(out)
#         out = self.layer13(out)
#         out = self.layer14(out)
#         out = self.layer15(out)
#         out = self.layer16(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc(out)
#         out = self.fc1(out)
#         out = self.fc2(out)
#         return out
        
model = models.vgg19_bn(num_classes = 10)
model = model.to(device)
    
# Loss function and optimizer
criterion = nn.CrossEntropyLoss() #load loss function
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                            weight_decay=0.005,momentum=0.9) #找到真正的最低點

# Train the model
for epoch in range(num_epochs):
    # running_loss = 0.0
    for i, (imgs , labels) in enumerate(trainloader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        outputs = model(imgs)
        # print(outputs)
        n_corrects = (outputs.argmax(axis=1)==labels).sum().item() #accurrency of test
        loss = criterion(outputs, labels) #計算loss rate
        
        #我不知道這是什麼
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
    # print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
    #                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        if (i+1) % 250 == 0:
            print('epoch {}/{},step: {}/{}: loss = {:.5f},acc = {:.2f}'
                  .format(epoch+1,num_epochs,i+1,total_step,loss,100*(n_corrects/labels.size(0))))

#save model
torch.save(model.state_dict(),"model.pth")
 
#test weight
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images) #trainset output
        loss = criterion(outputs, labels) #loss rate of validation
        _, predicted = torch.max(outputs.data, 1) #trainset 從2為中抓取1為資料
        total += labels.size(0) #total data
        correct += (predicted == labels).sum().item() #train vs test data accurency
        del images, labels, outputs

    print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))     
     