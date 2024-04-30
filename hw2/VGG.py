import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
from PIL import Image
import glob
import typing
import os

import torch
import torch.nn as nn
import torch.utils.data 
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary

#init global
batch_size = 32
#init model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg19_bn(num_classes = 10)
model = model.to(device)
#init optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
#init dataset
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(30),
                    transforms.Resize([32,32]),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    ])
val_transform = transforms.Compose([transforms.Resize([32,32]),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()])
train_set = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=train_transform,
)
val_set = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=val_transform,
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size,
                                        shuffle =True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                        shuffle=False)

#init loss function
loss_fn = nn.CrossEntropyLoss()

#init tensorboard writer
best_val_acc = 0
best_epoch = 0

num_epoch = 30
avg_train_losses = []
avg_val_losses = []
avg_train_accs = []
avg_val_accs = []

#train
for epoch in range(num_epoch):
    #training
    model.train()
    train_loss_history = []
    train_acc_history = []
    for x,y in train_loader:
        x,y = x.cuda(), y.cuda()
        # x = x.expand(-1,3,-1,-1)
        y_one_hot = nn.functional.one_hot(y, num_classes=10).float()
        y_pred = model(x)
        
        loss = loss_fn(y_pred, y_one_hot)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        acc = (y_pred.argmax(dim=1) == y).float().mean()
        train_loss_history.append(loss.item())
        train_acc_history.append(acc.item())
    
    #load train loss and acc
    avg_train_loss = sum(train_loss_history) / len(train_loss_history)
    avg_train_acc = sum(train_acc_history) / len(train_acc_history)
    avg_train_losses.append(avg_train_loss)
    avg_train_accs.append(avg_train_acc)

    #validation
    model.eval()
    val_loss_history = []
    val_acc_history = []

    for x,y in val_loader:
        x,y = x.cuda(), y.cuda()
        y_one_hot = nn.functional.one_hot(y, num_classes=10).float()
        with torch.no_grad():
            # x = x.expand(-1,3,-1,-1)
            y_pred = model(x)
            loss = loss_fn(y_pred, y_one_hot)
            _, predicted = torch.max(y_pred.data, 1) #trainset 從2為中抓取1為資料
            acc = (predicted == y).float().mean()
        val_loss_history.append(loss.item())
        val_acc_history.append(acc.item())
        
    #load validation loss and acc
    avg_val_loss = sum(val_loss_history) / len(val_loss_history)
    avg_val_acc = sum(val_acc_history) / len(val_acc_history)
    avg_val_losses.append(avg_val_loss)
    avg_val_accs.append(avg_val_acc)  

    #save model if acc is better
    if avg_val_acc >= best_val_acc:
        print("best model saved at epoch {}, acc: {:.4f}".format(epoch, avg_val_acc))
        best_val_acc = avg_val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), "best_model.pth")  
    
#show plot
plt.figure(figsize=(20,10))
#loss
ax = plt.subplot(1,2,1)
ax.plot(avg_train_losses, label="train loss")
ax.plot(avg_val_losses, label="val loss")
ax.legend()
ax.set_title("loss")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
#acc
ax = plt.subplot(1,2,2)
ax.plot(avg_train_accs, label="train acc")
ax.plot(avg_val_accs, label="val acc")
ax.legend()
ax.set_title("accuracy")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
plt.savefig("VGG_loss_acc.png")
print("best model saved at epoch {}, acc: {:.4f}".format(best_epoch, best_val_acc))
