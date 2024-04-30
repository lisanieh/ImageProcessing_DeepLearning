import os
import glob
import shutil
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data 
import torchvision
import torchvision.transforms as transforms
from torchvision import models

# data_dir = "/data"

# #create training dir
# training_dir = os.path.join(data_dir,"training_dataset")
# if not os.path.isdir(training_dir):
#   os.mkdir(training_dir)

# #create dog in training
# dog_training_dir = os.path.join(training_dir,"Dog")
# if not os.path.isdir(dog_training_dir):
#   os.mkdir(dog_training_dir)

# #create cat in training
# cat_training_dir = os.path.join(training_dir,"Cat")
# if not os.path.isdir(cat_training_dir):
#   os.mkdir(cat_training_dir)

# #create validation dir
# validation_dir = os.path.join(data_dir,"validation_dataset")
# if not os.path.isdir(validation_dir):
#   os.mkdir(validation_dir)

# #create dog in validation
# dog_validation_dir = os.path.join(validation_dir,"Dog")
# if not os.path.isdir(dog_validation_dir):
#   os.mkdir(dog_validation_dir)

# #create cat in validation
# cat_validation_dir = os.path.join(validation_dir,"Dat")
# if not os.path.isdir(cat_validation_dir):
#   os.mkdir(cat_validation_dir)
  
# #shuffle data
# split_size = 0.8
# cat_imgs_size = len(glob.glob("/data/training_dataset/Cat"))
# dog_imgs_size = len(glob.glob("/data/training_dataset/Dog"))

# for i, img in enumerate(glob.glob("/data/training_dataset/Cat")):
#   if i < (cat_imgs_size * split_size):
#       shutil.move(img, cat_training_dir)
#   else:
#       shutil.move(img,cat_validation_dir)
        
# for i, img in enumerate(glob.glob("/data/training_dataset/Dog")):
#   if i < (dog_imgs_size * split_size):
#       shutil.move(img, dog_training_dir)
#   else:
#       shutil.move(img,dog_validation_dir)
 
#function 
# def make_train_step(model, optimizer, loss_fn):
#   def train_step(x,y):
#     #make prediction
#     yhat = model(x)
#     #enter train mode
#     model.train()
#     #compute loss
#     loss = loss_fn(yhat,y)

#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     #optimizer.cleargrads()

#     return loss
#   return train_step
      
#global      
batch_size = 16

#init model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = models.resnet50(pretrained=True)
model2 = models.resnet50(pretrained=True)
# for params in model.parameters():
#   params.requires_grad_ = False
nr_filters1 = model1.fc.in_features
model1.fc = nn.Linear(nr_filters1,1)
model1 = model1.to(device)
nr_filters2 = model2.fc.in_features
model2.fc = nn.Linear(nr_filters2,1)
model2 = model2.to(device)
#init optimizer
optimizer1 = torch.optim.Adam(model1.fc.parameters())
optimizer2 = torch.optim.Adam(model2.fc.parameters())
#data load
traindir = "/data/training_dataset"
valdir = "/data/validation_dataset"

train_transformers1 = transforms.Compose([transforms.Resize(244,244),
                                         transforms.ToTensor(),
                                         torchvision.transforms.Normalize(
                                           mean=[0.485,0.456,0.406],
                                           std=[0.229,0.224,0.225]
                                         )])
val_transformers1 = transforms.Compose([transforms.Resize(244,244),
                                         transforms.ToTensor(),
                                         torchvision.transforms.Normalize(
                                           mean=[0.485,0.456,0.406],
                                           std=[0.229,0.224,0.225]
                                         )])
train_set1 = torchvision.datasets.ImageFolder(traindir,transform=train_transformers1)
val_set1 = torchvision.datasets.ImageFolder(valdir,transform=val_transformers1)
train_loader1 = torch.utils.data.DataLoader(train_set1, shuffle=True,batch_size=batch_size)
val_loader1 = torch.utils.data.DataLoader(val_set1, shuffle=True,batch_size=batch_size)

train_transformers2 = transforms.Compose([transforms.Resize(244,244),
                                         transforms.ToTensor(),
                                         torchvision.transforms.Normalize(
                                           mean=[0.485,0.456,0.406],
                                           std=[0.229,0.224,0.225]
                                         ),
                                         transforms.RandomErasing(),
                                         ])
val_transformers2 = transforms.Compose([transforms.Resize(244,244),
                                         transforms.ToTensor(),
                                         torchvision.transforms.Normalize(
                                           mean=[0.485,0.456,0.406],
                                           std=[0.229,0.224,0.225]
                                         ),
                                         transforms.RandomErasing(),
                                         ])
train_set2 = torchvision.datasets.ImageFolder(traindir,transform=train_transformers2)
val_set2 = torchvision.datasets.ImageFolder(valdir,transform=val_transformers2)
train_loader2 = torch.utils.data.DataLoader(train_set2, shuffle=True,batch_size=batch_size)
val_loader2 = torch.utils.data.DataLoader(val_set2, shuffle=True,batch_size=batch_size)

#init loss function
loss_fn = nn.modules.loss.BCEWithLogitsLoss()
#init tensorboard writer
# train_step = make_train_step(model, optimizer, loss_fn)
best_val_acc1 = 0
best_val_acc2 = 0
best_epoch1 = 0
best_epoch2 = 0

num_epoch = 10
# early_stopping_tolerance = 3
# early_stopping_threshold = 0.03
# avg_train_losses = []
# avg_val_losses = []
# avg_train_accs = []
# avg_val_accs = []

#train
for epoch in range(num_epoch):
    #training
    # train_loss_history = []
    # train_acc_history1 = []
    # train_acc_history2 = []
    for img,label in train_loader1:
        img = img.to(device)
        label = label.unsqueeze(1).float()
        label = label.to(device)
        y_pred = model1(img)
        model1.train()
        
        loss = loss_fn(y_pred, label)
        loss.backward()
        optimizer1.step()
        optimizer1.zero_grad()
        
        # acc = (y_pred.argmax(dim=1) == label).float().mean()
        # train_loss_history.append(loss.item())
        # train_acc_history1.append(acc.item())
    #load train loss and acc
    # avg_train_loss = sum(train_loss_history) / len(train_loss_history)
    # avg_train_acc1 = sum(train_acc_history1) / len(train_acc_history1)
    # avg_train_losses.append(avg_train_loss)
    # avg_train_accs.append(avg_train_acc)
    for img,label in train_loader2:
        img = img.to(device)
        label = label.unsqueeze(1).float()
        label = label.to(device)
        y_pred = model2(img)
        model2.train()
        
        loss = loss_fn(y_pred, label)
        loss.backward()
        optimizer2.step()
        optimizer2.zero_grad()
        
        # acc = (y_pred.argmax(dim=1) == label).float().mean()
        # train_loss_history.append(loss.item())
        # train_acc_history2.append(acc.item())
    # avg_train_acc2 = sum(train_acc_history2) / len(train_acc_history2)
    
    #validation
    model1.eval()
    model2.eval()
    # val_loss_history = []
    val_acc_history1 = []
    val_acc_history2 = []

    for img,label in val_loader1:
        with torch.no_grad():
            img = img.to(device)
            label = label.unsqueeze(1).float()
            label = label.to(device)
            model1.eval()
            
            y_pred = model1(img)
            loss = loss_fn(y_pred, label)
            _, predicted = torch.max(y_pred.data, 1) #trainset 從2為中抓取1為資料
            acc = (predicted == label).float().mean()
        # val_loss_history.append(loss.item())
        val_acc_history1.append(acc.item())
        
    #load validation loss and acc
    # avg_val_loss = sum(val_loss_history) / len(val_loss_history)
    avg_val_acc1 = sum(val_acc_history1) / len(val_acc_history1)
    # avg_val_losses.append(avg_val_loss)
    # avg_val_accs.append(avg_val_acc)  
    
    #save model if acc is better
    if avg_val_acc1 >= best_val_acc1:
        print("best model saved at epoch {}, acc: {:.4f}".format(epoch, avg_val_acc1))
        best_val_acc1 = avg_val_acc1
        best_epoch1 = epoch
    torch.save(model1.state_dict(), "resnet_model.pth") 
        
    for img,label in val_loader2:
        with torch.no_grad():
            img = img.to(device)
            label = label.unsqueeze(1).float()
            label = label.to(device)
            model2.eval()
            
            y_pred = model2(img)
            loss = loss_fn(y_pred, label)
            _, predicted = torch.max(y_pred.data, 1) #trainset 從2為中抓取1為資料
            acc = (predicted == label).float().mean()
        # val_loss_history.append(loss.item())
        val_acc_history2.append(acc.item())
    
    avg_val_acc2 = sum(val_acc_history2) / len(val_acc_history2)
    
    #save model if acc is better
    if avg_val_acc2 >= best_val_acc2:
        print("best model saved at epoch {}, acc: {:.4f}".format(epoch, avg_val_acc2))
        best_val_acc2 = avg_val_acc2
        best_epoch2 = epoch
    torch.save(model2.state_dict(), "resnet_better_model.pth") 
    
    # #early stopping
    # early_stopping_counter = 0
    # if best_val_acc > avg_val_acc:
    #   early_stopping_counter +=1

    # if (early_stopping_counter == early_stopping_tolerance) or (best_val_acc > early_stopping_threshold):
    #   print("/nTerminating: early stopping")
    #   break #terminate training
        
    # torch.save(model.state_dict(), "resnet_model.pth")  
        
#setting
x_bar = ["without random erasing","with random erasing"]
pred = [best_val_acc1,best_val_acc2]
#plot
plt.figure(figsize=(10,10))
plt.bar(range(2),pred)
# plt.bar(range(10),softmax)
plt.xticks(range(2),x_bar)
plt.ylabel("accurracy")
plt.title("accurracy comparison")
plt.savefig("compare.png")
 