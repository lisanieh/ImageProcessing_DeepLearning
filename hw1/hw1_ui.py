from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
from PIL import Image
import glob

import torch
import torch.nn as nn
import torch.utils.data 
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary


#global

#functions
def load_img():
    global img, file_path
    file_path, _ = QFileDialog.getOpenFileName()
    img = cv.imread(file_path)
    
def col_separation():
    cv.imshow("rgb",img)
    b,g,r = cv.split(img)
    zeros = np.zeros(img.shape[:2], dtype="uint8")
    imgb = cv.merge([b,zeros,zeros,])
    imgg = cv.merge([zeros,g,zeros])
    imgr = cv.merge([zeros,zeros,r])
    cv.imshow("b channel",imgb)
    cv.imshow("g channel",imgg)
    cv.imshow("r channel",imgr)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def col_transformation():
    #I1
    i1 = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #I2
    b,g,r = cv.split(img)
    tmp = cv.addWeighted(b, 0.5, g, 0.5, 0)
    i2 = cv.addWeighted(r, 1/3, tmp, 2/3,0)
    #outcome
    cv.imshow("I1 : opencv function",i1)
    cv.imshow("I2 : average weighted",i2)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def col_extraction():
    #I1
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    yellow_lower = np.array([10,0,0])
    green_upper = np.array([90,255,255])
    i1 = cv.inRange(hsv,yellow_lower,green_upper)
    #I2
    tmp = cv.cvtColor(i1,cv.COLOR_GRAY2BGR)
    i2 = cv.bitwise_not(tmp,img,i1)
    #outcome
    cv.imshow("mask I1",i1)
    cv.imshow("img without y and r",i2)
    cv.waitKey(0)
    cv.destroyAllWindows()

def gaussian():
    cv.imshow("gaussian_blur",img)
    cv.createTrackbar("magnitude","gaussian_blur",1,5,g_blur)
    cv.setTrackbarMin("magnitude","gaussian_blur",1)
    cv.setTrackbarPos("magnitude","gaussian_blur",1)
    
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def bilateral():
    cv.imshow("gaussian_blur",img)
    cv.createTrackbar("magnitude","gaussian_blur",1,5,b_blur)
    cv.setTrackbarMin("magnitude","gaussian_blur",1)
    cv.setTrackbarPos("magnitude","gaussian_blur",1)
    
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def median():
    cv.imshow("gaussian_blur",img)
    cv.createTrackbar("magnitude","gaussian_blur",1,5,m_blur)
    cv.setTrackbarMin("magnitude","gaussian_blur",1)
    cv.setTrackbarPos("magnitude","gaussian_blur",1)
    
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def g_blur(val):
    n = (2 * val) + 1
    img1 = cv.GaussianBlur(img,(n,n),0)
    cv.imshow("gaussian_blur",img1) 

def b_blur(val):
    n = (2 * val) + 1
    img1 = cv.bilateralFilter(img,n,90,90)
    cv.imshow("gaussian_blur",img1)     

def m_blur(val):
    n = (2 * val) + 1
    img1 = cv.medianBlur(img,n)
    cv.imshow("gaussian_blur",img1) 

def sobel_x():
    gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gaussian_img = cv.GaussianBlur(gray_img,(3,3),0)
    filter = [[-1,0,1],
              [-2,0,2],
              [-1,0,1]]
    #sobel x
    sobelx_img = np.zeros(img.shape[:2], dtype="uint8")
    for i in range(gaussian_img.shape[0] - 2): #y
        for j in range(gaussian_img.shape[1] - 2): #x
            f1 = gaussian_img[i][j] * filter[0][0] #y,x
            f2 = gaussian_img[i][j+1] * filter[0][1] #y,x+1
            f3 = gaussian_img[i][j+2] * filter[0][2] #y,x+2
            f4 = gaussian_img[i+1][j] * filter[1][0]
            f5 = gaussian_img[i+1][j+1] * filter[1][1]
            f6 = gaussian_img[i+1][j+2] * filter[1][2]
            f7 = gaussian_img[i+2][j] * filter[2][0]
            f8 = gaussian_img[i+2][j+1] * filter[2][1]
            f9 = gaussian_img[i+2][j+2] * filter[2][2]
            sum = f1+f2+f3+f4+f5+f6+f7+f8+f9
            if sum < 0:
                sum *= -1
            if sum > 255:
                sum = 255
            sobelx_img[i+1][j+1] = sum
    cv.imshow("sobel x",sobelx_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def sobel_y():
    gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gaussian_img = cv.GaussianBlur(gray_img,(3,3),0)
    filter = [[-1,-2,-1],
              [0,0,0],
              [1,2,1]]
    #sobel y
    sobely_img = np.zeros(img.shape[:2], dtype="uint8")
    for i in range(gaussian_img.shape[0] - 2): #y
        for j in range(gaussian_img.shape[1] - 2): #x
            f1 = gaussian_img[i][j] * filter[0][0] #y,x
            f2 = gaussian_img[i][j+1] * filter[0][1] #y,x+1
            f3 = gaussian_img[i][j+2] * filter[0][2] #y,x+2
            f4 = gaussian_img[i+1][j] * filter[1][0]
            f5 = gaussian_img[i+1][j+1] * filter[1][1]
            f6 = gaussian_img[i+1][j+2] * filter[1][2]
            f7 = gaussian_img[i+2][j] * filter[2][0]
            f8 = gaussian_img[i+2][j+1] * filter[2][1]
            f9 = gaussian_img[i+2][j+2] * filter[2][2]
            sum = f1+f2+f3+f4+f5+f6+f7+f8+f9
            if sum < 0:
                sum *= -1
            if sum > 255:
                sum = 255
            sobely_img[i+1][j+1] = sum
    cv.imshow("sobel y",sobely_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def combination():
    gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gaussian_img = cv.GaussianBlur(gray_img,(3,3),0)
    filter_x = [[-1,0,1],
              [-2,0,2],
              [-1,0,1]]
    filter_y = [[-1,-2,-1],
              [0,0,0],
              [1,2,1]]
    #sobel
    sobelx_img = np.zeros(img.shape[:2], dtype="uint8")
    sobely_img = np.zeros(img.shape[:2], dtype="uint8")
    filter = filter_x
    for i in range(gaussian_img.shape[0] - 2): #y
            for j in range(gaussian_img.shape[1] - 2): #x
                f1 = gaussian_img[i][j] * filter[0][0] #y,x
                f2 = gaussian_img[i][j+1] * filter[0][1] #y,x+1
                f3 = gaussian_img[i][j+2] * filter[0][2] #y,x+2
                f4 = gaussian_img[i+1][j] * filter[1][0]
                f5 = gaussian_img[i+1][j+1] * filter[1][1]
                f6 = gaussian_img[i+1][j+2] * filter[1][2]
                f7 = gaussian_img[i+2][j] * filter[2][0]
                f8 = gaussian_img[i+2][j+1] * filter[2][1]
                f9 = gaussian_img[i+2][j+2] * filter[2][2]
                sum = f1+f2+f3+f4+f5+f6+f7+f8+f9
                if sum < 0:
                    sum *= -1
                if sum > 255:
                    sum = 255
                sobelx_img[i+1][j+1] = sum
    filter = filter_y
    for i in range(gaussian_img.shape[0] - 2): #y
            for j in range(gaussian_img.shape[1] - 2): #x
                f1 = gaussian_img[i][j] * filter[0][0] #y,x
                f2 = gaussian_img[i][j+1] * filter[0][1] #y,x+1
                f3 = gaussian_img[i][j+2] * filter[0][2] #y,x+2
                f4 = gaussian_img[i+1][j] * filter[1][0]
                f5 = gaussian_img[i+1][j+1] * filter[1][1]
                f6 = gaussian_img[i+1][j+2] * filter[1][2]
                f7 = gaussian_img[i+2][j] * filter[2][0]
                f8 = gaussian_img[i+2][j+1] * filter[2][1]
                f9 = gaussian_img[i+2][j+2] * filter[2][2]
                sum = f1+f2+f3+f4+f5+f6+f7+f8+f9
                if sum < 0:
                    sum *= -1
                if sum > 255:
                    sum = 255
                sobely_img[i+1][j+1] = sum
    #combination
    c_img = np.zeros(img.shape[:2], dtype="uint8")
    for i in range(sobelx_img.shape[0]):
        for j in range(sobelx_img.shape[1]):
             result = math.sqrt((sobelx_img[i][j])**2 + (sobely_img[i][j])**2) / math.sqrt(2)
             c_img[i][j] = result
    #threshold
    threshold = 128
    t_img = np.zeros(img.shape[:2], dtype="uint8")
    for i in range(c_img.shape[0]):
        for j in range(c_img.shape[1]):
            if c_img[i][j] < threshold:
                t_img[i][j] = 0
            else:
                t_img[i][j] = 255
    #output
    cv.imshow("combination",c_img)
    cv.imshow("threshold",t_img)
    cv.waitKey(0)
    cv.destroyAllWindows
    
def gradient():
    ###combinition###
    gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gaussian_img = cv.GaussianBlur(gray_img,(3,3),0)
    filter_x = [[-1,0,1],
              [-2,0,2],
              [-1,0,1]]
    filter_y = [[-1,-2,-1],
              [0,0,0],
              [1,2,1]]
    sobelcx_img = np.zeros(img.shape[:2], dtype="uint8")
    sobelcy_img = np.zeros(img.shape[:2], dtype="uint8")
    filter = filter_x
    for i in range(gaussian_img.shape[0] - 2): #y
            for j in range(gaussian_img.shape[1] - 2): #x
                f1 = gaussian_img[i][j] * filter[0][0] #y,x
                f2 = gaussian_img[i][j+1] * filter[0][1] #y,x+1
                f3 = gaussian_img[i][j+2] * filter[0][2] #y,x+2
                f4 = gaussian_img[i+1][j] * filter[1][0]
                f5 = gaussian_img[i+1][j+1] * filter[1][1]
                f6 = gaussian_img[i+1][j+2] * filter[1][2]
                f7 = gaussian_img[i+2][j] * filter[2][0]
                f8 = gaussian_img[i+2][j+1] * filter[2][1]
                f9 = gaussian_img[i+2][j+2] * filter[2][2]
                sum = f1+f2+f3+f4+f5+f6+f7+f8+f9
                if sum < 0:
                    sum *= -1
                if sum > 255:
                    sum = 255
                sobelcx_img[i+1][j+1] = sum
    filter = filter_y
    for i in range(gaussian_img.shape[0] - 2): #y
            for j in range(gaussian_img.shape[1] - 2): #x
                f1 = gaussian_img[i][j] * filter[0][0] #y,x
                f2 = gaussian_img[i][j+1] * filter[0][1] #y,x+1
                f3 = gaussian_img[i][j+2] * filter[0][2] #y,x+2
                f4 = gaussian_img[i+1][j] * filter[1][0]
                f5 = gaussian_img[i+1][j+1] * filter[1][1]
                f6 = gaussian_img[i+1][j+2] * filter[1][2]
                f7 = gaussian_img[i+2][j] * filter[2][0]
                f8 = gaussian_img[i+2][j+1] * filter[2][1]
                f9 = gaussian_img[i+2][j+2] * filter[2][2]
                sum = f1+f2+f3+f4+f5+f6+f7+f8+f9
                if sum < 0:
                    sum *= -1
                if sum > 255:
                    sum = 255
                sobelcy_img[i+1][j+1] = sum
    #combination
    c_img = np.zeros(img.shape[:2], dtype="uint8")
    for i in range(sobelcx_img.shape[0]):
        for j in range(sobelcx_img.shape[1]):
             result = math.sqrt((sobelcx_img[i][j])**2 + (sobelcy_img[i][j])**2) / math.sqrt(2)
             c_img[i][j] = result       
    #sobel signed
    sobelx_img = np.zeros(img.shape[:2], dtype="int8")
    sobely_img = np.zeros(img.shape[:2], dtype="int8")
    filter = filter_x
    for i in range(gaussian_img.shape[0] - 2): #y
            for j in range(gaussian_img.shape[1] - 2): #x
                f1 = gaussian_img[i][j] * filter[0][0] #y,x
                f2 = gaussian_img[i][j+1] * filter[0][1] #y,x+1
                f3 = gaussian_img[i][j+2] * filter[0][2] #y,x+2
                f4 = gaussian_img[i+1][j] * filter[1][0]
                f5 = gaussian_img[i+1][j+1] * filter[1][1]
                f6 = gaussian_img[i+1][j+2] * filter[1][2]
                f7 = gaussian_img[i+2][j] * filter[2][0]
                f8 = gaussian_img[i+2][j+1] * filter[2][1]
                f9 = gaussian_img[i+2][j+2] * filter[2][2]
                sum = f1+f2+f3+f4+f5+f6+f7+f8+f9
                # if sum < 0:
                #     sum *= -1
                # if sum > 255:
                #     sum = 255
                sobelx_img[i+1][j+1] = sum
                # print("x : ",sobelx_img[i+1][j+1])
    filter = filter_y
    for i in range(gaussian_img.shape[0] - 2): #y
            for j in range(gaussian_img.shape[1] - 2): #x
                f1 = gaussian_img[i][j] * filter[0][0] #y,x
                f2 = gaussian_img[i][j+1] * filter[0][1] #y,x+1
                f3 = gaussian_img[i][j+2] * filter[0][2] #y,x+2
                f4 = gaussian_img[i+1][j] * filter[1][0]
                f5 = gaussian_img[i+1][j+1] * filter[1][1]
                f6 = gaussian_img[i+1][j+2] * filter[1][2]
                f7 = gaussian_img[i+2][j] * filter[2][0]
                f8 = gaussian_img[i+2][j+1] * filter[2][1]
                f9 = gaussian_img[i+2][j+2] * filter[2][2]
                sum = f1+f2+f3+f4+f5+f6+f7+f8+f9
                # if sum < 0:
                #     sum *= -1
                # if sum > 255:
                #     sum = 255
                sobely_img[i+1][j+1] = sum
    #mask
    '''
    120~180/210~330
    '''
    mask1 = np.zeros(img.shape[:2], dtype="uint8")
    mask2 = np.zeros(img.shape[:2], dtype="uint8")
    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            rad = math.atan2(sobely_img[i][j],sobelx_img[i][j])
            deg = rad * (180/ math.pi)
            # print("degree : ",deg)
            if deg < 0:
                deg += 360
            #mask1
            if (deg > 120) and (deg < 180):
                mask1[i][j] = 255
            else:
                mask1[i][j] = 0
            #mask2
            if (deg > 210) and (deg < 330):
                mask2[i][j] = 255
            else:
                mask2[i][j] = 0
    # cv.imshow("mask1",mask1)
    # cv.imshow("mask2",mask2)
    #bitwise and
    cm1_img = np.zeros(img.shape[:2], dtype="uint8")
    cm2_img = np.zeros(img.shape[:2], dtype="uint8")
    cm1_img = cv.bitwise_and(c_img,mask1)
    cm2_img = cv.bitwise_and(c_img,mask2)
    # cv.imshow("mask1",mask1)
    # cv.imshow("mask2",mask2)
    # cv.imshow("c",c_img)
    # #output
    cv.imshow("120~180",cm1_img)
    cv.imshow("210~330",cm2_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def transform():
    height, width = img.shape[:2]
    d = float(box4_1.text())
    s = float(box4_2.text())
    tx_val = int(box4_3.text())
    ty_val = int(box4_4.text())
    #rotate + scale
    center = (200,240)
    r_matrix = cv.getRotationMatrix2D(center=center,angle = d,scale = s)
    rs_img = cv.warpAffine(src = img,M = r_matrix,dsize = (width,height))
    #translation
    t_matrix = np.float32([[1,0,tx_val],[0,1,ty_val]])
    t_img = cv.warpAffine(src = rs_img,M = t_matrix,dsize = (width,height))
    cv.imshow("output",t_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def put_img():
    global img5,file_path5
    file_path5, _ = QFileDialog.getOpenFileName()
    pixmap = QPixmap(file_path5)
    pixmap = pixmap.scaled(128,128)
    image_space.setPixmap(pixmap)
    img5 = cv.imread(file_path5)
    
def augmented():
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #load images
    im_list = []
    for filename in glob.glob("./Q5_image/Q5_1/*"):
       im = Image.open(filename)
       im_list.append(im) 
       
    #data augumentation
    t1 = transforms.RandomHorizontalFlip()
    t2 = transforms.RandomVerticalFlip()
    t3 = transforms.RandomRotation(30)
    
    #adjust image
    for im in im_list:
       im = t1(im)
       im = t2(im)
       im = t3(im)
    
    #show image
    fig, ax = plt.subplots(3,3) #plot
    for k,(im) in enumerate(im_list):
        i = k % 3
        j = k / 3
        ax[i,j].imshow(im)    
        ax[i,j].set_title(classes[k])
    plt.tight_layout()
    plt.show()
    
def model_structure():
    model = models.vgg19_bn(num_classes = 10)
    summary(model,(3,244,244))
    
def accuracy_loss():
    #settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64  #check n photos at one time
    num_epochs = 40 #slice trainset into n parts , n = 40
    learning_rate = 0.005

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
    # print("train len : ",len(trainloader))

    # Download test data from open datasets.
    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)
    # print("test len : ",len(testloader))

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    total_step = len(trainloader)
        
    model = models.vgg19_bn(num_classes = 10)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss() #load loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                weight_decay=0.005,momentum=0.9) #找到真正的最低點

    # Train the model
    train_loss = []
    train_acc = []
    for epoch in range(num_epochs):
        # running_loss = 0.0
        for i, (imgs , labels) in enumerate(trainloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            outputs = model(imgs)
            n_corrects = (outputs.argmax(axis=1)==labels).sum().item() #accurrency of test
            loss = criterion(outputs, labels) #計算loss rate
            
            #我不知道這是什麼
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #collect data
            if i < len(testloader):
                loss = loss.tolist()
                loss_data = loss
                train_loss.append(loss_data)
                train_acc.append(100*(n_corrects/labels.size(0)))
            
            
        # print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
        #                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            if (i+1) % 250 == 0:
                print('epoch {}/{},step: {}/{}: loss = {:.5f},acc = {:.2f}'
                    .format(epoch+1,num_epochs,i+1,total_step,loss,100*(n_corrects/labels.size(0))))
    
    #test weight
    valid_loss = []
    valid_acc = []
    with torch.no_grad():
        correct = 0
        total = 0
        for i,(images, labels) in enumerate(testloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images) #testset output
            loss = criterion(outputs, labels) #loss rate of validation
            # n_corrects = (outputs.argmax(axis=1)==labels).sum().item() #accurrency of test
            _, predicted = torch.max(outputs.data, 1) #trainset 從2為中抓取1為資料
            total += labels.size(0) #total data
            correct += (predicted == labels).sum().item() #train vs test data accurency
            del images, labels, outputs
            loss = loss.tolist()
            loss_data = loss
            valid_loss.append(loss_data)
            valid_acc.append(100 * correct / total)
                

        # print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))  
    
    # print("train loss : ",len(train_loss))
    # print("train acc : ",len(train_acc))
    # print("valid loss : ",len(valid_loss))
    # print("valid acc : ",len(valid_acc))   
    # print the plot
    a1 = train_loss
    a2 = valid_loss
    b1 = train_acc
    b2 = valid_acc
    fig, ax = plt.subplots(2,1) #plot
    ax[0].plot(a1) #data    
    ax[0].plot(a2)
    ax[0].set_title("loss")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[1].plot(b1) #data    
    ax[1].plot(b2)
    ax[1].set_title("accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy(%)")
    plt.tight_layout()
    plt.show()
          
def inference():
    #settings
    model = torchvision.models.vgg19_bn(num_classes = 10)
    model.load_state_dict(torch.load("model.pth"))
    transform = transforms.ToTensor()
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    #start model
    image = transform(img5)
    image = image.unsqueeze(0)
    outputs = model(image)
    # print(outputs)
    
    #change tensor into numpy
    outputs = outputs.tolist()
    # print(outputs)
    
    # reset probability
    data = []
    for i in range(10):
        if outputs[0][i] < 0:
            outputs[0][i] = 0
        data.append(outputs[0][i])
    # print(data)
    
    #print histogtram
    plt.bar(classes,data)
    plt.title("probability of each calss")
    plt.xlabel("class")
    plt.ylabel("probility")
    plt.show()
    
    # probability, plabel = torch.max(outputs.data, 1)
    # print("prob : ",probability)
    # print("label : ",plabel)
    
  
#initialization
file_path5 = ""
app = QApplication([])
window = QWidget()
layout = QHBoxLayout()
glayout1 = QVBoxLayout()
glayout2 = QVBoxLayout()
first_button = QPushButton("Load image")
groupbox1 = QGroupBox("1. Image Processing")
button1_1 = QPushButton("1.1 Color Seperate")
button1_2 = QPushButton("1.2 Color Transform")
button1_3 = QPushButton("1.3 Color Extraction")
groupbox2 = QGroupBox("2. Image Smoothing")
button2_1 = QPushButton("2.1 Gaussian Blur")
button2_2 = QPushButton("2.2 Bilateral Filter")
button2_3 = QPushButton("2.3 Median Filter")
groupbox3 = QGroupBox("3. Edge Detection")
button3_1 = QPushButton("3.1 Sobel X")
button3_2 = QPushButton("3.2 Sobel Y")
button3_3 = QPushButton("3.3 Combination and Threshold")
button3_4 = QPushButton("3.4 Gradient Angle")
groupbox4 = QGroupBox("4. Transforms")
top_layout4 = QHBoxLayout()
left_layout4 = QVBoxLayout()
right_layout4 = QVBoxLayout()
mid_layout4 = QVBoxLayout()
buttom_layout4 = QVBoxLayout()
label4_1 = QLabel("Rotation:")
box4_1 = QLineEdit()
end4_1 = QLabel("deg")
label4_2 = QLabel("Scaling:")
box4_2 = QLineEdit()
end4_2 = QLabel("")
label4_3 = QLabel("Tx:")
box4_3 = QLineEdit()
end4_3 = QLabel("pixel")
label4_4 = QLabel("Ty:")
box4_4 = QLineEdit()
end4_4 = QLabel("pixel")
button4_1 = QPushButton("4. Transforms")
groupbox5 = QGroupBox("5. VGG19")
button5_0 = QPushButton("Load Image")
button5_1 = QPushButton("1. Show Augmented Images")
button5_2 = QPushButton("2. Show Model Structure")
button5_3 = QPushButton("3. Show Accuracy and Loss")
button5_4 = QPushButton("4. Inference")
image_space = QLabel("image")
pixmap = QPixmap(file_path5)
image_space.setPixmap(pixmap)
string = QLabel("Predicted : ")

#setting
layout1 = QVBoxLayout(groupbox1)
layout1.addWidget(button1_1)
layout1.addWidget(button1_2)
layout1.addWidget(button1_3)
layout2 = QVBoxLayout(groupbox2)
layout2.addWidget(button2_1)
layout2.addWidget(button2_2)
layout2.addWidget(button2_3)
layout3 = QVBoxLayout(groupbox3)
layout3.addWidget(button3_1)
layout3.addWidget(button3_2)
layout3.addWidget(button3_3)
layout3.addWidget(button3_4)
layout4 = QVBoxLayout(groupbox4)
left_layout4.addWidget(label4_1)
left_layout4.addWidget(label4_2)
left_layout4.addWidget(label4_3)
left_layout4.addWidget(label4_4)
mid_layout4.addWidget(box4_1)
mid_layout4.addWidget(box4_2)
mid_layout4.addWidget(box4_3)
mid_layout4.addWidget(box4_4)
right_layout4.addWidget(end4_1)
right_layout4.addWidget(end4_2)
right_layout4.addWidget(end4_3)
right_layout4.addWidget(end4_4)
buttom_layout4.addWidget(button4_1)
top_layout4.addLayout(left_layout4)
top_layout4.addLayout(mid_layout4)
top_layout4.addLayout(right_layout4)
layout4.addLayout(top_layout4)
layout4.addLayout(buttom_layout4)
layout5 = QVBoxLayout(groupbox5)
layout5.addWidget(button5_0)
layout5.addWidget(button5_1)
layout5.addWidget(button5_2) 
layout5.addWidget(button5_3)
layout5.addWidget(button5_4)
layout5.addWidget(string)
layout5.addWidget(image_space)
groupbox1.setLayout(layout1)
groupbox2.setLayout(layout2)
groupbox3.setLayout(layout3)
groupbox4.setLayout(layout4)
groupbox5.setLayout(layout5)
glayout1.addWidget(groupbox1)
glayout1.addWidget(groupbox2)
glayout1.addWidget(groupbox3)
glayout2.addWidget(groupbox4)
glayout2.addWidget(groupbox5)
layout.addWidget(first_button)
layout.addLayout(glayout1)
layout.addLayout(glayout2)
window.setLayout(layout)

#actions
first_button.clicked.connect(load_img)
button1_1.clicked.connect(col_separation)
button1_2.clicked.connect(col_transformation)
button1_3.clicked.connect(col_extraction)
button2_1.clicked.connect(gaussian)
button2_2.clicked.connect(bilateral)
button2_3.clicked.connect(median)
button3_1.clicked.connect(sobel_x)
button3_2.clicked.connect(sobel_y)
button3_3.clicked.connect(combination)
button3_4.clicked.connect(gradient)
button4_1.clicked.connect(transform)
button5_0.clicked.connect(put_img)
button5_1.clicked.connect(augmented)
button5_2.clicked.connect(model_structure)
button5_3.clicked.connect(accuracy_loss)
button5_4.clicked.connect(inference)

#output
window.show()
app.exec_()