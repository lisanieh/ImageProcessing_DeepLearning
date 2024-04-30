from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
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


#global

#definition
class DrawBoard(QFrame):
    #  i4 = np.zeros([200,200], dtype="uint8")
     def __init__(self):
         super(DrawBoard, self).__init__()
         self.tracing_xy = []
         self.lineHistory = []
         self.pen = QPen(Qt.white, 10, Qt.SolidLine)

     def paintEvent(self, QPaintEvent):
         self.painter = QPainter()
         self.painter.begin(self)
         self.painter.setPen(self.pen)

         start_x_temp = 0
         start_y_temp = 0

         if self.lineHistory:
             for line_n in range(len(self.lineHistory)):
                 for point_n in range(1, len(self.lineHistory[line_n])):
                     start_x, start_y = self.lineHistory[line_n][point_n-1][0], self.lineHistory[line_n][point_n-1][1]
                     end_x, end_y = self.lineHistory[line_n][point_n][0], self.lineHistory[line_n][point_n][1]
                     self.painter.drawLine(start_x, start_y, end_x, end_y)

         for x, y in self.tracing_xy:
             if start_x_temp == 0 and start_y_temp == 0:
                 self.painter.drawLine(self.start_xy[0][0], self.start_xy[0][1], x, y)
             else:
                 self.painter.drawLine(start_x_temp, start_y_temp, x, y)

             start_x_temp = x
             start_y_temp = y

         self.painter.end()

     def mousePressEvent(self, QMouseEvent):
         self.start_xy = [(QMouseEvent.pos().x(), QMouseEvent.pos().y())]

     def mouseMoveEvent(self, QMouseEvent):
         self.tracing_xy.append((QMouseEvent.pos().x(), QMouseEvent.pos().y()))
         self.update()

     def mouseReleaseEvent(self, QMouseEvent):
         self.lineHistory.append(self.start_xy+self.tracing_xy)
         self.tracing_xy = []

     def delete(self):
        self.lineHistory = []
        self.update()

#functions
def load_img():
    global img, file_path
    file_path, _ = QFileDialog.getOpenFileName()
    img = cv.imread(file_path)
    
def draw_counter():
    scale = 220
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width,height)
    i0 = cv.resize(img,dim,interpolation=cv.INTER_AREA)
    i1 = cv.resize(img,dim,interpolation=cv.INTER_AREA)
    i2 = np.zeros(i0.shape[:2], dtype="uint8")
    igray = cv.cvtColor(i0,cv.COLOR_BGR2GRAY)
    iblr = cv.GaussianBlur(igray,(5,5),0)
    circles = cv.HoughCircles(iblr,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=30,maxRadius=50)
    circles = np.uint16(np.around(circles))
    num = 0
    for i in circles[0,:]:
        cv.circle(i1,(i[0],i[1]),i[2],(0,255,0),2)
        cv.circle(i2,(i[0],i[1]),2,(255,255,255),3)
        num = num + 1
        
    print(num)    
    cv.imshow("img_src",i0)
    cv.imshow("img_process",i1)
    cv.imshow("circle_center",i2)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def count_coins():
    scale = 220
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width,height)
    i0 = cv.resize(img,dim,interpolation=cv.INTER_AREA)
    igray = cv.cvtColor(i0,cv.COLOR_BGR2GRAY)
    iblr = cv.GaussianBlur(igray,(5,5),0)
    circles = cv.HoughCircles(iblr,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=30,maxRadius=50)
    circles = np.uint16(np.around(circles))
    num = 0
    for i in circles[0,:]:
        num = num + 1
        
    string1.setText("There are {} coins in the image.".format(num)) 
  
def histogram_equ():
    i0 = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    i1 = cv.equalizeHist(i0)
    i2 = np.zeros(i0.shape[:2], dtype="uint8")
    x = []
    p0 = []
    y0 = []
    p1 = []
    y1 = []
    p2 = []
    y2 = []
    tmp = []
    for i in range(i0.shape[0]):
        for j in range(i0.shape[1]):
            p0.append(i0[i,j])
            p1.append(i1[i,j])
    mp0 = Frequency(p0,len(p0))
    mp1 = Frequency(p1,len(p1))
    for i in range(255):
        x.append(i)
        if i in mp0:
            y0.append(mp0[i])
            tmp.append(mp0[i])
        else:
            y0.append(0)
            tmp.append(0)
        if i in mp1:
            y1.append(mp1[i])
        else:
            y1.append(0)
    
    #normalization of img2
    min = 255
    max = 0
    for i in range(255):
        if i > 0:
            tmp[i] = tmp[i] + tmp[i-1]
            if tmp[i] < min : min = tmp[i]
            if tmp[i] > max : max = tmp[i]
    for i in range(i2.shape[0]):
        for j in range(i2.shape[1]):
            i2[i,j] = round(((tmp[i0[i,j]] - min) / max - min) * 255)
            p2.append(i2[i,j])
    #conduct img2
    mp2 = Frequency(p2,len(p2))
    for i in range(255):
        if i in mp2:
            y2.append(mp2[i])
        else:
            y2.append(0)
            
    plt.figure(figsize=(20,30))
    
    ax = plt.subplot(2,3,1)
    ax.imshow(i0)
    ax.set_title("Original Image")
    
    ax = plt.subplot(2,3,2)
    ax.imshow(i1)
    ax.set_title("Equalized with OpenCV")
    
    ax = plt.subplot(2,3,3)
    ax.imshow(i2)
    ax.set_title("Equalized Manually")
    
    ax = plt.subplot(2,3,4)
    ax.bar(x,y0)
    ax.set_xlabel("gray value")
    ax.set_ylabel("frequency")
    ax.set_title("Histogram of Original")
    
    ax = plt.subplot(2,3,5)
    ax.bar(x,y1)
    ax.set_xlabel("gray value")
    ax.set_ylabel("frequency")
    ax.set_title("Histogram of Equaliized(OpenCV)")
    
    ax = plt.subplot(2,3,6)
    ax.bar(x,y2)
    ax.set_xlabel("gray value")
    ax.set_ylabel("frequency")
    ax.set_title("Histogram of Equaliized(Manual)")
    
    plt.show()
    
def Frequency(arr, n):
    mp = {}
    for i in range(n):
        if arr[i] in mp:
            mp[arr[i]] += 1
        else:
            mp[arr[i]] = 1
    return mp       

def closing():
    igray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    i1 = np.zeros(igray.shape[:2], dtype="uint8")
    i2 = np.zeros(igray.shape[:2], dtype="uint8")
    filter = [[1,1,1],
              [1,1,1],
              [1,1,1]]
    #dilation
    for i in range(igray.shape[0] - 2): #y
        for j in range(igray.shape[1] - 2): #x
            max = 0
            f1 = igray[i][j] * filter[0][0] #y,x
            if f1 > max : max = f1
            f2 = igray[i][j+1] * filter[0][1] #y,x+1
            if f2 > max : max = f2
            f3 = igray[i][j+2] * filter[0][2] #y,x+2
            if f3 > max : max = f3
            f4 = igray[i+1][j] * filter[1][0]
            if f4 > max : max = f4
            f5 = igray[i+1][j+1] * filter[1][1]
            if f5 > max : max = f5
            f6 = igray[i+1][j+2] * filter[1][2]
            if f6 > max : max = f6
            f7 = igray[i+2][j] * filter[2][0]
            if f7 > max : max = f7
            f8 = igray[i+2][j+1] * filter[2][1]
            if f8 > max : max = f8
            f9 = igray[i+2][j+2] * filter[2][2] 
            if f9 > max : max = f9   
            i1[i+1][j+1] = max
    #erosion
    for i in range(igray.shape[0] - 2): #y
        for j in range(igray.shape[1] - 2): #x
            min = 255
            f1 = i1[i][j] * filter[0][0] #y,x
            if f1 < min : min = f1
            f2 = i1[i][j+1] * filter[0][1] #y,x+1
            if f2 < min : min = f2
            f3 = i1[i][j+2] * filter[0][2] #y,x+2
            if f3 < min : min = f3
            f4 = i1[i+1][j] * filter[1][0]
            if f4 < min : min = f4
            f5 = i1[i+1][j+1] * filter[1][1]
            if f5 < min : min = f5
            f6 = i1[i+1][j+2] * filter[1][2]
            if f6 < min : min = f6
            f7 = i1[i+2][j] * filter[2][0]
            if f7 < min : min = f7
            f8 = i1[i+2][j+1] * filter[2][1]
            if f8 < min : min = f8
            f9 = i1[i+2][j+2] * filter[2][2] 
            if f9 < min : min = f9   
            i2[i+1][j+1] = min
    
    cv.imshow("closing",i2)
    cv.waitKey(0)
    cv.destroyAllWindows()
  
def opening():
    igray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    i1 = np.zeros(igray.shape[:2], dtype="uint8")
    i2 = np.zeros(igray.shape[:2], dtype="uint8")
    filter = [[1,1,1],
              [1,1,1],
              [1,1,1]]
    
    #erosion
    for i in range(igray.shape[0] - 2): #y
        for j in range(igray.shape[1] - 2): #x
            min = 255
            f1 = igray[i][j] * filter[0][0] #y,x
            if f1 < min : min = f1
            f2 = igray[i][j+1] * filter[0][1] #y,x+1
            if f2 < min : min = f2
            f3 = igray[i][j+2] * filter[0][2] #y,x+2
            if f3 < min : min = f3
            f4 = igray[i+1][j] * filter[1][0]
            if f4 < min : min = f4
            f5 = igray[i+1][j+1] * filter[1][1]
            if f5 < min : min = f5
            f6 = igray[i+1][j+2] * filter[1][2]
            if f6 < min : min = f6
            f7 = igray[i+2][j] * filter[2][0]
            if f7 < min : min = f7
            f8 = igray[i+2][j+1] * filter[2][1]
            if f8 < min : min = f8
            f9 = igray[i+2][j+2] * filter[2][2] 
            if f9 < min : min = f9   
            i1[i+1][j+1] = min
            
    #dilation
    for i in range(igray.shape[0] - 2): #y
        for j in range(igray.shape[1] - 2): #x
            max = 0
            f1 = i1[i][j] * filter[0][0] #y,x
            if f1 > max : max = f1
            f2 = i1[i][j+1] * filter[0][1] #y,x+1
            if f2 > max : max = f2
            f3 = i1[i][j+2] * filter[0][2] #y,x+2
            if f3 > max : max = f3
            f4 = i1[i+1][j] * filter[1][0]
            if f4 > max : max = f4
            f5 = i1[i+1][j+1] * filter[1][1]
            if f5 > max : max = f5
            f6 = i1[i+1][j+2] * filter[1][2]
            if f6 > max : max = f6
            f7 = i1[i+2][j] * filter[2][0]
            if f7 > max : max = f7
            f8 = i1[i+2][j+1] * filter[2][1]
            if f8 > max : max = f8
            f9 = i1[i+2][j+2] * filter[2][2] 
            if f9 > max : max = f9   
            i2[i+1][j+1] = max
    
    cv.imshow("opening",i2)
    cv.waitKey(0)
    cv.destroyAllWindows()
  
def show_VGG_model():
    model = models.vgg19_bn(num_classes = 10)
    summary(model,(3,244,244))
  
def VGG_acc_and_loss():
    img = cv.imread("VGG_loss_acc.png")
    cv.imshow("show loss and acc of vgg",img)
    
def predict():
    #store img
    i4 = QPixmap(image_space4.size())
    image_space4.render(i4)
    i4.save("test.png")
    
    #load img
    testing_img = cv.imread("test.png")
    
    #test with model
    model = torchvision.models.vgg19_bn(num_classes=10)
    model.load_state_dict(torch.load("model/best_model.pth", map_location=torch.device("cpu")))
    # model = torch.load("model/vgg19_20231215_v4.pth", map_location=torch.device("cpu"))
    model.eval()
    class_names = ["0","1","2","3","4","5","6","7","8","9"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32,32)),
    ])
    
    # #check if image loaded
    # if getattr(self, "testing_img", None) is None:
    #     QMessageBox.warning(self, "Warning", "please load image first")
    #     return
    
    #img to tensor
    testing_img = transform(testing_img)
    testing_img = testing_img.unsqueeze(0)
    
    with torch.no_grad():
        pred = model(testing_img)
        pred = torch.softmax(pred, dim=1)
        pred_label_idx = pred.argmax(dim=1).item()
    # softmax = nn.Softmax(dim=1)
    # softmax = softmax(pred).toList()[0]
    # pred = torch.max(pred.data(),1)[1]
    # pred = class_names[pred.item()]
    # output4.setText(pred)
    # print("pred :",pred)
    
    output4.setText(class_names[pred_label_idx])    
    pred = pred.squeeze().numpy()
    
    #plot
    plt.figure(figsize=(10,10))
    plt.bar(range(10),pred)
    # plt.bar(range(10),softmax)
    plt.xticks(range(10),class_names)
    plt.xlabel("class")
    plt.ylabel("probability")
    plt.title("probability of esch class")
    plt.show()
    
def reset():
    image_space4.delete()
 
def show_img():
    dog_dir = []
    cat_dir = []
    for img in os.listdir("inference_dataset/Dog"):
        dog_dir.append(os.path.join("inference_dataset/Dog",img))
    for img in os.listdir("inference_dataset/Cat"):
        cat_dir.append(os.path.join("inference_dataset/Cat",img))    
        
    resize_fn = transforms.Compose([
                    transforms.Resize([224,224]),
                    ])
    
    plt.figure(figsize=(10,20))
    dog_img = Image.open(dog_dir[0])
    dog_img = resize_fn(dog_img)
    ax = plt.subplot(1,2,1)
    ax.imshow(dog_img)
    ax.set_title("dog")
    cat_img = Image.open(cat_dir[0])
    cat_img = resize_fn(cat_img)
    ax = plt.subplot(1,2,2)
    ax.imshow(cat_img)
    ax.set_title("cat")
    plt.show()
    
def show_ResNet_model():
    model = models.resnet50()
    model.fc = nn.Sequential(nn.Linear(2048,1),nn.Sigmoid())
    summary(model,(3,244,244))

def ResNet_acc_compare():
    img = cv.imread("compare.png")
    cv.imshow("accurracy comparison",img)

def put_img():
    global img5,file_path5
    file_path5, _ = QFileDialog.getOpenFileName()
    pixmap = QPixmap(file_path5)
    pixmap = pixmap.scaled(224,224)
    image_space5.setPixmap(pixmap)
    img5 = cv.imread(file_path5)

def inference():
    #img setting
    transform = transforms.ToTensor()
    test_data = transform(img5)
    test_data = test_data.unsqueeze(0)
    
    #train setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    model.load_state_dict(torch.load("model/resnet_better_model.pth"))
    idx = torch.randint(1, len(test_data), (1,))
    sample = torch.unsqueeze(test_data[idx][0], dim=0).to(device)

    if torch.sigmoid(model(sample)) < 0.5:
        string.setText("Prediction : Cat")
    else:
        string.setText("Prediction : Dog")
 
 
#initialization
file_path5 = ""
app = QApplication([])
window = QWidget()
layout = QHBoxLayout()
glayout1 = QVBoxLayout()
glayout2 = QVBoxLayout()
first_button = QPushButton("Load image")
groupbox1 = QGroupBox("1. Hough Circle Transform")
button1_1 = QPushButton("1.1 Draw Contour")
button1_2 = QPushButton("1.2 Count Coins")
string1 = QLabel("There are _coins in the image.")
groupbox2 = QGroupBox("2. Histogram Equalization")
button2_1 = QPushButton("2. Histogram Equalization")
groupbox3 = QGroupBox("3. Morphology Operation")
button3_1 = QPushButton("3.1 Closing")
button3_2 = QPushButton("3.2 Opening")
groupbox4 = QGroupBox("4. MNIST Classifier Using VGG19")
left_layout4 = QVBoxLayout()
right_layout4 = QVBoxLayout()
button4_1 = QPushButton("4.1 Show Model Structure")
button4_2 = QPushButton("4.2 Show Accuracy and Loss")
button4_3 = QPushButton("4.3 Predict")
button4_4 = QPushButton("4.4 Reset")
output4 = QLabel("")
image_space4 = DrawBoard()
image_space4.setFixedSize(800,400)
image_space4.setStyleSheet("background-color : black;")
groupbox5 = QGroupBox("5. ResNet50")
left_layout5 = QVBoxLayout()
right_layout5 = QVBoxLayout()
button5_0 = QPushButton("Load Image")
button5_1 = QPushButton("5.1 Show Images")
button5_2 = QPushButton("5.2 Show Model Structure")
button5_3 = QPushButton("5.3 Show Accuracy and Loss")
button5_4 = QPushButton("5.4 Inference")
image_space5 = QLabel("image")
pixmap = QPixmap(file_path5)
image_space5.setPixmap(pixmap)
string = QLabel("Predicted : ")

#setting
layout1 = QVBoxLayout(groupbox1)
layout1.addWidget(button1_1)
layout1.addWidget(button1_2)
layout1.addWidget(string1)
layout2 = QVBoxLayout(groupbox2)
layout2.addWidget(button2_1)
layout3 = QVBoxLayout(groupbox3)
layout3.addWidget(button3_1)
layout3.addWidget(button3_2)
layout4 = QHBoxLayout(groupbox4)
left_layout4.addWidget(button4_1)
left_layout4.addWidget(button4_2)
left_layout4.addWidget(button4_3)
left_layout4.addWidget(button4_4)
left_layout4.addWidget(output4)
right_layout4.addWidget(image_space4)
layout4.addLayout(left_layout4)
layout4.addLayout(right_layout4)
layout5 = QHBoxLayout(groupbox5)
left_layout5.addWidget(button5_0)
left_layout5.addWidget(button5_1)
left_layout5.addWidget(button5_2) 
left_layout5.addWidget(button5_3)
left_layout5.addWidget(button5_4)
right_layout5.addWidget(image_space5)
layout5.addLayout(left_layout5)
layout5.addLayout(right_layout5)
groupbox1.setLayout(layout1)
groupbox2.setLayout(layout2)
groupbox3.setLayout(layout3)
groupbox4.setLayout(layout4)
# groupbox4.setStyleSheet("background-color : black;")
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
button1_1.clicked.connect(draw_counter)
button1_2.clicked.connect(count_coins)
button2_1.clicked.connect(histogram_equ)
button3_1.clicked.connect(closing)
button3_2.clicked.connect(opening)
button4_1.clicked.connect(show_VGG_model)
button4_2.clicked.connect(VGG_acc_and_loss)
button4_3.clicked.connect(predict)
button4_4.clicked.connect(reset)
button5_0.clicked.connect(put_img)
button5_1.clicked.connect(show_img)
button5_2.clicked.connect(show_ResNet_model)
button5_3.clicked.connect(ResNet_acc_compare)
button5_4.clicked.connect(inference)

#output
# window.setStyleSheet("background-color : black;")
window.show()
app.exec_()