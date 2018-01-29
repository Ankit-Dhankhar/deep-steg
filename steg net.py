# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:02:04 2018

@author: ankit
"""

from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image
import matplotlib.pyplot as plt
import os
import random 
import scipy.misc
from skimage import io, transform
from tqdm import *
%matplotlib inline
#%%
cuda=False
#%%
### Constants ###
DATA_DIR = "./data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

IMG_SHAPE = (64, 64)
#%%
def load_dataset_small(num_images_per_class_train=10, num_images_test=500):
    """Loads training and test datasets, from Tiny ImageNet Visual Recogition Challenge.

    Arguments:
        num_images_per_class_train: number of images per class to load into training dataset.
        num_images_test: total number of images to load into training dataset.
    """
    X_train = []
    X_test = []
    
    # Create training set.
    for c in os.listdir(TRAIN_DIR):
        c_dir = os.path.join(TRAIN_DIR, c, 'images')
        c_imgs = os.listdir(c_dir)
        #print(len(c_imgs))
        random.shuffle(c_imgs)
        num_images= num_images_per_class_train
        
        while img_name_i in c_imgs[0:num_images]:
            img_i = io.imread(os.path.join(c_dir, img_name_i))
            img_i=img_i.astype(np.float)           
            if img_i.shape ==(64,64,3):            
                X_train.append(img_i)
                # print("Image added " + str(img_i.shape))
            else:
                num_images+=1
                # print("Image not added " + str(img_i.shape))
            #print(img_i.shape)
        print (len(X_train))
        print(num_images)
    random.shuffle(X_train)
    
    # Create test set.
    test_dir = os.path.join(TEST_DIR, 'images')
    test_imgs = os.listdir(test_dir)
    random.shuffle(test_imgs)
    for img_name_i in test_imgs[0:num_images_test]:
        img_i = io.imread(os.path.join(test_dir, img_name_i))
        img_i  = img_i.astype(np.float)        
        if img_i.shape ==(64,64,3):
            X_test.append(img_i)
            #print("Image addded " +str(img_i.shape))
        else:
            num_images_test+=1
            #print("Image not added "+ str(img_i.shape))
    # Return train and test data as numpy arrays.
    #X_train =np.array(X_train)
    # print(X_train.shape) 
    return np.array(X_train), np.array(X_test)
#%%
# Load dataset.
X_train_orig, X_test_orig = load_dataset_small(10,500)
#%%
# Normalize image vectors.
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Print statistics.
print ("Number of training examples = " + str(X_train.shape[0]))
print ("Number of test examples = " + str(X_train.shape[0]))
print ("X_train shape: " + str(X_train.shape)) # Should be (train_size, 64, 64, 3).
#%%
# We split training set into two halfs.
# First half is used for training as secret images, second images, second hald for cover images.

# S: secret image
input_S = X_train[0:X_train.shape[0] //2]

# C: cover image
input_C = X_train[X_train.shape[0] //2]
#%%

# Show samples images from the training dataset
fig = plt.figure(figsize =(8,8))
columns =4
rows =5
for i in range(1,columns*rows +1):
    #Randomly samples from training dataset
    img_idx = np.random.choice(X_train.shape[0])
    fig.add_subplot(rows, columns ,i)
    plt.imshow(X_train[img_idx])
plt.show()
#%%
# variable used to weight the loss of the secret and cover images (See paper for more details)
beta =1.0

# Loss for reveal network 
class rev_loss(nn.Module):
    def __init__(self):
        super(rev_loss,self).__init__()
    
    def forward(self,s_true,y_pred,beta):
        return  beta * torch.sum(torch.mul((s_true-s_pred),(s_true,s_pred)))

#Loss for the full model, used for preparation and hidding networks
class full_loss(nn.Module):
    def __init__(self):
        super(full_loss,self).__init__()
        
    def forward(self,y_true,y_pred):
        s_true, c_true = y_true[...,0:3], y_true[...,3:6]
        s_pred, c_pred = y_pred[...,0:3], y_true[...,3:6]
        s_loss = rev_loss(s_true, s_pred)
        c_loss = torch.sum(torch.mul(c_true - c_pred))
    
        return s_loss + c_loss

    
# Return the encoder as Keras model, composed by Preparation and Hiding Networks.
class make_encoder(nn.Module):
    def __init__(self):
        super(make_encoder, self).__init__()
        #Preparation Network
        self.pad=(0,1,0,1)
        self.conv1 = nn.Conv2d(3 ,50, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(3,10, kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d(3,5, kernel_size =5 , padding=2)
        self.conv4 = nn.Conv2d(65,50, kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(65,10, kernel_size=4, padding=1)
        self.conv6 = nn.Conv2d(65,5, kernel_size =5 , padding=2)
        # Hidden network
        self.conv7 = nn.Conv2d(65,50, kernel_size=3,padding=1)
        self.conv8 = nn.Conv2d(65,10, kernel_size=4, padding=1)
        self.conv9 = nn.Conv2d(65,5, kernel_size =5 , padding=2)
        self.conv10 = nn.Conv2d(65,50, kernel_size=3,padding=1)
        self.conv11 = nn.Conv2d(65,10, kernel_size=4, padding=1)
        self.conv12 = nn.Conv2d(65,5, kernel_size =5 , padding=2)
        self.conv13 = nn.Conv2d(65,50, kernel_size=3,padding=1)
        self.conv14 = nn.Conv2d(65,10, kernel_size=4, padding=1)
        self.conv15 = nn.Conv2d(65,5, kernel_size =5 , padding=2)
        self.conv16 = nn.Conv2d(65,50, kernel_size=3,padding=1)
        self.conv17 = nn.Conv2d(65,10, kernel_size=4, padding=1)
        self.conv18 = nn.Conv2d(65,5, kernel_size =5 , padding=2)
        self.conv19 = nn.Conv2d(65,50, kernel_size=3,padding=1)
        self.conv20 = nn.Conv2d(65,10, kernel_size=4, padding=1)
        self.conv21 = nn.Conv2d(65,5, kernel_size =5 , padding=2)
        self.conv22 = nn.Conv2d(65,3, kernel_size=3,padding=1)

    def forward(self,x,y):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x2 = F.pad(x2,pad,'constant',0)
        x3 =F.relu(self.conv3(x))
        x4 = torch.cat([x1,x2,x3])
        x1 = F.relu(self.conv4(x4))
        x2 = F.relu(self.conv5(x4))
        x2 = F.pad(x2,pad,'constant',0)
        x3 =F.relu(self.conv6(x4))            
        x4=torch.cat([x1,x2,x3])            
        x = torch.cat([y,x4])
        x1 = F.relu(self.conv7(x4))
        x2 = F.relu(self.conv8(x4))
        x2 = F.pad(x2,pad,'constant',0)
        x3 =F.relu(self.conv9(x4))
        x4 = torch.cat([x1,x2,x3])
        x1 = F.relu(self.conv10(x4))
        x2 = F.relu(self.conv11(x4))
        x2 = F.pad(x2,pad,'constant',0)
        x3 =F.relu(self.conv12(x4))
        x4 = torch.cat([x1,x2,x3])
        x1 = F.relu(self.conv13(x4))
        x2 = F.relu(self.conv14(x4))
        x2 = F.pad(x2,pad,'constant',0)
        x3 =F.relu(self.conv15(x4))
        x4 = torch.cat([x1,x2,x3])
        x1 = F.relu(self.conv16(x4))
        x2 = F.relu(self.conv17(x4))
        x2 = F.pad(x2,pad,'constant',0)
        x3 =F.relu(self.conv18(x4))
        x4 = torch.cat([x1,x2,x3])
        x1 = F.relu(self.conv19(x4))
        x2 = F.relu(self.conv20(x4))
        x2 = F.pad(x2,pad,'constant',0)
        x3 =F.relu(self.conv21(x4))
        x4 = torch.cat([x1,x2,x3])
        output =f.relu(self.conv22(x4))
        return output
            
#%%
class make_decoder(nn.Module):
    def __init__(self):
        super(make_decoder, self).__init__()
        self.conv1 = nn.Conv2d(3,50, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(3,10, kernel_size=4, padding=2)
        self.conv3 = nn.Conv2d(3,5, kernel_size =5 , padding=2)
        self.conv4 = nn.Conv2d(65,50, kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(65,10, kernel_size=4, padding=2)
        self.conv6 = nn.Conv2d(65,5, kernel_size =5 , padding=2)
        self.conv7 = nn.Conv2d(65,50, kernel_size=3,padding=1)
        self.conv8 = nn.Conv2d(65,10, kernel_size=4, padding=2)
        self.conv9 = nn.Conv2d(65,5, kernel_size =5 , padding=2)
        self.conv10 = nn.Conv2d(65,50, kernel_size=3,padding=1)
        self.conv11 = nn.Conv2d(65,10, kernel_size=4, padding=2)
        self.conv12 = nn.Conv2d(65,5, kernel_size =5 , padding=2)
        self.conv13 = nn.Conv2d(65,50, kernel_size=3,padding=1)
        self.conv14 = nn.Conv2d(65,10, kernel_size=4, padding=2)
        self.conv15 = nn.Conv2d(65,5, kernel_size =5 , padding=2)
        self.conv16 = nn.Conv2d(65,3, kernel_size=3,padding=1)
        
    def forward(self,x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x2 = F.pad(x2,pad,'constant',0)
        x3 =F.relu(self.conv3(x))
        x4 = torch.cat([x1,x2,x3])
        x1 = F.relu(self.conv4(x4))
        x2 = F.relu(self.conv5(x4))
        x2 = F.pad(x2,pad,'constant',0)
        x3 =F.relu(self.conv6(x4))            
        x4=torch.cat([x1,x2,x3])
        x1 = F.relu(self.conv7(x4))
        x2 = F.relu(self.conv8(x4))
        x2 = F.pad(x2,pad,'constant',0)
        x3 =F.relu(self.conv9(x4))
        x4 = torch.cat([x1,x2,x3])
        x1 = F.relu(self.conv10(x4))
        x2 = F.relu(self.conv11(x4))
        x2 = F.pad(x2,pad,'constant',0)
        x3 =F.relu(self.conv12(x4))
        x4 = torch.cat([x1,x2,x3])
        x1 = F.relu(self.conv13(x4))
        x2 = F.relu(self.conv14(x4))
        x2 = F.pad(x2,pad,'constant',0)
        x3 =F.relu(self.conv15(x4))
        x4 = torch.cat([x1,x2,x3])
        output = F.relu(self.conv16(x4))
           
#%%
class make_model(nn.Module):
    def __init__(self):
        super(make_model,self).__init__()
        self.encoder=make_encoder()
        self.decoder=make_decoder()
        
        
    def forward(self,input_S,input_C):
        output_Cprime =encoder(input_S,input_C)
        output_Sprime =decoder(output_Cprime)
        autoencoder=torch.cat([output_Sprime,output_Cprime])
        return autoencoder
#%%
model =make_model()
#%%

for m in model.modules():
    print(m)
        

#%%
if cuda:
    model.cuda()
optimizer = optim.Adam([{'params':model.parameters()}],lr=0.001)

NB_EPOCHS=1000
BATCH_SIZE=32
m=input_S.shape[0]
loss_history=[]
for  epoch in range(NB_EPOCHS):
    np.random.shuffle(input_S)
    np.random.shuffle(input_C)
    
    t = tqdm(range(0,input_S.shape[0],BATCH_SIZE),miniinterval=0)
    ae_loss = []
    rev_loss = []
    for indx in t :
        batch_S =input_S[idx:min(idx+Batch_SIZE,m)]
        batch_C = input_C[idx:min(idx+ BATCH_SIZE,m)]
        
        C_prime =model.encoder
        if iteration ==200:
            for param_group in optimizer.param_groups:
                param_group['lr'] =0.0003
        if  iteration == 400:
            for param_group in optimizer.param_groups:
                param_group['lr'] =0.0001
        
        if iteration==600:
            for param_group in optimizer.param_groups:
                param_groups['lr'] = 0.00003
        
#%%
