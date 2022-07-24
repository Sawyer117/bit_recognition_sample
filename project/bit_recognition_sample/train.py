# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 14:20:16 2022

@author: 36284
"""
#
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.models as models
from tqdm import tqdm

import os
#import pandas as pd
from torchvision.io import read_image
import numpy as np


EPOCH = 150
best_acc = 0

learning_rate = 0.0001 # was 0.0001
#transforms.RandomRotation(10),
image_transforms = {"train": transforms.Compose([transforms.Resize((128, 128)),
                                                 transforms.RandomHorizontalFlip(0.1),
                                                 transforms.RandomVerticalFlip(p=0.1),
                                                 transforms.ToTensor()]),
                    "test": transforms.Compose([transforms.Resize((128, 128)),transforms.ToTensor()])}

train_data = datasets.ImageFolder("D:/pytorch-project/data/bit_recog_data/train", transform=image_transforms["train"])
test_data = datasets.ImageFolder("D:/pytorch-project/data/bit_recog_data/validation", transform=image_transforms["test"])


train_loader = torch.utils.data.DataLoader(train_data, batch_size=1,shuffle=True)
val_loader = torch.utils.data.DataLoader(test_data, batch_size=1)


model = models.resnet50(pretrained=True)

# Finetune convertion
num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, 2)

#criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                     momentum=0.9, weight_decay=5e-4)
train_losses = []

accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}


# TRAINING
def train(epoch):
    model.train()
    correct = 0
    total = 0
    train_epoch_loss = 0
    loop = tqdm(train_loader,total = len(train_loader))
    for batch_idx, (x_train_batch, y_train_batch) in enumerate(loop):
        optimizer.zero_grad()
        output = model(x_train_batch) 
        train_loss = criterion(output, y_train_batch)
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()
        _, predicted = output.max(1)
        #print("\nprediction: {}  ground truth: {}".format(predicted.item(), y_train_batch.item()))
        total += y_train_batch.size(0)
        correct += predicted.eq(y_train_batch).sum().item()
        acc = correct/total
        
        loop.set_description(f"Epoch [{epoch}/{EPOCH}]")
        loop.set_postfix(loss=(train_epoch_loss/(batch_idx+1)), accuracy=acc)
    
# VALIDATION
def validation():
    global best_acc
    model.eval()
    val_epoch_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (x_val_batch, y_val_batch) in enumerate(val_loader):
            output = model(x_val_batch) 
            val_loss = criterion(output, y_val_batch)
            val_epoch_loss += val_loss.item()
            _, predicted = output.max(1)
            total += y_val_batch.size(0)
            correct += predicted.eq(y_val_batch).sum().item()
    acc = correct/total
    print("val acc: {}".format(acc))
    print("best acc: {}".format(best_acc))
    if acc > best_acc:
        best_acc = acc
        # add saving logic here
        torch.save(model, "./model_best.pth")

if __name__ == "__main__":
    for epoch in range(0, EPOCH):
        train(epoch)
        # val and save every 5 epochs
        if epoch%5 == 0:
            validation()
    torch.save(model, "./model_final.pth")