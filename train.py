# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 14:20:16 2022

@author: 36284
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.models as models
from tqdm.notebook import tqdm

import os
#import pandas as pd
from torchvision.io import read_image
import numpy as np

'''
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
'''


learning_rate = 0.0005
#image_transforms = {"train": transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()]),
#                    "test": transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])}

image_transforms = {"train": transforms.Compose([transforms.Resize((128, 128)),transforms.ToTensor()]),
                    "test": transforms.Compose([transforms.Resize((128, 128)),transforms.ToTensor()])}

train_data = datasets.ImageFolder("D:/playground/gui_bit_recog/data/train", transform=image_transforms["train"])
test_data = datasets.ImageFolder("D:/playground/gui_bit_recog/data/test", transform=image_transforms["test"])


train_loader = torch.utils.data.DataLoader(train_data, batch_size=1,shuffle=True)
val_loader = torch.utils.data.DataLoader(test_data, batch_size=1)


model = models.resnet18(pretrained=True)

#print(model)
model.fc = nn.Linear(512, 1)
model = nn.Sequential(
    model,
    nn.Sigmoid()
)
print(model)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = nn.Sigmoid(x)
        return x


#model = Net()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
train_losses = []
'''
for epoch in range(1, num_epochs=15):
    train_loss = 0.0
    model.train()
    for data, target in train_loader:
  
        optimizer.zero_grad()
        #forward-pass
        output = model(data)
        loss = criterion(output, target)
        #backward-pass
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Update the Training loss
        train_loss += loss.item() * data.size(0)
'''
def acc(y_pred, y_test):
    acc = 1-abs(y_pred-y_test)/y_test
    return acc
#def binary_acc(y_pred, y_test):
#    #y_pred_tag = torch.log_softmax(y_pred, dim = 0)
#    #_, y_pred_tags = torch.max(y_pred_tag, dim = 0)
#    y_pred_tag = torch.log_softmax(y_pred)
#    _, y_pred_tags = torch.max(y_pred_tag)
#    correct_results_sum = (y_pred_tags == y_test).sum().float()
#    acc = correct_results_sum/y_test.shape[0]
#    acc = torch.round(acc * 100)
#    return acc

accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}

#for e in range(1, 15):

# TRAINING
train_epoch_loss = 0
train_epoch_acc = 0
model.train()
for i in range(0,30):
  for X_train_batch, y_train_batch in train_loader:
    y_train_batch = y_train_batch
    #X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
    optimizer.zero_grad()
    y_train_pred = model(X_train_batch).view(-1)
    #print("prediction: {}".format(y_train_pred.size()))
    #print(y_train_pred)
    #print("ground truth: {}".format(y_train_batch.size()))
    #print(y_train_batch)
       
    train_loss = criterion(y_train_pred, y_train_batch.float())
    #train_loss = criterion(y_train_batch.float(), y_train_batch.float())
    train_acc = acc(y_train_pred, y_train_batch)
    print("pred: {}".format(y_train_pred))
    train_loss.backward()
    optimizer.step()
    train_epoch_loss += train_loss.item()
    train_epoch_acc += train_acc.item()
    print("train loss: {}".format(train_loss))
    print("train acc: {}\n".format(train_acc))
'''
# VALIDATION
with torch.no_grad():
    model.eval()
    val_epoch_loss = 0
    val_epoch_acc = 0
    for X_val_batch, y_val_batch in val_loader:
        #X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
        y_val_pred = model(X_val_batch).squeeze()
        y_val_pred = torch.unsqueeze(y_val_pred, 0)
        val_loss = criterion(y_val_pred, y_val_batch.float())
        val_acc = acc(y_val_pred, y_val_batch)
        val_epoch_loss += val_loss.item()
        val_epoch_acc += val_acc.item()
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
        #print(f'Epoch {e+0:02}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')
'''
torch.save(model, "./model.pth")