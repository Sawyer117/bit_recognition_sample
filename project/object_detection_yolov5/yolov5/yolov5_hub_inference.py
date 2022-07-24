# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 22:51:47 2022

@author: wen

# not working as 0318/2022

"""


import torch

# Model
path = "D:/pytorch-project/Gear_Recognition/object_detection_yolov5/yolov5"
model = torch.hub.load(path, 'yolov5s',pretrained=False,force_reload=True)  # or yolov5m, yolov5l, yolov5x, custom

# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()