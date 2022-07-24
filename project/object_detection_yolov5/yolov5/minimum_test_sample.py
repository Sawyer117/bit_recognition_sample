# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 22:11:35 2022

@author: wen
"""
import torch
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device, time_sync
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
from utils.augmentations import letterbox

def hex2rgb(h):  # rgb order (PIL)
    return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
       '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
color = hex2rgb('#' + hex[6])

# Model Parameters
conf_thres=0.25  # confidence threshold
iou_thres=0.45
max_det=1000
classes=None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False

# Visualization Parameters
line_thickness=3


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
IMG_PATH = "D:/pytorch-project/Gear_Recognition/object_detection_yolov5/yolov5/data/images/zidane.jpg"
# candidate model: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x
weights = ROOT / 'yolov5s.pt'  
data=ROOT / 'data/coco128.yaml'
device = ''
device = select_device(device)
#model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)


stride, names, pt = model.stride, model.names, model.pt
imgsz=(640, 640)
imgsz = check_img_size(imgsz, s=stride)  # check image size
model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))  # warmup

# load data
dt = np.array([0.0, 0.0, 0.0, 0.0])
t1 = time_sync()
img_show = cv2.imread(IMG_PATH)  # BGR
# refer to ./utils/augmentations.py
img = letterbox(img_show, imgsz, stride=stride, auto=pt)[0]
img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
img = np.ascontiguousarray(img)

# make a copy for annotation
img_raw  = img.copy()

img = img[None,...]
img = torch.from_numpy(img).to(device)
img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
img /= 255  # 0 - 255 to 0.0 - 1.0
if len(img.shape) == 3:
    img = img[None]  # expand for batch dim
t2 = time_sync()
dt[0] = t2 - t1
# inference
pred = model(img, augment=False, visualize=False)
t3 = time_sync()
dt[1] = t3 - t2
# NMS- non max suppresion
pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
t4 = time_sync()
dt[2] = t4 - t3
# Process predictions
# pred after NMS consists of [n,6] elements, whereas n = nums of detected samples
# 6 consists of [[x1,y1,x2,y2], conf, label] whereas conf = probability of pred, label = class num

# 1. recover actual x1,y1,x2,y2 with respect to original pictures
x,y,x2,y2 = 0,0,0,0
img_drawn = img_show
for i in range(pred[0].shape[0]):
    x,y,x2,y2 = pred[0].data[i][0].item(), pred[0].data[i][1].item(), pred[0].data[i][2].item(),pred[0].data[i][3].item()
    x,y = int(x*img_show.shape[1]/img_raw.shape[2]), int(y*img_show.shape[0]/img_raw.shape[1]) 
    x2,y2 = int(x2*img_show.shape[1]/img_raw.shape[2]), int(y2*img_show.shape[0]/img_raw.shape[1])
    
    # draw rectangle of object
    img_drawn = cv2.rectangle(img_drawn,(x,y),(x2,y2),color,line_thickness) #(0,255,0)
    # draw txt
    txt = "{}: {:.3f}".format(names[int(pred[0].data[i][5].item())] ,pred[0].data[i][4].item())
    # draw background of txt 
    (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_thickness)
    img_drawn = cv2.rectangle(img_drawn, (x, y - 20), (x + w, y), color, -1)
    img_drawn = cv2.putText(img_drawn, txt, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, lineType=cv2.LINE_AA)

img_final = cv2.cvtColor(img_drawn, cv2.COLOR_BGR2RGB)
im_pil = Image.fromarray(img_final)
dt[3] = time_sync()- t4
dt = dt*1000
im_pil.show()
print("Speed: {:.2f}ms pre-process, {:.2f}ms inference, {:.2f}ms NMS , {:.2f}ms post-process per image at shape {}".format(dt[0],dt[1],dt[2],dt[3],list(img.size())))



