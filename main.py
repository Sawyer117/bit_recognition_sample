# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 12:01:20 2022

@author: 36284
"""


import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QMessageBox, QWidget, QFileDialog, QRubberBand
)
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon, QPixmap, QImage  
from PyQt5 import QtCore,QtGui

from main_ui_0424 import Ui_BitRecognition

# Only needed for access to command line arguments
import sys
from PIL import ImageQt
import torch
import numpy as np

# Library and function for image recognition
import torch
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device, time_sync
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
from utils.augmentations import letterbox
from PIL import ImageQt
def hex2rgb(h):  # rgb order (PIL)
    return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
       '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')

def qtpixmap_to_cvimg(qtpixmap):

    qimg = qtpixmap.toImage()
    temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
    temp_shape += (4,)
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    result = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
    result = result[..., :3]

    return result


# You need one (and only one) QApplication instance per application.
# Pass in sys.argv to allow command line arguments for your app.
# If you know you won't use command line arguments QApplication([]) works too.
#app = QApplication(sys.argv)
class Window(QMainWindow, Ui_BitRecognition):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        
    #%%
        # global variables
        self.img_path = 0
        self.img_rect = QtCore.QRect(0,0,0,0)
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = QtCore.QPoint()
        self.cropQPixmap = None
        PATH = "./model/model_final.pth"
        self.model = torch.load(PATH)
        print(self.model)
    #%%
        #buttons and events
        self.LoadImage_button.clicked.connect(self.openFileNameDialog)
        self.Inference_button.clicked.connect(self.inference)
        self.Detection_button.clicked.connect(self.inference_recognition)
        #self.connectSignalsSlots()
    #%%    
    def mousePressEvent(self, event):
        if self.pixmap:
           if event.button() == QtCore.Qt.LeftButton:
           
               self.origin = QtCore.QPoint(event.pos())
               self.rubberBand.setGeometry(QtCore.QRect(self.origin, QtCore.QSize()))
               self.rubberBand.show()
       
    def mouseMoveEvent(self, event):
       
           if not self.origin.isNull():
               self.rubberBand.setGeometry(QtCore.QRect(self.origin, event.pos()).normalized())
       
    def mouseReleaseEvent(self, event):
       
           if event.button() == QtCore.Qt.LeftButton:
               self.rubberBand.hide()
           currentQRect = self.rubberBand.geometry()
           
           #calculate image relative position:
           img_label_coordinates = [self.ImgLabel.geometry().x(),self.ImgLabel.geometry().y(),self.ImgLabel.geometry().width(),self.ImgLabel.geometry().height()]
           #print(img_label_coordinates)
           #image_origin_pos = [20+int((820-self.img_rect.width())/2),30+int((600-self.img_rect.height())/2)]
           image_origin_pos = [img_label_coordinates[0]+int((img_label_coordinates[2]-self.img_rect.width())/2),img_label_coordinates[1]+int((img_label_coordinates[3]-self.img_rect.height())/2)]
           true_rect = QtCore.QRect(currentQRect.x()-image_origin_pos[0],currentQRect.y()-image_origin_pos[1],currentQRect.width(),currentQRect.height())
           #self.rubberBand.deleteLater()
           self.cropQPixmap = self.pixmap.copy(true_rect)
           self.cropQPixmap = self.cropQPixmap.scaled(self.CropLabel.width(), self.CropLabel.height(), QtCore.Qt.KeepAspectRatio,QtCore.Qt.FastTransformation)
           self.CropLabel.setPixmap(self.cropQPixmap)
           
           '''
           #delete later, only used for data generation
           
           filename = None
           for i in range(0,1000):
               if not os.path.isfile("D:/playground/gui_bit_recog/data/{}".format(i)+'.png'):
                   filename = "D:/playground/gui_bit_recog/data/{}".format(i)+".png"
                   break
           PIL_image = ImageQt.fromqpixmap(cropQPixmap)
           PIL_image.save(filename)
           '''
  
        
    #%%    
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Image Files (*.jpg,*.png,*jpeg)", options=options)
        if fileName:
            self.img_path = fileName
            self.pixmap_original = QPixmap(fileName)
            self.pixmap = self.pixmap_original.scaled(self.ImgLabel.width(), self.ImgLabel.height(), QtCore.Qt.KeepAspectRatio,QtCore.Qt.FastTransformation)
            #pixmap = pixmap.scaled(self.ImgLabel.width(), self.ImgLabel.height())
            self.img_rect = self.pixmap.rect()
            self.ImgLabel.setPixmap(self.pixmap)
            self.ImgNameTextEdit.setPlainText(os.path.basename(fileName)+" \nLoaded Sucessfully")
        else: self.ImgNameTextEdit.setPlainText("Invalid File or File not selected")
            #print(fileName)
    #%%        
    def inference(self):
        #print(self.cropQPixmap)
        if self.cropQPixmap:
            PIL_image = ImageQt.fromqpixmap(self.cropQPixmap)
            PIL_image = PIL_image.resize((128,128))
        else:
            self.ImgNameTextEdit.setPlainText("no ROI selected")  
            return
        input_np_array = np.rollaxis(np.array(PIL_image),2,0)[None,...]
        #print(input_np_array.shape)
        output = self.model(torch.from_numpy(input_np_array).float())
        _, prediction = output.max(1)
        self.probability_TextEdit.setPlainText("Current mode: image classification\n\nObject Information:\n   Probability of being bits: {}".format(prediction.item())) 
    #%%
    # image recognition inference code 
    ###########################################################################
    def inference_recognition(self): 
        self.probability_TextEdit.setPlainText("Current mode: image recognition\n\nObject Information:\n   Probability of being bits: {}".format(1)) 
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
        # candidate model: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x
        weights = ROOT / 'weights/best.pt'  
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
        img_show = qtpixmap_to_cvimg(self.pixmap_original)  
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
        #print(pred[0].shape)
        t4 = time_sync()
        dt[2] = t4 - t3
        # Process predictions
        # pred after NMS consists of [n,6] elements, whereas n = nums of detected samples
        # 6 consists of [[x1,y1,x2,y2], conf, label] whereas conf = probability of pred, label = class num
    
        # 1. recover actual x1,y1,x2,y2 with respect to original pictures
        x,y,x2,y2 = 0,0,0,0
        img_drawn = img_show.copy() 
        for i in range(pred[0].shape[0]):
            x,y,x2,y2 = pred[0].data[i][0].item(), pred[0].data[i][1].item(), pred[0].data[i][2].item(),pred[0].data[i][3].item()
            x,y = int(x*img_show.shape[1]/img_raw.shape[2]), int(y*img_show.shape[0]/img_raw.shape[1]) 
            x2,y2 = int(x2*img_show.shape[1]/img_raw.shape[2]), int(y2*img_show.shape[0]/img_raw.shape[1])
            
            # draw rectangle of object
            img_drawn = cv2.rectangle(img_drawn,(x,y),(x2,y2),color,line_thickness) #(0,255,0)
            # draw txt
            txt = "{:.3f}".format(pred[0].data[i][4].item())
            # draw background of txt 
            (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_thickness)
            img_drawn = cv2.rectangle(img_drawn, (x, y - 20), (x + w, y), color, -1)
            img_drawn = cv2.putText(img_drawn, txt, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, lineType=cv2.LINE_AA)
    
        
        img_final = cv2.cvtColor(img_drawn, cv2.COLOR_BGR2RGB)
        #im_pil = Image.fromarray(img_final)  
        
        height, width, channel = img_final.shape
        bytesPerLine = 3 * width
        qImg = QImage(img_final.data, width, height, bytesPerLine, QImage.Format_RGB888)

        im_qt= qImg.scaled(self.ImgLabel.width(), self.ImgLabel.height(), QtCore.Qt.KeepAspectRatio,QtCore.Qt.FastTransformation)
        self.ImgLabel.setPixmap(QPixmap(im_qt))
    
#%%
# Create a Qt widget, which will be our window.
#window = QWidget()
#window.show()  # IMPORTANT!!!!! Windows are hidden by default.

# Start the event loop.
#app.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())