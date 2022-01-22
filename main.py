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
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtCore,QtGui

from main_ui import Ui_BitRecognition

# Only needed for access to command line arguments
import sys
from PIL import ImageQt
import torch
import numpy as np

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
        PATH = "./model.pth"
        self.model = torch.load(PATH)
        print(self.model)
    #%%
        #buttons and events
        self.LoadImage_button.clicked.connect(self.openFileNameDialog)
        self.Inference_button.clicked.connect(self.inference)
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
           print(img_label_coordinates)
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
            self.pixmap = QPixmap(fileName)
            self.pixmap = self.pixmap.scaled(self.ImgLabel.width(), self.ImgLabel.height(), QtCore.Qt.KeepAspectRatio,QtCore.Qt.FastTransformation)
            #pixmap = pixmap.scaled(self.ImgLabel.width(), self.ImgLabel.height())
            self.img_rect = self.pixmap.rect()
            self.ImgLabel.setPixmap(self.pixmap)
            self.ImgNameTextEdit.setPlainText(os.path.basename(fileName)+" \nLoaded Sucessfully")
        else: self.ImgNameTextEdit.setPlainText("Invalid File or File not selected")
            #print(fileName)
    #%%        
    def inference(self):
        print(self.cropQPixmap)
        if self.cropQPixmap:
            PIL_image = ImageQt.fromqpixmap(self.cropQPixmap)
            PIL_image = PIL_image.resize((128,128))
        else:
            self.ImgNameTextEdit.setPlainText("no ROI selected")  
            return
        input_np_array = np.rollaxis(np.array(PIL_image),2,0)[None,...]
        print(input_np_array.shape)
        prediction = self.model(torch.from_numpy(input_np_array).float())
        self.probability_TextEdit.setPlainText("Probability: {}".format(prediction.item())) 
            
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