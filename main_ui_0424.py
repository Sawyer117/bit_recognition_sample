# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
# H.J. Eysenck
#Die wissenschäftliche Erforschung der Astrologie und die Förderung nach ‘naïven’ Versuchspersonen
#Zeitschrift für Parapsychologie und Grenzgebiete der Psychologie
#(1981)
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_BitRecognition(object):
    def setupUi(self, BitRecognition):
        BitRecognition.setObjectName("BitRecognition")
        BitRecognition.resize(1253, 812)
        self.groupBox = QtWidgets.QGroupBox(BitRecognition)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 861, 641))
        self.groupBox.setObjectName("groupBox")
        self.ImgLabel = QtWidgets.QLabel(self.groupBox)
        self.ImgLabel.setGeometry(QtCore.QRect(20, 30, 820, 600))
        self.ImgLabel.setText("")
        self.ImgLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.ImgLabel.setObjectName("ImgLabel")
        self.groupBox_2 = QtWidgets.QGroupBox(BitRecognition)
        self.groupBox_2.setGeometry(QtCore.QRect(910, 30, 300, 256))
        self.groupBox_2.setObjectName("groupBox_2")
        self.CropLabel = QtWidgets.QLabel(self.groupBox_2)
        self.CropLabel.setGeometry(QtCore.QRect(10, 40, 281, 201))
        self.CropLabel.setText("")
        self.CropLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.CropLabel.setObjectName("CropLabel")
        self.CropImage_button = QtWidgets.QPushButton(BitRecognition)
        self.CropImage_button.setGeometry(QtCore.QRect(200, 670, 141, 41))
        self.CropImage_button.setObjectName("CropImage_button")
        self.Inference_button = QtWidgets.QPushButton(BitRecognition)
        self.Inference_button.setGeometry(QtCore.QRect(370, 670, 141, 41))
        self.Inference_button.setObjectName("Inference_button")
        self.groupBox_3 = QtWidgets.QGroupBox(BitRecognition)
        self.groupBox_3.setGeometry(QtCore.QRect(910, 300, 300, 351))
        self.groupBox_3.setObjectName("groupBox_3")
        self.probability_TextEdit = QtWidgets.QPlainTextEdit(self.groupBox_3)
        self.probability_TextEdit.setGeometry(QtCore.QRect(10, 30, 261, 61))
        self.probability_TextEdit.setStyleSheet("background:transparent;\n"
"border:0")
        self.probability_TextEdit.setObjectName("probability_TextEdit")
        self.LoadImage_button = QtWidgets.QPushButton(BitRecognition)
        self.LoadImage_button.setGeometry(QtCore.QRect(30, 670, 141, 41))
        self.LoadImage_button.setObjectName("LoadImage_button")
        self.Detection_button = QtWidgets.QPushButton(BitRecognition)
        self.Detection_button.setGeometry(QtCore.QRect(530, 670, 141, 41))
        self.Detection_button.setObjectName("Detection_button")
        self.ImgNameTextEdit = QtWidgets.QPlainTextEdit(BitRecognition)
        self.ImgNameTextEdit.setGeometry(QtCore.QRect(30, 720, 301, 61))
        self.ImgNameTextEdit.setStyleSheet("background:transparent;\n"
"border:0")
        self.ImgNameTextEdit.setReadOnly(True)
        self.ImgNameTextEdit.setPlainText("")
        self.ImgNameTextEdit.setObjectName("ImgNameTextEdit")
        self.Clear_button = QtWidgets.QPushButton(BitRecognition)
        self.Clear_button.setGeometry(QtCore.QRect(680, 670, 141, 41))
        self.Clear_button.setObjectName("Clear_button")

        self.retranslateUi(BitRecognition)
        QtCore.QMetaObject.connectSlotsByName(BitRecognition)

    def retranslateUi(self, BitRecognition):
        _translate = QtCore.QCoreApplication.translate
        BitRecognition.setWindowTitle(_translate("BitRecognition", "BitRecognition"))
        self.groupBox.setTitle(_translate("BitRecognition", "Input Image"))
        self.groupBox_2.setTitle(_translate("BitRecognition", "Selected ROI"))
        self.CropImage_button.setText(_translate("BitRecognition", "Crop Image"))
        self.Inference_button.setText(_translate("BitRecognition", "Inference"))
        self.groupBox_3.setTitle(_translate("BitRecognition", "Inference Result"))
        self.probability_TextEdit.setPlainText(_translate("BitRecognition", "Object Information:\n"
""))
        self.LoadImage_button.setText(_translate("BitRecognition", "Load Image"))
        self.Detection_button.setText(_translate("BitRecognition", "Auto-Detection"))
        self.Clear_button.setText(_translate("BitRecognition", "Clear"))

