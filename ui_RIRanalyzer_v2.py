# -*- coding: utf-8 -*-

#
# Created by: PyQt5 UI code generator 5.9.2
#

from PyQt5 import QtCore, QtGui, QtWidgets

#QT Designer output code

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1037, 683)
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setFamily("Gadugi")
        font.setPointSize(14)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox_imput = QtWidgets.QGroupBox(Form)
        self.groupBox_imput.setObjectName("groupBox_imput")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_imput)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox_imput)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout_3.addWidget(self.pushButton_2, 1, 1, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.groupBox_imput)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout_3.addWidget(self.pushButton, 0, 1, 1, 1)
        self.radioButton_SiseSwp = QtWidgets.QRadioButton(self.groupBox_imput)
        self.radioButton_SiseSwp.setObjectName("radioButton_SiseSwp")
        self.gridLayout_3.addWidget(self.radioButton_SiseSwp, 1, 0, 1, 1)
        self.radioButton_IResponse = QtWidgets.QRadioButton(self.groupBox_imput)
        self.radioButton_IResponse.setObjectName("radioButton_IResponse")
        self.gridLayout_3.addWidget(self.radioButton_IResponse, 0, 0, 1, 1)
        self.horizontalLayout.addWidget(self.groupBox_imput)
        self.groupBox_stereo = QtWidgets.QGroupBox(Form)
        self.groupBox_stereo.setMaximumSize(QtCore.QSize(374, 16777215))
        self.groupBox_stereo.setObjectName("groupBox_stereo")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_stereo)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox_stereo)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout_5.addWidget(self.pushButton_3, 0, 1, 1, 1)
        self.checkBox = QtWidgets.QCheckBox(self.groupBox_stereo)
        self.checkBox.setObjectName("checkBox")
        self.gridLayout_5.addWidget(self.checkBox, 0, 0, 2, 1)
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox_stereo)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout_5.addWidget(self.pushButton_4, 1, 1, 1, 1)
        self.horizontalLayout.addWidget(self.groupBox_stereo)
        self.groupBox_filter = QtWidgets.QGroupBox(Form)
        self.groupBox_filter.setMaximumSize(QtCore.QSize(305, 16777215))
        self.groupBox_filter.setObjectName("groupBox_filter")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_filter)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.radioButton_Thirdoctave = QtWidgets.QRadioButton(self.groupBox_filter)
        self.radioButton_Thirdoctave.setObjectName("radioButton_Thirdoctave")
        self.gridLayout_4.addWidget(self.radioButton_Thirdoctave, 1, 0, 1, 1)
        self.checkBox_reversedRIR = QtWidgets.QCheckBox(self.groupBox_filter)
        self.checkBox_reversedRIR.setObjectName("checkBox_reversedRIR")
        self.gridLayout_4.addWidget(self.checkBox_reversedRIR, 1, 1, 1, 1)
        self.radioButton_Octaveband = QtWidgets.QRadioButton(self.groupBox_filter)
        self.radioButton_Octaveband.setObjectName("radioButton_Octaveband")
        self.gridLayout_4.addWidget(self.radioButton_Octaveband, 0, 0, 1, 1)
        self.horizontalLayout.addWidget(self.groupBox_filter)
        self.groupBox_MMFilter = QtWidgets.QGroupBox(Form)
        self.groupBox_MMFilter.setMaximumSize(QtCore.QSize(305, 16777215))
        self.groupBox_MMFilter.setObjectName("groupBox_MMFilter")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox_MMFilter)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.lineEdit_Wind_len = QtWidgets.QLineEdit(self.groupBox_MMFilter)
        self.lineEdit_Wind_len.setMaximumSize(QtCore.QSize(112, 16777215))
        self.lineEdit_Wind_len.setObjectName("lineEdit_Wind_len")
        self.gridLayout_6.addWidget(self.lineEdit_Wind_len, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox_MMFilter)
        self.label_2.setObjectName("label_2")
        self.gridLayout_6.addWidget(self.label_2, 0, 0, 1, 1)
        self.horizontalLayout.addWidget(self.groupBox_MMFilter)
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.pushButton_Analyze = QtWidgets.QPushButton(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_Analyze.setFont(font)
        self.pushButton_Analyze.setStyleSheet("background-color: rgb(170, 170, 255);")
        self.pushButton_Analyze.setObjectName("pushButton_Analyze")
        self.gridLayout_10.addWidget(self.pushButton_Analyze, 0, 0, 1, 1)
        self.pushButton_Cancel = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_Cancel.setObjectName("pushButton_Cancel")
        self.gridLayout_10.addWidget(self.pushButton_Cancel, 1, 0, 1, 1)
        self.horizontalLayout.addWidget(self.groupBox)
        self.pushButton_ExportResult = QtWidgets.QPushButton(Form)
        self.pushButton_ExportResult.setObjectName("pushButton_ExportResult")
        self.horizontalLayout.addWidget(self.pushButton_ExportResult)
        self.gridLayout_2.addLayout(self.horizontalLayout, 1, 0, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.graphicsView = QtWidgets.QGraphicsView(Form)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 0, 1, 1, 1)
        self.groupBox_Results = QtWidgets.QGroupBox(Form)
        self.groupBox_Results.setMinimumSize(QtCore.QSize(151, 301))
        self.groupBox_Results.setMaximumSize(QtCore.QSize(151, 301))
        self.groupBox_Results.setObjectName("groupBox_Results")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_Results)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox_7 = QtWidgets.QGroupBox(self.groupBox_Results)
        self.groupBox_7.setMaximumSize(QtCore.QSize(140, 80))
        self.groupBox_7.setTitle("")
        self.groupBox_7.setObjectName("groupBox_7")
        self.label_stereoOmono = QtWidgets.QLabel(self.groupBox_7)
        self.label_stereoOmono.setGeometry(QtCore.QRect(10, 10, 130, 16))
        self.label_stereoOmono.setObjectName("label_stereoOmono")
        self.radioButton_Rchannel = QtWidgets.QRadioButton(self.groupBox_7)
        self.radioButton_Rchannel.setGeometry(QtCore.QRect(10, 52, 70, 17))
        self.radioButton_Rchannel.setObjectName("radioButton_Rchannel")
        self.radioButton_Lchannel = QtWidgets.QRadioButton(self.groupBox_7)
        self.radioButton_Lchannel.setGeometry(QtCore.QRect(10, 29, 68, 17))
        self.radioButton_Lchannel.setObjectName("radioButton_Lchannel")
        self.verticalLayout.addWidget(self.groupBox_7)
        self.groupBox_6 = QtWidgets.QGroupBox(self.groupBox_Results)
        self.groupBox_6.setMinimumSize(QtCore.QSize(131, 0))
        self.groupBox_6.setMaximumSize(QtCore.QSize(400, 182))
        self.groupBox_6.setTitle("")
        self.groupBox_6.setObjectName("groupBox_6")
        self.radioButton_Timedomain = QtWidgets.QRadioButton(self.groupBox_6)
        self.radioButton_Timedomain.setGeometry(QtCore.QRect(10, 10, 82, 17))
        self.radioButton_Timedomain.setObjectName("radioButton_Timedomain")
        self.radioButton_Frecdomain = QtWidgets.QRadioButton(self.groupBox_6)
        self.radioButton_Frecdomain.setGeometry(QtCore.QRect(10, 33, 110, 17))
        self.radioButton_Frecdomain.setObjectName("radioButton_Frecdomain")
        self.radioButton_EDT_RT = QtWidgets.QRadioButton(self.groupBox_6)
        self.radioButton_EDT_RT.setGeometry(QtCore.QRect(10, 56, 62, 17))
        self.radioButton_EDT_RT.setObjectName("radioButton_EDT_RT")
        self.radioButton_C5080 = QtWidgets.QRadioButton(self.groupBox_6)
        self.radioButton_C5080.setGeometry(QtCore.QRect(10, 79, 68, 17))
        self.radioButton_C5080.setObjectName("radioButton_C5080")
        self.radioButton_IACC = QtWidgets.QRadioButton(self.groupBox_6)
        self.radioButton_IACC.setGeometry(QtCore.QRect(10, 125, 82, 17))
        self.radioButton_IACC.setObjectName("radioButton_IACC")
        self.radioButton = QtWidgets.QRadioButton(self.groupBox_6)
        self.radioButton.setGeometry(QtCore.QRect(10, 102, 83, 17))
        self.radioButton.setObjectName("radioButton")
        self.verticalLayout.addWidget(self.groupBox_6)
        self.gridLayout.addWidget(self.groupBox_Results, 0, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 2, 0, 3, 1)
        self.tableView_2 = QtWidgets.QTableWidget(Form)
        self.tableView_2.setObjectName("tableWidget")
        # self.tableWidget.setColumnCount(0)
        # self.tableWidget.setRowCount(0)
        self.gridLayout_2.addWidget(self.tableView_2, 5, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "RIR-Analyzer "))
        self.groupBox_imput.setTitle(_translate("Form", "Input Signal (Stereo/ Mono)"))
        self.pushButton_2.setText(_translate("Form", "Inverse filter file"))
        self.pushButton.setText(_translate("Form", "IR file (stereo/ mono)"))
        self.radioButton_SiseSwp.setText(_translate("Form", "Sine Sweep"))
        self.radioButton_IResponse.setText(_translate("Form", "Impulse Response"))
        self.groupBox_stereo.setTitle(_translate("Form", "Input Signal Stereo"))
        self.pushButton_3.setText(_translate("Form", "Channel L"))
        self.checkBox.setText(_translate("Form", "Split Input"))
        self.pushButton_4.setText(_translate("Form", "Channel R"))
        self.groupBox_filter.setTitle(_translate("Form", "Band Filtering"))
        self.radioButton_Thirdoctave.setText(_translate("Form", "Third-octave bands"))
        self.checkBox_reversedRIR.setText(_translate("Form", "Reversed RIR"))
        self.radioButton_Octaveband.setText(_translate("Form", "Octave bands"))
        self.groupBox_MMFilter.setTitle(_translate("Form", "Moving Median Filter"))
        self.lineEdit_Wind_len.setText(_translate("Form", "50"))
        self.label_2.setText(_translate("Form", "Window length [ms]"))
        self.pushButton_Analyze.setText(_translate("Form", "Analyze"))
        self.pushButton_Cancel.setText(_translate("Form", "Cancel"))
        self.pushButton_ExportResult.setText(_translate("Form", "Export results"))
        self.groupBox_Results.setTitle(_translate("Form", "Results"))
        self.radioButton_Rchannel.setText(_translate("Form", "R channel"))
        self.radioButton_Lchannel.setText(_translate("Form", "L channel"))
        self.radioButton_Timedomain.setText(_translate("Form", "Time domain"))
        self.radioButton_Frecdomain.setText(_translate("Form", "Frecuency domain"))
        self.radioButton_EDT_RT.setText(_translate("Form", "EDT, RT"))
        self.radioButton_C5080.setText(_translate("Form", "C50, C80"))
        self.radioButton_IACC.setText(_translate("Form", "IACC (early)"))
        self.radioButton.setText(_translate("Form", "EDTTt, Tt"))
