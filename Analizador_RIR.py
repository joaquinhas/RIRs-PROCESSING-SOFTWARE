# -*- coding: utf-8 -*-
"""
@author: Joaquín
"""
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT 
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog,QMessageBox, QTableWidgetItem
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt
from ui_RIRanalyzer_v2 import Ui_Form
import numpy as np
import soundfile as sf  
import data_processing_v2 as dp
from scipy import signal
import csv
import pandas as pd  

#Event configuration code in the graphical user interface

class Analizador(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        
        self.fig = Figure()
        self.fig.set_tight_layout(True) 
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self) 
        self.ui.gridLayout.addWidget(self.toolbar, 2, 1, 1, 2)
        self.ui.gridLayout.addWidget(self.canvas, 0, 1, 2, 6)
        
        #Default parameters
        self.ui.radioButton_IResponse.setChecked(True)
        self.ui.radioButton_Octaveband.setChecked(True)
        self.ui.pushButton_2.setEnabled(False)
        self.ui.pushButton_3.setEnabled(False)
        self.ui.pushButton_4.setEnabled(False)
        self.ui.checkBox_reversedRIR.setEnabled(False)
        self.ui.radioButton_Lchannel.setEnabled(False)
        self.ui.radioButton_Rchannel.setEnabled(False)
        self.ui.radioButton_Timedomain.setChecked(True)
        self.ui.radioButton_IACC.setEnabled(False)
        self.fileType = 'IR'
        self.invertirIR = False
        self.filter = 'octava'
        self.signal = 'Mono'
        self.window = self.ui.lineEdit_Wind_len.text()
        self.flag = 0
        self.flag2 = 0
        
        # Window settings
        self.setWindowTitle("RIR-Analyzer")
        self.setWindowIcon(QtGui.QIcon("gui/sine_.png"))
        
        #Button events
        self.ui.pushButton.clicked.connect(self.cargar_audio)
        self.ui.pushButton_3.clicked.connect(self.cargar_audioL)
        self.ui.pushButton_4.clicked.connect(self.cargar_audioR)
        self.ui.pushButton_2.clicked.connect(self.cargar_InvFilt)
        self.ui.pushButton_Cancel.clicked.connect(self.close)
        self.ui.radioButton_SiseSwp.toggled.connect(self.toggleSweep)
        self.ui.radioButton_IResponse.toggled.connect(self.toggleIR)
        self.ui.radioButton_Thirdoctave.toggled.connect(self.toggleFiltTER)
        self.ui.radioButton_Octaveband.toggled.connect(self.toggleFiltOCT)
        self.ui.checkBox_reversedRIR.stateChanged.connect(lambda x: self.reverse_IR() if x else self.reverse_IR())
        self.ui.checkBox.stateChanged.connect(lambda x: self.stereo_on() if x else self.stereo_off())
        self.ui.pushButton_Analyze.clicked.connect(self.analize)
        self.ui.radioButton_Timedomain.toggled.connect(self.plot)
        self.ui.radioButton_EDT_RT.toggled.connect(self.plot)
        self.ui.radioButton_Frecdomain.toggled.connect(self.plot)
        self.ui.radioButton_C5080.toggled.connect(self.plot)
        self.ui.radioButton_IACC.toggled.connect(self.plot)
        self.ui.radioButton_Lchannel.toggled.connect(self.plot)
        self.ui.radioButton_Rchannel.toggled.connect(self.plot)
        self.ui.pushButton_ExportResult.clicked.connect(self.export)
        self.ui.radioButton.toggled.connect(self.plot)
        
    def cargar_audio(self):
        filtro = 'Audio en wav (*wav) ;; FLAC (*flac)'
        self.ruta = QFileDialog.getOpenFileName(filter = filtro)[0]
        self.file_name = self.ruta.split('/')[-1]
        if self.ruta != '':
            self.audio, self.fs = sf.read(self.ruta)
            self.flag2 = 1  
            if np.ndim(self.audio) == 2:
                self.signal = 'Stereo'
                self.audioL = self.audio[:,0]
                self.file_nameL = self.file_name+'(Channel L)'
                self.audioR = self.audio[:,1]
                self.file_nameR = self.file_name+'(Channel R)'
                self.ui.radioButton_IACC.setEnabled(True)
                self.ui.radioButton_Lchannel.setEnabled(True)
                self.ui.radioButton_Rchannel.setEnabled(True)
                _translate = QtCore.QCoreApplication.translate
                self.ui.label_stereoOmono.setText(_translate("Form", f'{self.signal}.   Fs: {self.fs} Hz'))
                self.ui.radioButton_Lchannel.setChecked(True)
    
            else:
                self.signal = 'Mono'
                self.ui.radioButton_IACC.setEnabled(False)
                self.ui.radioButton_Timedomain.setChecked(True)
                self.ui.radioButton_Lchannel.setEnabled(False)
                self.ui.radioButton_Rchannel.setEnabled(False)
                _translate = QtCore.QCoreApplication.translate
                self.ui.label_stereoOmono.setText(_translate("Form", f'{self.signal}.  Fs: {self.fs} Hz'))
        
    
    def cargar_InvFilt(self):
        filtro = 'Audio en wav (*wav) ;; FLAC (*flac)'
        ruta = QFileDialog.getOpenFileName(filter = filtro)[0]
        if ruta != '':
            self.invFilt, self.fsInvFilt = sf.read(ruta)
            self.flag2 = 1
    
    def cargar_audioL(self):
        filtro = 'Audio en wav (*wav) ;; FLAC (*flac)'
        ruta = QFileDialog.getOpenFileName(filter = filtro)[0]
        if ruta != '':
            self.file_nameL = ruta.split('/')[-1]
            self.audioL, self.fs = sf.read(ruta)
            self.flag2 = 1
            _translate = QtCore.QCoreApplication.translate
            self.ui.label_stereoOmono.setText(_translate("Form", f'{self.signal}.   Fs: {self.fs} Hz'))
                    
    def cargar_audioR(self):
        filtro = 'Audio en wav (*wav) ;; FLAC (*flac)'
        ruta = QFileDialog.getOpenFileName(filter = filtro)[0]
        if ruta != '':
            self.file_nameR = ruta.split('/')[-1]
            self.audioR, self.fs = sf.read(ruta)
            self.flag2 = 1
            _translate = QtCore.QCoreApplication.translate
            self.ui.label_stereoOmono.setText(_translate("Form", f'{self.signal}.   Fs: {self.fs} Hz'))
        
    def toggleSweep(self):
        self.ui.pushButton_2.setEnabled(True)
        self.fileType = 'sweep'
    
    def toggleIR(self):
        self.ui.pushButton_2.setEnabled(False)
        self.fileType = 'IR'
        
    def toggleFiltOCT(self):
        self.ui.checkBox_reversedRIR.setEnabled(False)
        self.ui.checkBox_reversedRIR.setChecked(False)
        self.filter = 'octava'
        
    def toggleFiltTER(self):
        self.ui.checkBox_reversedRIR.setEnabled(True)
        self.filter = 'tercios'
    
    def reverse_IR(self):
        self.invertirIR = not self.invertirIR
        
    def stereo_on(self):
        self.ui.pushButton_3.setEnabled(True)
        self.ui.pushButton_4.setEnabled(True)
        self.ui.pushButton.setEnabled(False)
        self.ui.radioButton_Lchannel.setChecked(True)
        self.ui.radioButton_IACC.setEnabled(True)
        self.ui.radioButton_Lchannel.setEnabled(True)
        self.ui.radioButton_Rchannel.setEnabled(True)
        self.signal = 'Stereo'
    
    def stereo_off(self):
        self.ui.pushButton_3.setEnabled(False)
        self.ui.pushButton_4.setEnabled(False)
        self.ui.pushButton.setEnabled(True)
        self.ui.radioButton_Timedomain.setChecked(True)
        self.ui.radioButton_IACC.setEnabled(False)
        self.ui.radioButton_Lchannel.setEnabled(False)
        self.ui.radioButton_Rchannel.setEnabled(False)
        self.signal = 'Mono'
        
    def export(self):
        if self.flag == 1:
            filename, _ = QFileDialog.getSaveFileName(self, 'Save File', 'Results', filter = "CSV (*.csv);; Excel Spreadsheet (*.xlsx)")
            if filename[-4:] != '.csv' and filename[-5:] != '.xlsx':
                filename = filename + '.csv'
            
            if len(self.parametros.fc_list) == 10:
                columnHeaders = ['31,5', '63', '125','250','500','1K', '2K','4K','8K', '16K', '250-2K','500-4K']
            else:
                columnHeaders = ['25', '31.5', '40', '50', '63', '80', '100', '125', '160',
                             '200', '250', '315', '400', '500', '630', '800', '1k',
                             '1.3K', '1.6K', '2K', '2.5K', '3.2K', '4K', '5K', 
                             '6.3K', '8K', '10K', '12.5K', '16K', '20K', '250-2K','500-4K']
             
            if self.parametros.mono == 1:
                EDT = list(np.round(self.parametros.edt_final,2))
                T20 = list(np.round(self.parametros.t20_final,2))
                T30 = list(np.round(self.parametros.t30_final,2))
                C50 = list(np.round(self.parametros.c50,2))
                C80 = list(np.round(self.parametros.c80,2))
                Tt = list(np.round(self.parametros.Tt,2))
                EDTTt = list(np.round(self.parametros.EDTTt,2))
                
                columnHeaders.insert(0, "f [Hz]")
                EDT.insert(0, "EDT [s]")
                T20.insert(0, "T20 [s]")
                T30.insert(0, "T30 [s]")
                C50.insert(0, "C50 [dB]")
                C80.insert(0, "C80 [dB]")
                Tt.insert(0, "Tt [s]")
                EDTTt.insert(0, "EDTTt [s]")
                
                # Write CSV
                with open(filename, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows([columnHeaders, EDT, T20, T30, C50, C80, Tt, EDTTt])
                
                columnHeaders.pop(0)
                # If XLSX selected, convert CSV to Excel
                if filename[-5:] == '.xlsx':
                    columns = ["EDT [s]", "T20 [s]", "T30 [s]", "C50 [dB]", "C80 [dB]", "Tt [s]", "EDTTt [s]"]
                    xlsx = pd.read_csv(filename)
                    xlsx.pop('f [Hz]')
                    xlsx.index = columns
                    xlsx = xlsx.to_excel(filename, columns = columnHeaders, index_label = "f [Hz]")
            else:

                EDTL = list(np.round(self.parametros.edt_finalL,2))
                EDTR = list(np.round(self.parametros.edt_finalR,2))
                T20L = list(np.round(self.parametros.t20_finalL,2))
                T20R = list(np.round(self.parametros.t20_finalR,2))
                T30L = list(np.round(self.parametros.t30_finalL,2))
                T30R = list(np.round(self.parametros.t30_finalR,2))
                C50L = list(np.round(self.parametros.c50L,2))
                C50R = list(np.round(self.parametros.c50R,2))
                C80L = list(np.round(self.parametros.c80L,2))
                C80R = list(np.round(self.parametros.c80R,2))
                TtL = list(np.round(self.parametros.TtL,2))
                TtR = list(np.round(self.parametros.TtR,2))
                EDTTtL = list(np.round(self.parametros.EDTTtL,2))
                EDTTtR = list(np.round(self.parametros.EDTTtR,2))
                IACC = list(np.round(self.parametros.IACC,2))
                
                columnHeaders.insert(0, "f [Hz]")
                EDTL.insert(0, "EDT_L [s]")
                EDTR.insert(0, "EDT_R [s]")
                T20L.insert(0, "T20_L [s]")
                T20R.insert(0, "T20_R [s]")
                T30L.insert(0, "T30_L [s]")
                T30R.insert(0, "T30_R [s]")
                C50L.insert(0, "C50_L [dB]")
                C50R.insert(0, "C50_R [dB]")
                C80L.insert(0, "C80_L [dB]")
                C80R.insert(0, "C80_R [dB]")
                TtL.insert(0, "Tt_L [s]")
                TtR.insert(0, "Tt_R [s]")
                EDTTtL.insert(0, "EDTTt_L [s]")
                EDTTtR.insert(0, "EDTTt_R [s]")
                IACC.insert(0, "IACC_e")
                
                # Write CSV
                with open(filename, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows([columnHeaders, EDTL, EDTR, T20L, T20R, T30L, T30R, C50L, C50R, 
                                      C80L, C80R, TtL, TtR,EDTTtL, EDTTtR, IACC])
                
                columnHeaders.pop(0)
                # If XLSX selected, convert CSV to Excel
                if filename[-5:] == '.xlsx':
                    columns = ["EDT_L [s]", "EDT_R [s]", "T20_L [s]", "T20_R [s]", "T30_L [s]", "T30_R [s]", 
                               "C50_L [dB]", "C50_R [dB]", "C80_L [dB]", "C80_R [dB]", 
                               "Tt_L [s]", "Tt_R [s]", "EDTTt_L [s]", "EDTTt_R [s]", "IACC_e"]
                    xlsx = pd.read_csv(filename)
                    xlsx.pop('f [Hz]')
                    xlsx.index = columns
                    xlsx = xlsx.to_excel(filename, columns = columnHeaders, index_label = "f [Hz]")
    
    def plot(self):
        if self.flag == 1:
            
            self.axes.cla()
            if self.signal =='Mono':
                if ventana.ui.radioButton_EDT_RT.isChecked():
                    self.linea = self.axes.plot(self.frecuencia, self.parametros.edt_final[:-2])[0]
                    self.linea = self.axes.plot(self.frecuencia, self.parametros.t20_final[:-2])[0]
                    self.linea = self.axes.plot(self.frecuencia, self.parametros.t30_final[:-2])[0]
                    self.axes.legend(['EDT', 'RT20','RT30'])
                    self.axes.set_xlabel('Frecuencia (Hz)')
                    self.axes.set_ylabel('Time (s)')
                    self.axes.set_title(self.file_name)
                    self.axes.set_xlim(20, 20000)
                    self.axes.grid()
                    self.axes.semilogx()
                    self.axes.set_xticks(self.xticks)
                    self.axes.set_xticklabels(self.xlabels)
                    self.linea.figure.canvas.draw()
                    self.linea.figure.canvas.flush_events()
                elif ventana.ui.radioButton_Timedomain.isChecked():
                    self.linea = self.axes.plot(self.tiempo, self.audio,'r')[0]
                    self.axes.set_xlabel('Time (s)')
                    self.axes.set_ylabel('Amplitude')
                    self.axes.set_title(self.file_name)
                    self.axes.grid()
                    self.linea.figure.canvas.draw()
                    self.linea.figure.canvas.flush_events()
                elif ventana.ui.radioButton_C5080.isChecked():
                    self.linea = self.axes.plot(self.frecuencia, self.parametros.c50[:-2])[0]
                    self.linea = self.axes.plot(self.frecuencia, self.parametros.c80[:-2])[0]
                    self.axes.legend(['C_50', 'C_80'])
                    self.axes.set_xlabel('Frecuencia (Hz)')
                    self.axes.set_ylabel('dB')
                    self.axes.set_title(self.file_name)
                    self.axes.set_xlim(20, 20000)
                    self.axes.grid()
                    self.axes.semilogx()
                    self.axes.set_xticks(self.xticks)
                    self.axes.set_xticklabels(self.xlabels)
                    self.linea.figure.canvas.draw()
                    self.linea.figure.canvas.flush_events() 
                elif ventana.ui.radioButton_Frecdomain.isChecked():
                    self.linea = self.axes.plot(self.frecuencia_audio, 10*np.log10(self.espectro/self.espectro.max()),'green')[0]
                    self.axes.set_xlabel('Frecuencia (Hz)')
                    self.axes.set_ylabel('dB Normalized')
                    self.axes.set_title(self.file_name)
                    self.axes.set_xlim(20, 20000)
                    self.axes.grid()
                    self.axes.semilogx()
                    self.axes.set_xticks(self.xticks)
                    self.axes.set_xticklabels(self.xlabels)
                    self.linea.figure.canvas.draw()
                    self.linea.figure.canvas.flush_events()
                elif ventana.ui.radioButton.isChecked():
                    self.linea = self.axes.plot(self.frecuencia, self.parametros.Tt[:-2], 'black')[0]
                    self.linea = self.axes.plot(self.frecuencia, self.parametros.EDTTt[:-2], 'red')[0]
                    self.axes.legend(['Tt', 'EDTTt'])
                    self.axes.set_xlabel('Frecuencia (Hz)')
                    self.axes.set_ylabel('[s]')
                    self.axes.set_title(self.file_name)
                    self.axes.set_xlim(20, 20000)
                    self.axes.grid()
                    self.axes.semilogx()
                    self.axes.set_xticks(self.xticks)
                    self.axes.set_xticklabels(self.xlabels)
                    self.linea.figure.canvas.draw()
                    self.linea.figure.canvas.flush_events()
            else:
                if ventana.ui.radioButton_Lchannel.isChecked():
                    if ventana.ui.radioButton_EDT_RT.isChecked():
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.edt_finalL[:-2])[0]
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.t20_finalL[:-2])[0]
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.t30_finalL[:-2])[0]
                        self.axes.legend(['EDT', 'RT20','RT30'])
                        self.axes.set_xlabel('Frecuencia (Hz)')
                        self.axes.set_ylabel('Time (s)')
                        self.axes.set_title(self.file_nameL)
                        self.axes.set_xlim(20, 20000)
                        self.axes.grid()
                        self.axes.semilogx()
                        self.axes.set_xticks(self.xticks)
                        self.axes.set_xticklabels(self.xlabels)
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events()
                    elif ventana.ui.radioButton_Timedomain.isChecked():
                        self.linea = self.axes.plot(self.tiempo, self.audioL,'r')[0]
                        self.axes.set_xlabel('Time (s)')
                        self.axes.set_ylabel('Amplitude')
                        self.axes.set_title(self.file_nameL)
                        self.axes.grid()
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events()
                    elif ventana.ui.radioButton_C5080.isChecked():
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.c50L[:-2])[0]
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.c80L[:-2])[0]
                        self.axes.legend(['C_50', 'C_80'])
                        self.axes.set_xlabel('Frecuencia (Hz)')
                        self.axes.set_ylabel('dB')
                        self.axes.set_title(self.file_nameL)
                        self.axes.set_xlim(20, 20000)
                        self.axes.grid()
                        self.axes.semilogx()
                        self.axes.set_xticks(self.xticks)
                        self.axes.set_xticklabels(self.xlabels)
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events() 
                    elif ventana.ui.radioButton_Frecdomain.isChecked():
                        self.linea = self.axes.plot(self.frecuencia_audioL, 10*np.log10(self.espectroL/self.espectroL.max()),'green')[0]
                        self.axes.set_xlabel('Frecuencia (Hz)')
                        self.axes.set_ylabel('dB Normalized')
                        self.axes.set_title(self.file_nameL)
                        self.axes.set_xlim(20, 20000)
                        self.axes.grid()
                        self.axes.semilogx()
                        self.axes.set_xticks(self.xticks)
                        self.axes.set_xticklabels(self.xlabels)
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events()
                    elif ventana.ui.radioButton_IACC.isChecked():
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.IACC[:-2],'yellow')[0]
                        self.axes.set_xlabel('Frecuencia (Hz)')
                        self.axes.set_ylabel('Relative')
                        # self.axes.set_title(self.file_name)
                        self.axes.set_xlim(20, 20000)
                        self.axes.grid()
                        self.axes.semilogx()
                        self.axes.set_xticks(self.xticks)
                        self.axes.set_xticklabels(self.xlabels)
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events()
                    elif ventana.ui.radioButton.isChecked():
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.TtL[:-2], 'black')[0]
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.EDTTtL[:-2], 'red')[0]
                        self.axes.legend(['Tt_L', 'EDTTt_L'])
                        self.axes.set_xlabel('Frecuencia (Hz)')
                        self.axes.set_ylabel('[s]')
                        self.axes.set_title(self.file_nameL)
                        self.axes.set_xlim(20, 20000)
                        self.axes.grid()
                        self.axes.semilogx()
                        self.axes.set_xticks(self.xticks)
                        self.axes.set_xticklabels(self.xlabels)
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events()
                else:
                    if ventana.ui.radioButton_EDT_RT.isChecked():
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.edt_finalR[:-2])[0]
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.t20_finalR[:-2])[0]
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.t30_finalR[:-2])[0]
                        self.axes.legend(['EDT', 'RT20','RT30'])
                        self.axes.set_xlabel('Frecuencia (Hz)')
                        self.axes.set_ylabel('Time (s)')
                        self.axes.set_title(self.file_nameR)
                        self.axes.set_xlim(20, 20000)
                        self.axes.grid()
                        self.axes.semilogx()
                        self.axes.set_xticks(self.xticks)
                        self.axes.set_xticklabels(self.xlabels)
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events()
                    elif ventana.ui.radioButton_Timedomain.isChecked():
                        self.linea = self.axes.plot(self.tiempo, self.audioR,'r')[0]
                        self.axes.set_xlabel('Time (s)')
                        self.axes.set_ylabel('Amplitude')
                        self.axes.set_title(self.file_nameR)
                        self.axes.grid()
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events()
                    elif ventana.ui.radioButton_C5080.isChecked():
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.c50R[:-2])[0]
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.c80R[:-2])[0]
                        self.axes.legend(['C_50', 'C_80'])
                        self.axes.set_xlabel('Frecuencia (Hz)')
                        self.axes.set_ylabel('dB')
                        self.axes.set_title(self.file_nameR)
                        self.axes.set_xlim(20, 20000)
                        self.axes.grid()
                        self.axes.semilogx()
                        self.axes.set_xticks(self.xticks)
                        self.axes.set_xticklabels(self.xlabels)
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events() 
                    elif ventana.ui.radioButton_Frecdomain.isChecked():
                        self.linea = self.axes.plot(self.frecuencia_audioR, 10*np.log10(self.espectroR/self.espectroR.max()),'green')[0]
                        self.axes.set_xlabel('Frecuencia (Hz)')
                        self.axes.set_ylabel('dB Normalized')
                        self.axes.set_title(self.file_nameR)
                        self.axes.set_xlim(20, 20000)
                        self.axes.grid()
                        self.axes.semilogx()
                        self.axes.set_xticks(self.xticks)
                        self.axes.set_xticklabels(self.xlabels)
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events()
                    elif ventana.ui.radioButton_IACC.isChecked():
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.IACC[:-2],'yellow')[0]
                        self.axes.set_xlabel('Frecuencia (Hz)')
                        self.axes.set_ylabel('Relative')
                        # self.axes.set_title(self.file_name)
                        self.axes.set_xlim(20, 20000)
                        self.axes.grid()
                        self.axes.semilogx()
                        self.axes.set_xticks(self.xticks)
                        self.axes.set_xticklabels(self.xlabels)
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events()
                    elif ventana.ui.radioButton.isChecked():
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.TtR[:-2], 'black')[0]
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.EDTTtR[:-2], 'red')[0]
                        self.axes.legend(['Tt_R', 'EDTTt_R'])
                        self.axes.set_xlabel('Frecuencia (Hz)')
                        self.axes.set_ylabel('[s]')
                        self.axes.set_title(self.file_nameR)
                        self.axes.set_xlim(20, 20000)
                        self.axes.grid()
                        self.axes.semilogx()
                        self.axes.set_xticks(self.xticks)
                        self.axes.set_xticklabels(self.xlabels)
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events()
                
    
    def analize(self):
        if self.flag2 == 1:
            
            #MENSAJE DE ADVERTENCIA----------------------------------------------------
            mensaje = QMessageBox()
            mensaje.setIcon(QMessageBox.Information)
            mensaje.setText('This operation may take a few minutes. Be patient!')
            mensaje.setWindowTitle('Information')
            mensaje.exec_()
            #---------------------------------------------------------------------------
            
            if self.ui.lineEdit_Wind_len.text() == '':
                self.window = 50
            else:
                self.window = int(self.ui.lineEdit_Wind_len.text())
            
            self.window2 = int((self.window /1000) * self.fs)
            if self.signal == 'Mono':
                if self.fileType == 'sweep':
                    self.audio = signal.fftconvolve(self.audio, self.invFilt, mode='same')
                    self.audio = np.trim_zeros(self.audio)
                    
                self.parametros = dp.ParametrosMono(self.audio, self.fs, self.window2, self.filter, self.invertirIR)
                self.parametros.calcula_parametros()
            else:
               if self.fileType == 'sweep':
                   self.audioL = signal.fftconvolve(self.audioL, self.invFilt, mode='same')
                   self.audioL = np.trim_zeros(self.audioL)
                   self.audioR = signal.fftconvolve(self.audioR, self.invFilt, mode='same')
                   self.audioR = np.trim_zeros(self.audioR)
                   
               self.parametros = dp.ParametroStereo(self.audioL, self.audioR, self.fs, self.window2, self.filter, self.invertirIR)
               self.parametros.calcula_parametros() 
            
            
            
            #Table-------------------------------------------------------------------------
            if self.signal == 'Mono':
                self.ui.tableView_2.setRowCount(7)
                encabezado_filas= ['EDT [s]','RT20 [s]', 'RT30 [s]', 'C50 [dB]', 'C80 [dB]', 'Tt [s]', 'EDTTt [s]']   
            else:
                self.ui.tableView_2.setRowCount(15)
                encabezado_filas= ['EDT_L [s]','EDT_R [s]', 'RT20_L [s]', 'RT20_R [s]', 'RT30_L [s]', 'RT30_R [s]', 
                                   'C50_L [dB]',  'C50_R [dB]', 'C80_L [dB]', 'C80_R [dB]','Tt_L [s]', 'Tt_R [s]',
                                   'EDTTt_L [s]', 'EDTTt_R [s]', 'IACC (early)']
            
            if self.filter == 'octava':
                self.ui.tableView_2.setColumnCount(12)
                encabezado_columnas= ['31,5 Hz', '63 Hz', '125 Hz', '250 Hz', '500 Hz', '1000 Hz', '2000 Hz', 
                                      '4000 Hz', '8000 Hz', '16000 Hz', '250-2K Hz','500-4K Hz']
            else:
                self.ui.tableView_2.setColumnCount(32)
                encabezado_columnas= ['25 Hz', '31,5 Hz', '40 Hz', '50 Hz','63 Hz', '80 Hz','100 Hz','125 Hz', '160 Hz',
                                      '200 Hz','250 Hz', '315 Hz','400 Hz','500 Hz', '630 Hz','800 Hz','1000 Hz', 
                                      '1250 Hz','1600 Hz','2000 Hz', '2500 Hz','3150 Hz','4000 Hz', '5000 Hz',
                                      '6300 Hz','8000 Hz', '10000 Hz','12500 Hz','16000 Hz','20000 Hz', '250-2K Hz','500-4K Hz']
                
            self.ui.tableView_2.setTextElideMode(Qt.ElideRight)
            self.ui.tableView_2.setVerticalHeaderLabels(encabezado_filas)
            self.ui.tableView_2.setHorizontalHeaderLabels(encabezado_columnas)
    
            if self.signal =='Mono':
                for i in range(len(self.parametros.fc_list)+2):
                    self.ui.tableView_2.setItem(0, i, QTableWidgetItem(str(np.round(self.parametros.edt_final[i],2))))
                    self.ui.tableView_2.setItem(1, i, QTableWidgetItem(str(np.round(self.parametros.t20_final[i],2))))
                    self.ui.tableView_2.setItem(2, i, QTableWidgetItem(str(np.round(self.parametros.t30_final[i],2))))
                    self.ui.tableView_2.setItem(3, i, QTableWidgetItem(str(np.round(self.parametros.c50[i],2))))
                    self.ui.tableView_2.setItem(4, i, QTableWidgetItem(str(np.round(self.parametros.c80[i],2))))
                    self.ui.tableView_2.setItem(5, i, QTableWidgetItem(str(np.round(self.parametros.Tt[i],2))))
                    self.ui.tableView_2.setItem(6, i, QTableWidgetItem(str(np.round(self.parametros.EDTTt[i],2))))
                    self.ui.tableView_2.setColumnWidth(i, 65)
            else:
                for i in range(len(self.parametros.fc_list)+2):
                    self.ui.tableView_2.setItem(0, i, QTableWidgetItem(str(np.round(self.parametros.edt_finalL[i],2))))
                    self.ui.tableView_2.setItem(2, i, QTableWidgetItem(str(np.round(self.parametros.t20_finalL[i],2))))
                    self.ui.tableView_2.setItem(4, i, QTableWidgetItem(str(np.round(self.parametros.t30_finalL[i],2))))
                    self.ui.tableView_2.setItem(6, i, QTableWidgetItem(str(np.round(self.parametros.c50L[i],2))))
                    self.ui.tableView_2.setItem(8, i, QTableWidgetItem(str(np.round(self.parametros.c80L[i],2))))
                    self.ui.tableView_2.setItem(1, i, QTableWidgetItem(str(np.round(self.parametros.edt_finalR[i],2))))
                    self.ui.tableView_2.setItem(3, i, QTableWidgetItem(str(np.round(self.parametros.t20_finalR[i],2))))
                    self.ui.tableView_2.setItem(5, i, QTableWidgetItem(str(np.round(self.parametros.t30_finalR[i],2))))
                    self.ui.tableView_2.setItem(7, i, QTableWidgetItem(str(np.round(self.parametros.c50R[i],2))))
                    self.ui.tableView_2.setItem(9, i, QTableWidgetItem(str(np.round(self.parametros.c80R[i],2))))
                    self.ui.tableView_2.setItem(10, i, QTableWidgetItem(str(np.round(self.parametros.TtL[i],2))))
                    self.ui.tableView_2.setItem(11, i, QTableWidgetItem(str(np.round(self.parametros.TtR[i],2))))
                    self.ui.tableView_2.setItem(12, i, QTableWidgetItem(str(np.round(self.parametros.EDTTtL[i],2))))
                    self.ui.tableView_2.setItem(13, i, QTableWidgetItem(str(np.round(self.parametros.EDTTtR[i],2))))
                    self.ui.tableView_2.setItem(14, i, QTableWidgetItem(str(np.round(self.parametros.IACC[i],2))))
                    self.ui.tableView_2.setColumnWidth(i, 65) 
            
            # PLOT--------------------------------------------------------------------
            if self.flag == 0:
                self.axes = self.fig.subplots()
            if self.flag == 1:
                self.axes.cla()
                
            self.frecuencia = ventana.parametros.fc_list
            self.xticks = [31.5,63,126,252,504,1008,2016,4032,8064,16128]
            self.xlabels = ['31.5','63','125','250','500','1K','2K','4K','8K','16K']
                
            if self.signal =='Mono':
                
                self.frecuencia_audio = np.fft.rfftfreq(self.audio.shape[0], 1/self.fs)
                self.espectro =  abs(np.fft.rfft(self.audio))
                self.tiempo = np.linspace(0, len(self.audio)/self.fs, len(self.audio))
                
                if ventana.ui.radioButton_EDT_RT.isChecked():
                    self.linea = self.axes.plot(self.frecuencia, self.parametros.edt_final[:-2])[0]
                    self.linea = self.axes.plot(self.frecuencia, self.parametros.t20_final[:-2])[0]
                    self.linea = self.axes.plot(self.frecuencia, self.parametros.t30_final[:-2])[0]
                    self.axes.legend(['EDT', 'RT20','RT30'])
                    self.axes.set_xlabel('Frecuencia (Hz)')
                    self.axes.set_ylabel('Time (s)')
                    self.axes.set_title(self.file_name)
                    self.axes.set_xlim(20, 20000)
                    self.axes.grid()
                    self.axes.semilogx()
                    self.axes.set_xticks(self.xticks)
                    self.axes.set_xticklabels(self.xlabels)
                    self.linea.figure.canvas.draw()
                    self.linea.figure.canvas.flush_events()
                elif ventana.ui.radioButton_Timedomain.isChecked():
                    self.linea = self.axes.plot(self.tiempo, self.audio,'r')[0]
                    self.axes.set_xlabel('Time (s)')
                    self.axes.set_ylabel('Amplitude')
                    self.axes.set_title(self.file_name)
                    self.axes.grid()
                    self.linea.figure.canvas.draw()
                    self.linea.figure.canvas.flush_events()
                elif ventana.ui.radioButton_C5080.isChecked():
                    self.linea = self.axes.plot(self.frecuencia, self.parametros.c50[:-2])[0]
                    self.linea = self.axes.plot(self.frecuencia, self.parametros.c80[:-2])[0]
                    self.axes.legend(['C_50', 'C_80'])
                    self.axes.set_xlabel('Frecuencia (Hz)')
                    self.axes.set_ylabel('dB')
                    self.axes.set_title(self.file_name)
                    self.axes.set_xlim(20, 20000)
                    self.axes.grid()
                    self.axes.semilogx()
                    self.axes.set_xticks(self.xticks)
                    self.axes.set_xticklabels(self.xlabels)
                    self.linea.figure.canvas.draw()
                    self.linea.figure.canvas.flush_events() 
                elif ventana.ui.radioButton_Frecdomain.isChecked():
                    self.linea = self.axes.plot(self.frecuencia_audio, 10*np.log10(self.espectro/self.espectro.max()),'green')[0]
                    self.axes.set_xlabel('Frecuencia (Hz)')
                    self.axes.set_ylabel('dB Normalized')
                    self.axes.set_title(self.file_name)
                    self.axes.set_xlim(20, 20000)
                    self.axes.grid()
                    self.axes.semilogx()
                    self.axes.set_xticks(self.xticks)
                    self.axes.set_xticklabels(self.xlabels)
                    self.linea.figure.canvas.draw()
                    self.linea.figure.canvas.flush_events()
                elif ventana.ui.radioButton.isChecked():
                    self.linea = self.axes.plot(self.frecuencia, self.parametros.Tt[:-2], 'black')[0]
                    self.linea = self.axes.plot(self.frecuencia, self.parametros.EDTTt[:-2], 'red')[0]
                    self.axes.legend(['Tt', 'EDTTt'])
                    self.axes.set_xlabel('Frecuencia (Hz)')
                    self.axes.set_ylabel('[s]')
                    self.axes.set_title(self.file_name)
                    self.axes.set_xlim(20, 20000)
                    self.axes.grid()
                    self.axes.semilogx()
                    self.axes.set_xticks(self.xticks)
                    self.axes.set_xticklabels(self.xlabels)
                    self.linea.figure.canvas.draw()
                    self.linea.figure.canvas.flush_events() 
            else:
                self.frecuencia_audioL = np.fft.rfftfreq(self.audioL.shape[0], 1/self.fs)
                self.espectroL =  abs(np.fft.rfft(self.audioL))
                self.frecuencia_audioR = np.fft.rfftfreq(self.audioR.shape[0], 1/self.fs)
                self.espectroR =  abs(np.fft.rfft(self.audioR))
                self.tiempo = np.linspace(0, len(self.audioR)/self.fs, len(self.audioR))
                
                if ventana.ui.radioButton_Lchannel.isChecked():
                    if ventana.ui.radioButton_EDT_RT.isChecked():
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.edt_finalL[:-2])[0]
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.t20_finalL[:-2])[0]
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.t30_finalL[:-2])[0]
                        self.axes.legend(['EDT', 'RT20','RT30'])
                        self.axes.set_xlabel('Frecuencia (Hz)')
                        self.axes.set_ylabel('Time (s)')
                        self.axes.set_title(self.file_nameL)
                        self.axes.set_xlim(20, 20000)
                        self.axes.grid()
                        self.axes.semilogx()
                        self.axes.set_xticks(self.xticks)
                        self.axes.set_xticklabels(self.xlabels)
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events()
                    elif ventana.ui.radioButton_Timedomain.isChecked():
                        self.linea = self.axes.plot(self.tiempo, self.audioL,'r')[0]
                        self.axes.set_xlabel('Time (s)')
                        self.axes.set_ylabel('Amplitude')
                        self.axes.set_title(self.file_nameL)
                        self.axes.grid()
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events()
                    elif ventana.ui.radioButton_C5080.isChecked():
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.c50L[:-2])[0]
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.c80L[:-2])[0]
                        self.axes.legend(['C_50', 'C_80'])
                        self.axes.set_xlabel('Frecuencia (Hz)')
                        self.axes.set_ylabel('dB')
                        self.axes.set_title(self.file_nameL)
                        self.axes.set_xlim(20, 20000)
                        self.axes.grid()
                        self.axes.semilogx()
                        self.axes.set_xticks(self.xticks)
                        self.axes.set_xticklabels(self.xlabels)
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events() 
                    elif ventana.ui.radioButton_Frecdomain.isChecked():
                        self.linea = self.axes.plot(self.frecuencia_audioL, 10*np.log10(self.espectroL/self.espectroL.max()),'green')[0]
                        self.axes.set_xlabel('Frecuencia (Hz)')
                        self.axes.set_ylabel('dB Normalized')
                        self.axes.set_title(self.file_nameL)
                        self.axes.set_xlim(20, 20000)
                        self.axes.grid()
                        self.axes.semilogx()
                        self.axes.set_xticks(self.xticks)
                        self.axes.set_xticklabels(self.xlabels)
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events()
                    elif ventana.ui.radioButton_IACC.isChecked():
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.IACC[:-2],'yellow')[0]
                        self.axes.set_xlabel('Frecuencia (Hz)')
                        self.axes.set_ylabel('Relative')
                        # self.axes.set_title(self.file_name)
                        self.axes.set_xlim(20, 20000)
                        self.axes.grid()
                        self.axes.semilogx()
                        self.axes.set_xticks(self.xticks)
                        self.axes.set_xticklabels(self.xlabels)
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events()
                    elif ventana.ui.radioButton.isChecked():
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.TtL[:-2], 'black')[0]
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.EDTTtL[:-2], 'red')[0]
                        self.axes.legend(['Tt_L', 'EDTTt_L'])
                        self.axes.set_xlabel('Frecuencia (Hz)')
                        self.axes.set_ylabel('[s]')
                        self.axes.set_title(self.file_nameL)
                        self.axes.set_xlim(20, 20000)
                        self.axes.grid()
                        self.axes.semilogx()
                        self.axes.set_xticks(self.xticks)
                        self.axes.set_xticklabels(self.xlabels)
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events()
                else:
                    if ventana.ui.radioButton_EDT_RT.isChecked():
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.edt_finalR[:-2])[0]
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.t20_finalR[:-2])[0]
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.t30_finalR[:-2])[0]
                        self.axes.legend(['EDT', 'RT20','RT30'])
                        self.axes.set_xlabel('Frecuencia (Hz)')
                        self.axes.set_ylabel('Time (s)')
                        self.axes.set_title(self.file_nameR)
                        self.axes.set_xlim(20, 20000)
                        self.axes.grid()
                        self.axes.semilogx()
                        self.axes.set_xticks(self.xticks)
                        self.axes.set_xticklabels(self.xlabels)
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events()
                    elif ventana.ui.radioButton_Timedomain.isChecked():
                        self.linea = self.axes.plot(self.tiempo, self.audioR,'r')[0]
                        self.axes.set_xlabel('Time (s)')
                        self.axes.set_ylabel('Amplitude')
                        self.axes.set_title(self.file_nameR)
                        self.axes.grid()
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events()
                    elif ventana.ui.radioButton_C5080.isChecked():
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.c50R[:-2])[0]
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.c80R[:-2])[0]
                        self.axes.legend(['C_50', 'C_80'])
                        self.axes.set_xlabel('Frecuencia (Hz)')
                        self.axes.set_ylabel('dB')
                        self.axes.set_title(self.file_nameR)
                        self.axes.set_xlim(20, 20000)
                        self.axes.grid()
                        self.axes.semilogx()
                        self.axes.set_xticks(self.xticks)
                        self.axes.set_xticklabels(self.xlabels)
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events() 
                    elif ventana.ui.radioButton_Frecdomain.isChecked():
                        self.linea = self.axes.plot(self.frecuencia_audioR, 10*np.log10(self.espectroR/self.espectroR.max()),'green')[0]
                        self.axes.set_xlabel('Frecuencia (Hz)')
                        self.axes.set_ylabel('dB Normalized')
                        self.axes.set_title(self.file_nameR)
                        self.axes.set_xlim(20, 20000)
                        self.axes.grid()
                        self.axes.semilogx()
                        self.axes.set_xticks(self.xticks)
                        self.axes.set_xticklabels(self.xlabels)
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events()
                    elif ventana.ui.radioButton_IACC.isChecked():
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.IACC[:-2],'yellow')[0]
                        self.axes.set_xlabel('Frecuencia (Hz)')
                        self.axes.set_ylabel('Relative')
                        # self.axes.set_title(self.file_name)
                        self.axes.set_xlim(20, 20000)
                        self.axes.grid()
                        self.axes.semilogx()
                        self.axes.set_xticks(self.xticks)
                        self.axes.set_xticklabels(self.xlabels)
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events()
                    elif ventana.ui.radioButton.isChecked():
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.TtR[:-2], 'black')[0]
                        self.linea = self.axes.plot(self.frecuencia, self.parametros.EDTTtR[:-2], 'red')[0]
                        self.axes.legend(['Tt_R', 'EDTTt_R'])
                        self.axes.set_xlabel('Frecuencia (Hz)')
                        self.axes.set_ylabel('[s]')
                        self.axes.set_title(self.file_nameR)
                        self.axes.set_xlim(20, 20000)
                        self.axes.grid()
                        self.axes.semilogx()
                        self.axes.set_xticks(self.xticks)
                        self.axes.set_xticklabels(self.xlabels)
                        self.linea.figure.canvas.draw()
                        self.linea.figure.canvas.flush_events()
            
            self.flag = 1
        
app = QApplication([])     
ventana = Analizador()
ventana.show()
app.exec_()   
        