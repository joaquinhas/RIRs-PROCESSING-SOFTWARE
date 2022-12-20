# -*- coding: utf-8 -*-
"""
@author: Joaquín
"""

import numpy as np
from scipy import signal 
from scipy.signal import hilbert
from collections import deque
from bisect import insort, bisect_left
from itertools import islice


def C_(signal, tiempo, fs, final):
    'calculation of parameter C (clarity)'
    
    signal = signal[np.argmax(signal):]
    t = int((tiempo / 1000)* fs)
    c = 10* np.log10((np.sum(signal[:t]**2) / np.sum(signal[t:final]**2)))
    return c

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

def running_mean(x, N):                                                  
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def lundeby(signal, fs, window = 1000):
    'calculation of the crossover point where the decay ends and the response is noise using the steps' 
    'indicated in the study by Lundeby et. to the. (Uncertainties of Measurements in Room Acoustics)'
    
    signal = signal[np.argmax(signal):]
    
    #1  Average squared impulse response
    padded = np.pad(signal, (window//2,  0), mode='edge')
    promediada = running_mean(padded**2, window) 
    
    #2 Estimate background noise level using the tail
    ruido = int(len(promediada)/10)
    ruido_rms = rms(promediada[-ruido:])
    ruido_dBrms = 10* np.log10(ruido_rms)
    if ruido_dBrms <= -91:                                             
        ruido_dBrms = -100 
    
    #3 Estimate slope of decay from 0 dB to noise level
    with np.errstate(divide='ignore'): 
        promediada_dB = 10* np.log10(promediada)
    punto = np.where(promediada_dB >= 20 + ruido_dBrms)[0]
    if len(punto) == 0:
        punto = np.where(promediada_dB >= 10 + ruido_dBrms)[0]
        if len(punto) == 0:
            return ''
        else:
            punto = punto[-1]
    else:
        punto = punto[-1]
    if ruido_dBrms <= -91:                                              
        promediada_dB = promediada_dB[:punto+ 7*fs]                     
        promediada = promediada[:punto+ 7*fs]  
    x = np.arange(punto)
    coeficientes = np.polyfit(x, promediada_dB[:punto], 1)
    
    #Reinforcement---------------------------------------------------------------

    if np.isnan(coeficientes[0]):
        promediada = np.convolve(padded**2, np.ones((window,))/window, mode='valid')
        
        #2 Estimate background noise level using the tail
        ruido = int(len(promediada)/70)
        ruido_rms = rms(promediada[-ruido:])
        ruido_dBrms = 10* np.log10(ruido_rms)
        if ruido_dBrms <= -91:                                            
            ruido_dBrms = -100                                            
        
        #3 Estimate slope of decay from 0 dB to noise level
        promediada_dB = 10* np.log10(promediada)
        punto = np.where(promediada_dB >= 20 + ruido_dBrms)[0][-1]
        if len(punto) == 0:
            punto = np.where(promediada_dB >= 10 + ruido_dBrms)[0]
            if len(punto) == 0:
                return ''
            else:
                punto = punto[-1]
        else:
            punto = punto[-1]
        if ruido_dBrms <= -91:                                              
            promediada_dB = promediada_dB[:punto+ 5*fs]                     
            promediada = promediada[:punto+ 5*fs]                          
        x = np.arange(punto)
        coeficientes = np.polyfit(x, promediada_dB[:punto], 1)
                                  
    #------------------------------------------------------------------------------
    
    x2 = np.arange(len(promediada))
    ajustada = np.polyval(coeficientes, x2)
    if ajustada[0] > ajustada[1]:
    
        #4 Find preliminary crosspoint
        crosspoint_pre = np.where(ajustada >= ruido_dBrms)[0]
        if len(crosspoint_pre)== 0:
            crosspoint_pre = ''
        else:
            crosspoint_pre = crosspoint_pre[-1] 
        
            #5 Find new local time interval length
            new_interval = (np.where(ajustada >= ajustada[0]-10)[0][-1])//10
            
            #6  Average squared impulse response in new local time intervals
            if ruido_dBrms <= -91:                                                          
                padded2 = np.pad(signal[:punto+ 7*fs], (new_interval//2,  0), mode='edge')  
            else:                                                                           
                padded2 = np.pad(signal, (new_interval//2,  0), mode='edge') 
            promediada2 = running_mean(padded2**2, new_interval) 
            with np.errstate(divide='ignore'):                                     
               promediada2_dB = 10* np.log10(promediada2)
            
            #Reinforcement 2----------------------------------------------------------------------------------------------
            if np.isin(True, np.isinf(promediada2_dB), assume_unique = False) == True:    
                promediada2 = np.convolve(padded2**2, np.ones((new_interval,))/new_interval, mode='valid')
                promediada2_dB = 10* np.log10(promediada2)
            #---------------------------------------------------------------------------------------------------------
            
            #7,8,9 Find crosspoint
            i = 0
            while i<5:
                #7
                nivel_final_dB = ajustada[crosspoint_pre]-10
                nivel_final = 10**(nivel_final_dB/10)
                ruido_dBrms_final = 10* np.log10(rms(promediada2[np.where(promediada2 >= nivel_final)[0][-1]:]))
                
                #8
                valores_sobre_ruido = np.where(promediada2_dB >= 5 + ruido_dBrms_final)[0]
                if len(valores_sobre_ruido)== 0:
                    crosspoint_pre = ''
                    break
                else:
                    punto2 = valores_sobre_ruido[-1]
                    x = np.arange(punto2)
                    coeficientes2 = np.polyfit(x, promediada2_dB[:punto2], 1)
                    x2 = np.arange(len(ajustada))
                    ajustada2 = np.polyval(coeficientes2, x2)
                    
                    #9
                    crosspoint_pre = np.where(ajustada2 >= ruido_dBrms_final)[0]
                    if len(crosspoint_pre) ==0:
                        return ''
                    else:
                        crosspoint_pre = crosspoint_pre[-1]
                    i +=1
    else:
        crosspoint_pre = ''
    
    
    return crosspoint_pre

def running_median_insort(seq, window_size):
    'moving median filter'
    
    seq = iter(seq)
    d = deque()
    s = []
    result = []
    for item in islice(seq, window_size):
        d.append(item)
        insort(s, item)
        result.append(s[len(d)//2])
    m = window_size // 2
    for item in seq:
        old = d.popleft()
        d.append(item)
        del s[bisect_left(s, old)]
        insort(s, item)
        result.append(s[m])
    return result

def RT(audio, fs, MM_window):
   'calculation of the reverberation time according to the ISO 3382 standard'
   
   maximo = np.argmax(audio)
   hilb = np.abs(hilbert(audio[maximo:])) #Transformada de Hilbert
   median = np.array(running_median_insort(hilb, MM_window)) #Mediana movil
   pos_lundeby = lundeby(median, fs, 10000)
   if pos_lundeby == '':
       return 0,0,0,[],[]
   else:
       
       median2 = median[:pos_lundeby]
       
       sch = (np.cumsum(median2[::-1]**2)/(np.sum(median2)**2))[::-1] #schroeder
       with np.errstate(divide='ignore'): #ignore divide by zero warning
           sch = 10 * np.log10(sch / np.max(sch))
       
       pmenos10= np.argmax(sch) -10
       pmenos5= np.argmax(sch) -5
       pmenos25= np.argmax(sch) -25
       pmenos35= np.argmax(sch) -35
       
       #EDT
       ini_edt = np.argmin(np.abs(sch))
       fin_edt = np.argmin(np.abs(sch - pmenos10))
       t_edt = np.arange(ini_edt, fin_edt) / fs
       pendiente_edt, interseccion_edt =np.polyfit(t_edt, sch[ini_edt:fin_edt],1)
       tiempo_edt = -10 / pendiente_edt
       tiempo_edt *= 6
       
       #T20
       ini_t20 = np.argmin(np.abs(sch - pmenos5))
       fin_t20 = np.argmin(np.abs(sch - pmenos25))
       t_20 = np.arange(ini_t20, fin_t20) / fs
       pendiente_t20, interseccion_t20 =np.polyfit(t_20, sch[ini_t20:fin_t20],1)
       tiempo_20 = -20 / pendiente_t20
       tiempo_20 *= 3
       
       #T30
       ini_t30 = np.argmin(np.abs(sch - pmenos5))
       fin_t30 = np.argmin(np.abs(sch - pmenos35))
       t_30 = np.arange(ini_t30, fin_t30) / fs
       pendiente_t30, interseccion_t30 =np.polyfit(t_30, sch[ini_t30:fin_t30],1)
       tiempo_30 = -30 / pendiente_t30
       tiempo_30 *= 2
       
       return tiempo_edt, tiempo_20, tiempo_30, pos_lundeby, sch

def IACC(signalL, signalR, fs):
    'IACC early'
    
    t= int(0.08 * fs) #80ms samples
    maximo = np.argmax(signalL)
    signalL = signalL[maximo: maximo+t]
    signalR = signalR[maximo: maximo+t]
    
    signalL2 = signalL **2
    signalR2= signalR **2
    
    IACF = signal.correlate(signalL, signalR, method='fft') / np.sqrt(np.sum(signalL2) * np.sum(signalR2))
    return np.max(np.abs(IACF))

def Tt(imput, fs, f_low, crosspoint_pre):
    'Transition time'
    
    imput = imput[np.argmax(imput) : np.argmax(imput)+crosspoint_pre]
    fnyq = fs/2
    f_low = f_low * fnyq
    
    # if f_low > 100:
    window = int(fs/f_low)
    if window % 2 == 0:
        window = window + 1
    # window = int(window)
    ETC_median = np.abs(np.array(running_median_insort(imput, window)))
    
    with np.errstate(divide='ignore'): 
        dcer = 10*np.log10(np.abs(imput)) + ETC_median
    dcer = 10**(dcer/10)
    
    actual_EDF = np.cumsum(dcer)
    actual_EDF = actual_EDF/np.max(actual_EDF)
    
    Tt = np.argmin(np.abs(actual_EDF-0.99*np.max(actual_EDF)))/fs
    # else:
    #     Tt = 0
    return Tt

def EDTTt(decay, Tt, fs):
    if Tt > 0.01:

        s_init=int(0.005*fs)
        s_end = int(Tt * fs)
        factor=6

        decay=decay[s_init:s_end]
        x = np.arange(s_init, s_end)
        y=decay
       
        slope, intercept =np.polyfit(x,y,1)
        line = slope * np.arange(decay.size) + intercept
        init = line[0]
        t = np.argmin(np.abs(line - (init - 10))) / fs
        EDTTt= factor * t
    else:
        EDTTt = 0
    
    return EDTTt    

#---------------------------------------------------------------------------------
#A class is created according to the type of input signal, whether mono or stereo.

class ParametrosMono:
    def __init__(self, impulso, fs, MM_window, filtro='octava', rev = False):
        self.impulso = impulso
        self.fs = fs
        self.filtro = filtro   #filter type
        self.MM_window = MM_window  #Window size
        self.fc_list = []   #filter center frequencies
        self.edt_final = [] 
        self.t20_final= []
        self.t30_final = []
        self.c50 = []
        self.c80 = []
        self.Tt = []
        self.EDTTt = []
        self.rev = rev      #Reversed RIR

    def calcula_parametros(self):
        
        if self.MM_window % 2 == 0:     #MM_window must be odd
            self.MM_window +=1
        
        n = 0
        fi_list = np.array([])
        fs_list = np.array([])
        
        if self.filtro == 'octava':
            fc = 31.5
            while n <=9:
                fi = (2**(-1/2))*fc   #lower frequency of each band
                fsup = (2**(1/2))*fc  #upper frequency of each band
                self.fc_list  = np.append(self.fc_list ,fc)
                fi_list = np.append(fi_list,fi)
                fs_list = np.append(fs_list,fsup)
                fc = 2 * fc
                n = n+1
        
        elif self.filtro == 'tercios':   
            fc = 25
            while n <=29:
                fi = (2**(-1/6))*fc   #lower frequency of each band
                fsup = (2**(1/6))*fc  #upper frequency of each band
                self.fc_list  = np.append(self.fc_list ,fc)
                fi_list = np.append(fi_list,fi)
                fs_list = np.append(fs_list,fsup)
                fc = (2**(1/3)) * fc
                n = n+1
        else:
            return print('Filtro invalido')
        
        fi_list = np.append(fi_list, [250, 2000])
        fs_list = np.append(fs_list, [500, 4000])
        N =3
        fnyq = self.fs/2
        fi_list = fi_list/fnyq
        fs_list = fs_list/fnyq
        
        fs_list[fs_list >= 1] = 0.99999999
        
        if self.rev:
            self.impulso = self.impulso[::-1]
            
        tot = len(fi_list)
        progress = 1
        
        for low, high in zip (fi_list, fs_list):
            sos = signal.butter(N, [low,high], analog=False, btype = 'bandpass', output= 'sos') #filter design
            output = signal.sosfilt(sos, self.impulso)     #Filtering
            if self.rev:
                output = output[::-1]
                
            
            edt, t20, t30, crosspoint, sch = RT(output,self.fs, self.MM_window)
            if type(crosspoint)== list:
                c_50 = 0
                c_80 = 0
                Tt_ = 0
                EDTTt_ = 0
            else:
                
                c_50 = C_(output, 50, self.fs, crosspoint-self.MM_window)
                c_80 = C_(output, 80, self.fs, crosspoint-self.MM_window)
                Tt_ = Tt(output, self.fs, low, crosspoint-self.MM_window)
                if len(sch) != 0:
                    EDTTt_ = EDTTt(sch, Tt_, self.fs)
                else:
                    EDTTt_ = 0
            
            
            self.edt_final.append(edt)
            self.t20_final.append(t20)
            self.t30_final.append(t30)
            self.c50.append(c_50)
            self.c80.append(c_80)
            self.Tt.append(Tt_)
            self.EDTTt.append(EDTTt_)
            self.mono = 1
            
            progress_ = int((progress/tot) * 100)
            print(f'Progress: {progress_}%')
            if progress_ == 100:
                print('-------COMPLETE-------')
                print()
            progress +=1
    

class ParametroStereo:
    def __init__(self, impulsoL, impulsoR, fs, MM_window, filtro='octava', rev = False):
        self.impulsoL = impulsoL
        self.impulsoR = impulsoR
        self.fs = fs
        self.filtro = filtro
        self.MM_window = MM_window
        self.fc_list = []
        self.edt_finalL = [] 
        self.t20_finalL= []
        self.t30_finalL = []
        self.c50L = []
        self.c80L = []
        self.edt_finalR = [] 
        self.t20_finalR= []
        self.t30_finalR = []
        self.c50R = []
        self.c80R = []
        self.IACC = []
        self.TtL = []
        self.TtR = []
        self.EDTTtL = []
        self.EDTTtR = []
        self.rev = rev

    def calcula_parametros(self):
        
        if self.MM_window % 2 == 0:
            self.MM_window +=1
        
        n = 0
        fi_list = np.array([])
        fs_list = np.array([])
        
        if self.filtro == 'octava':
            fc = 31.5
            while n <=9:
                fi = (2**(-1/2))*fc
                fsup = (2**(1/2))*fc
                self.fc_list  = np.append(self.fc_list ,fc)
                fi_list = np.append(fi_list,fi)
                fs_list = np.append(fs_list,fsup)
                fc = 2 * fc
                n = n+1
        
        elif self.filtro == 'tercios':   
            fc = 25
            while n <=29:
                fi = (2**(-1/6))*fc
                fsup = (2**(1/6))*fc
                self.fc_list  = np.append(self.fc_list ,fc)
                fi_list = np.append(fi_list,fi)
                fs_list = np.append(fs_list,fsup)
                fc = (2**(1/3)) * fc
                n = n+1
        else:
            return print('Filtro invalido')
        
        fi_list = np.append(fi_list, [250, 2000])
        fs_list = np.append(fs_list, [500, 4000])
        N =3
        fnyq = self.fs/2
        fi_list = fi_list/fnyq
        fs_list = fs_list/fnyq
        
        fs_list[fs_list >= 1] = 0.99999999
        
        if self.rev:
            self.impulsoL = self.impulsoL[::-1]
            self.impulsoR = self.impulsoR[::-1]
        
        tot = len(fi_list)
        progress = 1
        
        for low, high in zip (fi_list, fs_list):
            sos = signal.butter(N, [low,high], analog=False, btype = 'bandpass', output= 'sos')
            outputL = signal.sosfilt(sos, self.impulsoL)
            outputR = signal.sosfilt(sos, self.impulsoR)
            if self.rev:
                outputL = outputL[::-1]
                outputR = outputR[::-1]
                
            
            edtL, t20L, t30L, crosspointL, schL = RT(outputL,self.fs, self.MM_window)
            if type(crosspointL)== list:
                c_50L = 0
                c_80L = 0
                Tt_L = 0
                EDTTt_L = 0
            else:
                c_50L = C_(outputL, 50, self.fs, crosspointL-self.MM_window)
                c_80L = C_(outputL, 80, self.fs, crosspointL-self.MM_window)
                Tt_L = Tt(outputL, self.fs, low, crosspointL-self.MM_window)
                if len(schL) != 0:
                    EDTTt_L = EDTTt(schL, Tt_L, self.fs)
                else:
                    EDTTt_L = 0
            
            
            edtR, t20R, t30R, crosspointR, schR = RT(outputR,self.fs, self.MM_window)
            if type(crosspointR)== list:
                c_50R = 0
                c_80R = 0
                Tt_R = 0
                EDTTt_R = 0
            else:
                c_50R = C_(outputR, 50, self.fs, crosspointR-self.MM_window)
                c_80R = C_(outputR, 80, self.fs, crosspointR-self.MM_window)
                Tt_R = Tt(outputR, self.fs, low, crosspointR-self.MM_window)
                if len(schL) != 0:
                    EDTTt_R = EDTTt(schR, Tt_R, self.fs)
                else:
                    EDTTt_R = 0
            iacc_ = IACC(outputL, outputR, self.fs)
            
            self.edt_finalL.append(edtL)
            self.t20_finalL.append(t20L)
            self.t30_finalL.append(t30L)
            self.c50L.append(c_50L)
            self.c80L.append(c_80L)
            self.TtL.append(Tt_L)
            self.EDTTtL.append(EDTTt_L)
            
            self.edt_finalR.append(edtR)
            self.t20_finalR.append(t20R)
            self.t30_finalR.append(t30R)
            self.c50R.append(c_50R)
            self.c80R.append(c_80R)
            self.TtR.append(Tt_R)
            self.EDTTtR.append(EDTTt_R)
            
            self.IACC.append(iacc_)
            self.mono = 0
            
            progress_ = int((progress/tot) * 100)
            print(f'Progress: {progress_}%')
            if progress_ == 100:
                print('-------COMPLETE-------')
                print()
            progress +=1

