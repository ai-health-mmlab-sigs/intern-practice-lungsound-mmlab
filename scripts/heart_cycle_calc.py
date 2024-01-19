# -*- coding:utf-8 -*-
import wave
import struct
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import fftpack
from scipy.fftpack import fft, ifft
from scipy.interpolate import make_interp_spline
from numpy.lib.function_base import append

# from scipy.io import wavfile

def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re
# 呼吸周期与心脏跳动周期的分离
def heart_calc(audio_path):
    # 读取wav文件
    filename = audio_path
    wavefile = wave.open(filename, 'r')  
    nchannels = wavefile.getnchannels()
    sample_width = wavefile.getsampwidth() 
    framerate = wavefile.getframerate() 
    numframes = wavefile.getnframes() 
    y = np.zeros(numframes)
    for i in range(numframes):
        val = wavefile.readframes(1) 
        left = val[0:2]
        # right = val[2:4]
        v = struct.unpack('h', left)[0]
        y[i] = v
    # framerate就是声音的采用率，文件初读取的值。
    Fs = framerate
    time = np.arange(0, numframes) * (1.0 / framerate) #时间=帧数/帧率
    #time = np.arange(0, numframes)
    fft_y = fft(y)
    fft_y_breath = fft(y)
    fft_y_heart = fft(y)
    abs_y = np.abs(fft_y)
    for i in range(0,numframes):
        # 筛选心脏跳动周期 20-200Hz
        if(i>200 or i<20):
            fft_y_heart[i] = 0.0
        # 筛选出呼吸周期 400-700Hz
        if(i>700 or i<400):
            fft_y_breath[i] = 0.0
            #print(f'傅里叶变换后的模为:{abs_y[i]}')
    #选择频率前后对比
    plt.figure(figsize=(20,5))
    plt.plot(time[:5000],abs_y[:5000],label='fft')
    plt.legend()
    plt.plot(time[:5000],np.abs(fft_y_heart)[:5000],'r',label='heart fft')
    plt.legend()
    plt.plot(time[:5000],np.abs(fft_y_breath)[:5000],'orange',label='breath fft')
    plt.legend()
    plt.show()
    # 进行ifft从频域回到时域
    sig = ifft(fft_y)
    sig_heart = ifft(fft_y_heart)
    sig_breath = ifft(fft_y_breath)
    # 显示各自周期
    plt.figure(figsize=(20,5))
    plt.plot(time[:50000],sig[:50000],label='fft')
    plt.legend()
    plt.plot(time[:50000],sig_heart[:50000],'r',label='heart fft')
    plt.legend()
    plt.plot(time[:50000],sig_breath[:50000],'orange',label='breath fft')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    audio_path = '102_1b1_Ar_sc_Meditron.wav'
    heart_calc(audio_path)
