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
import os
# from scipy.io import wavfile

def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re
# 包络提取算法
def oscillogram_spectrum(audio_path):
    """
    画出音频文件audio_path的声波图和频谱图
    :param audio_path:音频文件路径
    :return:
    """
    # 读取wav文件
    filename = audio_path
    wavefile = wave.open(filename, 'r')  # open for writing
    # 读取wav文件的四种信息的函数。其中numframes表示一共读取了几个frames。
    nchannels = wavefile.getnchannels()
    sample_width = wavefile.getsampwidth() #比特宽度，每一帧的字节数
    framerate = wavefile.getframerate() #采样率
    numframes = wavefile.getnframes() # 采样点数
    t_audio = numframes / framerate
    # 建一个y的数列，用来保存后面读的每个frame的波幅amplitude。
    y = np.zeros(numframes)
    for i in range(numframes):
        val = wavefile.readframes(1) #n是读取的帧数,返回二进制字符串对象
        left = val[0:2]
        # right = val[2:4]
        v = struct.unpack('h', left)[0]
        y[i] = v
    # framerate就是声音的采用率，文件初读取的值。
    Fs = framerate
    time = np.arange(0, numframes) * (1.0 / framerate) #时间=帧数/帧率
    #time = np.arange(0, numframes)
    fft_y = fft(y)
    abs_y = np.abs(fft_y)
    for i in range(0,numframes):
        #print(i)
        if(i>700 or i<100):
            fft_y[i] = 0.0
            #print(f'傅里叶变换后的模为:{abs_y[i]}')
    # 进行ifft从频域回到时域
    sig = ifft(fft_y)
    plt.figure(figsize=(20,5))
    plt.subplot(211)
    plt.plot(time,y,label='original')
    plt.ylabel('amptitude')
    plt.legend()
    # 求希尔伯特变换
    hx = fftpack.hilbert(sig)    
    # 计算呼吸周期
    height_thre = 0.35 * np.max(np.abs(sig)) # 阈值的设置是为了区分出呼吸信号
    #print(f'np.max的值为：{np.max(hx)}')
    peaks, _ = signal.find_peaks(hx, height=height_thre, distance=int(numframes*sample_width/50)) #peaks,对应峰值的索引
    #print(f'峰值个数为:{len(peaks)}')
    # 开始画出平滑曲线
    y_average = moving_average(y, 5000)
    temp_thre = 0.45* np.max(np.abs(y_average))
    temp, _ = signal.find_peaks(y_average, height=temp_thre, distance=20/12*framerate/4)

    breath_T = np.diff(peaks) / Fs #计算相邻两个元素间的差值
    # 计算呼吸频率（每分钟呼吸次数)
     
    breath_rate = 60 / np.mean(breath_T)
    #print(breath_T)
    try:
        breath_rate = math.ceil(breath_rate)
        return breath_rate, t_audio
        #print(f'每分钟的呼吸周期为{math.ceil(breath_rate)}次')
    except:
        #print('没有检测到')
        return 0, t_audio

if __name__ == '__main__':
    path = r'..\data\icbhi_dataset\audio_test_data'
    files = os.listdir(path)
    files = files[:20]
    files_path = [path + '\\' + f for f in files if f.endswith('.wav')]
    print('要等待比较长的时间...')
    #print(files)
    t_audio = 0
    t_audio_list = []
    breath_rate_truth = []
    breath_rate_pred = []
    import time
    start = time.time()
    for audio_path in files_path:
        #print(audio_path)
        i = 0
        breath_rate, t_audio = oscillogram_spectrum(audio_path)
        t_audio_list.append(t_audio)
        breath_rate_pred.append(breath_rate)
        #print(f'在{files[i]}中检测到的呼吸周期为:{breath_rate}')

    files_txt_path = [path + '\\' + f for f in files if f.endswith('.txt')]
    for txt_path in files_txt_path:
        with open(txt_path,'r') as file:
            lines = file.readlines()
        breath_rate_truth.append(len(lines))
    for i in range(len(t_audio_list)):
        breath_rate_truth[i] = int(breath_rate_truth[i]*60/t_audio_list[i])
    print(f'breath_rate_truth:{len(breath_rate_truth)}')
    print(f'breath_rate_pred:{len(breath_rate_pred)}')
    print(f't_audio_list:{len(t_audio_list)}')
    # 比较两个列表的相似性
    judge_list = []
    flagt = 0
    flagp = 0
    thre = 2
    for i in range(len(breath_rate_truth)):
        if(breath_rate_truth[i]<breath_rate_pred[i]+thre and breath_rate_truth[i]>breath_rate_pred[i]-thre):
            judge_list.append(1)
            flagt = flagt+1
        # if(breath_rate_truth[i] == breath_rate_pred[i]):
        #     flagt = flagt + 1
        else:
            judge_list.append(0)
            flagp = flagp+1
    accuracy = flagt/(flagp+flagt)
    print(flagt)
    print(flagp)
    print(f'最终的精度为:{accuracy}')
    end = time.time()
    print(f'总共用时{end-start}s')
    #envelope(audio_path)
    #test()
