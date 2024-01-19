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
    print("通道数channel:", nchannels)
    print("比特sample_width:", sample_width)
    print("采样率framerate:", framerate)
    print("采样点数numframes:", numframes)
    # 建一个y的数列，用来保存后面读的每个frame的波幅amplitude。
    y = np.zeros(numframes)
    print(f'y的长度为:{len(y)}')
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
    # 选择频率前后对比
    # plt.figure(figsize=(20,5))
    # plt.plot(time[:5000],abs_y[:5000],label='fft')
    # plt.legend()
    # plt.plot(time[:5000],np.abs(fft_y)[:5000],'orange',label='selected fft')
    # plt.legend()
    # plt.show()
    # 进行ifft从频域回到时域
    sig = ifft(fft_y)
    plt.figure(figsize=(20,5))
    plt.subplot(211)
    plt.plot(time,y,label='original')
    plt.ylabel('amptitude')
    plt.legend()
    # 求希尔伯特变换
    hx = fftpack.hilbert(sig)    
    #hx = np.sqrt(hx**2+sig*2)
    #hx = np.abs(hx)
    # 计算呼吸周期
    height_thre = 0.35 * np.max(np.abs(sig)) # 阈值的设置是为了区分出呼吸信号
    #print(f'np.max的值为：{np.max(hx)}')
    peaks, _ = signal.find_peaks(hx, height=height_thre, distance=int(numframes*sample_width/50)) #peaks,对应峰值的索引
    #print(peaks)
    point_x = []
    point_y = []
    for i in peaks:
        point_x.append(i)
        point_y.append(hx[i])
    point_x = np.array(point_x)
    point_y = np.array(point_y)
    print(f'峰值个数为:{len(point_y)}')
    # 开始画出平滑曲线
    plt.subplot(212)
    y_average = moving_average(y, 5000)
    temp_thre = 0.45* np.max(np.abs(y_average))
    temp, _ = signal.find_peaks(y_average, height=temp_thre, distance=20/12*framerate/4)
    print(f'滤波后峰值为：{temp} {len(temp)}')
    plt.plot(time,y_average,'orange',label='move_average')
    #plt.scatter(time[peaks],y_average[peaks],c='red')
    plt.plot(time[temp],y_average[temp],'.',color='red') #把'r'调至.前面会连线?具体有时间再去研究
    plt.xlabel('time')
    plt.ylabel('amptitude')
    plt.legend()
    plt.show()
    breath_T = np.diff(peaks) / Fs #计算相邻两个元素间的差值
    # 计算呼吸频率（每分钟呼吸次数)
    breath_rate = 60 / np.mean(breath_T)
    #print(breath_T)
    print(f'每分钟的呼吸周期为{math.ceil(breath_rate)}次')

if __name__ == '__main__':
    audio_path = '102_1b1_Ar_sc_Meditron.wav'
    oscillogram_spectrum(audio_path)
    #envelope(audio_path)
    #test()
