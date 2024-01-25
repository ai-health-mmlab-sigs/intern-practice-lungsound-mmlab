import wave
from random import sample
import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import numpy as np
from math import sqrt

def hampel_filter(data, window_size=2500, threshold=2):
    """
    Hampel滤波器
    
    Parameters:
    - data: 输入的时间序列数据
    - window_size: 窗口大小，用于计算中位数
    - threshold: 阈值，用于定义异常值的判定标准
    
    Returns:
    - filtered_data: 经过Hampel滤波后的数据
    """
    n = len(data)
    filtered_data = np.copy(data)

    for i in range(window_size, n - window_size):
        # 计算窗口中的中位数和中位数绝对偏差
        window = data[i - window_size:i + window_size + 1]
        median = np.median(window)
        mad = np.median(np.abs(window - median))

        # 判断是否为异常值
        if np.abs(data[i] - median) > threshold * mad:
            # 将异常值替换为中位数
            filtered_data[i] = median

    return filtered_data

def adaptive_mean_filter(signal, window_size, threshold):
    filtered_signal = np.zeros_like(signal)
    buffer = np.zeros(window_size)
    buffer_index = 0

    for i, x in enumerate(signal):
        buffer[buffer_index] = x
        buffer_index = (buffer_index + 1) % window_size

        local_mean = np.mean(buffer)
        diff = np.abs(x - local_mean)

        if diff > threshold:
            filtered_signal[i] = local_mean
        else:
            filtered_signal[i] = x

    return filtered_signal


def detect_peaks(x, mph=0.015, mpd=50000, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's
    See this IPython Notebook [1]_.
    References
    ----------
    "Marcos Duarte, https://github.com/demotu/BMC"
    [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indexes of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indexes by their occurrence
        ind = np.sort(ind[~idel])

    return ind


def AMPD(data):
    """
    实现AMPD算法
    :param data: 1-D numpy.ndarray 
    :return: 波峰所在索引值的列表
    """
    p_data = np.zeros_like(data, dtype=np.int32)
    count = data.shape[0]
    arr_rowsum = []
    for k in range(1, count // 2 + 1):
        row_sum = 0
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                row_sum -= 1
        arr_rowsum.append(row_sum)
    min_index = np.argmin(arr_rowsum)
    max_window_length = min_index
    for k in range(1, max_window_length + 1):
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                p_data[i] += 1
    return np.where(p_data == max_window_length)[0]



def compute_envelope(signal):
    analytic_signal = hilbert(signal)
    #analytic_signal = signal + 1j * np.imag(signal)  # 构建解析信号
    envelope = np.abs(analytic_signal)  # 提取振幅包络

    return envelope

def reduce_peak(signal, threshold, reduction_factor):
    peaks = np.where(signal > threshold)[0]  # 找到超出阈值的峰值的索引
    signal[peaks] *= reduction_factor  # 将超出阈值的峰值进行幅度减小
    return signal


def enhance_small_peaks(signal, threshold, enhancement_factor):#对数放大
    small_peaks = signal < threshold  # 找到较小的峰值
    signal[small_peaks] = np.log(1 + enhancement_factor * signal[small_peaks])  # 对较小的峰值进行对数变换增强
    return signal






# 打开音频文件
filename = 'data/icbhi_dataset/audio_test_data/102_1b1_Ar_sc_Meditron.wav'
audio, sample = sf.read(filename)

fs=44100
# 设计陷波滤波器
f0 = 50.0  # 陷波中心频率
Q = 30.0  # 品质因数，控制陷波宽度
b, a = signal.iirnotch(f0, Q, fs)

# 应用陷波滤波器
filtered_signal = signal.lfilter(b, a, audio)

# 绘制原始信号和滤波后的信号
 
filtered_signal = signal.lfilter(b, a, filtered_signal)
filtered_signal = signal.lfilter(b, a, filtered_signal)
filtered_signal = signal.lfilter(b, a, filtered_signal)
filtered_signal = signal.lfilter(b, a, filtered_signal)
filtered_signal = signal.lfilter(b, a, filtered_signal)



f0 = 500
b, a = signal.iirnotch(f0, Q, fs)
filtered_signal = signal.lfilter(b, a, filtered_signal)
filtered_signal = signal.lfilter(b, a, filtered_signal)
filtered_signal = signal.lfilter(b, a, filtered_signal)
filtered_signal = signal.lfilter(b, a, filtered_signal)
filtered_signal = signal.lfilter(b, a, filtered_signal)


plt.figure()
plt.plot(audio, label='original signal')
plt.xlabel('Sampling Spots')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
plt.savefig('origin signal.png')

plt.figure()
plt.plot(filtered_signal, label='signal processed by a notch filter')
plt.xlabel('Sampling Spots')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
plt.savefig('signal processed by notch filters.png')






# 采样率
sample_rate = 44100
order = 4  # 滤波器阶数，可以根据需要进行调整
cutoff_freq = 1500  # 截止频率，以Hz为单位
ripple_db = 1  # 过渡带波纹，以dB为单位
stop_attenuation_db = 40  # 阻带衰减，以dB为单位

    # 将频率转换为归一化频率
nyquist_freq = 0.5*sample_rate  # Nyquist频率为采样率的一半
normalized_cutoff_freq = cutoff_freq / nyquist_freq

    # 设计椭圆低通滤波器
b, a = signal.ellip(order, ripple_db, stop_attenuation_db, normalized_cutoff_freq, btype='low')

    # 应用滤波器
filtered_audio = signal.filtfilt(b, a, filtered_signal)
print(filtered_audio)

plt.figure()
plt.plot(filtered_audio, label='signal processed by the first elliptical filter')
plt.xlabel('Sampling Spots')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
plt.savefig('signal processed by the first elliptical filter.png')


filtered2_audio = signal.filtfilt(b, a, filtered_audio)

print(filtered2_audio)

plt.figure()
plt.plot(filtered2_audio, label='signal processed by the second elliptical filter')
plt.xlabel('Sampling Spots')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
plt.savefig('signal processed by the second elliptical filter.png')


#####################################################################

# 设计巴特沃斯滤波器
order = 4  # 滤波器阶数
fs = sample_rate  # 采样率
f1 = 100  # 通带起始频率
f2 = 400 # 通带终止频率
Wn = np.array([f1, f2]) / (fs / 2)  # 归一化频率
b, a = signal.butter(order, Wn, btype='bandpass')


filtered3_signal = signal.filtfilt(b, a, filtered2_audio)

plt.figure()
plt.plot(filtered3_signal, label='signal processed by the first Butterworth filter')
plt.xlabel('Sampling Spots')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
plt.savefig('signal processed by the first Butterworth filter.png')


#第二个butter滤波
order = 4  # 滤波器阶数
fs = sample_rate  # 采样率
f1 = 70  # 通带起始频率
f2 = 400 # 通带终止频率
Wn = np.array([f1, f2]) / (fs / 2)  # 归一化频率
b, a = signal.butter(order, Wn, btype='bandpass')

filtered3_signal = signal.filtfilt(b, a, filtered3_signal)



plt.figure()
plt.plot(filtered3_signal, label='signal processed by the second Butterworth filter')
plt.xlabel('Sampling Spots')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
plt.savefig('signal processed by the second Butterworth filter.png')


#########################################################
#使用对数进行滤波，最后选择先hampel后对数滤波
filter4=enhance_small_peaks(filtered3_signal,0.01,3)
plt.figure()
plt.plot(filter4, label='Logarithmic filter')
plt.xlabel('Samples Spots')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
plt.savefig('Logarithmic filter.png')


##########################################################
#希尔伯特包络，但效果不佳,因此未使用
envelope = compute_envelope(filter4)
plt.figure()
plt.plot(envelope, label='envelope')
plt.xlabel('Sampling Spots')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
plt.savefig('envelope.png')

filter3=hampel_filter(filtered3_signal)#进行hampel滤波，去除异常值
filter3=enhance_small_peaks(filter3,0.01,2)#放大较小的波峰

##########################################################
#检测波峰并标记，利用波峰计算周期
peaks = detect_peaks(filter3)
plt.figure()
plt.plot(filter3)
plt.plot(peaks, filter3[peaks], "o")
plt.xlabel('Sampling Spots')
plt.ylabel('Amplitude')
plt.title("Final signal and identified peaks")
plt.legend()
plt.show()
plt.savefig('Final signal and identified peaks.png')

print(len(peaks))
time=len(filtered3_signal)/44100
print("time")
print(time)
breathing_rate=(60/time)*len(peaks)
print(f"每分钟呼吸率: {breathing_rate:.2f}")


'''
####################################################

#寻找功率谱密度，使用welch方法  
frequencies, spectrum = signal.periodogram(filter4, fs=sample_rate)

print(frequencies)
peak_indices, _ = signal.find_peaks(spectrum)

#np.set_printoptions(threshold=np.inf)

    # 估计主要峰值对应的频率
main_peak_frequency = frequencies[peak_indices[0]]
print(peak_indices)
print("mainfrequency")
print(main_peak_frequency)

    # 估计呼吸周期大小
breathing_period = 1 / main_peak_frequency
    
breathing_rate= 60/breathing_period

print(f"每分钟呼吸率: {breathing_rate:.2f}")

'''


