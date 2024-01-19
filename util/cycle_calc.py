import random
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import signal

def filter_and_amplify(audio, sr, low_cutoff, high_cutoff, amplification_factor):
    # 设计带通滤波器
    nyquist = 0.8 * sr
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = signal.butter(4, [low, high], btype='band')

    # 应用滤波器
    filtered_audio = signal.filtfilt(b, a, audio)

    # 放大低峰
    amplified_audio = filtered_audio * amplification_factor

    return amplified_audio
def calc_breath_cycle(filename, picture = False):
    audio, sr = librosa.load(filename, sr=8000)
    time_audio = np.arange(len(audio)) / sr
    # 滤波并放大低峰
    low_cutoff = 1000  # 低通滤波器截止频率
    high_cutoff = 4000  # 高通滤波器截止频率
    amplification_factor = 5  # 放大因子

    # sampled_indices = random.sample(range(len(audio)), k=1000)  # 选择1000个点进行绘制
    # #print(sampled_indices)
    # sampled_indices.sort()
    # time_audio = time_audio[sampled_indices]
    # audio = audio[sampled_indices]
    #print(time_audio.shape, audio.shape)
    filtered_and_amplified_audio = filter_and_amplify(audio, sr, low_cutoff, high_cutoff, amplification_factor)
    # 计算低峰的值
    duration = len(audio) / sr
    samples_to_keep = int(duration * sr)
    low_peaks, _ = find_peaks(-filtered_and_amplified_audio[:samples_to_keep])  # 注意这里取负值，因为 find_peaks 寻找的是峰而不是谷
    time_filtered = np.arange(len(filtered_and_amplified_audio)) / sr
    if picture:
        plt.figure(figsize=(24, 8))
        plt.plot(time_audio, audio, label="Original Audio")
        # plt.plot(time_filtered, filtered_and_amplified_audio, label="Filtered and Amplified Audio")
        plt.scatter(low_peaks / sr, filtered_and_amplified_audio[low_peaks], c='r', label="Low Peaks")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

    # 计算20s所有低峰的值
    low_peaks_values = filtered_and_amplified_audio[low_peaks]
    # print(f"Values of Low Peaks in the first {duration} seconds: {low_peaks_values}")
    return len(low_peaks_values) * 3 / sr


