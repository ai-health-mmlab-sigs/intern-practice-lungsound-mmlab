import os
import shutil
from scipy import signal
import matplotlib.pyplot as plt
import torch
import torchaudio
from torchaudio.compliance.kaldi import fbank
import numpy as np


def cycle_calc(filename: str, debug_tag=False):
    '''
    :param filename : 文件路径，不包含后缀
    :return : 预测的每个周期的开始时间
    :return : 预测周期率（呼吸次数/分钟）
    :return : 标签的每个周期开始时间
    :return : 标签周期率（呼吸次数/分钟）
    '''
    audio_path = filename + '.wav'
    label_path = filename + '.txt'
    
    audio, sample_rate = torchaudio.load(audio_path)    
    audio_time = len(audio[0]) / sample_rate
    
    if debug_tag:
        plt.clf()
        plt.figure(figsize=(15, 5))
        plt.plot(audio[0].numpy())
        plt.savefig(os.path.join(picture_path, 'cycle_calc(raw_wave).png'), 
                    bbox_inches='tight')  

    
    # 只保留低频段 1/2
    _fbank = fbank(audio, htk_compat=True, sample_frequency=sample_rate,
                use_energy=False, window_type='hanning', num_mel_bins=256,
                dither=0.0, frame_shift=10)

    if debug_tag:
        plt.clf()
        plt.imshow(_fbank.numpy(), cmap='hot', interpolation='nearest')
        plt.savefig(os.path.join(picture_path, 'cycle_calc(raw_fbank).png'),
                    bbox_inches='tight')  

    # 取低频段，再进行线性预处理，让频谱明暗变化更明显
    audio = _fbank[:, 5 : 64 + 5]
    audio = (audio - torch.mean(audio)) * 2
    audio = audio.clamp(-1, 1)
    
    if debug_tag:
        plt.clf()
        plt.imshow(audio.numpy(), cmap='hot', interpolation='nearest')
        plt.savefig(os.path.join(picture_path, 'cycle_calc(after_linear).png'),
                    bbox_inches='tight')  

    # 将频率轴压缩
    audio = torch.sum(audio, dim=1).numpy()
    
    if debug_tag:
        plt.clf()
        plt.figure(figsize=(25, 5))
        plt.plot(audio)
        plt.savefig(os.path.join(picture_path, 'cycle_calc(after_compress).png'),
                    bbox_inches='tight')  

    # 中值平滑
    medfilt_ksize, conv_ksize = 11, 11

    audio = signal.medfilt(audio, kernel_size=medfilt_ksize)
    
    for i in range(medfilt_ksize):
        audio[i] = audio[medfilt_ksize + 2]
        audio[-(i + 1)] = audio[-medfilt_ksize - 2]
        
    if debug_tag:
        plt.clf()
        plt.figure(figsize=(25, 5))
        plt.plot(audio)
        plt.savefig(os.path.join(picture_path, 'cycle_calc(after_medfilt).png'),
                    bbox_inches='tight')  

    # 均值平滑
    audio = signal.convolve(audio, np.ones(conv_ksize) / float(conv_ksize))
    # audio = signal.convolve(audio, np.ones(conv_ksize) / float(conv_ksize))
    # 似乎过度平滑效果也不好

    # 线性处理，归一化
    audio -= np.min(audio)
    audio = audio * (400 / np.max(audio))
    if debug_tag:
        plt.clf()
        plt.figure(figsize=(25, 5))
        plt.plot(audio)
        plt.savefig(os.path.join(picture_path, 'cycle_calc(reback_wave).png'),
                    bbox_inches='tight')  
    

    # 取出所有波峰波谷，间隔阈值为 0.2 秒
    peaks, _ = signal.find_peaks(audio, width=len(audio)/(audio_time)*0.2)  
    valleys, _ = signal.find_peaks(-audio, width=len(audio)/(audio_time)*0.2)
    # 到这里，数据处理就做好了，现在 audio 是一个一维的 numpy 
        
    # 所有波峰波谷组成的混合列表，(波峰/谷的时间，幅值，标签(1峰，-1谷))，然后按时间排序
    mixed_data = [[peaks[i], audio[peaks[i]], 1] for i in range(len(peaks))]
    mixed_data.extend([[valleys[i], audio[valleys[i]], -1] for i in range(len(valleys))])
    mixed_data = sorted(mixed_data)
    
    if debug_tag:
        print("Mixed Data:")
        for i in mixed_data:
            print(i)

    # 合并连续的峰或谷，取每个峰之前的最后一个谷作为呼吸周期的开始
    res = []
    for i in range(len(mixed_data) - 1):
        if (mixed_data[i][2] == -1 and mixed_data[i + 1][2] == 1):
            res.append(mixed_data[i][0])
    time_ret = [round(1.0 * res[i] * audio_time / len(audio), 3) for i in range(len(res))]
    
    # 获取标签
    labels, label_ret = [], []
    with open(label_path) as file:
        lines = file.readlines()
        for line in lines:
            x = float(line.split('\t')[0])
            label_ret.append(round(x, 3))
            labels.append(x / audio_time * len(audio))

    if debug_tag:
        plt.clf()
        plt.figure(figsize=(25, 5))
        plt.plot(audio)
        plt.scatter(peaks, audio[peaks], s=40, c='r')
        plt.scatter(valleys, audio[valleys], s=40, c='g')
        plt.vlines(res, 0, 400, colors='purple', label='pred')
        plt.savefig(os.path.join(picture_path, 'cycle_calc(final_curve).png'),
                    bbox_inches='tight')

        plt.clf()
        plt.figure(figsize=(25, 5))
        plt.vlines(res, 0, 400, colors='purple', label='pred')
        plt.vlines(labels, 0, 400, colors='green', label='standerd')
        plt.savefig(os.path.join(picture_path, 'cycle_calc(final_result).png'),
                    bbox_inches='tight')
        
    return time_ret, len(time_ret) * 60 / audio_time, label_ret, len(label_ret) * 60 / audio_time


def main():
    global picture_path
    picture_path = 'pictures/cycle_calc'
    global dataset_path
    dataset_path = 'data/icbhi_dataset/audio_test_data'

    pred_list, pred_freq, label_list, label_freq = cycle_calc(os.path.join(dataset_path, '102_1b1_Ar_sc_Meditron'), True)
    print('\nPridiction:')
    print('Time Series: ', pred_list)
    print('Cycles/Minute: ', pred_freq)

    print('\nLabel:')
    print('Time Series: ', label_list)
    print('Cycles/Minute: ', label_freq)


main()