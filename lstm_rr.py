from copy import deepcopy
import os
import sys
import json
import warnings

import math
import time
import random
import pickle
import argparse
import numpy as np
from scipy import signal

import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torchvision import transforms
import torchaudio
from torchaudio import transforms as T
from torchaudio.compliance.kaldi import fbank

from util.time_warping import sparse_image_warp
from util.augmentation import SpecAugment
from util.misc import adjust_learning_rate, warmup_learning_rate, AverageMeter
from method import PatchMixLoss, PatchMixConLoss
warnings.filterwarnings("ignore")


class Args:
    def __init__(self):
        '''模型选择相关参数'''
        self.lstm_bidirection = False # 是否启用双向 lstm
        self.use_fbank = True
        self.use_augment = False
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.categories = 100

        '''文件路径相关参数'''
        self.resz = None
        self.split_file_path = 'data/icbhi_dataset/official_split.txt'
        self.data_folder_path = 'data/icbhi_dataset/audio_test_data'
        self.lstm_rr_save_dir = 'lstm_rr_save'
        self.debug_dir_path = 'cq_debug_info'
        self.abnormal_samples_file = 'cq_debug_info/abnormal_samples.txt'

        '''训练强度相关的参数'''
        self.data_slice = 100        # 切片间隔，越小数据量越大
        self.print_frequency = 100
        self.epochs = 100
        self.batch_size = 1
        self.num_workers = 0
        
        '''Specaugment相关的参数'''
        self.specaug_policy = 'LB'
        self.specaug_mask = 'mean'
        
        '''学习率相关的参数'''
        self.learning_rate = 0.01
        self.cosine = True          # 在 util.misc.adjust_learning_rate() 引用，不知道具体有什么用
        self.lr_decay_rate = 0.1    # 在 util.misc.adjust_learning_rate() 引用，不知道具体有什么用
        self.warm = True
        self.warm_epochs = 120
        
        if self.warm:
            self.warmup_from = self.learning_rate * 0.1
            self.warm_epochs = 10
            if self.cosine:
                eta_min = self.learning_rate * (self.lr_decay_rate ** 3)
                self.warmup_to = eta_min + (self.learning_rate - eta_min) * (
                        1 + math.cos(math.pi * self.warm_epochs / self.epochs)) / 2
            else:
                self.warmup_to = self.learning_rate

        '''数据预归一化规格相关的参数'''
        self.std_audio_time = 10        # 将所有音频的归一化时长
        self.std_sample_rate = 16000    # 标准采样频率
        self.std_audio_len = self.std_audio_time * self.std_sample_rate
        self.std_fbank_len = 998        # 频谱图的标准长度，由时长得到
        self.std_fbank_width = 128      # 标准频谱图宽度
        self.std_lstm_len = (self.std_fbank_len + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.std_lstm_width = (self.std_fbank_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        # 我希望的是，卷积和池化都不要改变形状
        assert self.std_lstm_len == self.std_fbank_len and self.std_lstm_width == self.std_fbank_width
        
        '''DEBUG 相关的符号'''
        self.debug_preprocessor = False
        self.debug_lstm = False
        self.debug_dataset = False
        self.debug_training = False
        self.debug_performance = False


def draw_curl_pic(x, path : str, title=''):
    plt.clf()
    plt.plot(x)
    plt.title(title)
    plt.show()
    plt.savefig(path)
    
    
def draw_hot_pic(x, path : str, title=''):
    plt.clf()
    plt.imshow(x, cmap='hot', interpolation='nearest')
    plt.title(title)
    plt.show()
    plt.savefig(path)  
    

class Preprocessor(nn.Module):
    '''
    图像预处理器。
    '''
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=args.kernel_size, stride=args.stride, padding=args.padding, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=args.kernel_size, stride=args.stride, padding=args.padding)
        
    def forward(self, audio):
        '''
        输入 audio 是一个三维 Tensor（频谱图），第一维是batchsize，第二、三维是频谱图（时间、频率，值是能量）
        变换后生成的 _fbank 也是二维 Tensor，第一维是时间，第二维是频率。
        返回的 audio 是一个一维 Tensor，代表不同时间点对应的振幅。
        '''        
        
        # 考虑是否启用卷积和池化
        audio = audio.reshape(args.batch_size, 1, args.std_fbank_len, args.std_fbank_width)
        audio = self.conv(audio)
        audio = self.pool(audio)
        audio = audio.reshape(args.batch_size, args.std_lstm_len, args.std_lstm_width)
        if self.training and args.debug_preprocessor:
            draw_hot_pic(audio[0].cpu().numpy(), os.path.join(args.debug_dir_path, 
                                                           'image_of_preprocessed_data(after conv).png'))
        
        # ? 存疑，fbank的频谱图储存的是能量密度的对数，是否需要取指数否则数据过小
        # ? 如果取了指数，会造成梯度爆炸，但是想不明白为什么   
        # audio = torch.exp(audio)

        return audio


class LSTM(nn.Module):
    '''
    通过 args.lstm_bidirection 判断是否启用双向 lstm
    '''
    def __init__(self):
        super().__init__()
        self._input_size = args.std_fbank_width
        self._hidden_layer_size = args.std_lstm_width
        self._lstm = nn.LSTM(self._input_size, self._hidden_layer_size, batch_first=True, bidirectional=args.lstm_bidirection)

    def forward(self, x):
        '''
        :param x : 输入的 x 是一个三维 Tensor：第一维是 batch，第二维是时间，第三位是 preprocessor 处理过后的波形。
        :return : 返回的是一个三维 Tensor：第一维是batch，第二维是时间，第三维是输出的 lstm 隐藏层。
        '''
        # 确保输入是经过归一化的
        assert x.shape == torch.Size([args.batch_size, args.std_lstm_len, args.std_lstm_width])
        
        h0 = torch.zeros((1 + args.lstm_bidirection), args.batch_size, self._hidden_layer_size).cuda()
        c0 = torch.zeros((1 + args.lstm_bidirection), args.batch_size, self._hidden_layer_size).cuda()

        # output 是一个三维 Tensor：第一维是 batch，第二维是时间片，第三维是隐藏层参数
        output, _ = self._lstm(x, (h0, c0))
        
        if self.training == args.debug_training and args.debug_lstm:
            draw_curl_pic(x[0].cpu().detach().numpy(), os.path.join(args.debug_dir_path, 'lstm_data(input).png'))
            draw_curl_pic(output[0].cpu().detach().numpy(), os.path.join(args.debug_dir_path, 'lstm_data(output).png'))
            print()
        
        output = output.reshape(args.batch_size * (1 + args.lstm_bidirection) * args.std_lstm_len * args.std_lstm_width)
        return output
    

class SimpleRegressor(nn.Module):
    '''
    按照任务书中的仓库，简单的全连接层。在 ICBHI 上效果不好。
    '''
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(args.batch_size * (1 + args.lstm_bidirection) * args.std_lstm_len * args.std_lstm_width, 
                                args.batch_size)
    
    def forward(self, x):
        x = self.linear(x)
        return x


class DNNRegressor(nn.Module):    
    '''
    多层的全连接回归器，用于将 LSTM 生成的所有时间片输出综合计算一个回归值
    很难搞。设计的简单了计算回归没效果，设计复杂了又梯度爆炸。
    '''
    def __init__(self):
        super().__init__()
        self._layers = nn.Sequential(
            nn.Linear(args.batch_size * (1 + args.lstm_bidirection) * args.std_lstm_len * args.std_lstm_width, 
                      args.batch_size * args.std_lstm_width),
            nn.ReLU(),
            nn.Linear(args.batch_size * args.std_lstm_width, args.batch_size * args.std_lstm_width),
            nn.ReLU(),
            nn.Linear(args.batch_size * args.std_lstm_width, args.batch_size * args.std_lstm_width),
            nn.ReLU(),
            nn.Linear(args.batch_size * args.std_lstm_width, args.batch_size)
        )

    def forward(self, x):
        '''
        :param x : 输入是二维 Tensor。第一维是 batch，第二维是一个时间序列
        :return : 输出是一个一维 Tensor，即一个batch的预测结果
        '''
        x = x.reshape(args.batch_size * (1 + args.lstm_bidirection) * args.std_lstm_len * args.std_lstm_width)
        x = self._layers(x)
        return x


class ResRegressor(nn.Module):
    '''
    不是 Resnet，意图是使用残差思想避免梯度爆炸。对结果提升有效，但是不能避免 MSE 的梯度爆炸。
    '''
    class ResnetBlock(nn.Module):
        def __init__(self, size):
            super().__init__()
            self.relu = nn.ReLU()
            self.linear = nn.Linear(size, size)
        
        def forward(self, x):
            y = self.linear(x)
            y = self.relu(y)
            return x + y
        
    def __init__(self):
        super().__init__()
        self._depth = 54
        midsize = args.batch_size * args.std_lstm_width
        self.layers = nn.Sequential()
        for i in range(self._depth):
            self.layers.add_module('block' + str(i), ResRegressor.ResnetBlock(midsize))

        self.linear_in = nn.Linear(args.batch_size * (1 + args.lstm_bidirection) * args.std_lstm_len * args.std_lstm_width, 
                      args.batch_size * args.std_lstm_width)
        self.linear_out = nn.Linear(args.batch_size * args.std_lstm_width, args.batch_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.reshape(args.batch_size * (1 + args.lstm_bidirection) * args.std_lstm_len * args.std_lstm_width)
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.layers(x)
        x = self.linear_out(x)
        return x


# TODO 写了一半否决了经典 ResNet，但是后来想想似乎可行？毕竟卷积有潜力充分利用频谱图像的性质！
# class ResNetRegressor(nn.Module):
#     '''
#     写了一个 Resnet 用来算回归。
#     '''
#     class ResnetBlock(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
#             self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
#             self.relu = nn.ReLU()
#             self.bn = nn.BatchNorm2d(1)

#         def forward(self, x):
#             y = self.conv1(x)
#             y = self.bn(y)
#             y = self.relu(y)
#             y = self.conv2(x)
#             y = self.bn(y)
#             y += x
#             y = self.relu(y)
#             return y


#     def __init__(self):
#         super().__init__()
#         self._depth = 6
#         midsize = args.batch_size * args.std_lstm_width
#         self.layers = nn.Sequential()
#         for i in range(self._depth):
#             self.layers.add_module('block' + str(i), ResNetRegressor.ResnetBlock(midsize))

#         self.linear_in = nn.Linear(args.batch_size * (1 + args.lstm_bidirection) * args.std_lstm_len * args.std_lstm_width, 
#                       args.batch_size * args.std_lstm_width)
#         self.linear_out = nn.Linear(args.batch_size * args.std_lstm_width, args.batch_size)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         y = self.linear_in(x)



class CQModel(nn.Module):
    '''
    抽象出本任务的一般模型：预处理层，循环层，分类层。
    update: 舍弃了预处理层，希望让模型学到更多东西。
    '''
    def __init__(self, preprocessor, recurrenter, regressor):
        super().__init__()
        self._layers = nn.Sequential(
            # preprocessor,
            recurrenter,
            regressor
        )
        
    def forward(self, x):
        return self._layers(x)
    
    
class CQCriterion(nn.Module):
    '''
    损失函数：类 Tanh 函数, tahn(loss)*2-1,  期望让更多值预测正确
    效果似乎不好，会梯度爆炸，不如 MSE 和 L1Loss
    '''
    def __init__(self):
        super().__init__()
    
    def forward(self, prediction, label):
        assert prediction.shape == label.shape
        loss = torch.div(prediction, label)
        loss = torch.sigmoid(loss) * 2 - torch.ones(args.batch_size).cuda()
        return loss.abs().mean()
    

class ICBHIDataset(Dataset):    
    def __get_device_id(self, filename : str):
        '''
        :param filename: str 不包含前缀目录的文件名
        :return: int 设备类型编号
        '''
        return self._device_to_id[filename.strip().split('_')[-1]]
    
    def __preprocess_audio(self, audio, metadata, label):
        '''
        用来对数据进行预处理。
        :param audio : 二维Tensor。第一维是通道，第二维是时间轴上的波形
        :param metadata, label : 文件元信息，仅用于调试打印
        '''
        if args.debug_preprocessor:
            draw_curl_pic(audio[0].cpu().numpy(), 
                          os.path.join(args.debug_dir_path, 'dataset(raw_wave).png'),
                          "file:{}, [{}, {}], label={}".format(metadata[0], metadata[1], metadata[2], label))
        
        # 只保留低频段 1/2
        _fbank = fbank(audio, htk_compat=True, sample_frequency=args.std_sample_rate,
                    use_energy=False, window_type='hanning', num_mel_bins=args.std_fbank_width * 2,
                    dither=0.0, frame_shift=10)

        if args.debug_dataset:
            draw_hot_pic(_fbank.cpu().numpy(), os.path.join(args.debug_dir_path, 'dataset(fbank).png'))

        audio = _fbank[:, 5 : args.std_fbank_width + 5]
        # audio = _fbank

        # 感觉下面这两行的效果不错，不知道可不可以更优
        # for i in range(args.batch_size):
        audio = (audio - torch.mean(audio)) * 2
        audio = audio.clamp(-1, 1)
        
        if args.debug_dataset:
            draw_hot_pic(audio.cpu().numpy(), os.path.join(args.debug_dir_path, 'dataset(after_linear).png'))

        # 将频率轴压缩 -> 中值平滑 -> 均值平滑
        medfilt_ksize, conv_ksize = 21, 100
        # audio = torch.sum(audio, dim=1).numpy()
        audio = signal.medfilt(audio.numpy(), kernel_size=(medfilt_ksize, 1))
        audio = torch.from_numpy(audio)
        for i in range(medfilt_ksize):
            audio[i] = audio[medfilt_ksize + 2]
            audio[-(i + 1)] = audio[-medfilt_ksize - 2]
            
        # if args.debug_dataset:
        #     draw_curl_pic(audio, os.path.join(args.debug_dir_path, 'dataset(after_medfilt).png'))

        # 不知道这里是否需要均值平滑操作
        # ! 要么在这里平滑，要么加上 preprocessor 层
        # audio = signal.convolve2d(audio, (conv_ksize, 1), mode='same')

        audio -= torch.min(audio)
        audio = audio * (400 / torch.max(audio))
        
        # if args.debug_dataset:
            # draw_curl_pic(audio.cpu().numpy(), os.path.join(args.debug_dir_path, 'dataset(after_smooth).png'))
            # print(metadata)
            # print(label)
        return audio
    
    def __preprocess_label(self, filename : str, start, end):
        '''计算标签标签（该文件某个时间段内的呼吸数量）
        :param filename(str) : 文件名
        :param start(float) : 起始时间
        :param end(float) : 结束时间
        '''
        label_path = os.path.join(args.data_folder_path, filename + '.txt')
        start_list = []
        cycle_cnt = 0
        with open(label_path) as file:
            lines = file.readlines()
            for line in lines:
                cur = float(line.split('\t')[0])
                if start <= cur <= end:
                    start_list.append(cur)
        
        # 报告这是一个劣质时间窗口
        if len(start_list) <= 1:
            return -1
        
        # 取完整的周期，求平均，推广到这段序列        
        average = (start_list[-1] - start_list[0]) / (len(start_list) - 1)
        ret = (end - start) / (average)
        return ret

    
    def __get_file_data(self, filename : str):
        '''
        :param filename: str 不包含前缀目录的文件名
        :return List[Tensor] 由这个文件新生成的数据列表（如果是训练集包含增强的数据），数据是频谱图
        :return List[float] 新生成的数据对应的标签，含义是平均呼吸周期
        :return List[int] 新生成的数据对应的原信息，一开始想统计采集设备等，后来设为了文件名，方便调试
        '''
        
        # 重采样
        audio_path = os.path.join(args.data_folder_path, filename + '.wav')
        audio, raw_sample_rate = torchaudio.load(audio_path)
        if raw_sample_rate != args.std_sample_rate:
            resample = T.Resample(raw_sample_rate, args.std_sample_rate)
            audio = resample(audio)
            
        # ! 如果时长不够，就暴力地复制拼接。这里不知道是否需要优化
        while len(audio[0]) < args.std_audio_len:
            audio = torch.cat((audio, audio), dim=1)

        audio_time = len(audio[0]) / raw_sample_rate

        # # 处理标签：每个呼吸周期的时长
        # label_path = os.path.join(args.data_folder_path, filename + '.txt')
        # with open(label_path) as file:
        #     cycle_cnt = len(file.readlines())
        # label = audio_time / cycle_cnt
        
        # 返回的三个列表，分别是从这个文件获取的：音频 tensor列表，标签列表，元信息列表
        ret_audio, ret_label, ret_metadata = [], [], []
        
        # 重点：归一化，生成频谱图
        for start in range(0, len(audio[0]) - args.std_audio_len, args.std_sample_rate * args.data_slice):
            end = start + args.std_audio_len
            metadata = (filename, start / args.std_sample_rate, end / args.std_sample_rate)
            label = self.__preprocess_label(filename, start / args.std_sample_rate, 
                                            end / args.std_sample_rate)
            
            # 认为是一段不好的数据，会影响训练
            if self._train and label == -1:
                continue
            
            audio_slice = audio[:, start : end]
            audio_slice = self.__preprocess_audio(audio_slice, metadata, label)
            assert args.std_fbank_len == len(audio_slice)
            
            # 元信息记录了文件名字，起始时间和结束时间
            ret_audio.append(audio_slice)
            ret_label.append(label)
            ret_metadata.append(metadata)
            
            # 数据增强
            if args.use_augment and self._train:    
                for augment in args.augment_methods:
                    new_audio_data = augment(audio_slice)
                    ret_audio.append(new_audio_data)
                    ret_label.append(label)
                    ret_metadata.append(metadata)
        
        return ret_audio, ret_label, ret_metadata

    def __init__(self, train : bool):
        '''
        这里最初想把元信息（听诊器，部位）等也考虑进来，但是实际上没完成。
        '''
        # 运用原仓库中的 Specaugment 增强
        self._augment = SpecAugment(args)

        self._train = train
        self._device_to_id = {'Meditron': 0, 'LittC2SE': 1, 'Litt3200': 2, 'AKGC417L': 3}
        self._filenames = set()

        with open(args.split_file_path) as official_split:
            lines = official_split.readlines()
            for line in lines:
                filename, tag = line.strip().split('\t')
                if (('train' in tag and train) or ('test' in tag and not train)):
                    self._filenames.add(filename)

        self._name_to_id = {}
        self._id_to_name = {}

        self._audio_images = []
        self._audio_labels = []
        self._audio_metadatas = []
        
        # 遍历所有数据文件，每个数据文件可能得到多组数据
        for idx, filename in enumerate(self._filenames, start=0):
            self._name_to_id[filename] = idx
            self._id_to_name[idx] = filename
            
            new_images, new_labels, new_metadatas = self.__get_file_data(filename)
            self._audio_images.extend(new_images)
            self._audio_labels.extend(new_labels)
            self._audio_metadatas.extend(new_metadatas)

        # for x in enumerate(self._audio_images):
        #     label_p = torch.zeros(args.categories)
        #     for i in range (0.7 * x, min(categories, 1.3 * x)):
        #         label_p[i] = cos((i - x) / 10)
        #     self._audio_labels_probabilities.append(label_p)

        print('Successfully loaded {} set.'.format('train' if self._train else 'test '))

    def __len__(self):
        return len(self._audio_images)

    def __getitem__(self, index):
        audio = self._audio_images[index]
        if self._train: # 只在训练集上使用数据增强
            # 原仓库中的 specaugment 带有通道数，所以这里要增一维
            audio = audio.reshape(1, args.std_fbank_len, args.std_fbank_width)
            audio = self._augment(audio)
            audio = audio.reshape(args.std_fbank_len, args.std_fbank_width)
        return audio, self._audio_labels[index], self._audio_metadatas[index]


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    losses = []

    for idx, (image, label, metadata) in enumerate(train_loader):
        image = image.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # 原仓库中实现的函数
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        # with torch.autograd.detect_anomaly():
        with torch.cuda.amp.autocast():
            prediction = model(image)
            current_loss = criterion(prediction.double(), label.double())
            
        losses.append(current_loss.item())
        optimizer.zero_grad()
        current_loss.backward()
        if current_loss.item() > 10000:
            print(current_loss)
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1, norm_type=2)
        optimizer.step()

        if (idx + 1) % args.print_frequency == 0:
            print('Train: [{}][{}/{}]'.format(epoch + 1, idx + 1, len(train_loader)))
            sys.stdout.flush()

    return np.mean(losses)


def validate(validate_loader, model, criterion, epoch=-1):
    save_bool = False
    model.eval()
    losses = []
    
    performance = []
    # if os.path.exist(args.abnormal_samples_list):
    #     os.remove(args.abnormal_samples_list)

    with torch.no_grad():
        for idx, (image, label, metadata) in enumerate(validate_loader):
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            
            with torch.cuda.amp.autocast():
                prediction = model(image)
                current_loss = criterion(prediction, label)
                losses.append(current_loss.item())
            
            error = abs(prediction[0] - label[0]) / (label[0])
            performance.append(error)

            # 把明显的错误数据导出来
            if epoch == -1 and current_loss.item() > 10:
                for i in range(args.batch_size):
                    with open('lstm_rr_results', 'w') as file:
                        file.write("metadata: {}\nlabel:{:.3f}\nprediction:{:.3f}\n".format(metadata[i], label[i].item(), prediction[i].item()))
                # draw_curl_pic(image[0].cpu().numpy(), os.path.join(args.debug_dir_path, metadata))
                
        if (idx + 1) % args.print_frequency == 0:
                print('Validate: [{}][{}/{}]\t'.format(epoch + 1, idx + 1, len(validate_loader)))
                sys.stdout.flush()

    succ = [0 for i in range(6)]
    for i in range(6):
        succ[i] = sum(1 for x in performance if x < i / 10) / len(performance)

    print('\033[031m')
    with open('lstm_log.txt', 'a') as log:
        for i in range(1, 6):
            current_res = '0.{} Success ratio: {:.5f}'.format(i, succ[i])
            print(current_res)
            log.write(current_res + '\n')  
        print('Loss: {:.5f}'.format(np.mean(losses)))
        log.write('Loss: {:.5f}\n\n'.format(np.mean(losses)))
    print('\033[0m')

    return np.var(losses), succ


def work(recurrenter, regressor):    
    # 数据集
    train_dataset = ICBHIDataset(True)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               sampler=None,
                                               drop_last=True)

    validate_dataset = ICBHIDataset(False)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           sampler=None,
                                           drop_last=True)

    # 模型
    preprocessor=None
    model = CQModel(preprocessor, recurrenter, regressor).cuda()

    # 使用 MSE 总是会不定时地发生梯度爆炸。个人总结为：数据脏，训练的时候偶尔会产生极大的误差，对训练结果产生极大影响。
    criterion=nn.L1Loss()
    # criterion=nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    
    # 数据集建好之后，更新一下标准归一大小
    if args.std_fbank_len != -1:
        args.std_audio_len = args.std_fbank_len

    for epoch in range(args.epochs):
        print("\033[034mStart epoch {}/{}\033[0m".format(epoch + 1, args.epochs))
        adjust_learning_rate(args, optimizer, epoch)   # 这是原仓库中的函数，不知道有没有实际作用
        loss = train(train_loader, model, criterion, optimizer, epoch)
        loss, succ = validate(validate_loader, model, criterion, epoch)
    
    validate(validate_loader, model, criterion, -1)

    print('Finished all epochs.')
    
    # 把实验结果写到日志里面
    with open('lstm_rr_results.txt', 'a') as log:
        log.write(str(datetime.now()) + '\nFinished New Experiment.' + '\n\n\n')
        log.write('recurrenter: {}\n'.format(type(recurrenter).__name__))
        log.write('regressor: {}\n'.format(type(regressor).__name__))
        log.write('criterion: {}\n'.format(type(criterion).__name__))
        log.write('optimizer: {}\n'.format(type(optimizer).__name__))
        
        for i in range(1, 6):
            current_res = '0.{} Success ratio: {:.5f}'.format(i, succ[i])
            log.write(current_res + '\n')  

        params = [attr for attr in dir(args) if not callable(getattr(args, attr)) and not attr.startswith("__")]
        for v in params:
            log.write('{}: {}\n'.format(v, getattr(args, v)))
        log.write('\n\n\n\n\n\n\n\n\n\n')


def main():
    global args

    args = Args()
    work(LSTM(), ResRegressor())

    args = Args()
    work(LSTM(), DNNRegressor())

    args = Args()
    work(LSTM(), SimpleRegressor())

    args = Args()
    args.data_slice = 3
    work(LSTM(), ResRegressor())
    
    args = Args()
    args.data_slice = 3
    work(LSTM(), DNNRegressor())
    
    args = Args()
    args.data_slice = 3
    work(LSTM(), SimpleRegressor())


    # 排列组合进行试验
    for i in range(2):
        for j in range(3):
            for k in range(2):
                args = Args()

                # i 决定是否启用双向 lstm
                # 注意：lstm 更新到高版本后，bidirectional参数不能使用int类型（0或1），要使用bool类型（False或True），要不会报错。
                args.lstm_bidirection = bool(i)
                recurrenter = LSTM()

                # j 决定使用什么回归器
                if j == 0:
                    regressor = SimpleRegressor()
                elif j == 1:
                    regressor = DNNRegressor()
                elif j == 2:
                    regressor = ResRegressor()

                # k 决定是否使用高强度的 data augment
                if k == 0:
                    args.data_slice = 100
                else:
                    args.data_slice = 3

                work(recurrenter, regressor)

main()
    