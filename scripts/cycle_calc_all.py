import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import os
import numpy as np
from scipy.signal import find_peaks

# 检查是否有 GPU 可用，并据此设置 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, audio_dir, transform=None):
        self.audio_dir = audio_dir
        self.audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        self.transform = transform

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        waveform, sample_rate = torchaudio.load(audio_path)
        # 获取标签
        labels = self.get_label_from_txt(audio_path)

        # 确定最大标签数量
        max_labels = 50  # 假设最多有50个呼吸周期

        # 创建一个固定大小的张量来存储标签，使用占位符填充
        label_tensor = torch.zeros((max_labels, 2))  # 假设每个标签是两个数

        # 将实际标签复制到张量中
        for i, label in enumerate(labels):
            if i >= max_labels:
                break
            label_tensor[i] = torch.tensor(label)
        
        if self.transform:
            waveform = self.transform(waveform)
            
        # 确定 Mel 频谱的目标大小
        target_size = (64, 882)  # 例如，64个 Mel 滤波器和882个时间步长

        # 如果 Mel 频谱太大，裁剪它
        if waveform.size(2) > target_size[1]:
            waveform = waveform[:, :, :target_size[1]]
        # 如果 Mel 频谱太小，填充它
        elif waveform.size(2) < target_size[1]:
            padding_size = target_size[1] - waveform.size(2)
            waveform = torch.nn.functional.pad(waveform, (0, padding_size))

        return waveform, label_tensor

    def get_label_from_txt(self, audio_path):
        # 获取对应的 txt 文件路径
        txt_path = audio_path.replace('.wav', '.txt')

        # 确保 txt 文件存在
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Label file not found for {audio_path}")

        # 读取 txt 文件并解析
        labels = []
        with open(txt_path, 'r') as file:
            for line in file:
                start_time, end_time, _, _ = line.strip().split('\t')
                labels.append((float(start_time), float(end_time)))

        return labels

class AudioClassifier(nn.Module):
    def __init__(self, fixed_length, in_features):
        super(AudioClassifier, self).__init__()
        
        # 将 fixed_length 保存为类的属性
        self.fixed_length = fixed_length

        # 示例：定义一些卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 定义全连接层，输出大小为时间序列长度乘以每个时间点的值数量
        self.fc = nn.Linear(in_features, self.fixed_length * 2)

    def forward(self, x):
        # 通过卷积层
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # 展平特征以供全连接层使用
        x = x.view(x.size(0), -1)
        # 通过全连接层
        x = self.fc(x)
        # 调整输出尺寸以匹配时间序列的长度
        x = x.view(-1, self.fixed_length, 2)  # 使用类的属性
        x = torch.sigmoid(x)  # 确保输出在 0 到 1 之间
        return x

def train(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 前向传播
            output = model(data)

            # 计算损失
            loss = criterion(output, target)

            # 后向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')
                
def post_process_output(output, sample_rate, threshold=0.5, min_distance=30):
    """
    从模型输出中后处理以计算呼吸周期数和每分钟呼吸率。

    参数:
        output (torch.Tensor): 模型的输出张量，形状为 (N,)，表示每个时间点的呼吸事件概率。
        sample_rate (int): 音频的采样率。
        threshold (float): 用于确定呼吸事件的阈值。
        min_distance (int): 用于峰值检测的最小距离，避免检测到过于密集的峰值。

    返回:
        int: 计算出的呼吸周期数。
        float: 每分钟呼吸率。
    """
    # 将模型输出转换为 numpy 数组
    probabilities = output.cpu().numpy()

    # 寻找概率超过阈值的峰值
    peaks, _ = find_peaks(probabilities, height=threshold, distance=min_distance)

    # 计算呼吸周期数
    num_cycles = len(peaks)

    # 计算每分钟呼吸率
    duration_in_seconds = len(probabilities) / sample_rate
    rate_per_minute = num_cycles / duration_in_seconds * 60

    return num_cycles, rate_per_minute

def evaluate_model(model, file_path, transform, device):
    # 加载音频
    waveform, sample_rate = torchaudio.load(file_path)

    # 应用转换（例如 Mel 频谱）
    if transform is not None:
        waveform = transform(waveform)

    # 确保音频是适当长度（与训练时相同）
    # 此处需要根据您的具体情况调整
    target_length = 882000  # 20秒 * 44100样本/秒
    if waveform.size(1) > target_length:
        waveform = waveform[:, :target_length]
    elif waveform.size(1) < target_length:
        padding_size = target_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, padding_size))

    # 将音频数据转换为适合模型的格式并移到设备上
    waveform = waveform.unsqueeze(0).to(device)  # 增加一个批次维度并转移到设备

    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        output = model(waveform)

    # 后处理输出以计算呼吸声个数
    # 这将取决于您的模型输出的具体性质
    num_breaths = post_process_output(output)

    return num_breaths

if __name__ == "__main__":
    # 数据集准备
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_mels=64)
    dataset = AudioDataset(audio_dir='/home/rlg/projects/sig/mmlab-sigs_practice_lungsound/data/icbhi_dataset/audio_test_data', transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 设置模型的时间序列长度和输入特征数量
    fixed_length = 50  # 示例值，根据您的数据调整
    in_features = 56320  # 示例值，根据您的数据调整

    # 模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioClassifier(fixed_length, in_features).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train(model, train_loader, criterion, optimizer, epochs=50)
    
    # 使用函数
    file_path = '/home/rlg/projects/sig/mmlab-sigs_practice_lungsound/data/icbhi_dataset/audio_test_data/102_1b1_Ar_sc_Meditron.wav'
    num_breaths = evaluate_model(model, file_path, transform, device)
    print("Predicted number of breaths:", num_breaths)