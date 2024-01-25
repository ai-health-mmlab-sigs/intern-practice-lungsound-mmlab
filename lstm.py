import torch
import torch.nn as nn
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler  # ��׼��
import os
import glob
import soundfile as sf

file_path='data/icbhi_dataset/audio_test_data/'  #matlab�����ļ�·��
Tolerable_relative_error=0.2                #�����������������


def create_inout_sequences(file_path):
    inout_seq = []
    txt_files = glob.glob(os.path.join(file_path, "*.txt"))#extract periods from .txt first
    for file_name in txt_files:
        fid=open(file_name)
        source_in_lines = fid.readlines()
        fid.close()
        differences = []
        for line in source_in_lines:
            values = line.strip().split('\t')
            start = float(values[0])
            end = float(values[1])
            difference = end - start
            differences.append(difference)
        average_difference = sum(differences) / len(differences)   #average periods

        file_name_without_extension = os.path.splitext(file_name)[0]
        new_file_name = file_name_without_extension + ".wav" #extract signals
        audio, sample = sf.read(new_file_name)


        train_seq = audio
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_data_normalized = scaler.fit_transform(train_seq.reshape(-1, 1))
        train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

        train_label = average_difference
        inout_seq.append((train_data_normalized ,train_label))
    return inout_seq


#separate train set and test set
inout_seq=create_inout_sequences(file_path)
total_num=len(inout_seq)
train_inout_seq=inout_seq[:int(0.7*total_num)]
test_inout_seq=inout_seq[-int(0.3*total_num):]
print("Train set "+str(len(train_inout_seq))+" files"+"      "+"Test set "+str(len(test_inout_seq))+" files")

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1): #output_size=1
        super().__init__()
        self.hidden_layer_size = hidden_layer_size                  # ���ز�ڵ���100
        self.lstm = nn.LSTM(input_size, hidden_layer_size)          #����ά��1�����ز�100

        self.linear = nn.Linear(hidden_layer_size, output_size)     #���ز���ȫ���Ӳ���������Ϊ���

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]                                      #���Ԥ��������ʣ�����ֵ


model = LSTM()

loss_function = nn.MSELoss()  
                                     #��ʧ����ΪMSE
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)          #Adam�Ż�����ѧϰ��0.001



#model train
epochs = 300
loss_list=[]
for i in range(epochs):
    loss_list_epoch = []
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(seq)    ##get killed
        y_pred = y_pred.clone().detach().requires_grad_(True)#set grad true
        #labels = labels.clone().detach().requires_grad_(True)
        labels = torch.tensor(labels, dtype=torch.float32, requires_grad=True)
        single_loss = loss_function(y_pred, labels)      
        loss_list_epoch.append(single_loss.item())
        single_loss.backward()                          #ǰ�򴫲�
        optimizer.step()
    mean_loss_epoch=sum(loss_list_epoch)/len(loss_list_epoch)
    loss_list.append(mean_loss_epoch)
    if i%10 == 0:
        print(f'epoch: {i:3}    loss: {mean_loss_epoch:10.8f}')
epoch_list=np.arange(0,len(loss_list))
loss_list=np.array(loss_list)
plt.plot(epoch_list,loss_list)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

#model test
model.eval()

i=0
j=0
for seq, labels in train_inout_seq:
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(seq)
        y_pred=y_pred.numpy()
        labels = labels.numpy()
        if  abs(y_pred[0]-labels[0])/labels[0]<=Tolerable_relative_error and abs(y_pred[1]-labels[1])/labels[1]<=Tolerable_relative_error:
            j=j+1
    i=i+1
accuracy=j/i
print("Train set accuracy: "+str(accuracy))


i=0
j=0
for seq, labels in test_inout_seq:
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(seq)
        y_pred=y_pred.numpy()
        labels = labels.numpy()
        if  abs(y_pred[0]-labels[0])/labels[0]<=Tolerable_relative_error and abs(y_pred[1]-labels[1])/labels[1]<=Tolerable_relative_error:
            j=j+1
    i=i+1
accuracy=j/i
print("Test set accuracy: "+str(accuracy))




