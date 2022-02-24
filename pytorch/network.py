# encoding=utf-8
"""
    Created on 10:41 2018/11/10 
    @author: Jindong Wang
"""

"""
神经网络的记录:
输入为 N x channel x height x weight, 每一条记录包含 9 个数据当做 channel，height做为1，width为数据长度
第一层： 输入 32 channel，kernel size 卷积核大小为 1 x 9，stride 步长为默认值 1，得 9 x 1 x 128 -> 32 x 1 x 120
最大池化层： kernel size 为 1 x 2，stride 为 2，32 x 1 x 120 -> 32 x 1 x 60
第二层：input 32 channel，output 64 channel，kernel 为 1 x 9，stride 为 1，32 x 1 x 60 -> 64 x 1 x 52
最大池化层： kernel size 为 1 x 2，stride 为 2，64 x 1 x 62 -> 64 x 1 x 26
合为一维向量，64 * 26
三个全连接层： 64 * 26 -> 1000 -> 5000 -> 6
"""

import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=32, kernel_size=(1, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64 * 26, out_features=1000),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=500, out_features=6)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.reshape(-1, 64 * 26)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
