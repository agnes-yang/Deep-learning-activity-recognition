# encoding=utf-8
"""
    Created on 21:11 2018/11/8 
    @author: Jindong Wang
"""


from itertools import accumulate
import data_preprocess
import matplotlib.pyplot as plt
import network as net
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
result = []

# 用来获取输入参数
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nepoch', type=int, default=50)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--momentum', type=float, default=.9)
    parser.add_argument('--data_folder', type=str, default='../data/')
    parser.add_argument('--result_folder', type=str, default='../results/')
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()
    return args


# 训练函数，加上测试的功能，每一轮训练都加测试一遍
def train(model, optimizer, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()  # 定义损失函数，为交叉熵损失

    for e in range(args.nepoch):  # 对每一个 epoch 的循环
        model.train()             # 定义模型在训练模式
        correct, total_loss = 0, 0   #
        total = 0
        for sample, target in train_loader:   # 对于训练集中的每一个 batch size 的样本和对应标签
            sample, target = sample.to(
                DEVICE).float(), target.to(DEVICE).long()  # 将数据放到 gpu 上
            sample = sample.view(-1, 9, 1, 128)            #
            output = model(sample)                         # 计算输入模型，计算输出
            loss = criterion(output, target)               # 通过输出与真实标签计算 loss
            optimizer.zero_grad()                          # 将优化器梯度置零
            loss.backward()                                # 将 loss 反向传播
            optimizer.step()                               # 优化器开始优化
            total_loss += loss.item()                      # 叠加 loss
            _, predicted = torch.max(output.data, 1)       # 取 输出中最大的概率 对应的标签 做为结果
            total += target.size(0)                        # total 累加每个 batch size 参与训练的样本数
            correct += (predicted == target).sum()         # correct 累加分类正确的样本数
        acc_train = float(correct) * 100.0 / len(train_loader.dataset)  # 计算准确率

        # Testing
        acc_test = valid(model, test_loader)
        # 输出每轮轮次，loss 为叠加每个 batch 的 loss 除以 batch 的总数，训练 accuracy 为 正确分类样本数 除以 总样本数，
        print(f'Epoch: [{e}/{args.nepoch}], /'
              f'total_loss:{total_loss:.4f}, len(train_loader):{len(train_loader)}, '
              f'loss:{total_loss / len(train_loader):.4f}, /'
              f'correct:{float(correct)}, len(train_loader.dataset): {len(train_loader.dataset)}, '
              f'train_acc: {acc_train:.2f}, /'
              f'correct:{float(correct)}, total: {total}, '
              f'test_acc: {float(correct) * 100 / total:.2f}')
        result.append([acc_train, acc_test])
        result_np = np.array(result, dtype=float)
        # np.savetxt('result.csv', result_np, fmt='%.2f', delimiter=',')


# 在测试集上测试结果
def valid(model, test_loader):
    model.eval()
    with torch.no_grad():    # 声明不计算梯度
        correct, total = 0, 0
        for sample, target in test_loader:                     # 对于测试集中的每一个 batch size 的样本和对应标签
            sample, target = sample.to(
                DEVICE).float(), target.to(DEVICE).long()      # 搬移数据到 gpu 上
            sample = sample.view(-1, 9, 1, 128)                #
            output = model(sample)                             # 计算输出
            _, predicted = torch.max(output.data, 1)           # 取 输出中最大的概率 对应的标签 做为结果
            total += target.size(0)                            # total 累加每个 batch size 参与测试的样本数
            correct += (predicted == target).sum()             # correct 累加分类正确的样本数
    acc_test = float(correct) * 100 / total                    # 计算准确率
    return acc_test


def plot(result_folder):
    data = np.loadtxt(result_folder + 'result.csv', delimiter=',')
    plt.figure()
    plt.plot(range(1, len(data[:, 0]) + 1),
             data[:, 0], color='blue', label='train')
    plt.plot(range(1, len(data[:, 1]) + 1),
             data[:, 1], color='red', label='test')
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Train and Test Accuracy', fontsize=16)
    plt.savefig(result_folder + 'plot.png')


if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    train_loader, test_loader = data_preprocess.load(
        args.data_folder, batch_size=args.batchsize)
    model = net.Network().to(DEVICE)
    optimizer = optim.SGD(params=model.parameters(
    ), lr=args.lr, momentum=args.momentum)
    train(model, optimizer, train_loader, test_loader)
    result = np.array(result, dtype=float)
    np.savetxt(args.result_folder + 'result.csv', result, fmt='%.2f', delimiter=',')
    plot(args.result_folder)
