import torch
import torch.nn as nn
import torch.nn.functional as F

class TrashNet(nn.Module):
    def __init__(self):
        super(TrashNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.fc1 = nn.Linear(64 * 56 * 56, 512)  # 确保输入维度与卷积层输出匹配
        self.fc2 = nn.Linear(512, 6)  # 假设有 6 类输出

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # 池化
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 池化
        x = x.view(-1, 64 * 56 * 56)  # 扁平化
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
