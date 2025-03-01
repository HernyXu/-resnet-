import torch
import torch.nn as nn
import torch.nn.functional as F

class TrashNet(nn.Module):
    def __init__(self, num_classes=12):  # 默认12个类别
        super(TrashNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)  # 假设输入图像为224x224
        self.fc2 = nn.Linear(512, num_classes)  # 输出层根据类别数调整

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 56 * 56)  # 展平张量
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x