import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# 修改 model.py 中的 TrashNet 构建部分，确保没有加载预训练权重。
class TrashNet(nn.Module):
    def __init__(self, num_classes=6):  # 添加 num_classes 参数
        super(TrashNet, self).__init__()
        # 自定义的卷积层，不是预训练的 ResNet 层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)  # 根据需要的类别数量进行修改

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 在加载模型时，使用这个自定义模型结构，不使用预训练权重：
model = TrashNet(num_classes=6)  # 传入类别数
model.load_state_dict(torch.load('../trashnet_model.pth'))
model.eval()


