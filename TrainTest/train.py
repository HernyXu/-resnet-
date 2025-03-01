import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import os

# 设置训练设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = datasets.ImageFolder(os.path.join('../data', 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join('../data', 'val'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 定义模型（这里以简单的CNN为例）
class TrashNet(nn.Module):
    def __init__(self):
        super(TrashNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 添加池化层
        self.fc1 = nn.Linear(64 * 56 * 56, 512)  # 修改全连接层的输入维度
        self.fc2 = nn.Linear(512, 6)  # 假设有 6 类输出

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 使用池化层
        x = self.pool(torch.relu(self.conv2(x)))  # 使用池化层
        x = x.view(-1, 64 * 56 * 56)  # 展平，确保与模型中的维度一致
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 初始化模型、损失函数和优化器
model = TrashNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 设置TensorBoard日志
log_dir = '../runs/trashnet'
writer = SummaryWriter(log_dir=log_dir)

# 训练模型
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

# 保存模型
torch.save(model.state_dict(), '../trashnet_model.pth')

print("训练完成！模型已保存！")
