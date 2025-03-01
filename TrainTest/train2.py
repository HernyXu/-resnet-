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

# 数据预处理与增强
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),   # 随机水平翻转
    transforms.RandomVerticalFlip(),     # 随机垂直翻转
    transforms.RandomRotation(20),       # 随机旋转 20 度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机颜色变化
    transforms.RandomResizedCrop(224),   # 随机裁剪并缩放为224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = datasets.ImageFolder(os.path.join('../data', 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join('../data', 'val'), transform=transform)

# 打印类别信息以验证
print("Training dataset classes:", train_dataset.classes)
print("Number of classes:", len(train_dataset.classes))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 获取数据集中的类别数量
num_classes = len(train_dataset.classes)

# 定义模型
class TrashNet(nn.Module):
    def __init__(self, num_classes):
        super(TrashNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.fc1 = nn.Linear(64 * 56 * 56, 512)  # 全连接层输入维度
        self.fc2 = nn.Linear(512, num_classes)  # 动态设置输出类别数

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # 池化
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 池化
        x = x.view(-1, 64 * 56 * 56)  # 扁平化
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型、损失函数和优化器
model = TrashNet(num_classes).to(device)
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