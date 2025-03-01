import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model3 import TrashNet  # 假设你的模型定义在 model3.py 中

def main():
    # 1. 设置设备：如果有GPU可用则使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on device: {device}")

    # 2. 数据增强：随机裁剪、翻转等
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 3. 加载数据集
    train_dataset = datasets.ImageFolder('data/train', transform=train_transform)
    val_dataset   = datasets.ImageFolder('data/val', transform=val_transform)

    # Windows 下使用多进程时，需要把训练逻辑放在 main() 中
    # num_workers>0 代表使用多进程
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)

    # 4. 初始化模型
    model = TrashNet(num_classes=12).to(device)

    # 5. 定义损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. 训练循环
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
        print(f"[Epoch {epoch+1}/{epochs}]  Loss: {avg_loss:.4f}")

        # 简单验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total * 100
        print(f"Validation Accuracy: {acc:.2f}%")

    # 7. 保存模型
    os.makedirs('output', exist_ok=True)
    torch.save(model.state_dict(), 'output/trashnet_model.pth')
    print("[INFO] 模型训练完成，已保存到 output/trashnet_model.pth")

# Windows 多进程入口
if __name__ == '__main__':
    main()
