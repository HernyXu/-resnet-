import torch
from torchvision import transforms
from PIL import Image
from ModelTest.model import TrashNet  # 确保模型类定义文件存在

# 加载模型
model = TrashNet()
model.load_state_dict(torch.load('trashnet_model.pth'))
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 预测函数
def predict(img_path):
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0)  # 增加一个批量维度
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# 测试预测
img_path = 'path_to_test_image.jpg'  # 更改为实际的图片路径
predicted_class = predict(img_path)
print(f'Predicted class: {predicted_class}')
