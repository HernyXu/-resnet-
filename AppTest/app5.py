import torch
from flask import Flask, render_template, request, jsonify
from torchvision import transforms
from PIL import Image
from ModelTest.model2 import TrashNet  # 从 model2.py 中导入 TrashNet

app = Flask(__name__)

# 设置最大上传文件大小为 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 定义12个类别的映射
label_map = {
    0: '电池',
    1: '生物',
    2: '棕色玻璃',
    3: '纸板',
    4: '服装',
    5: '绿色玻璃',
    6: '金属',
    7: '纸张',
    8: '塑料',
    9: '鞋子',
    10: '垃圾',
    11: '白色玻璃'
}

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载模型
model = TrashNet(num_classes=12)  # 设置为12类
model.load_state_dict(torch.load('../trashnet_model.pth'))  # 加载训练好的权重
model.eval()  # 设置为评估模式

# 主页路由
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index2.html")  # 需要一个前端页面

# 预测接口
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(file.stream)
        img = transform(img).unsqueeze(0)  # 增加批量维度

        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
            label_index = predicted.item()

        label = label_map.get(label_index, "未知")
        return jsonify({"label": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)