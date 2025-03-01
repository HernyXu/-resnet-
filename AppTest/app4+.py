import torch
from flask import Flask, render_template, request, jsonify
from torchvision import transforms
from PIL import Image
from ModelTest.model1 import TrashNet  # 确保导入的是 model1.py 中修改后的 TrashNet 类

app = Flask(__name__)

# 设置最大上传文件大小为 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 最大上传大小 16MB

# 定义类别映射，假设模型有6个类别
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
# 预处理转化器
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载模型
model = TrashNet(num_classes=6)
model.load_state_dict(torch.load('../trashnet_model.pth'))
model.eval()  # 设置模型为评估模式


# 图片上传和预测接口
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index2.html")


@app.route("/predict", methods=["POST"])
def predict():
    # 检查是否有文件上传
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]

    # 如果没有选择文件
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # 打开上传的图片并进行预处理
        img = Image.open(file.stream)
        img = transform(img).unsqueeze(0)  # 增加一个批量维度

        # 使用模型进行预测
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
            label_index = predicted.item()

        # 获取对应的标签
        label = label_map.get(label_index, "未知")

        return jsonify({"label": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
