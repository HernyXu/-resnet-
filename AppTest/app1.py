import torch
from flask import Flask, render_template, request, jsonify
from torchvision import transforms
from PIL import Image
from ModelTest.model import TrashNet  # 确保 model.py 文件和 TrashNet 类存在

app = Flask(__name__)

# 预处理转化器
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载模型
model = TrashNet()
model.load_state_dict(torch.load('../trashnet_model.pth'))
model.eval()  # 设置模型为评估模式

# 图片上传和预测接口
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")  # 使用 get() 方法防止KeyError
        if file:
            try:
                # 打开上传的图片并进行预处理
                img = Image.open(file.stream)
                img = transform(img).unsqueeze(0)  # 增加一个批量维度

                # 使用模型进行预测
                with torch.no_grad():
                    output = model(img)
                    _, predicted = torch.max(output, 1)
                    label = predicted.item()

                return jsonify({"label": label})
            except Exception as e:
                return jsonify({"error": str(e)}), 500  # 捕获异常并返回 500 错误
        else:
            return jsonify({"error": "No image file provided"}), 400  # 返回错误信息

    return render_template("index1.html")

if __name__ == "__main__":
    app.run(debug=True)

