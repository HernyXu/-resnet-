import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from model3 import TrashNet  # 你的模型类

app = Flask(__name__)

# 设置最大上传文件大小为 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 定义类别映射
label_map = {
    0: 'battery',
    1: 'biological',
    2: 'brown-glass',
    3: 'cardboard',
    4: 'clothes',
    5: 'green-glass',
    6: 'metal',
    7: 'paper',
    8: 'plastic',
    9: 'shoes',
    10: 'trash',
    11: 'white-glass'
}

# 预处理转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载模型
model = TrashNet(num_classes=12)
model.load_state_dict(torch.load('trashnet_model.pth'))
model.eval()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, 'data', 'test')

@app.route("/", methods=["GET"])
def index():
    return render_template("index2.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(file.stream)
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
            label_index = predicted.item()

        label = label_map.get(label_index, "未知")
        return jsonify({"label": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/evaluate", methods=["GET"])
def evaluate():
    if not os.path.exists(TEST_DIR):
        return jsonify({"error": f"测试集目录 '{TEST_DIR}' 不存在，请检查路径！"}), 400

    empty_classes = [class_name for class_name in label_map.values()
                     if not os.path.exists(os.path.join(TEST_DIR, class_name)) or not os.listdir(os.path.join(TEST_DIR, class_name))]

    if empty_classes:
        return jsonify({"error": f"以下类别的文件夹为空或不存在: {', '.join(empty_classes)}"}), 400

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.values(), yticklabels=label_map.values())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('static/confusion_matrix.png')

    # 生成分类报告
    report = classification_report(all_labels, all_preds, target_names=label_map.values(), output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    report_path = os.path.join("static", "classification_report.csv")
    df_report.to_csv(report_path, index=True)

    return render_template("evaluation.html", confusion_matrix="static/confusion_matrix.png", report_path=report_path)

@app.route("/download_report")
def download_report():
    return send_file("static/classification_report.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)