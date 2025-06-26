
# 垃圾图像分类系统

本项目基于 PyTorch 和 Flask 构建了一个垃圾图像分类系统。利用预训练的 ResNet18 模型，通过迁移学习实现对 12 类垃圾图像的分类，同时提供 Web 前端接口供用户上传图片进行实时预测，以及对测试集进行评估和生成混淆矩阵、分类报告。

## 项目结构

```
trash-classification
├── AppTest            # 应用层测试代码（调试过程中使用）
├── data               # 数据集文件夹，包括 train、val、test 等子目录
├── DataTest           # 数据处理相关的测试代码
├── ModelTest          # 模型相关测试代码
├── output             # 训练后模型保存目录
├── runs               # 训练过程中保存的运行日志或中间结果（如 tensorboard 日志）
├── static             # 静态文件目录，存放混淆矩阵图片、分类报告等
├── templates          # HTML 模板文件目录，包括 index、evaluation、result 页面
├── TrainTest          # 训练代码调试相关内容
├── uploads            # 用户上传的图片保存目录（如需持久化，可使用）
├── app6.py            # Flask Web 应用入口文件，包含预测与评估接口
├── model3.py          # 模型定义文件，基于 ResNet18 构建 TrashNet 模型
├── predict.py         # 独立的预测脚本，可用于命令行测试
├── README.md          # 本说明文件
├── train3.py          # 模型训练脚本，包含数据增强、训练、验证及模型保存逻辑
└── trashnet_model.pth # 预训练或训练生成的模型权重文件
```

> **说明**：  
> * 文件夹 **AppTest、DataTest、ModelTest、TrainTest** 为调试和测试过程中产生的内容，不影响正式运行。

## 环境要求

确保你的 Python 环境中已安装以下主要依赖包（推荐使用 [virtualenv](https://docs.python.org/3/library/venv.html) 或 Conda 创建虚拟环境）：

- Python 3.6+
- [PyTorch](https://pytorch.org/)  
- torchvision
- Flask
- pandas
- seaborn
- matplotlib
- scikit-learn
- Pillow

可通过以下命令安装依赖（假设已安装 pip）：

```bash
pip install torch torchvision flask pandas seaborn matplotlib scikit-learn pillow
```

## 数据准备

项目数据需按照以下结构组织在 `data` 文件夹下：

```
data
├── train         # 训练集目录，每个类别对应一个子文件夹
├── val           # 验证集目录，每个类别对应一个子文件夹
└── test          # 测试集目录，每个类别对应一个子文件夹，用于评估生成混淆矩阵和分类报告
```

确保每个子文件夹中包含相应类别的图像文件。

## 模型训练

在开始 Web 应用部署之前，需要对模型进行训练。训练脚本为 `train3.py`，执行以下命令即可开始训练：

```bash
python train3.py
```

训练过程中会输出每个 epoch 的训练损失和验证准确率。训练完成后，模型权重将保存在 `output/trashnet_model.pth`。

> **注意**：若已有训练好的模型，可将 `output/trashnet_model.pth` 拷贝到项目根目录或修改代码中的模型加载路径。

## 模型推理与 Web 服务

### 1. Web 应用

项目使用 Flask 提供 Web 前端接口，支持图片上传预测和测试集评估。主要入口文件为 `app6.py`。

运行命令：

```bash
python app6.py
```

启动后，在浏览器中访问 `http://127.0.0.1:5000/`，即可看到图片上传页面。上传图片后，系统会返回预测的垃圾类别。

同时，通过访问 `/evaluate` 接口（例如 `http://127.0.0.1:5000/evaluate`）可以对测试集进行评估，生成混淆矩阵和分类报告，并支持 CSV 文件下载。

### 2. 命令行预测脚本

独立的预测脚本 `predict.py` 可用于命令行下测试单张图片的分类结果。编辑脚本中的 `img_path` 为测试图片路径后，执行：

```bash
python predict.py
```

脚本会输出预测的类别索引（可根据需要修改代码映射为类别名称）。

## 项目原理与实现亮点

- **迁移学习**：采用预训练的 ResNet18 模型，通过替换最后一层实现垃圾分类，有效提升在有限数据集上的性能。
- **数据增强**：在训练脚本中加入了随机裁剪、水平翻转等数据增强技术，提高模型泛化能力。
- **Web 前后端分离**：利用 Flask 搭建 RESTful API，前端 HTML 模板实现用户友好交互，同时支持模型评估与结果下载。
- **评估报告生成**：通过 sklearn 生成混淆矩阵和分类报告，结合 Seaborn 和 Matplotlib 进行可视化展示。

## 后续优化建议

- **数据增强策略**：可尝试更多数据增强方法（如旋转、色彩变换）进一步提升模型鲁棒性。
- **模型结构**：除 ResNet18 外，可尝试更深层次网络（如 ResNet50、MobileNet）对比性能，或采用模型剪枝、量化技术以提升推理速度。
- **生产部署**：开发完成后建议关闭 Flask 的 debug 模式，并使用 Gunicorn、uWSGI 等 WSGI 服务器进行部署。
- **代码模块化**：将预处理、模型加载、预测、评估等逻辑进一步拆分为独立模块，便于后续维护和扩展。

## 参考文献

如需深入了解相关技术，可以参考以下资料：
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [Flask 官方文档](https://flask.palletsprojects.com/)
- [Torchvision 模型库](https://pytorch.org/vision/stable/models.html)

---

以上即为项目的基本介绍及使用说明。欢迎各位有兴趣的同学进行调试、优化及进一步扩展。
