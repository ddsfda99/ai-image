# AI创意图像生成项目
## 一、项目概述
本项目"ai-image"是一个基于 Stable Diffusion 模型的图像创意生成平台，集成了文本生成图像与图像风格迁移两大功能，支持用户自定义描述文字或者内容图，并可以选择风格模型，快速生成艺术图像，满足创意生成与审美改造的需求。

## 二、项目结构
```
ai-image/
├── frontend/                # 前端界面
│   ├── app.py                 
├── backend/                 # Flask 后端服务
│   ├── app.py               # 主接口逻辑
│   └── static/              # 图像输出目录
├── models/
│   ├── stable-diffusion-v1-5/   # 预下载基础模型
│   └── lora/                   # 经过微调的各种风格的 LoRA 模型
├── requirements.txt
└── README.md
```

## 三、实验环境与运行方式
### 3.1 AI创意图像工坊平台环境
#### 克隆并进入项目
```bash
git clone https://github.com/ddsfda99/ai-image.git
cd ai-image
```
#### 配置 Python 虚拟环境 + 安装依赖
```bash
python3 -m venv .venv           
source .venv/bin/activate      
cd backend
pip install -r requirements.txt 
```
#### 启动后端 Flask 服务
```bash
cd backend
python app.py
```
浏览器访问：[http://localhost:5000/](http://localhost:5000/)

## 四. 项目资料
AI图像生成项目录屏演示：AI图像生成项目录屏演示：https://www.bilibili.com/video/BV1tfNszmEpp/?vd_source=de633d4318be770bdffc3275f1e20c2c

或者在https://github.com/ddsfda99/ai-image/blob/main/%E9%A1%B9%E7%9B%AE%E6%BC%94%E7%A4%BA.mp4 下载观看

项目展示ppt链接：https://github.com/ddsfda99/ai-image/blob/main/AI%E5%88%9B%E6%84%8F%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90%E9%A1%B9%E7%9B%AEppt%E5%B1%95%E7%A4%BA.pptx

AI图像生成项目地址：https://github.com/ddsfda99/ai-image

微调训练+模型比较分析项目地址：https://github.com/ddsfda99/fine_tuning/tree/main


