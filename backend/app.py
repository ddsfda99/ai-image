from flask import Flask, request, jsonify, send_from_directory
import os
import sys
import uuid
from datetime import datetime
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file

# 项目路径配置
BASE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

# 模型基础路径
MODEL_DIR = os.path.join(PROJECT_DIR, 'models', 'stable-diffusion-v1-5')
LORA_DIR = os.path.join(PROJECT_DIR, 'models', 'lora')
print(f"正在加载基础模型：{MODEL_DIR}")

def load_base_pipe():
    return StableDiffusionPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16
    ).to("cuda")

# 初始化模型
pipe = load_base_pipe()
print("模型加载完成")

# 多风格 LoRA 权重目录（只加载 .safetensors 文件）
LORA_WEIGHTS = {
    "monet": os.path.join(LORA_DIR, "monet.safetensors"),
    "vangogh": os.path.join(LORA_DIR, "van_gogh.safetensors"),
}

# 当前激活的风格，用于避免重复加载
current_style = None

# 手动加载 LoRA 权重（简单叠加，不可切换回原始）
def apply_lora_weights(pipe, lora_path, alpha=1.0):
    state_dict = load_file(lora_path)
    for key, value in state_dict.items():
        if key in pipe.unet.state_dict() and pipe.unet.state_dict()[key].shape == value.shape:
            pipe.unet.state_dict()[key] += alpha * value.to(pipe.unet.device)
    print(f"✅ LoRA 权重已加载：{os.path.basename(lora_path)}")

# 初始化 Flask
app = Flask(__name__)

@app.route('/api/text-to-image', methods=['POST'])
def text_to_image():
    global current_style
    global pipe

    prompt = request.form.get("prompt", "")
    style = request.form.get("style", "monet").lower()

    if not prompt:
        return jsonify({"error": "缺少 prompt 参数"}), 400
    if style not in LORA_WEIGHTS:
        return jsonify({"error": f"不支持的风格：{style}"}), 400

    try:
        # 如果风格变化，重载基础模型 + 应用新 LoRA
        if current_style != style:
            print(f"🎨 切换风格为：{style}")
            pipe = load_base_pipe()
            apply_lora_weights(pipe, LORA_WEIGHTS[style])
            current_style = style

        styled_prompt = f"<{style}-style> {prompt}"

        image = pipe(styled_prompt).images[0]
        filename = f"text2img_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
        output_path = os.path.join(STATIC_DIR, filename)
        image.save(output_path)

        return jsonify({"output_url": f"/static/{filename}"})
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return jsonify({"error": str(e)}), 500

# 前端页面服务
@app.route('/')
def serve_index():
    return send_from_directory(os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend')), 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend')), path)

if __name__ == '__main__':
    print("🚀 启动 Flask 后端...")
    app.run(debug=True, host='0.0.0.0', port=5000)
