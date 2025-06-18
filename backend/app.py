from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
from PIL import Image
import os
import uuid
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from safetensors.torch import load_file
import time

# ------------------ 路径配置 ------------------
BASE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
STATIC_DIR = os.path.join(BASE_DIR, "static")
FRONTEND_DIR = os.path.join(PROJECT_DIR, 'frontend')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models', 'stable-diffusion-v1-5')
LORA_DIR = os.path.join(PROJECT_DIR, 'models', 'lora')
TI_PATH = os.path.join(PROJECT_DIR, 'models', 'textual-inversion', 'anime-style', 'learned_embeds.safetensors')
os.makedirs(STATIC_DIR, exist_ok=True)

# ------------------ 风格配置 ------------------
LORA_WEIGHTS = {
    "monet": os.path.join(LORA_DIR, "monet.safetensors"),
    "vangogh": os.path.join(LORA_DIR, "vangogh.safetensors"),
    "francis": os.path.join(LORA_DIR, "francis.safetensors"),
}
TI_PLACEHOLDERS = {
    "anime-style": "<anime-style>"
}

# ------------------ 模型加载 ------------------
print("正在加载模型...")
text2img_pipe = StableDiffusionPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).to("cuda")
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).to("cuda")
anime_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).to("cuda")

if os.path.exists(TI_PATH):
    text2img_pipe.load_textual_inversion(TI_PATH, token=TI_PLACEHOLDERS["anime-style"])
    anime_pipe.load_textual_inversion(TI_PATH, token=TI_PLACEHOLDERS["anime-style"])
    print("已加载 Textual Inversion 模型")
else:
    print("未找到 Textual Inversion 模型")

# ------------------ 工具函数 ------------------
def save_image(img, prefix):
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
    path = os.path.join(STATIC_DIR, filename)
    img.save(path)
    return f"/static/{filename}"

def apply_lora(pipe, lora_path, alpha=1.0):
    print(f"加载 LoRA 权重：{lora_path}")
    state_dict = load_file(lora_path)
    for k, v in state_dict.items():
        if k in pipe.unet.state_dict() and pipe.unet.state_dict()[k].shape == v.shape:
            pipe.unet.state_dict()[k] += alpha * v.to(pipe.unet.device)

# ------------------ Flask 应用 ------------------
app = Flask(__name__)

@app.route('/api/text-to-image', methods=['POST'])
def text_to_image():
    start = time.time()
    prompt = request.form.get("prompt", "")
    style = request.form.get("style", "none")

    if not prompt:
        return jsonify({"error": "缺少 prompt 参数"}), 400

    final_prompt = prompt
    if style in LORA_WEIGHTS:
        apply_lora(text2img_pipe, LORA_WEIGHTS[style])
        final_prompt = f"<{style}-style> {prompt}"
    elif style in TI_PLACEHOLDERS:
        final_prompt = f"{TI_PLACEHOLDERS[style]} {prompt}"

    image = text2img_pipe(prompt=final_prompt, guidance_scale=7.5).images[0]
    output_url = save_image(image, "text2img")
    return jsonify({"output_url": output_url, "inference_time": time.time() - start})

@app.route('/api/style-transfer', methods=['POST'])
def style_transfer():
    start = time.time()
    content_file = request.files.get("content")
    style = request.form.get("style", "monet")
    if not content_file:
        return jsonify({"error": "缺少内容图像"}), 400

    image = Image.open(content_file).convert("RGB").resize((512, 512))
    if style in LORA_WEIGHTS:
        apply_lora(img2img_pipe, LORA_WEIGHTS[style])
    result = img2img_pipe(prompt=f"<{style}-style>", image=image, strength=0.75, guidance_scale=7.5).images[0]
    output_url = save_image(result, "style_transfer")
    return jsonify({"output_url": output_url, "inference_time": time.time() - start})


# ------------------ 静态资源与页面 ------------------
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:path>')
def catch_all(path):
    return send_from_directory(FRONTEND_DIR, path)

@app.route('/samples/<path:filename>')
def serve_samples(filename):
    return send_from_directory(os.path.join(FRONTEND_DIR, 'samples'), filename)

if __name__ == '__main__':
    print("🚀 后端服务已启动：http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
