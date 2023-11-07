import datetime
import torch
from diffusers import StableDiffusionXLPipeline

# 模型路径
model_path = "models/sd/sd_xl_base_1.0.safetensors"
lora_model_path = "models/lora"

# 初始化Pipeline
pipe = StableDiffusionXLPipeline.from_single_file(model_path)
# pipe.load_lora_weights(lora_model_path, weight_name="pixel-art-xl.safetensors")
# pipe.fuse_lora(lora_scale=0.8)

# LoRA one.
pipe.load_lora_weights(lora_model_path, weight_name="cyborg_style_xl-off.safetensors")
pipe.fuse_lora(lora_scale=0.7)

# LoRA two.
pipe.load_lora_weights(lora_model_path, weight_name="pikachu.safetensors")
pipe.fuse_lora(lora_scale=0.7)

# 确定设备
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# 设置种子
seed = 12345
generator = torch.Generator(device=device).manual_seed(seed)

# 设置参数
prompt = "cyborg style pikachu"
negative_prompt = ""
num_inference_steps = 30
guidance_scale = 7.5  # 控制prompt的影响，较大的值将生成与prompt更紧密相关的图像
width = 1024           # 生成图像的宽度
height = 1024          # 生成图像的高度
num_images = 1        # 生成图像的数量

# 生成图像
images = pipe(
    generator=generator,
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    width=width,
    height=height,
    num_images=num_images,
).images

# 保存图像
for i, image in enumerate(images):
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    image.save(f"output/sdxl_{timestamp}_{i}.png")
