import datetime
import torch
from diffusers import StableDiffusionXLPipeline

# 模型路径
model_path = "models/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors"
lora_model_path = "models/pixel-art-xl/pixel-art-xl.safetensors"

# 初始化Pipeline
pipe = StableDiffusionXLPipeline.from_single_file(model_path)
pipe.load_lora_weights(lora_model_path)

# 确定设备
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# 设置种子
seed = 12345
generator = torch.Generator(device=device).manual_seed(seed)

# 设置参数
prompt = "a photo of an astronaut riding a horse on mars"
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
