import datetime
import torch
from diffusers import StableDiffusionXLPipeline

model_path = "models/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors"
lora_model_path = "models/pixel-art-xl/pixel-art-xl.safetensors"

# 创建输出文件夹的代码可以放在这里，如果文件夹不存在的话
# os.makedirs("output", exist_ok=True)

pipe = StableDiffusionXLPipeline.from_single_file(model_path)
pipe.load_lora_weights(lora_model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# 使用torch.Generator来设置种子
seed = 12345
generator = torch.Generator(device=device).manual_seed(seed)

prompt = "a photo of an astronaut riding a horse on mars"
negative_prompt = ""

num_inference_steps = 30

image = pipe(
    generator=generator,
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=num_inference_steps,
).images[0]

# 生成文件名，包含当前时间
filename = f"output/sdxl_with_lora_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
image.save(filename)
