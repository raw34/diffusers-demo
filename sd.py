import datetime

from diffusers import DiffusionPipeline
import torch

model_path = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

# 将模型转移到合适的设备
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

n_steps = 30

prompt = "An astronaut riding a green horse"

image = pipe(
    prompt=prompt,
    num_inference_steps=n_steps,
).images[0]

filename = f"output/sdxl{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
image.save(filename)
