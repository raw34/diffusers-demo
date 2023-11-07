import datetime

import torch
from diffusers import StableDiffusionXLPipeline

# model_path = "stabilityai/stable-diffusion-xl-base-1.0"
# lora_model_path = "nerijs/pixel-art-xl"
model_path = "models/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors"
lora_model_path = "models/pixel-art-xl/pixel-art-xl.safetensors"

pipe = StableDiffusionXLPipeline.from_single_file(
    model_path, torch_dtype=torch.float16
)
pipe.load_lora_weights(lora_model_path)


pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
negative_prompt = ""

num_inference_steps = 30

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=num_inference_steps,
).images[0]

image.save(f"output/sdxl{datetime.time}.png")


