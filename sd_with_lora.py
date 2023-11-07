import datetime

from diffusers import DiffusionPipeline
import torch

# model_path = "stabilityai/stable-diffusion-xl-base-1.0"
# lora_model_path = "nerijs/pixel-art-xl"
model_path = "models/stable-diffusion-xl-base-1.0"
lora_model_path = "models/pixel-art-xl"
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.load_lora_weights(lora_model_path)

pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "A pokemon with blue eyes."
n_steps = 30

image = pipe(prompt, num_inference_steps=n_steps, guidance_scale=7.5, cross_attention_kwargs={"scale": 0.5}).images[0]

filename = f"output/sd_with_lora_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
image.save(filename)
