from diffusers import StableDiffusionPipeline
import torch

model_path = "stabilityai/stable-diffusion-xl-base-1.0"
lora_model_path = "ostris/ikea-instructions-lora-sdxl"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.unet.load_attn_procs(lora_model_path)
pipe.to("cuda")

prompt = "A pokemon with blue eyes."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5, cross_attention_kwargs={"scale": 0.5}).images[0]
image.save("output/base_lora.png")
