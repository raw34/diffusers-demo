from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline
import torch

model_path = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

# 加载模型
# model_path = "./models/sd_xl_base_1.0.safetensors"
# pipe = StableDiffusionPipeline.from_single_file(model_path, use_safetensors=True)
# pipe.save_pretrained("/Users/randy/Downloads/models/sd")

# 将模型转移到合适的设备
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

n_steps = 40

prompt = "An astronaut riding a green horse"

images = pipe(
    prompt=prompt,
    num_inference_steps=n_steps,
).images[0]

images.save("output/base.png")
