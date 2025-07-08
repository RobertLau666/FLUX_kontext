import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16
).to(device)

input_img = load_image("path/to/image.png").convert("RGB")
out = pipe(
    image=input_img,
    prompt="在猫的头上加顶红色帽子",
    guidance_scale=2.5,
    generator=torch.Generator(device).manual_seed(1234)
)
out.images[0].save("edited.png")
print("推理完成，结果保存为 edited.png")
