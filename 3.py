import os
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image


def main():
    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # 加载模型（使用模型自带的 scheduler，避免 DDIM 报错）
    pipe = FluxKontextPipeline.from_pretrained(
        FLUX1_Kontext_dev_path,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)

    # 显存优化
    pipe.enable_sequential_cpu_offload()

    # 加载图像
    input_image = load_image(input_image_path).convert("RGB")

    # 推理
    output = pipe(
        image=input_image,
        prompt="Restore and colorize this image. Remove any scratches or imperfections",
        guidance_scale=2.5,
        num_inference_steps=25
    )
    edited = output.images[0]

    # 内容安全检查（可选）
    if use_check:
        import numpy as np
        from flux.content_filters import PixtralContentFilter
        check = PixtralContentFilter(torch.device(device))
        img_arr = np.array(edited) / 255.0
        img_arr = 2 * img_arr - 1
        img_tensor = torch.from_numpy(img_arr).to(device, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
        if check.test_image(img_tensor):
            raise ValueError("⚠️ 内容安全检查未通过，请更换提示词或图像")

    # 保存结果
    edited.save(output_image_path)
    print(f"✅ 推理完成，结果已保存为：{output_image_path}")


if __name__ == "__main__":
    use_check = False  # True 则启用安全检查
    FLUX1_Kontext_dev_path = "/data/code/songtao.tian/models/base_models/FLUX.1-Kontext-dev"
    input_image_path = "input_images/1.jpg"
    output_image_path = f"output_images/{os.path.basename(input_image_path)}"
    main()
