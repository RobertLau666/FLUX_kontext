import os
import torch
from typing import List
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

from PIL import Image
import os
import torch
import numpy as np
import contextlib
from io import StringIO
from typing import List
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import datetime
from tqdm import tqdm


def get_current_time():
    # 获取当前的日期和时间
    now = datetime.datetime.now()
    # 将日期和时间格式化为字符串
    return now.strftime("%Y-%m-%d %H:%M:%S")


class Real_Esrgan:
    def __init__(self, model_name="RealESRGAN_x4plus", scale_factor=4, half_precision=False, tile=0, tile_pad=10, pre_pad=0):
        self.scale_factor = scale_factor
        self.half_precision = half_precision
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        # check model
        if model_name == "RealESRGAN_x4plus":
            upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.netscale = 4
            file_url = "https://huggingface.co/SliverHand/RealESRGAN_x4plus/resolve/main/RealESRGAN_x4plus.pth" # "https://mirror.ghproxy.com/https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        elif model_name == "RealESRNet_x4plus":
            upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.netscale = 4
            file_url = "https://mirror.ghproxy.com/https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
        elif model_name == "RealESRGAN_x4plus_anime_6B":
            upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            self.netscale = 4
            file_url = "https://mirror.ghproxy.com/https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        elif model_name == "RealESRGAN_x2plus":
            upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            self.netscale = 2
            file_url = "https://mirror.ghproxy.com/https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        else:
            raise NotImplementedError("Model name not supported")
        # download model
        # model_path = download_file(file_url, path="./upscaler-model", progress=False, interrupt_check=False)

        # declare the upscaler
        self.upsampler = RealESRGANer(
            scale=self.netscale,
            model_path=os.path.join("./", model_name + ".pth"),
            dni_weight=None,
            model=upscale_model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=self.half_precision,
            gpu_id=None,
        )
        return

    def factorize(self, num, max_value):
        result = []
        while num > max_value:
            result.append(max_value)
            num /= max_value
        result.append(round(num, 4))
        return result

    def sr_inference(self, img_list) -> List[Image.Image]:
        if isinstance(img_list, Image.Image):
            img_list = [img_list]
        torch.cuda.empty_cache()
        upscaled_imgs = []
        for i, img in enumerate(img_list):
            img = np.array(img)

            outscale_list = self.factorize(self.scale_factor, self.netscale)
            with contextlib.redirect_stdout(StringIO()):
                for outscale in outscale_list:
                    curr_img = self.upsampler.enhance(img, outscale=outscale)[0]
                    img = curr_img
                upscaled_imgs.append(Image.fromarray(img))
        torch.cuda.empty_cache()
        return upscaled_imgs


class FluxKontextRunner:
    def __init__(self,
                 model_path: str,
                 device: str = None,
                 dtype: torch.dtype = None,
                 enable_safety_check: bool = False):
        """
        初始化 FluxKontextPipeline 模型

        :param model_path: 本地模型路径
        :param device: 'cuda' 或 'cpu'，默认自动检测
        :param dtype: torch.bfloat16 或 torch.float32，默认根据 device 设置
        :param enable_safety_check: 是否启用图像内容安全检测
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)
        self.model_path = model_path
        self.use_check = enable_safety_check

        print(f"🚀 加载模型到 {self.device}，精度为 {self.dtype} ...")
        self.pipe = FluxKontextPipeline.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype=self.dtype,
        ).to(self.device)

        self.pipe.enable_sequential_cpu_offload()
        print("✅ 模型加载完毕")

        if self.use_check:
            from flux.content_filters import PixtralContentFilter
            self.checker = PixtralContentFilter(torch.device(self.device))

        self.real_esrgan = Real_Esrgan(
            model_name="RealESRGAN_x4plus",
            scale_factor=4,
            # scale_factor=2,
            half_precision=False,
            tile=700,
        )


    def process_images(self,
                       input_image_paths: List[str],
                       output_dir: str,
                       prompt: str = "Restore and colorize this image. Remove any scratches or imperfections",
                       guidance_scale: float = 2.5,
                       num_inference_steps: int = 25):
        """
        对多个图像进行编辑推理

        :param input_image_paths: 输入图像路径列表
        :param output_dir: 输出图像保存目录
        :param prompt: 提示词
        :param guidance_scale: 引导系数
        :param num_inference_steps: 迭代步数
        """
        os.makedirs(output_dir, exist_ok=True)

        for img_path in tqdm(input_image_paths):
            for gen_time in tqdm(range(gen_times)):
                print(f"🖼️ 处理图像: {img_path}")
                input_image = load_image(img_path).convert("RGB")

                output = self.pipe(
                    image=input_image,
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )
                edited = output.images[0]

                edited = self.real_esrgan.sr_inference(edited)[0]

                # 可选：内容检测
                if self.use_check:
                    import numpy as np
                    img_arr = np.array(edited) / 255.0
                    img_arr = 2 * img_arr - 1
                    img_tensor = torch.from_numpy(img_arr).to(self.device, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
                    if self.checker.test_image(img_tensor):
                        print(f"⚠️ 内容检测未通过，跳过保存: {img_path}")
                        continue

                name_only, ext = os.path.splitext(os.path.basename(img_path))
                output_path = os.path.join(output_dir, f"{name_only}_gentime{gen_time}_{get_current_time()}{ext}")
                edited.save(output_path)
                print(f"✅ 已保存: {output_path}")

if __name__ == "__main__":
    gen_times = 3
    model_path = "/data/code/songtao.tian/models/base_models/FLUX.1-Kontext-dev"
    # input_image_paths = [
    #     "input_images/1.jpg",
    #     "input_images/2.jpg"
    # ]
    input_dir = "input_images/20250703"
    input_image_paths = [os.path.join(input_dir, input_image_name) for input_image_name in os.listdir(input_dir)]
    output_dir = f"output_images/{os.path.basename(input_dir)}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    runner = FluxKontextRunner(model_path, enable_safety_check=False)
    runner.process_images(input_image_paths, output_dir)