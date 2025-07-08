# FLUX_kontext

## Install
```
conda create -n fluxkontext python=3.11
conda activate fluxkontext
pip install torch transformers accelerate sentencepiece protobuf realesrgan
pip install git+https://github.com/huggingface/diffusers.git
```

## Run
```
CUDA_VISIBLE_DEVICES=5 python 3_class.py
```