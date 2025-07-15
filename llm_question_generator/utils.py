import torch


def clear_gpu_cache():
    torch.cuda.empty_cache()
