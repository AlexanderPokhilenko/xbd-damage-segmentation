import os
import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")


def device_info(device: torch.device) -> str:
    if device.type == "cuda":
        return f"CUDA: {torch.cuda.get_device_name(0)}"
    if device.type == "mps":
        return "Apple MPS"
    return "CPU"
