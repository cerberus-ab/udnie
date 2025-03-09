import os

import torch

def _cuda_available():
    return torch.cuda.is_available()

def _mps_available():
    return torch.backends.mps.is_available()

def _tpu_available():
    return "COLAB_TPU_ADDR" in os.environ or "XRT_TPU_CONFIG" in os.environ

def get_device(device_type=None):
    # Get the device to run the model on
    if device_type is not None:
        print(f"Trying to run on device: {device_type}")

    if _cuda_available() and (device_type is None or device_type == "cuda"):
        d_type = 'cuda' # GPU is the best for CNN (e.g. VGG)
    elif _mps_available() and (device_type is None or device_type == "mps"):
        d_type = 'mps' # Apple GPU acceleration for PyTorch (Metal)
    elif _tpu_available() and (device_type is None or device_type == "tpu"):
        d_type = 'tpu' # Google's AI chip, optimized via XLA
    else:
        d_type = 'cpu' # default

    if device_type is not None and d_type != device_type:
        print(f"Device {device_type} not available, running on: {d_type}")
    else:
        print(f"Running on device: {d_type}")

    if d_type == 'tpu':
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    else:
        return torch.device(d_type)
