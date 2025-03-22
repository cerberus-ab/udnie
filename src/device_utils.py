import os

import psutil
import torch

def _cuda_available():
    return torch.cuda.is_available()

def _mps_available():
    return torch.backends.mps.is_available()

def _tpu_available():
    return "COLAB_TPU_ADDR" in os.environ or "XRT_TPU_CONFIG" in os.environ

def _bytes_to_mb(bytes_value):
    return bytes_value / (1024 * 1024)

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

def get_memory_usage():
    # Get memory usage
    process = psutil.Process(os.getpid())
    process_memory = _bytes_to_mb(process.memory_info().rss)

    torch_memory = {}
    if _cuda_available():
        torch.cuda.synchronize()
        torch_memory['cuda'] = {
            'allocated': _bytes_to_mb(torch.cuda.memory_allocated()),
            'reserved': _bytes_to_mb(torch.cuda.memory_reserved())
        }

    if hasattr(torch, 'mps') and hasattr(torch.mps, 'current_allocated_memory'):
        torch_memory['mps'] = {
            'allocated': _bytes_to_mb(torch.mps.current_allocated_memory())
        }

    return {
        'process': process_memory,
        'torch': torch_memory
    }

def print_memory_usage():
    # Print memory usage
    memory = get_memory_usage()
    print(f"\nFinal memory usage:")
    print(f"  Process memory: {memory['process']:.2f} MB")

    for device, stats in memory['torch'].items():
        print(f"  {device.upper()} memory:")
        for key, value in stats.items():
            print(f"    {key}: {value:.2f} MB")
    print()