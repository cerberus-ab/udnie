import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image


def _get_norm(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(t.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(t.device)
    return mean, std

def _normalize(t):
    mean, std = _get_norm(t)
    return (t - mean) / std

def _denormalize(t):
    mean, std = _get_norm(t)
    return t * std + mean

def image_to_tensor(image, device):
    # Load PIL image to device as a normalized tensor;
    tensor = pil_to_tensor(image).float().div(255).unsqueeze(0).contiguous()
    tensor = tensor.to(device)
    tensor = _normalize(tensor)

    return tensor

def tensor_to_image(tensor):
    # Get a PIL image from a normalized tensor
    tensor = _denormalize(tensor)
    image = to_pil_image(tensor.squeeze(0).clamp(0, 1))

    return image

def init_gen_tensor(t_input, t_style, init_method):
    if init_method == "input":
        # Clone the input tensor
        return t_input.clone().requires_grad_(True)
    elif init_method == "random":
        # Create a random tensor with the same size as the input tensor
        return torch.randn_like(t_input, requires_grad=True)
    elif init_method == "blend":
        # Create a tensor that is a blend of the input and style tensors
        return (0.9 * t_input + 0.1 * t_style).requires_grad_(True)
    else:
        print(f"Unknown init method: {init_method}")
