import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from img_utils import load_image, save_image


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

def load_image_tensor(image_path, device, max_size=None):
    # Load image to device as a normalized tensor;
    # returns the tensor and the original size of the image
    pil_image, orig_size = load_image(image_path, max_size)

    tensor = pil_to_tensor(pil_image).float().div(255).unsqueeze(0).contiguous()
    tensor = tensor.to(device)
    tensor = _normalize(tensor)

    return tensor, orig_size

def save_image_tensor(tensor, image_path, orig_size, show_image=True):
    # Save a normalized tensor as an image;
    # respects the original size of the image
    tensor = _denormalize(tensor)
    image = to_pil_image(tensor.squeeze(0).clamp(0, 1))

    save_image(image, image_path, orig_size, show_image)
