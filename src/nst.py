import torch

from Params import Params
from vgg import VGGFeatureExtractor
from tensor_utils import load_image_tensor, save_image_tensor
from transfer_style import transfer_style


def get_device():
    # Get the device to run the model on
    if torch.cuda.is_available():
        d_type = "cuda"
    elif torch.backends.mps.is_available():
        d_type = "mps"
    else:
        d_type = "cpu"
    return torch.device(d_type)

if __name__ == "__main__":
    params = Params.of_args()

    device = get_device()
    print(f"Using device: {device.type}")

    print(f"Loading input image: {params.input_path}, size: {params.size}")
    t_input, orig_size = load_image_tensor(params.input_path, device, params.size)
    print(f"Loading style image for \"{params.style_name}\": {params.style_path}, size: {max(orig_size)}")
    t_style, _ = load_image_tensor(params.style_path, device, max(orig_size))
    print(f"Output image path: {params.output_path}")

    print("Loading model")
    vgg = VGGFeatureExtractor().to(device)

    t_output = transfer_style(t_input, t_style, vgg, params.optim, params.steps)
    print(f"Saving styled image to: {params.output_path}")
    save_image_tensor(t_output, params.output_path, orig_size, params.show)
