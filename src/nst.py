from Params import Params
from device_utils import get_device, print_memory_usage
from img_utils import load_image, save_image
from vgg import VGGFeatureExtractor
from tensor_utils import init_gen_tensor, image_to_tensor, tensor_to_image
from transfer_style import transfer_style

if __name__ == "__main__":
    params = Params.of_args()

    device = get_device(params.device_type)

    print(f"Loading input image: {params.input_path}, size: {params.size}")
    img_input, orig_size, new_size = load_image(params.input_path, params.size)
    t_input = image_to_tensor(img_input, device)

    print(f"Loading style image for \"{params.style_name}\": {params.style_path}, size: {max(new_size)}")
    img_style = load_image(params.style_path, max(new_size))[0]
    t_style = image_to_tensor(img_style, device)

    print(f"Init generated image by \"{params.init_method}\", path: {params.output_path}")
    t_gen = init_gen_tensor(t_input, t_style, params.init_method)

    print("Loading model...")
    vgg = VGGFeatureExtractor().to(device)

    t_output = transfer_style(t_gen, t_input, t_style, vgg, params.optim, params.steps)
    print(f"Saving styled image to: {params.output_path}")
    img_output = tensor_to_image(t_output)
    save_image(img_output, params.output_path, orig_size, params.show)

    if params.track_memory:
        print_memory_usage()
