import time

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from torchvision.models import VGG19_Weights

from src.Params import Params
from src.Paint import Paint


def get_device():
    # Get the device to run the model on
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print(f"Using device: {device_type}")
    return torch.device(device_type)

def load_model():
    # Load pre-trained VGG-19 model
    layers = {
        '0': 'conv1_1', # Captures fine details, textures, and brushstrokes
        '5': 'conv2_1', # Extracts larger textural patterns
        '10': 'conv3_1', # Influences color distribution and mid-level textures
        '19': 'conv4_1', # Determines overall stylistic composition
        '21': 'conv4_2', # Typically used for content loss to preserve structure
        '28': 'conv5_1' # Captures high-level stylistic elements (brushstrokes, shading)
    }
    model = models.vgg19(weights=VGG19_Weights.DEFAULT).features[:29]
    for param in model.parameters():
        param.requires_grad = False
    return model, layers

def load_image(image_path, max_size=512):
    # Load an image as a tensor
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize(max_size),
        transforms.ToTensor()
    ])

    image = transform(image).unsqueeze(0)
    return image

def save_image(tensor, image_path, show_image=True):
    # Save and show an image from tensor
    image = tensor.cpu().clone().detach().squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(image_path)
    if show_image:
        image.show()

def get_features(tensor, model, layers):
    # Get image features from model by layers
    features = {}
    x = tensor
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def content_loss(content, target):
    return torch.mean((content - target) ** 2)

def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    return torch.mm(tensor, tensor.t()) / (c * h * w)

def style_loss(gen, style):
    return torch.mean((gram_matrix(gen) - gram_matrix(style)) ** 2)

def generate_styled_image(tensor_input, tensor_style, model, layers, steps = 150):
    # Setup parameters
    alpha = 0.8
    beta = 90000
    style_weights = {
        "conv1_1": 6.0,
        "conv2_1": 5.0,
        "conv3_1": 12.0,
        "conv4_1": 8.0,
        "conv5_1": 2.0
    }
    start_input = 1 # 1..0

    features_style = get_features(tensor_style, model, layers)
    features_input = get_features(tensor_input, model, layers)

    tensor_output = (start_input * tensor_input + (1 - start_input) * tensor_style).clone().requires_grad_(True)

    optimizer = optim.LBFGS([tensor_output])

    def closure():
        optimizer.zero_grad()

        features_output = get_features(tensor_output, model, layers)
        loss_c = content_loss(features_output['conv4_2'], features_input['conv4_2'])

        loss_s = 0
        for layer in style_weights:
            loss_s += style_weights[layer] * style_loss(features_output[layer], features_style[layer])

        loss = alpha * loss_c + beta * loss_s
        loss.backward()

        return loss

    print(f"Generating styled image with {steps} steps")
    start_time = time.time()
    for s in range(steps):
        optimizer.step(closure)

        with torch.no_grad():
            tensor_output.clamp_(0, 1)

        print(f"Step {s+1}/{steps} ({(s+1)/steps*100:.1f}%), time elapsed: {time.time() - start_time:.2f}s")

    return tensor_output

if __name__ == "__main__":
    params = Params.of_args()

    device = get_device()

    print(f"Loading input image: {params.input_path}")
    img_input = load_image(params.input_path).to(device)
    print(f"Loading style image for \"{params.style_name}\": {params.style_path}")
    img_style = load_image(params.style_path).to(device)
    print(f"Output image path: {params.output_path} with post-processing: {params.post_process}")

    print("Loading model")
    vgg_model, vgg_layers = load_model()
    vgg_model = vgg_model.to(device)

    img_output = generate_styled_image(img_input, img_style, vgg_model, vgg_layers, params.steps)
    print(f"Saving styled image to: {params.output_path}")
    save_image(img_output, params.output_path)

    if params.post_process:
        print(f"Post-processing styled image to: {params.output_post_path}")
        Paint.match_histogram(params.output_path, params.style_path, params.output_post_path)
        Paint.enhance_colors(params.output_post_path, params.output_post_path)
