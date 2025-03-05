import time

import torch
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torchvision.models import VGG19_Weights
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import pil_to_tensor

from src.Params import Params


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
    model = models.vgg19(weights=VGG19_Weights.DEFAULT).features
    for param in model.parameters():
        param.requires_grad = False
    return model, layers

def get_norm(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    return mean, std

def normalize(tensor):
    mean, std = get_norm(tensor)
    return (tensor - mean) / std

def denormalize(tensor):
    mean, std = get_norm(tensor)
    return tensor * std + mean

def load_image(image_path, max_size=None):
    image = Image.open(image_path).convert('RGB')

    w, h = image.size
    if max_size is not None and max(w, h) > max_size:
        scale_factor = max_size / max(w, h)
        new_size = (int(w * scale_factor), int(h * scale_factor))
        image = image.resize(new_size, Image.LANCZOS)

    return image, (w, h)

def load_image_tensor(image_path, device, max_size=None):
    pil_image, orig_size = load_image(image_path, max_size)

    tensor = pil_to_tensor(pil_image).float().div(255).unsqueeze(0).contiguous()
    tensor = tensor.to(device)
    tensor = normalize(tensor)

    return tensor, orig_size

def save_image(image, image_path, orig_size, show_image=True):
    new_size = image.size
    if new_size[0] < orig_size[0] or new_size[1] < orig_size[1]:
        image = image.resize(orig_size, Image.BICUBIC)

    image.save(image_path)
    if show_image:
        image.show()

def save_image_tensor(tensor, image_path, orig_size, show_image=True):
    tensor = denormalize(tensor)
    image = to_pil_image(tensor.squeeze(0).clamp(0, 1))

    save_image(image, image_path, orig_size, show_image)

def get_features(tensor, model, layers):
    # Get image features from model by layers
    features = {}
    x = tensor
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def content_loss(gen, content):
    return torch.nn.functional.mse_loss(gen, content)

def compute_content_loss(features_gen, features_content, layer):
    loss_c = content_loss(features_gen[layer], features_content[layer])
    return loss_c

def gram_matrix(tensor):
    b, c, h, w = tensor.shape
    features = tensor.view(c, h * w)
    gram = torch.mm(features, features.t())
    return gram / (c * h * w)

def style_loss(gen, style):
    return torch.nn.functional.mse_loss(gram_matrix(gen), gram_matrix(style))

def compute_style_loss(features_gen, features_style, layers_weights):
    loss_s = sum(
        layers_weights[layer] * style_loss(features_gen[layer], features_style[layer])
        for layer in layers_weights
    )
    return loss_s

def generate_styled_image(tensor_input, tensor_style, model, layers, steps = 300):
    # Classic parameters
    content_weight = 1
    content_layer = 'conv4_2'
    style_weight = 1e5
    style_layers_weights = {
        'conv1_1': 1.0,
        'conv2_1': 0.75,
        'conv3_1': 0.2,
        'conv4_1': 0.2,
        'conv5_1': 0.2,
    }

    features_style = get_features(tensor_style, model, layers)
    features_content = get_features(tensor_input, model, layers)

    tensor_gen = tensor_input.clone().requires_grad_(True)

    optimizer = optim.LBFGS([tensor_gen])

    def closure():
        optimizer.zero_grad()

        features_gen = get_features(tensor_gen, model, layers)

        loss_c = compute_content_loss(features_gen, features_content, content_layer)
        loss_s = compute_style_loss(features_gen, features_style, style_layers_weights)

        loss = content_weight * loss_c + style_weight * loss_s
        loss.backward()

        return loss

    print(f"Generating styled image with {steps} steps")
    start_time = time.time()
    for s in range(steps):
        optimizer.step(closure)
        print(f"Step {s+1}/{steps} ({(s+1)/steps*100:.1f}%), time elapsed: {time.time() - start_time:.2f}s")

    return tensor_gen

if __name__ == "__main__":
    params = Params.of_args()

    device = get_device()

    print(f"Loading input image: {params.input_path}, size: {params.size}")
    img_input, img_size = load_image_tensor(params.input_path, device, params.size)
    print(f"Loading style image for \"{params.style_name}\": {params.style_path}, size: {params.size}")
    img_style, _ = load_image_tensor(params.style_path, device, params.size)
    print(f"Output image path: {params.output_path}")

    print("Loading model")
    vgg_model, vgg_layers = load_model()
    vgg_model = vgg_model.to(device)

    img_output = generate_styled_image(img_input, img_style, vgg_model, vgg_layers, params.steps)
    print(f"Saving styled image to: {params.output_path}")
    save_image_tensor(img_output, params.output_path, img_size, params.show)
