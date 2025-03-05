import time

import torch
import torch.optim as optim


def _content_loss(gen, content):
    return torch.nn.functional.mse_loss(gen, content)

def _compute_content_loss(fs_gen, fs_content, layer):
    loss_c = _content_loss(fs_gen[layer], fs_content[layer])
    return loss_c

def _gram_matrix(t):
    b, ch, h, w = t.shape
    features = t.view(ch, h * w)
    gram = torch.mm(features, features.t())
    return gram / (ch * h * w)

def _style_loss(gen, style):
    return torch.nn.functional.mse_loss(_gram_matrix(gen), _gram_matrix(style))

def _compute_style_loss(fs_gen, fs_style, layers_weights):
    loss_s = sum(
        layers_weights[layer] * _style_loss(fs_gen[layer], fs_style[layer])
        for layer in layers_weights
    )
    return loss_s

def _compute_loss(fs_gen, fs_content, fs_style):
    # Classic parameters
    content_weight = 1
    content_layer = 'conv4_2'
    style_weight = 1e6
    style_layers_weights = {
        'conv1_1': 1.0,
        'conv2_1': 0.75,
        'conv3_1': 0.2,
        'conv4_1': 0.2,
        'conv5_1': 0.2,
    }

    loss_c = _compute_content_loss(fs_gen, fs_content, content_layer)
    loss_s = _compute_style_loss(fs_gen, fs_style, style_layers_weights)

    loss = content_weight * loss_c + style_weight * loss_s
    return loss

def transfer_style_lbfgs(tensor_input, tensor_style, extract_features, steps):
    # Based on L-BFGS optimizer
    features_content = extract_features(tensor_input)
    features_style = extract_features(tensor_style)

    tensor_gen = tensor_input.clone().requires_grad_(True)

    optimizer = optim.LBFGS([tensor_gen])

    def closure():
        optimizer.zero_grad()
        features_gen = extract_features(tensor_gen)
        loss = _compute_loss(features_gen, features_content, features_style)
        loss.backward()

        return loss

    print(f"Generating styled image with {steps} steps of L-BFGS optimizer")
    start_time = time.time()
    for s in range(steps):
        optimizer.step(closure)
        print(f"Step {s+1}/{steps} ({(s+1)/steps*100:.1f}%), time elapsed: {time.time() - start_time:.2f}s")

    return tensor_gen

def transfer_style_adam(tensor_input, tensor_style, extract_features, steps):
    # Based on Adam optimizer
    adam_params = {
        'lr': 0.04,
        'betas': (0.8, 0.99),
        'eps': 1e-8
    }
    features_content = extract_features(tensor_input)
    features_style = extract_features(tensor_style)

    tensor_gen = tensor_input.clone().requires_grad_(True)

    optimizer = optim.Adam([tensor_gen], **adam_params)

    print(f"Generating styled image with {steps} steps of Adam optimizer")
    start_time = time.time()
    for s in range(steps):
        optimizer.zero_grad()
        features_gen = extract_features(tensor_gen)
        loss = _compute_loss(features_gen, features_content, features_style)
        loss.backward()
        optimizer.step()
        if (s + 1) % 10 == 0:
            print(f"Step {s+1}/{steps} ({(s+1)/steps*100:.1f}%), time elapsed: {time.time() - start_time:.2f}s")

    return tensor_gen

def transfer_style(tensor_input, tensor_style, extract_features, optim = 'lbfgs', steps = 300):
    # Transfer the style using the selected optimizer algorithm
    if optim == 'lbfgs':
        return transfer_style_lbfgs(tensor_input, tensor_style, extract_features, steps)
    elif optim == 'adam':
        return transfer_style_adam(tensor_input, tensor_style, extract_features, steps)
    else:
        print(f"Unknown optimizer: {optim}")
