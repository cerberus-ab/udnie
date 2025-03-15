import time

import torch
import torch.optim as optim


def _content_loss(fm_gen, fm_content):
    return torch.nn.functional.mse_loss(fm_gen, fm_content)

def _compute_content_loss(fs_gen, fs_content, layer):
    loss_c = _content_loss(fs_gen[layer], fs_content[layer])
    return loss_c

def _gram_matrix(fm):
    b, ch, h, w = fm.shape
    features = fm.view(ch, h * w)
    gram = torch.mm(features, features.t())
    return gram / (ch * h * w)

def _style_loss(fm_gen, fm_style):
    return torch.nn.functional.mse_loss(_gram_matrix(fm_gen), _gram_matrix(fm_style))

def _compute_style_loss(fs_gen, fs_style, layers_weights):
    loss_s = sum(
        layers_weights[layer] * _style_loss(fs_gen[layer], fs_style[layer])
        for layer in layers_weights
    )
    return loss_s

def _compute_total_variation_loss(img):
    loss_h = torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    loss_w = torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    return loss_h + loss_w

def _compute_loss(img_gen, fs_gen, fs_content, fs_style):
    # Loss function parameters
    content_weight = 1 # alpha
    content_layer = 'conv4_2'
    style_weight = 1e6 # beta
    style_layers_weights = {
        'conv1_1': 1.0,
        'conv2_1': 0.75,
        'conv3_1': 0.2,
        'conv4_1': 0.2,
        'conv5_1': 0.2,
    }
    tv_weight = 1e-6 # gamma

    loss_c = _compute_content_loss(fs_gen, fs_content, content_layer)
    loss_s = _compute_style_loss(fs_gen, fs_style, style_layers_weights)
    loss_tv = _compute_total_variation_loss(img_gen)

    loss = content_weight * loss_c + style_weight * loss_s + tv_weight * loss_tv
    return loss

def transfer_style_lbfgs(t_gen, t_content, t_style, extract_features, steps):
    # Based on L-BFGS optimizer
    features_content = extract_features(t_content)
    features_style = extract_features(t_style)

    optimizer = optim.LBFGS([t_gen])

    def closure():
        optimizer.zero_grad()
        features_gen = extract_features(t_gen)
        loss = _compute_loss(t_gen, features_gen, features_content, features_style)
        loss.backward()

        return loss

    print(f"Generating styled image with {steps} steps of L-BFGS optimizer")
    start_time = time.time()
    for s in range(steps):
        optimizer.step(closure)
        print(f"Step {s+1}/{steps} ({(s+1)/steps*100:.1f}%), time elapsed: {time.time() - start_time:.2f}s")

    return t_gen

def transfer_style_adam(t_gen, t_content, t_style, extract_features, steps):
    # Based on Adam optimizer
    adam_params = {
        'lr': 0.04,
        'betas': (0.8, 0.99),
        'eps': 1e-8
    }
    features_content = extract_features(t_content)
    features_style = extract_features(t_style)

    optimizer = optim.Adam([t_gen], **adam_params)

    print(f"Generating styled image with {steps} steps of Adam optimizer")
    start_time = time.time()
    for s in range(steps):
        optimizer.zero_grad()
        features_gen = extract_features(t_gen)
        loss = _compute_loss(t_gen, features_gen, features_content, features_style)
        loss.backward()
        optimizer.step()
        if (s + 1) % 10 == 0:
            print(f"Step {s+1}/{steps} ({(s+1)/steps*100:.1f}%), time elapsed: {time.time() - start_time:.2f}s")

    return t_gen

def transfer_style(t_gen, t_content, t_style, extract_features, optim = 'lbfgs', steps = 300):
    # Transfer the style using the selected optimizer algorithm
    if optim == 'lbfgs':
        return transfer_style_lbfgs(t_gen, t_content, t_style, extract_features, steps)
    elif optim == 'adam':
        return transfer_style_adam(t_gen, t_content, t_style, extract_features, steps)
    else:
        print(f"Unknown optimizer: {optim}")
