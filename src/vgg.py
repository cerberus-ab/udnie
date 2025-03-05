import torch
import torchvision.models as models

class VGGFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        self.layers = {
            '0': 'conv1_1', # Captures fine details, textures, and brushstrokes
            '5': 'conv2_1', # Extracts larger textural patterns
            '10': 'conv3_1', # Influences color distribution and mid-level textures
            '19': 'conv4_1', # Determines overall stylistic composition
            '21': 'conv4_2', # Typically used for content loss to preserve structure
            '28': 'conv5_1' # Captures high-level stylistic elements (brushstrokes, shading)
        }
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Extract features from input tensor
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features
