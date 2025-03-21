import pytest
import torch
from PIL import Image
import numpy as np

@pytest.fixture
def device():
    """Return a CPU device for testing."""
    return torch.device('cpu')

@pytest.fixture
def sample_image():
    """Create a small test image."""
    # Create a 10x10 RGB image with random values
    img_array = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    return Image.fromarray(img_array)

@pytest.fixture
def sample_tensor(device):
    """Create a small test tensor."""
    # Create a 1x3x10x10 tensor with random values between 0 and 1
    return torch.rand(1, 3, 10, 10, device=device)

@pytest.fixture
def normalized_tensor(device):
    """Create a normalized tensor for testing."""
    # Create a tensor with values that would result from normalization
    tensor = torch.rand(1, 3, 10, 10, device=device)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    return (tensor - mean) / std

@pytest.fixture
def mock_features():
    """Create mock VGG features for testing."""
    return {
        'conv1_1': torch.rand(1, 64, 10, 10),
        'conv2_1': torch.rand(1, 128, 5, 5),
        'conv3_1': torch.rand(1, 256, 3, 3),
        'conv4_1': torch.rand(1, 512, 2, 2),
        'conv4_2': torch.rand(1, 512, 2, 2),
        'conv5_1': torch.rand(1, 512, 1, 1)
    }
