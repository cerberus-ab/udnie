import pytest
import torch
import numpy as np
from PIL import Image
from src.tensor_utils import _get_norm, _normalize, _denormalize, image_to_tensor, tensor_to_image, init_gen_tensor

class TestTensorUtils:
    def test_get_norm(self, device):
        """Test that _get_norm returns the correct mean and std tensors."""
        # Create a test tensor
        t = torch.rand(1, 3, 10, 10, device=device)

        # Get the normalization parameters
        mean, std = _get_norm(t)

        # Check that the mean and std have the correct values and shapes
        assert mean.shape == (1, 3, 1, 1)
        assert std.shape == (1, 3, 1, 1)
        assert torch.allclose(mean, torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        assert torch.allclose(std, torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))
        assert mean.device == device
        assert std.device == device

    def test_normalize(self, device):
        """Test that _normalize correctly normalizes a tensor."""
        # Create a test tensor with known values
        t = torch.ones(1, 3, 10, 10, device=device)  # All ones

        # Normalize the tensor
        normalized = _normalize(t)

        # Get the expected values
        mean, std = _get_norm(t)
        expected = (t - mean) / std

        # Check that the normalization was done correctly
        assert torch.allclose(normalized, expected)

    def test_denormalize(self, device, normalized_tensor):
        """Test that _denormalize correctly reverses normalization."""
        # Denormalize the tensor
        denormalized = _denormalize(normalized_tensor)

        # Get the expected values
        mean, std = _get_norm(normalized_tensor)
        expected = normalized_tensor * std + mean

        # Check that the denormalization was done correctly
        assert torch.allclose(denormalized, expected)

    def test_normalize_denormalize_inverse(self, device, sample_tensor):
        """Test that normalization followed by denormalization returns the original tensor."""
        # Normalize and then denormalize
        normalized = _normalize(sample_tensor)
        denormalized = _denormalize(normalized)

        # Check that we get back the original tensor (with a small tolerance for floating-point precision issues)
        assert torch.allclose(denormalized, sample_tensor, rtol=1e-5, atol=1e-5)

    def test_image_to_tensor(self, device, sample_image):
        """Test that image_to_tensor correctly converts a PIL image to a tensor."""
        # Convert the image to a tensor
        tensor = image_to_tensor(sample_image, device)

        # Check that the tensor has the correct shape and device
        assert tensor.shape[0] == 1  # Batch dimension
        assert tensor.shape[1] == 3  # RGB channels
        assert tensor.shape[2:] == torch.Size(sample_image.size[::-1])  # Height, Width
        assert tensor.device == device

        # Check that the tensor is normalized
        assert tensor.min() < 0 and tensor.max() > 1  # Normalized values are outside [0, 1]

    def test_tensor_to_image(self, normalized_tensor):
        """Test that tensor_to_image correctly converts a tensor to a PIL image."""
        # Convert the tensor to an image
        image = tensor_to_image(normalized_tensor)

        # Check that we get a PIL image with the correct size
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'
        assert image.size == (normalized_tensor.shape[3], normalized_tensor.shape[2])

    def test_image_tensor_conversion_roundtrip(self, device, sample_image):
        """Test that converting an image to a tensor and back gives a similar image."""
        # Convert image to tensor and back
        tensor = image_to_tensor(sample_image, device)
        image = tensor_to_image(tensor)

        # Check that the image has the same size
        assert image.size == sample_image.size

        # Convert both images to numpy arrays for comparison
        # Note: There will be some differences due to quantization and normalization
        orig_array = np.array(sample_image)
        new_array = np.array(image)

        # Check that the images are similar (not exact due to processing)
        # We use a high tolerance because normalization and denormalization can cause significant changes
        assert np.abs(orig_array - new_array).mean() < 50  # Average pixel difference should be small

    def test_init_gen_tensor_input(self, device, sample_tensor):
        """Test that init_gen_tensor correctly initializes a tensor from the input."""
        # Initialize with 'input' method
        gen_tensor = init_gen_tensor(sample_tensor, None, "input")

        # Check that the tensor is a clone of the input and requires gradients
        assert torch.allclose(gen_tensor, sample_tensor)
        assert gen_tensor is not sample_tensor  # Should be a different tensor object
        assert gen_tensor.requires_grad

    def test_init_gen_tensor_random(self, device, sample_tensor):
        """Test that init_gen_tensor correctly initializes a random tensor."""
        # Initialize with 'random' method
        gen_tensor = init_gen_tensor(sample_tensor, None, "random")

        # Check that the tensor has the same shape as the input and requires gradients
        assert gen_tensor.shape == sample_tensor.shape
        assert gen_tensor.device == sample_tensor.device
        assert not torch.allclose(gen_tensor, sample_tensor)  # Should be different values
        assert gen_tensor.requires_grad

    def test_init_gen_tensor_blend(self, device, sample_tensor):
        """Test that init_gen_tensor correctly initializes a blended tensor."""
        # Create a style tensor
        style_tensor = torch.rand_like(sample_tensor)

        # Initialize with 'blend' method
        gen_tensor = init_gen_tensor(sample_tensor, style_tensor, "blend")

        # Check that the tensor is a blend of input and style and requires gradients
        expected = (0.9 * sample_tensor + 0.1 * style_tensor)
        assert torch.allclose(gen_tensor, expected)
        assert gen_tensor.requires_grad

    def test_init_gen_tensor_unknown(self, device, sample_tensor, capfd):
        """Test that init_gen_tensor handles unknown initialization methods."""
        # Initialize with an unknown method
        gen_tensor = init_gen_tensor(sample_tensor, None, "unknown")

        # Check that a warning was printed
        out, _ = capfd.readouterr()
        assert "Unknown init method: unknown" in out

        # The function should return None for unknown methods
        assert gen_tensor is None
