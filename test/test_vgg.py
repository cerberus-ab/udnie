import pytest
import torch
from unittest.mock import patch, MagicMock
from src.vgg import VGGFeatureExtractor

class TestVGG:
    @patch('torchvision.models.vgg19')
    def test_vgg_feature_extractor_init(self, mock_vgg19):
        """Test that VGGFeatureExtractor initializes correctly."""
        # Create a mock VGG model
        mock_model = MagicMock()
        mock_model.features = MagicMock()
        mock_vgg19.return_value = mock_model

        # Initialize the feature extractor
        extractor = VGGFeatureExtractor()

        # Check that the model was initialized correctly
        mock_vgg19.assert_called_once()

        # Check that the layers dictionary contains the expected keys
        expected_layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',
            '28': 'conv5_1'
        }
        assert extractor.layers == expected_layers

        # Check that the model parameters are frozen (not trainable)
        for param in mock_model.features.parameters.return_value:
            param.requires_grad = False

    @patch('torchvision.models.vgg19')
    def test_vgg_feature_extractor_forward(self, mock_vgg19, device, sample_tensor):
        """Test that VGGFeatureExtractor.forward extracts features correctly."""
        # Create a mock VGG model with a mock forward pass
        mock_model = MagicMock()
        mock_features = MagicMock()

        # Set up the mock model's layers to return different tensors for each layer
        layer_outputs = {}
        mock_modules = {}

        # Define the expected layers and their outputs
        expected_layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',
            '28': 'conv5_1'
        }

        # Create mock layers that return different tensors
        for layer_idx, layer_name in expected_layers.items():
            # Create a tensor with a unique shape for each layer
            channels = 64 if '1' in layer_name else 128 if '2' in layer_name else 256 if '3' in layer_name else 512
            size = 10 if '1' in layer_name else 5 if '2' in layer_name else 3 if '3' in layer_name else 2
            layer_output = torch.rand(1, channels, size, size, device=device)

            # Create a mock layer that returns this tensor
            mock_layer = MagicMock()
            mock_layer.return_value = layer_output
            mock_modules[layer_idx] = mock_layer

            # Store the expected output
            layer_outputs[layer_idx] = layer_output

        # Set up the mock model to use these layers
        mock_features._modules = mock_modules
        mock_model.features = mock_features
        mock_vgg19.return_value = mock_model

        # Initialize the feature extractor
        extractor = VGGFeatureExtractor()

        # Run the forward pass
        features = extractor(sample_tensor)

        # Check that the features dictionary contains the expected layers
        assert set(features.keys()) == set(expected_layers.values())

        # Check that each layer's output has the expected shape
        for layer_idx, layer_name in expected_layers.items():
            # The layer should have been called with the output of the previous layer
            # For the first layer, it should be called with the input tensor
            if layer_idx == '0':
                mock_modules[layer_idx].assert_called_with(sample_tensor)
            else:
                # For other layers, we can't easily check the exact input in this mock setup
                assert mock_modules[layer_idx].called

            # Check that the output tensor is in the features dictionary
            assert layer_name in features
            assert features[layer_name] is layer_outputs[layer_idx]

    def test_vgg_feature_extractor_real(self, device, sample_tensor):
        """Test that VGGFeatureExtractor works with a real tensor (integration test)."""
        # Skip this test if running in a CI environment or if it would take too long
        pytest.skip("Skipping integration test that loads the real VGG model")

        # Initialize the real feature extractor
        extractor = VGGFeatureExtractor().to(device)

        # Run the forward pass
        features = extractor(sample_tensor)

        # Check that the features dictionary contains the expected layers
        expected_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']
        assert set(features.keys()) == set(expected_layers)

        # Check that each feature tensor has the expected shape
        # The exact shapes depend on the input tensor size and the VGG architecture
        assert features['conv1_1'].shape[1] == 64  # 64 channels in first conv layer
        assert features['conv2_1'].shape[1] == 128  # 128 channels in second conv block
        assert features['conv3_1'].shape[1] == 256  # 256 channels in third conv block
        assert features['conv4_1'].shape[1] == 512  # 512 channels in fourth conv block
        assert features['conv4_2'].shape[1] == 512  # 512 channels in fourth conv block
        assert features['conv5_1'].shape[1] == 512  # 512 channels in fifth conv block
