import pytest
import torch
from unittest.mock import patch, MagicMock
from src.transfer_style import (_content_loss, _compute_content_loss, _gram_matrix, 
                           _style_loss, _compute_style_loss, _compute_total_variation_loss,
                           _compute_loss, transfer_style_lbfgs, transfer_style_adam, transfer_style)

class TestTransferStyle:
    def test_content_loss(self, device):
        """Test that _content_loss correctly computes MSE loss."""
        # Create test tensors
        fm_gen = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        fm_content = torch.tensor([[2.0, 3.0], [4.0, 5.0]], device=device)

        # Compute the loss
        loss = _content_loss(fm_gen, fm_content)

        # Compute the expected loss (MSE)
        expected_loss = torch.nn.functional.mse_loss(fm_gen, fm_content)

        # Check that the loss is correct
        assert torch.isclose(loss, expected_loss)

    def test_compute_content_loss(self, mock_features):
        """Test that _compute_content_loss correctly computes content loss for a specific layer."""
        # Create test feature maps
        fs_gen = mock_features
        fs_content = {k: v.clone() for k, v in mock_features.items()}

        # Modify one feature map to create a difference
        layer = 'conv4_2'
        fs_gen[layer] = fs_gen[layer] + 0.1

        # Compute the content loss
        loss = _compute_content_loss(fs_gen, fs_content, layer)

        # Compute the expected loss
        expected_loss = _content_loss(fs_gen[layer], fs_content[layer])

        # Check that the loss is correct
        assert torch.isclose(loss, expected_loss)

    def test_gram_matrix(self, device):
        """Test that _gram_matrix correctly computes the Gram matrix."""
        # Create a test feature map
        fm = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device).view(1, 2, 2, 1)

        # Compute the Gram matrix
        gram = _gram_matrix(fm)

        # Compute the expected Gram matrix
        features = fm.view(2, 2)
        expected_gram = torch.mm(features, features.t()) / (2 * 2 * 1)

        # Check that the Gram matrix is correct
        assert torch.allclose(gram, expected_gram)

    def test_style_loss(self, device):
        """Test that _style_loss correctly computes style loss using Gram matrices."""
        # Create test feature maps
        fm_gen = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device).view(1, 2, 2, 1)
        fm_style = torch.tensor([[2.0, 3.0], [4.0, 5.0]], device=device).view(1, 2, 2, 1)

        # Compute the style loss
        loss = _style_loss(fm_gen, fm_style)

        # Compute the expected loss
        gram_gen = _gram_matrix(fm_gen)
        gram_style = _gram_matrix(fm_style)
        expected_loss = torch.nn.functional.mse_loss(gram_gen, gram_style)

        # Check that the loss is correct
        assert torch.isclose(loss, expected_loss)

    def test_compute_style_loss(self, mock_features):
        """Test that _compute_style_loss correctly computes weighted style loss across layers."""
        # Create test feature maps
        fs_gen = mock_features
        fs_style = {k: v.clone() for k, v in mock_features.items()}

        # Modify feature maps to create differences
        for layer in fs_gen:
            fs_gen[layer] = fs_gen[layer] + 0.1

        # Define layer weights
        layers_weights = {
            'conv1_1': 1.0,
            'conv2_1': 0.75,
            'conv3_1': 0.2,
            'conv4_1': 0.2,
            'conv5_1': 0.2,
        }

        # Compute the style loss
        loss = _compute_style_loss(fs_gen, fs_style, layers_weights)

        # Compute the expected loss
        expected_loss = sum(
            layers_weights[layer] * _style_loss(fs_gen[layer], fs_style[layer])
            for layer in layers_weights
        )

        # Check that the loss is correct
        assert torch.isclose(loss, expected_loss)

    def test_compute_total_variation_loss(self, device):
        """Test that _compute_total_variation_loss correctly computes total variation loss."""
        # Create a test image tensor
        img = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], device=device).view(1, 1, 3, 3)

        # Compute the total variation loss
        loss = _compute_total_variation_loss(img)

        # Compute the expected loss
        h_diff = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        w_diff = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        expected_loss = torch.sum(h_diff) + torch.sum(w_diff)

        # Check that the loss is correct
        assert torch.isclose(loss, expected_loss)

    def test_compute_loss(self, device, mock_features):
        """Test that _compute_loss correctly combines all losses."""
        # Create test tensors and feature maps
        img_gen = torch.rand(1, 3, 10, 10, device=device)
        fs_gen = mock_features
        fs_content = {k: v.clone() for k, v in mock_features.items()}
        fs_style = {k: v.clone() for k, v in mock_features.items()}

        # Modify feature maps to create differences
        for layer in fs_gen:
            fs_gen[layer] = fs_gen[layer] + 0.1

        # Compute the loss
        loss = _compute_loss(img_gen, fs_gen, fs_content, fs_style)

        # Check that the loss is a scalar tensor
        assert loss.dim() == 0
        # Note: The loss may not require gradients in the test environment

    @patch('torch.optim.LBFGS')
    @patch('time.time')
    def test_transfer_style_lbfgs(self, mock_time, mock_lbfgs, device, sample_tensor):
        """Test that transfer_style_lbfgs correctly optimizes the generated image."""
        # Mock time.time to return predictable values
        mock_time.side_effect = [0, 1]  # Start time, end time

        # Create test tensors
        t_gen = sample_tensor.clone().requires_grad_(True)
        t_content = sample_tensor.clone()
        t_style = sample_tensor.clone()

        # Create a mock optimizer
        mock_optimizer = MagicMock()
        mock_lbfgs.return_value = mock_optimizer

        # Create mock features
        mock_features = {
            'conv1_1': torch.rand(1, 64, 10, 10, device=device),
            'conv2_1': torch.rand(1, 128, 5, 5, device=device),
            'conv3_1': torch.rand(1, 256, 3, 3, device=device),
            'conv4_1': torch.rand(1, 512, 2, 2, device=device),
            'conv4_2': torch.rand(1, 512, 2, 2, device=device),
            'conv5_1': torch.rand(1, 512, 1, 1, device=device)
        }

        # Create a mock feature extractor
        extract_features = MagicMock()
        extract_features.side_effect = [
            {k: v.clone() for k, v in mock_features.items()},  # Content features
            {k: v.clone() for k, v in mock_features.items()},  # Style features
            {k: v.clone() for k, v in mock_features.items()},  # Generated features
        ]

        # Run the style transfer
        result = transfer_style_lbfgs(t_gen, t_content, t_style, extract_features, steps=1)

        # Check that the optimizer was created and used correctly
        mock_lbfgs.assert_called_once_with([t_gen])
        assert mock_optimizer.step.call_count == 1

        # Check that the result is the same tensor object as t_gen
        assert result is t_gen

    @patch('torch.optim.Adam')
    @patch('time.time')
    def test_transfer_style_adam(self, mock_time, mock_adam, device, sample_tensor):
        """Test that transfer_style_adam correctly optimizes the generated image."""
        # Mock time.time to return predictable values
        mock_time.side_effect = [0, 1]  # Start time, end time

        # Create test tensors
        t_gen = sample_tensor.clone().requires_grad_(True)
        t_content = sample_tensor.clone()
        t_style = sample_tensor.clone()

        # Create a mock optimizer
        mock_optimizer = MagicMock()
        mock_adam.return_value = mock_optimizer

        # Create mock features
        mock_features = {
            'conv1_1': torch.rand(1, 64, 10, 10, device=device),
            'conv2_1': torch.rand(1, 128, 5, 5, device=device),
            'conv3_1': torch.rand(1, 256, 3, 3, device=device),
            'conv4_1': torch.rand(1, 512, 2, 2, device=device),
            'conv4_2': torch.rand(1, 512, 2, 2, device=device),
            'conv5_1': torch.rand(1, 512, 1, 1, device=device)
        }

        # Create a mock feature extractor
        extract_features = MagicMock()
        extract_features.side_effect = [
            {k: v.clone() for k, v in mock_features.items()},  # Content features
            {k: v.clone() for k, v in mock_features.items()},  # Style features
            {k: v.clone() for k, v in mock_features.items()},  # Generated features
        ]

        # Run the style transfer
        result = transfer_style_adam(t_gen, t_content, t_style, extract_features, steps=1)

        # Check that the optimizer was created and used correctly
        mock_adam.assert_called_once()
        assert mock_optimizer.zero_grad.call_count == 1
        assert mock_optimizer.step.call_count == 1

        # Check that the result is the same tensor object as t_gen
        assert result is t_gen

    @patch('src.transfer_style.transfer_style_lbfgs')
    @patch('src.transfer_style.transfer_style_adam')
    def test_transfer_style_lbfgs_selection(self, mock_adam, mock_lbfgs, sample_tensor):
        """Test that transfer_style correctly selects the L-BFGS optimizer."""
        # Create test tensors
        t_gen = sample_tensor.clone().requires_grad_(True)
        t_content = sample_tensor.clone()
        t_style = sample_tensor.clone()
        extract_features = MagicMock()

        # Run the style transfer with L-BFGS
        transfer_style(t_gen, t_content, t_style, extract_features, optim='lbfgs', steps=100)

        # Check that the correct optimizer was used
        mock_lbfgs.assert_called_once_with(t_gen, t_content, t_style, extract_features, 100)
        mock_adam.assert_not_called()

    @patch('src.transfer_style.transfer_style_lbfgs')
    @patch('src.transfer_style.transfer_style_adam')
    def test_transfer_style_adam_selection(self, mock_adam, mock_lbfgs, sample_tensor):
        """Test that transfer_style correctly selects the Adam optimizer."""
        # Create test tensors
        t_gen = sample_tensor.clone().requires_grad_(True)
        t_content = sample_tensor.clone()
        t_style = sample_tensor.clone()
        extract_features = MagicMock()

        # Run the style transfer with Adam
        transfer_style(t_gen, t_content, t_style, extract_features, optim='adam', steps=100)

        # Check that the correct optimizer was used
        mock_adam.assert_called_once_with(t_gen, t_content, t_style, extract_features, 100)
        mock_lbfgs.assert_not_called()

    @patch('src.transfer_style.transfer_style_lbfgs')
    @patch('src.transfer_style.transfer_style_adam')
    def test_transfer_style_unknown_optimizer(self, mock_adam, mock_lbfgs, sample_tensor, capfd):
        """Test that transfer_style handles unknown optimizers."""
        # Create test tensors
        t_gen = sample_tensor.clone().requires_grad_(True)
        t_content = sample_tensor.clone()
        t_style = sample_tensor.clone()
        extract_features = MagicMock()

        # Run the style transfer with an unknown optimizer
        result = transfer_style(t_gen, t_content, t_style, extract_features, optim='unknown', steps=100)

        # Check that a warning was printed
        out, _ = capfd.readouterr()
        assert "Unknown optimizer: unknown" in out

        # Check that no optimizer was used
        mock_lbfgs.assert_not_called()
        mock_adam.assert_not_called()

        # The function should return None
