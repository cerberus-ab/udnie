import os
import pytest
from unittest.mock import patch, MagicMock
import torch
from src.get_device import get_device, _cuda_available, _mps_available, _tpu_available

class TestGetDevice:
    def test_cuda_available(self):
        """Test that _cuda_available returns the correct value."""
        # This just verifies that the function returns the same value as torch.cuda.is_available()
        assert _cuda_available() == torch.cuda.is_available()

    def test_mps_available(self):
        """Test that _mps_available returns the correct value."""
        # This just verifies that the function returns the same value as torch.backends.mps.is_available()
        assert _mps_available() == torch.backends.mps.is_available()

    def test_tpu_available(self):
        """Test that _tpu_available returns the correct value based on environment variables."""
        # Test when TPU is not available
        with patch.dict(os.environ, {}, clear=True):
            assert not _tpu_available()

        # Test when TPU is available via COLAB_TPU_ADDR
        with patch.dict(os.environ, {"COLAB_TPU_ADDR": "10.0.0.1"}):
            assert _tpu_available()

        # Test when TPU is available via XRT_TPU_CONFIG
        with patch.dict(os.environ, {"XRT_TPU_CONFIG": "tpu_config"}):
            assert _tpu_available()

    @patch('src.get_device._cuda_available')
    @patch('src.get_device._mps_available')
    @patch('src.get_device._tpu_available')
    def test_get_device_with_cuda(self, mock_tpu, mock_mps, mock_cuda):
        """Test that get_device returns a CUDA device when CUDA is available."""
        # Mock device availability
        mock_cuda.return_value = True
        mock_mps.return_value = False
        mock_tpu.return_value = False

        # Test with no device type specified
        device = get_device()
        assert device.type == 'cuda'

        # Test with CUDA explicitly requested
        device = get_device('cuda')
        assert device.type == 'cuda'

    @patch('src.get_device._cuda_available')
    @patch('src.get_device._mps_available')
    @patch('src.get_device._tpu_available')
    def test_get_device_with_mps(self, mock_tpu, mock_mps, mock_cuda):
        """Test that get_device returns an MPS device when MPS is available and CUDA is not."""
        # Mock device availability
        mock_cuda.return_value = False
        mock_mps.return_value = True
        mock_tpu.return_value = False

        # Test with no device type specified
        device = get_device()
        assert device.type == 'mps'

        # Test with MPS explicitly requested
        device = get_device('mps')
        assert device.type == 'mps'

    @pytest.mark.skip(reason="torch_xla not installed")
    @patch('src.get_device._cuda_available')
    @patch('src.get_device._mps_available')
    @patch('src.get_device._tpu_available')
    @patch('torch_xla.core.xla_model.xla_device')
    def test_get_device_with_tpu(self, mock_xla_device, mock_tpu, mock_mps, mock_cuda):
        """Test that get_device returns a TPU device when TPU is available."""
        # Mock device availability
        mock_cuda.return_value = False
        mock_mps.return_value = False
        mock_tpu.return_value = True

        # Mock the xla_device function to return a device
        mock_device = MagicMock()
        mock_device.type = 'xla'
        mock_xla_device.return_value = mock_device

        # Test with no device type specified
        device = get_device()
        assert device.type == 'xla'

        # Test with TPU explicitly requested
        device = get_device('tpu')
        assert device.type == 'xla'

    @patch('src.get_device._cuda_available')
    @patch('src.get_device._mps_available')
    @patch('src.get_device._tpu_available')
    def test_get_device_with_cpu(self, mock_tpu, mock_mps, mock_cuda):
        """Test that get_device returns a CPU device when no accelerators are available."""
        # Mock device availability
        mock_cuda.return_value = False
        mock_mps.return_value = False
        mock_tpu.return_value = False

        # Test with no device type specified
        device = get_device()
        assert device.type == 'cpu'

        # Test with CPU explicitly requested
        device = get_device('cpu')
        assert device.type == 'cpu'

    @patch('src.get_device._cuda_available')
    @patch('src.get_device._mps_available')
    @patch('src.get_device._tpu_available')
    def test_get_device_fallback(self, mock_tpu, mock_mps, mock_cuda):
        """Test that get_device falls back to CPU when requested device is not available."""
        # Mock device availability
        mock_cuda.return_value = False
        mock_mps.return_value = False
        mock_tpu.return_value = False

        # Test with CUDA requested but not available
        device = get_device('cuda')
        assert device.type == 'cpu'

        # Test with MPS requested but not available
        device = get_device('mps')
        assert device.type == 'cpu'

        # Test with TPU requested but not available
        device = get_device('tpu')
        assert device.type == 'cpu'
