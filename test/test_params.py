from unittest.mock import patch, MagicMock
from src.Params import Params, _get_output_path, _style_map

class TestParams:
    def test_get_output_path(self):
        """Test that _get_output_path correctly constructs the output path."""
        input_path = "data/input/test.jpg"
        style_name = "udnie"
        init_method = "input"
        optim = "lbfgs"
        steps = 150

        expected_path = "data/output/test-udnie-input-lbfgs-150.jpg"
        actual_path = _get_output_path(input_path, style_name, init_method, optim, steps)

        assert actual_path == expected_path

    def test_get_output_path_with_different_extension(self):
        """Test that _get_output_path preserves the file extension."""
        input_path = "data/input/test.png"
        style_name = "udnie"
        init_method = "input"
        optim = "lbfgs"
        steps = 150

        expected_path = "data/output/test-udnie-input-lbfgs-150.png"
        actual_path = _get_output_path(input_path, style_name, init_method, optim, steps)

        assert actual_path == expected_path

    @patch('argparse.ArgumentParser.parse_args')
    def test_params_of_args_default_values(self, mock_parse_args):
        """Test that Params.of_args() correctly sets default values."""
        # Mock the return value of parse_args
        mock_args = MagicMock()
        mock_args.input = "data/input/test.jpg"
        mock_args.style = "udnie"
        mock_args.device = None
        mock_args.init = "input"
        mock_args.optim = "lbfgs"
        mock_args.steps = 150
        mock_args.size = 512
        mock_args.show = False
        mock_parse_args.return_value = mock_args

        params = Params.of_args()

        assert params.style_name == "udnie"
        assert params.style_path == _style_map["udnie"]
        assert params.input_path == "data/input/test.jpg"
        assert params.output_path == "data/output/test-udnie-input-lbfgs-150.jpg"
        assert params.device_type is None
        assert params.init_method == "input"
        assert params.optim == "lbfgs"
        assert params.steps == 150
        assert params.size == 512
        assert params.show is False

    @patch('argparse.ArgumentParser.parse_args')
    def test_params_of_args_custom_values(self, mock_parse_args):
        """Test that Params.of_args() correctly sets custom values."""
        # Mock the return value of parse_args
        mock_args = MagicMock()
        mock_args.input = "data/input/custom.jpg"
        mock_args.style = "starry_night"
        mock_args.device = "cpu"
        mock_args.init = "random"
        mock_args.optim = "adam"
        mock_args.steps = 300
        mock_args.size = 1024
        mock_args.show = True
        mock_parse_args.return_value = mock_args

        params = Params.of_args()

        assert params.style_name == "starry_night"
        assert params.style_path == _style_map["starry_night"]
        assert params.input_path == "data/input/custom.jpg"
        assert params.output_path == "data/output/custom-starry_night-random-adam-300.jpg"
        assert params.device_type == "cpu"
        assert params.init_method == "random"
        assert params.optim == "adam"
        assert params.steps == 300
        assert params.size == 1024
        assert params.show is True
