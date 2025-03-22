import pytest
from unittest.mock import patch
from PIL import Image
from src.img_utils import load_image, save_image

class TestImgUtils:
    def test_load_image_no_resize(self, tmp_path):
        """Test that load_image correctly loads an image without resizing."""
        # Create a test image
        img_path = tmp_path / "test.jpg"
        test_img = Image.new('RGB', (100, 100), color='red')
        test_img.save(img_path)

        # Load the image with a max_size larger than the image
        img, orig_size, new_size = load_image(img_path, 200)

        # Check that the image was loaded correctly and not resized
        assert isinstance(img, Image.Image)
        assert orig_size == (100, 100)
        assert new_size == (100, 100)

    def test_load_image_with_resize(self, tmp_path):
        """Test that load_image correctly resizes an image if it's larger than max_size."""
        # Create a test image
        img_path = tmp_path / "test.jpg"
        test_img = Image.new('RGB', (200, 200), color='red')
        test_img.save(img_path)

        # Load the image with a max_size smaller than the image
        img, orig_size, new_size = load_image(img_path, 100)

        # Check that the image was loaded correctly and resized
        assert isinstance(img, Image.Image)
        assert orig_size == (200, 200)
        assert new_size == (100, 100)

    def test_save_image_no_resize(self, tmp_path, sample_image):
        """Test that save_image correctly saves an image without resizing."""
        # Create a path for the output image
        output_path = tmp_path / "output.jpg"

        # Save the image with orig_size equal to the image size
        orig_size = sample_image.size
        save_image(sample_image, output_path, orig_size, show_image=False)

        # Check that the image was saved correctly
        assert output_path.exists()
        saved_img = Image.open(output_path)
        assert saved_img.size == orig_size

    def test_save_image_with_resize(self, tmp_path, sample_image):
        """Test that save_image correctly resizes an image if it's smaller than orig_size."""
        # Create a path for the output image
        output_path = tmp_path / "output.jpg"

        # Save the image with orig_size larger than the image size
        orig_size = (sample_image.size[0] * 2, sample_image.size[1] * 2)
        save_image(sample_image, output_path, orig_size, show_image=False)

        # Check that the image was saved correctly and resized
        assert output_path.exists()
        saved_img = Image.open(output_path)
        assert saved_img.size == orig_size

    @patch('PIL.Image.Image.show')
    def test_save_image_with_show(self, mock_show, tmp_path, sample_image):
        """Test that save_image correctly shows the image when show_image is True."""
        # Create a path for the output image
        output_path = tmp_path / "output.jpg"

        # Save the image with show_image=True
        save_image(sample_image, output_path, sample_image.size, show_image=True)

        # Check that the image.show() method was called
        mock_show.assert_called_once()

    def test_load_image_invalid_path(self):
        """Test that load_image raises an exception when given an invalid path."""
        with pytest.raises(FileNotFoundError):
            load_image("nonexistent_file.jpg", 100)
