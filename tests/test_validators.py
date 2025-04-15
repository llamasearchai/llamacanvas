"""
Tests for validator functions in LlamaCanvas.

This module contains comprehensive tests for data validation utilities.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage

from llama_canvas.utils.validators import (
    normalize_color,
    validate_aspect_ratio,
    validate_color,
    validate_file_exists,
    validate_fps,
    validate_image_array,
    validate_image_format,
    validate_image_size,
    validate_model_name,
    validate_range,
    validate_video_format,
)


class TestImageValidation:
    """Tests for image validation utilities."""

    def test_validate_image_size(self):
        """Test validating image dimensions."""
        # Valid size
        assert validate_image_size(512, 512) == (512, 512)
        assert validate_image_size(100, 200) == (100, 200)

        # Below minimum (typical minimum is around 32 or 64)
        with pytest.raises(ValueError, match="minimum"):
            validate_image_size(10, 512)

        # Above maximum (typical maximum is several thousand pixels)
        with pytest.raises(ValueError, match="maximum"):
            validate_image_size(512, 10000)

        # Default values when None provided
        with patch("llama_canvas.utils.validators.settings") as mock_settings:
            mock_settings.get.side_effect = lambda k, d: {
                "default_image_width": 800,
                "default_image_height": 600,
                "min_image_dimension": 32,
                "max_image_dimension": 4096,
            }.get(k, d)

            assert validate_image_size(None, None) == (800, 600)
            assert validate_image_size(400, None) == (400, 600)
            assert validate_image_size(None, 400) == (800, 400)

        # With custom min/max
        assert validate_image_size(100, 100, min_dim=50, max_dim=1000) == (100, 100)

        with pytest.raises(ValueError):
            validate_image_size(10, 100, min_dim=50)

        with pytest.raises(ValueError):
            validate_image_size(2000, 100, max_dim=1000)

    def test_validate_aspect_ratio(self):
        """Test validating aspect ratio."""
        # Valid aspect ratio
        assert validate_aspect_ratio(800, 600) == (800, 600)
        assert validate_aspect_ratio(1920, 1080) == (1920, 1080)

        # Square is always valid
        assert validate_aspect_ratio(500, 500) == (500, 500)

        # Extreme aspect ratio
        with pytest.raises(ValueError, match="extreme"):
            validate_aspect_ratio(2000, 100)  # 20:1 ratio

        # Adjust ratio to fit within limits
        corrected_width, corrected_height = validate_aspect_ratio(800, 100, adjust=True)

        # Should adjust to be within the max ratio (probably 4:1 or similar)
        assert corrected_width < 800 or corrected_height > 100
        assert 0.25 <= corrected_width / corrected_height <= 4.0  # Common limit

        # Test with custom max ratio
        with pytest.raises(ValueError):
            validate_aspect_ratio(600, 100, max_ratio=5.0)  # 6:1 ratio

        assert validate_aspect_ratio(500, 100, max_ratio=5.0) == (500, 100)  # 5:1 ratio

        # Test adjustment with custom max ratio
        corrected_width, corrected_height = validate_aspect_ratio(
            600, 100, adjust=True, max_ratio=5.0
        )
        assert (
            corrected_width == 500 or corrected_height == 120
        )  # Either width reduced or height increased

    def test_validate_image_array(self):
        """Test validating numpy image arrays."""
        # Valid RGB array
        rgb_array = np.zeros((100, 100, 3), dtype=np.uint8)
        assert validate_image_array(rgb_array) is rgb_array

        # Valid RGBA array
        rgba_array = np.zeros((100, 100, 4), dtype=np.uint8)
        assert validate_image_array(rgba_array) is rgba_array

        # Invalid shape (1D)
        with pytest.raises(ValueError, match="dimensions"):
            validate_image_array(np.zeros(100))

        # Invalid shape (too many dimensions)
        with pytest.raises(ValueError, match="dimensions"):
            validate_image_array(np.zeros((100, 100, 3, 3)))

        # Invalid channels
        with pytest.raises(ValueError, match="channels"):
            validate_image_array(np.zeros((100, 100, 2)))

        # Invalid data type
        float_array = np.zeros((100, 100, 3), dtype=np.float32)

        # Should be converted to uint8
        converted = validate_image_array(float_array)
        assert converted.dtype == np.uint8

        # Test with allowed channels
        grayscale = np.zeros((100, 100), dtype=np.uint8)

        # Not allowed by default
        with pytest.raises(ValueError):
            validate_image_array(grayscale)

        # Allowed when specified
        assert validate_image_array(grayscale, allowed_channels=[1, 3, 4]) is grayscale

    def test_validate_image_format(self):
        """Test validating image file formats."""
        # Valid formats
        assert validate_image_format("test.png") == "test.png"
        assert validate_image_format("test.jpg") == "test.jpg"
        assert validate_image_format("test.jpeg") == "test.jpeg"
        assert validate_image_format("test.gif") == "test.gif"

        # Invalid format
        with pytest.raises(ValueError, match="format"):
            validate_image_format("test.txt")

        # Case insensitivity
        assert validate_image_format("TEST.PNG") == "TEST.PNG"

        # With custom allowed formats
        assert validate_image_format("test.svg", allowed_formats=[".svg"]) == "test.svg"

        with pytest.raises(ValueError):
            validate_image_format("test.webp", allowed_formats=[".png", ".jpg"])


class TestColorValidation:
    """Tests for color validation utilities."""

    def test_validate_color(self):
        """Test validating color values."""
        # RGB tuple
        assert validate_color((255, 0, 0)) == (255, 0, 0)
        assert validate_color((0, 255, 0)) == (0, 255, 0)
        assert validate_color((0, 0, 255)) == (0, 0, 255)

        # RGBA tuple
        assert validate_color((255, 0, 0, 128)) == (255, 0, 0, 128)

        # Hex string
        assert validate_color("#FF0000") == (255, 0, 0)
        assert validate_color("#00FF00") == (0, 255, 0)

        # Short hex
        assert validate_color("#F00") == (255, 0, 0)

        # Named color
        assert validate_color("red") == (255, 0, 0)
        assert validate_color("green") == (0, 128, 0)

        # Invalid RGB value
        with pytest.raises(ValueError, match="valid color"):
            validate_color((256, 0, 0))

        # Invalid hex
        with pytest.raises(ValueError, match="valid color"):
            validate_color("#XYZ")

        # Invalid name
        with pytest.raises(ValueError, match="valid color"):
            validate_color("not_a_color")

        # Invalid type
        with pytest.raises(ValueError, match="valid color"):
            validate_color(123)

    def test_normalize_color(self):
        """Test normalizing colors to RGB/RGBA tuples."""
        # RGB tuple unchanged
        assert normalize_color((255, 0, 0)) == (255, 0, 0)

        # RGBA tuple unchanged
        assert normalize_color((255, 0, 0, 128)) == (255, 0, 0, 128)

        # Hex string to RGB
        assert normalize_color("#FF0000") == (255, 0, 0)

        # Short hex to RGB
        assert normalize_color("#F00") == (255, 0, 0)

        # Named color to RGB
        assert normalize_color("red") == (255, 0, 0)

        # With alpha specified
        assert normalize_color("#FF0000", alpha=128) == (255, 0, 0, 128)
        assert normalize_color("blue", alpha=200) == (0, 0, 255, 200)

        # Force RGB
        assert normalize_color((255, 0, 0, 128), force_rgb=True) == (255, 0, 0)

        # Force RGBA
        assert normalize_color((255, 0, 0), force_rgba=True) == (255, 0, 0, 255)

        # Invalid color
        with pytest.raises(ValueError):
            normalize_color("not_a_color")


class TestFileValidation:
    """Tests for file validation utilities."""

    def test_validate_file_exists(self):
        """Test validating that a file exists."""
        import os
        import tempfile

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file_path = temp_file.name

        try:
            # File exists
            assert validate_file_exists(file_path) == file_path

            # File does not exist
            with pytest.raises(FileNotFoundError):
                validate_file_exists("/path/to/nonexistent/file.txt")

            # With create_ok
            nonexistent = "/tmp/nonexistent_dir/test.txt"
            with patch("os.path.exists", return_value=False), patch(
                "os.makedirs"
            ) as mock_makedirs:
                assert validate_file_exists(nonexistent, create_ok=True) == nonexistent
                assert mock_makedirs.called

            # Directory not file
            with tempfile.TemporaryDirectory() as temp_dir:
                with pytest.raises(ValueError, match="directory"):
                    validate_file_exists(temp_dir, must_be_file=True)
        finally:
            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_validate_video_format(self):
        """Test validating video file formats."""
        # Valid formats
        assert validate_video_format("test.mp4") == "test.mp4"
        assert validate_video_format("test.avi") == "test.avi"
        assert validate_video_format("test.mov") == "test.mov"

        # Invalid format
        with pytest.raises(ValueError, match="format"):
            validate_video_format("test.txt")

        # Case insensitivity
        assert validate_video_format("TEST.MP4") == "TEST.MP4"

        # With custom allowed formats
        assert validate_video_format("test.flv", allowed_formats=[".flv"]) == "test.flv"

        with pytest.raises(ValueError):
            validate_video_format("test.mp4", allowed_formats=[".flv", ".mov"])


class TestMiscValidation:
    """Tests for miscellaneous validation utilities."""

    def test_validate_range(self):
        """Test validating numeric ranges."""
        # Valid values
        assert validate_range(5, 0, 10) == 5
        assert validate_range(0, 0, 10) == 0
        assert validate_range(10, 0, 10) == 10

        # Below minimum
        with pytest.raises(ValueError, match="minimum"):
            validate_range(-1, 0, 10)

        # Above maximum
        with pytest.raises(ValueError, match="maximum"):
            validate_range(11, 0, 10)

        # Default when None
        assert validate_range(None, 0, 10, default=5) == 5

        # Clamp instead of error
        assert validate_range(-1, 0, 10, clamp=True) == 0
        assert validate_range(11, 0, 10, clamp=True) == 10

        # With custom error message
        with pytest.raises(ValueError, match="custom error"):
            validate_range(11, 0, 10, error_message="custom error")

    def test_validate_fps(self):
        """Test validating frames per second."""
        # Valid values
        assert validate_fps(24) == 24
        assert validate_fps(30) == 30
        assert validate_fps(60) == 60

        # Too low
        with pytest.raises(ValueError, match="minimum"):
            validate_fps(0)

        # Too high
        with pytest.raises(ValueError, match="maximum"):
            validate_fps(1000)

        # Default when None
        assert validate_fps(None) == 30  # Typical default

        # With custom min/max
        assert validate_fps(10, min_fps=5) == 10

        with pytest.raises(ValueError):
            validate_fps(10, min_fps=15)

    def test_validate_model_name(self):
        """Test validating AI model names."""
        # Supported models
        assert validate_model_name("claude-3-opus") == "claude-3-opus"
        assert validate_model_name("stable-diffusion-xl") == "stable-diffusion-xl"

        # Unknown model
        with pytest.raises(ValueError, match="supported"):
            validate_model_name("non-existent-model")

        # Default when None
        assert validate_model_name(None) == "claude-3-opus"  # Typical default

        # With custom supported models
        assert (
            validate_model_name(
                "custom-model", supported_models=["custom-model", "another-model"]
            )
            == "custom-model"
        )

        # Case insensitivity
        assert (
            validate_model_name(
                "CUSTOM-MODEL",
                supported_models=["custom-model", "another-model"],
                case_sensitive=False,
            )
            == "CUSTOM-MODEL"
        )


if __name__ == "__main__":
    pytest.main()
