"""
Tests for the Canvas class in LlamaCanvas.

This module contains tests for all canvas functionality and operations.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from llama_canvas.core.canvas import Canvas
from llama_canvas.core.image import Image


@pytest.fixture
def sample_canvas():
    """Create a sample canvas for testing."""
    return Canvas(width=200, height=200, background_color=(240, 240, 240))


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    width, height = 100, 100
    array = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            r = int(255 * y / height)
            g = int(255 * x / width)
            b = int(128)
            array[y, x] = [r, g, b]

    return Image(array)


@pytest.fixture
def temp_image_path():
    """Create a temporary file path for saving images."""
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


def test_canvas_init():
    """Test canvas initialization with different parameters."""
    # Default initialization
    canvas = Canvas()
    assert canvas.width == 512
    assert canvas.height == 512
    assert canvas.background_color == (255, 255, 255)

    # Custom dimensions and background color
    canvas = Canvas(width=300, height=400, background_color=(0, 0, 0))
    assert canvas.width == 300
    assert canvas.height == 400
    assert canvas.background_color == (0, 0, 0)

    # Verify the base image has the correct dimensions
    assert canvas.base_image.width == 300
    assert canvas.base_image.height == 400

    # With agents disabled
    canvas = Canvas(use_agents=False)
    assert canvas.agents is None


def test_generate_from_text_placeholder(sample_canvas, mocker):
    """Test generate_from_text method using the placeholder implementation."""
    # Mock the agents to ensure we use the placeholder implementation
    sample_canvas.agents = None

    # Generate an image from text
    image = sample_canvas.generate_from_text("Test prompt")

    assert isinstance(image, Image)
    assert image.width == sample_canvas.width
    assert image.height == sample_canvas.height

    # Verify the image was added to layers
    assert len(sample_canvas.layers) == 1
    assert sample_canvas.layers[0] is image


def test_generate_from_text_with_agents(mocker):
    """Test generate_from_text when agents are available."""
    # Mock the AgentManager
    mock_agent_manager = mocker.MagicMock()
    mock_generated_image = mocker.MagicMock(spec=Image)
    mock_agent_manager.generate_image.return_value = mock_generated_image

    # Create canvas with mocked agent manager
    canvas = Canvas(width=100, height=100)
    canvas.agents = mock_agent_manager

    # Generate image
    result = canvas.generate_from_text("Test prompt", model="test-model")

    # Verify agent was called correctly
    mock_agent_manager.generate_image.assert_called_once_with(
        "Test prompt", model="test-model"
    )
    assert result is mock_generated_image


def test_apply_style_placeholder(sample_canvas, sample_image):
    """Test apply_style method using the placeholder implementation."""
    # Mock the agents to ensure we use the placeholder implementation
    sample_canvas.agents = None

    # Apply style
    styled_image = sample_canvas.apply_style(sample_image, "test_style")

    assert isinstance(styled_image, Image)
    assert styled_image.width == sample_image.width
    assert styled_image.height == sample_image.height

    # The placeholder implementation should return a different image
    assert styled_image is not sample_image


def test_apply_style_with_agents(sample_image, mocker):
    """Test apply_style when agents are available."""
    # Mock the AgentManager
    mock_agent_manager = mocker.MagicMock()
    mock_styled_image = mocker.MagicMock(spec=Image)
    mock_agent_manager.apply_style.return_value = mock_styled_image

    # Create canvas with mocked agent manager
    canvas = Canvas(width=100, height=100)
    canvas.agents = mock_agent_manager

    # Apply style using Claude
    result = canvas.apply_style(sample_image, "abstract", strength=0.7, use_claude=True)

    # Verify agent was called correctly
    mock_agent_manager.apply_style.assert_called_once_with(
        sample_image, "abstract", strength=0.7
    )
    assert result is mock_styled_image


def test_enhance_resolution_placeholder(sample_canvas, sample_image):
    """Test enhance_resolution method using the placeholder implementation."""
    # Mock the agents to ensure we use the placeholder implementation
    sample_canvas.agents = None

    # Enhance resolution
    enhanced = sample_canvas.enhance_resolution(sample_image, scale=2, use_agents=False)

    assert isinstance(enhanced, Image)
    assert enhanced.width == sample_image.width * 2
    assert enhanced.height == sample_image.height * 2

    # The placeholder implementation should return a different image
    assert enhanced is not sample_image


def test_enhance_resolution_with_agents(sample_image, mocker):
    """Test enhance_resolution when agents are available."""
    # Mock the AgentManager
    mock_agent_manager = mocker.MagicMock()
    mock_enhanced_image = mocker.MagicMock(spec=Image)
    mock_agent_manager.enhance_resolution.return_value = mock_enhanced_image

    # Create canvas with mocked agent manager
    canvas = Canvas(width=100, height=100)
    canvas.agents = mock_agent_manager

    # Enhance resolution
    result = canvas.enhance_resolution(sample_image, scale=4)

    # Verify agent was called correctly
    mock_agent_manager.enhance_resolution.assert_called_once_with(sample_image, scale=4)
    assert result is mock_enhanced_image


def test_blend_images(sample_canvas, sample_image):
    """Test blending two images."""
    # Create a second image with different colors
    width, height = 100, 100
    array2 = np.zeros((height, width, 3), dtype=np.uint8)
    array2[:, :, 0] = 255  # Red image
    image2 = Image(array2)

    # Blend images
    blended = sample_canvas.blend_images(sample_image, image2, alpha=0.3)

    assert isinstance(blended, Image)
    assert blended.width == sample_image.width
    assert blended.height == sample_image.height

    # The result should be different from both inputs
    assert not np.array_equal(blended.array, sample_image.array)
    assert not np.array_equal(blended.array, image2.array)


def test_blend_images_different_sizes(sample_canvas, sample_image):
    """Test blending images of different sizes."""
    # Create a larger image
    array2 = np.zeros((200, 200, 3), dtype=np.uint8)
    array2[:, :, 0] = 255  # Red image
    image2 = Image(array2)

    # Blend images
    blended = sample_canvas.blend_images(sample_image, image2)

    # Image2 should have been resized to match image1
    assert blended.width == sample_image.width
    assert blended.height == sample_image.height


def test_blend_images_different_modes(sample_canvas, sample_image):
    """Test blending images with different modes."""
    # Create an RGBA image
    array2 = np.zeros((100, 100, 4), dtype=np.uint8)
    array2[:, :, 0] = 255  # Red
    array2[:, :, 3] = 128  # Semi-transparent
    image2 = Image(array2)

    # Blend images
    blended = sample_canvas.blend_images(sample_image, image2)

    assert blended.width == sample_image.width
    assert blended.height == sample_image.height


def test_create_animation(sample_canvas, sample_image, mocker):
    """Test creating an animation from frames."""
    # Create multiple frames
    frames = [sample_image.copy() for _ in range(5)]

    # Mock the Video class
    mock_video = mocker.MagicMock()
    mocker.patch("llama_canvas.core.video.Video", return_value=mock_video)

    # Create animation
    video = sample_canvas.create_animation(frames, fps=30)

    assert video is mock_video


def test_save_empty_canvas(sample_canvas, temp_image_path):
    """Test saving an empty canvas."""
    # Save the empty canvas
    sample_canvas.save(temp_image_path)

    # Check that file exists
    assert os.path.exists(temp_image_path)

    # Load the file and check dimensions
    loaded = PILImage.open(temp_image_path)
    assert loaded.width == sample_canvas.width
    assert loaded.height == sample_canvas.height


def test_save_with_layers(sample_canvas, sample_image, temp_image_path):
    """Test saving a canvas with layers."""
    # Add image to layers
    sample_canvas.layers.append(sample_image)

    # Create a second layer
    array2 = np.zeros((100, 100, 4), dtype=np.uint8)
    array2[:, :, 0] = 255  # Red
    array2[:, :, 3] = 128  # Semi-transparent
    image2 = Image(array2)
    sample_canvas.layers.append(image2)

    # Save the canvas
    sample_canvas.save(temp_image_path)

    # Check that file exists
    assert os.path.exists(temp_image_path)

    # Load the file and check dimensions
    loaded = PILImage.open(temp_image_path)
    assert loaded.width == sample_image.width
    assert loaded.height == sample_image.height


def test_clear_canvas(sample_canvas, sample_image):
    """Test clearing a canvas."""
    # Add image to layers
    sample_canvas.layers.append(sample_image)
    assert len(sample_canvas.layers) == 1

    # Clear the canvas
    sample_canvas.clear()

    # Verify layers are empty
    assert len(sample_canvas.layers) == 0


if __name__ == "__main__":
    pytest.main()
