"""
Shared fixtures for LlamaCanvas tests.

This module contains pytest fixtures shared across multiple test files.
"""

import os
import tempfile

import numpy as np
import pytest
from PIL import Image as PILImage

from llama_canvas.core.canvas import Canvas
from llama_canvas.core.image import Image
from llama_canvas.core.video import Video


@pytest.fixture
def sample_image():
    """Create a sample gradient image for testing."""
    # Create a simple gradient image
    width, height = 100, 100
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            r = int(255 * y / height)
            g = int(255 * x / width)
            b = int(255 * (1 - (x + y) / (width + height)))
            gradient[y, x] = [r, g, b]

    return Image(gradient)


@pytest.fixture
def temp_image_path():
    """Create a temporary file path for saving images."""
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def sample_canvas():
    """Create a sample canvas for testing."""
    return Canvas(width=200, height=200, background_color=(240, 240, 240))


@pytest.fixture
def sample_frames():
    """Create a list of sample frames for testing."""
    frames = []
    width, height = 100, 100

    # Create 10 gradient frames with different colors
    for i in range(10):
        array = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                r = int(255 * y / height)
                g = int(255 * x / width)
                b = int(255 * i / 10)  # Change blue based on frame number
                array[y, x] = [r, g, b]
        frames.append(Image(array))

    return frames


@pytest.fixture
def sample_video(sample_frames):
    """Create a sample video for testing."""
    return Video(sample_frames, fps=24)


@pytest.fixture
def temp_video_path():
    """Create a temporary file path for saving videos."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def sample_base64_image():
    """Create a sample base64 encoded image for testing."""
    import base64
    import io

    # Create a simple test image
    width, height = 100, 100
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, :, 0] = 255  # Red image

    pil_image = PILImage.fromarray(array)

    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir
