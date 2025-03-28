"""
Tests for the Image class in LlamaCanvas.

This module contains tests for all image manipulation functionality provided by the Image class.
"""

import os
import tempfile
from pathlib import Path
import pytest
import numpy as np
from PIL import Image as PILImage, ImageFilter

from llama_canvas.core.image import Image


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
    fd, path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


def test_image_creation_from_array(sample_image):
    """Test creating an image from a numpy array."""
    # From numpy array
    array = np.zeros((50, 50, 3), dtype=np.uint8)
    array[:, :, 0] = 255  # Red channel
    image = Image(array)
    
    assert image.width == 50
    assert image.height == 50
    assert image.shape == (50, 50, 3)
    assert np.array_equal(image.array[:, :, 0], np.full((50, 50), 255))


def test_image_creation_from_pil():
    """Test creating an image from a PIL Image."""
    pil_image = PILImage.new('RGB', (30, 40), color='blue')
    image = Image(pil_image)
    
    assert image.width == 30
    assert image.height == 40
    assert image.mode == 'RGB'
    # Check that the image is blue
    assert np.all(image.array[:, :, 2] > 240)  # Blue channel


def test_image_creation_from_path(temp_image_path, sample_image):
    """Test creating an image from a file path."""
    # Save and reload
    sample_image.save(temp_image_path)
    
    # Create from path string
    image1 = Image(temp_image_path)
    assert image1.width == sample_image.width
    assert image1.height == sample_image.height
    
    # Create from Path object
    image2 = Image(Path(temp_image_path))
    assert image2.width == sample_image.width
    assert image2.height == sample_image.height


def test_image_creation_from_another_image(sample_image):
    """Test creating an image from another Image instance."""
    image = Image(sample_image)
    
    assert image.width == sample_image.width
    assert image.height == sample_image.height
    
    # Ensure it's a copy, not the same object
    assert image is not sample_image
    assert image._image is not sample_image._image


def test_image_properties(sample_image):
    """Test image property accessors."""
    assert sample_image.width == 100
    assert sample_image.height == 100
    assert sample_image.shape == (100, 100, 3)
    assert sample_image.mode == 'RGB'
    assert isinstance(sample_image.array, np.ndarray)
    assert isinstance(sample_image.pil_image, PILImage.Image)


def test_image_copy(sample_image):
    """Test copying an image."""
    copy = sample_image.copy()
    
    assert copy is not sample_image
    assert copy._image is not sample_image._image
    assert copy.width == sample_image.width
    assert copy.height == sample_image.height
    assert np.array_equal(copy.array, sample_image.array)


def test_resize_with_tuple(sample_image):
    """Test resizing an image with a tuple size."""
    resized = sample_image.resize((50, 75))
    
    assert resized.width == 50
    assert resized.height == 75
    assert resized.shape == (75, 50, 3)


def test_resize_with_scale_factor(sample_image):
    """Test resizing an image with a scale factor."""
    resized = sample_image.resize(0.5)
    
    assert resized.width == 50
    assert resized.height == 50


def test_resize_maintain_aspect_ratio(sample_image):
    """Test resizing an image while maintaining aspect ratio."""
    # Original is 100x100, aspect ratio 1:1
    resized = sample_image.resize((200, 100), maintain_aspect_ratio=True)
    
    # Should be 100x100 to maintain 1:1 aspect ratio
    assert resized.width == 100
    assert resized.height == 100


def test_crop(sample_image):
    """Test cropping an image."""
    cropped = sample_image.crop((25, 25, 75, 75))
    
    assert cropped.width == 50
    assert cropped.height == 50
    
    # Check that the cropped image contains the expected portion of the original
    original_portion = sample_image.array[25:75, 25:75]
    assert np.array_equal(cropped.array, original_portion)


def test_adjust_brightness(sample_image):
    """Test adjusting image brightness."""
    brightened = sample_image.adjust_brightness(1.5)
    darkened = sample_image.adjust_brightness(0.5)
    
    # Check dimensions remain the same
    assert brightened.width == sample_image.width
    assert brightened.height == sample_image.height
    
    # Verify overall brightness increased/decreased
    assert np.mean(brightened.array) > np.mean(sample_image.array)
    assert np.mean(darkened.array) < np.mean(sample_image.array)


def test_adjust_contrast(sample_image):
    """Test adjusting image contrast."""
    increased = sample_image.adjust_contrast(1.5)
    decreased = sample_image.adjust_contrast(0.5)
    
    # Check dimensions remain the same
    assert increased.width == sample_image.width
    assert increased.height == sample_image.height
    
    # Verify standard deviation (measure of contrast) increased/decreased
    assert np.std(increased.array) > np.std(sample_image.array)
    assert np.std(decreased.array) < np.std(sample_image.array)


def test_apply_filter(sample_image):
    """Test applying filters to an image."""
    filters = ["blur", "sharpen", "contour", "detail", "edge_enhance", "find_edges", "smooth"]
    
    for filter_type in filters:
        filtered = sample_image.apply_filter(filter_type)
        
        # Check dimensions remain the same
        assert filtered.width == sample_image.width
        assert filtered.height == sample_image.height
        
        # Filtered image should be different from original
        assert not np.array_equal(filtered.array, sample_image.array)


def test_apply_invalid_filter(sample_image):
    """Test applying an invalid filter raises ValueError."""
    with pytest.raises(ValueError):
        sample_image.apply_filter("nonexistent_filter")


def test_rotate(sample_image):
    """Test rotating an image."""
    # Without expansion
    rotated = sample_image.rotate(45, expand=False)
    assert rotated.width == sample_image.width
    assert rotated.height == sample_image.height
    
    # With expansion
    rotated_expanded = sample_image.rotate(45, expand=True)
    assert rotated_expanded.width > sample_image.width
    assert rotated_expanded.height > sample_image.height


def test_flip_horizontal(sample_image):
    """Test flipping an image horizontally."""
    flipped = sample_image.flip_horizontal()
    
    assert flipped.width == sample_image.width
    assert flipped.height == sample_image.height
    
    # Check that the image is flipped horizontally
    assert np.array_equal(flipped.array[:, 0], sample_image.array[:, -1])
    assert np.array_equal(flipped.array[:, -1], sample_image.array[:, 0])


def test_flip_vertical(sample_image):
    """Test flipping an image vertically."""
    flipped = sample_image.flip_vertical()
    
    assert flipped.width == sample_image.width
    assert flipped.height == sample_image.height
    
    # Check that the image is flipped vertically
    assert np.array_equal(flipped.array[0], sample_image.array[-1])
    assert np.array_equal(flipped.array[-1], sample_image.array[0])


def test_convert(sample_image):
    """Test converting an image to different modes."""
    # Convert to grayscale
    grayscale = sample_image.convert('L')
    assert grayscale.mode == 'L'
    assert grayscale.width == sample_image.width
    assert grayscale.height == sample_image.height
    
    # Convert to RGBA
    rgba = sample_image.convert('RGBA')
    assert rgba.mode == 'RGBA'
    assert rgba.shape == (100, 100, 4)


def test_compose(sample_image):
    """Test composing images."""
    # Create a smaller overlay image
    overlay_array = np.zeros((50, 50, 4), dtype=np.uint8)
    overlay_array[:, :, 0] = 255  # Red
    overlay_array[:, :, 3] = 128  # Alpha (semi-transparent)
    overlay = Image(overlay_array)
    
    # Compose images
    composed = sample_image.compose(overlay, position=(25, 25))
    
    assert composed.width == sample_image.width
    assert composed.height == sample_image.height
    
    # Check that the overlay was applied
    # The region should be different from the original due to the overlay
    assert not np.array_equal(
        composed.array[25:75, 25:75],
        sample_image.array[25:75, 25:75]
    )


def test_save_and_load(sample_image, temp_image_path):
    """Test saving and loading an image."""
    # Save to a file
    sample_image.save(temp_image_path)
    assert os.path.exists(temp_image_path)
    
    # Load the saved file
    loaded = Image(temp_image_path)
    
    assert loaded.width == sample_image.width
    assert loaded.height == sample_image.height
    
    # The saved and loaded image might not be byte-for-byte identical due to 
    # compression, but should be very similar
    correlation = np.corrcoef(
        loaded.array.flatten(), 
        sample_image.array.flatten()
    )[0, 1]
    assert correlation > 0.99


def test_blend_images(sample_image):
    """Test blending two images."""
    # Create a second image with different colors
    width, height = 100, 100
    array2 = np.zeros((height, width, 3), dtype=np.uint8)
    array2[:, :, 0] = 255  # Red image
    image2 = Image(array2)
    
    # Blend with alpha=0.5
    blended = Image.blend(sample_image, image2, 0.5)
    
    assert blended.width == sample_image.width
    assert blended.height == sample_image.height
    
    # Check that the blend is between the two images
    # The red channel should be increased due to the second image
    assert np.mean(blended.array[:, :, 0]) > np.mean(sample_image.array[:, :, 0])


def test_blend_different_sized_images():
    """Test blending images of different sizes."""
    # Create two images of different sizes
    image1 = Image(np.zeros((100, 100, 3), dtype=np.uint8))
    image2 = Image(np.zeros((50, 50, 3), dtype=np.uint8))
    
    # This should raise a ValueError
    with pytest.raises(ValueError):
        Image.blend(image1, image2, 0.5)


if __name__ == "__main__":
    pytest.main() 