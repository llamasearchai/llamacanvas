"""
Tests for the API functionality in LlamaCanvas.

This module contains tests for the FastAPI web API endpoints.
"""

import os
import base64
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from llama_canvas.api.app import app
from llama_canvas.core.image import Image
from llama_canvas.core.canvas import Canvas
from llama_canvas.core.video import Video


@pytest.fixture
def api_client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    import numpy as np
    from PIL import Image as PILImage
    
    # Create a simple test image
    width, height = 100, 100
    array = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            r = int(255 * y / height)
            g = int(255 * x / width)
            b = int(128)
            array[y, x] = [r, g, b]
    
    pil_image = PILImage.fromarray(array)
    
    # Save to a temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        pil_image.save(tmp.name)
        yield tmp.name
    
    # Cleanup
    if os.path.exists(tmp.name):
        os.remove(tmp.name)


@pytest.fixture
def sample_base64_image():
    """Create a sample base64 encoded image for testing."""
    import numpy as np
    from PIL import Image as PILImage
    import io
    
    # Create a simple test image
    width, height = 100, 100
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, :, 0] = 255  # Red image
    
    pil_image = PILImage.fromarray(array)
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def test_health_endpoint(api_client):
    """Test the health check endpoint."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_generate_image_endpoint(api_client):
    """Test the generate image endpoint."""
    # Mock the Canvas.generate_from_text method
    with patch('llama_canvas.api.routes.images.Canvas') as mock_canvas_cls:
        # Setup mock
        mock_canvas = MagicMock()
        mock_image = MagicMock()
        mock_canvas.generate_from_text.return_value = mock_image
        mock_canvas_cls.return_value = mock_canvas
        
        # Mock image saving to b64
        with patch('llama_canvas.api.routes.images.Image.to_base64') as mock_to_base64:
            mock_to_base64.return_value = "mock_base64_image_data"
            
            # Make API request
            response = api_client.post(
                "/images/generate",
                json={
                    "prompt": "A mountain landscape",
                    "width": 512,
                    "height": 512,
                    "model": "stable-diffusion"
                }
            )
            
            # Check response
            assert response.status_code == 200
            data = response.json()
            assert data["image"] == "mock_base64_image_data"
            
            # Verify canvas was created with correct parameters
            mock_canvas_cls.assert_called_once_with(width=512, height=512, background_color=(255, 255, 255))
            
            # Verify generate was called with correct parameters
            mock_canvas.generate_from_text.assert_called_once_with(
                "A mountain landscape",
                model="stable-diffusion"
            )


def test_style_image_endpoint(api_client, sample_base64_image):
    """Test the style image endpoint."""
    # Mock the Canvas.apply_style method
    with patch('llama_canvas.api.routes.images.Canvas') as mock_canvas_cls, \
         patch('llama_canvas.api.routes.images.Image') as mock_image_cls:
        
        # Setup mocks
        mock_canvas = MagicMock()
        mock_image = MagicMock()
        mock_styled_image = MagicMock()
        mock_image_cls.from_base64.return_value = mock_image
        mock_canvas.apply_style.return_value = mock_styled_image
        mock_canvas_cls.return_value = mock_canvas
        
        # Mock image saving to b64
        mock_styled_image.to_base64.return_value = "mock_styled_base64_image"
        
        # Make API request
        response = api_client.post(
            "/images/style",
            json={
                "image": sample_base64_image,
                "style": "van-gogh",
                "strength": 0.7,
                "use_claude": True
            }
        )
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["image"] == "mock_styled_base64_image"
        
        # Verify image was loaded from base64
        mock_image_cls.from_base64.assert_called_once_with(sample_base64_image)
        
        # Verify style was applied with correct parameters
        mock_canvas.apply_style.assert_called_once_with(
            mock_image, "van-gogh", strength=0.7, use_claude=True
        )


def test_enhance_image_endpoint(api_client, sample_base64_image):
    """Test the enhance image endpoint."""
    # Mock the Canvas.enhance_resolution method
    with patch('llama_canvas.api.routes.images.Canvas') as mock_canvas_cls, \
         patch('llama_canvas.api.routes.images.Image') as mock_image_cls:
        
        # Setup mocks
        mock_canvas = MagicMock()
        mock_image = MagicMock()
        mock_enhanced_image = MagicMock()
        mock_image_cls.from_base64.return_value = mock_image
        mock_canvas.enhance_resolution.return_value = mock_enhanced_image
        mock_canvas_cls.return_value = mock_canvas
        
        # Mock image saving to b64
        mock_enhanced_image.to_base64.return_value = "mock_enhanced_base64_image"
        
        # Make API request
        response = api_client.post(
            "/images/enhance",
            json={
                "image": sample_base64_image,
                "scale": 2
            }
        )
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["image"] == "mock_enhanced_base64_image"
        
        # Verify image was loaded from base64
        mock_image_cls.from_base64.assert_called_once_with(sample_base64_image)
        
        # Verify enhance was called with correct parameters
        mock_canvas.enhance_resolution.assert_called_once_with(mock_image, scale=2)


def test_blend_images_endpoint(api_client, sample_base64_image):
    """Test the blend images endpoint."""
    # Mock the Canvas.blend_images method
    with patch('llama_canvas.api.routes.images.Canvas') as mock_canvas_cls, \
         patch('llama_canvas.api.routes.images.Image') as mock_image_cls:
        
        # Setup mocks
        mock_canvas = MagicMock()
        mock_image1 = MagicMock()
        mock_image2 = MagicMock()
        mock_blended_image = MagicMock()
        mock_image_cls.from_base64.side_effect = [mock_image1, mock_image2]
        mock_canvas.blend_images.return_value = mock_blended_image
        mock_canvas_cls.return_value = mock_canvas
        
        # Mock image saving to b64
        mock_blended_image.to_base64.return_value = "mock_blended_base64_image"
        
        # Make API request
        response = api_client.post(
            "/images/blend",
            json={
                "image1": sample_base64_image,
                "image2": sample_base64_image,
                "alpha": 0.3,
                "mode": "overlay"
            }
        )
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["image"] == "mock_blended_base64_image"
        
        # Verify images were loaded from base64
        assert mock_image_cls.from_base64.call_count == 2
        
        # Verify blend was called with correct parameters
        mock_canvas.blend_images.assert_called_once_with(
            mock_image1, mock_image2, alpha=0.3, mode="overlay"
        )


def test_apply_filter_endpoint(api_client, sample_base64_image):
    """Test the apply filter endpoint."""
    # Mock the Image.apply_filter method
    with patch('llama_canvas.api.routes.images.Image') as mock_image_cls:
        
        # Setup mocks
        mock_image = MagicMock()
        mock_filtered_image = MagicMock()
        mock_image.apply_filter.return_value = mock_filtered_image
        mock_image_cls.from_base64.return_value = mock_image
        
        # Mock image saving to b64
        mock_filtered_image.to_base64.return_value = "mock_filtered_base64_image"
        
        # Make API request
        response = api_client.post(
            "/images/filter",
            json={
                "image": sample_base64_image,
                "filter": "blur"
            }
        )
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["image"] == "mock_filtered_base64_image"
        
        # Verify image was loaded from base64
        mock_image_cls.from_base64.assert_called_once_with(sample_base64_image)
        
        # Verify filter was applied with correct parameters
        mock_image.apply_filter.assert_called_once_with("blur")


def test_resize_image_endpoint(api_client, sample_base64_image):
    """Test the resize image endpoint."""
    # Mock the Image.resize method
    with patch('llama_canvas.api.routes.images.Image') as mock_image_cls:
        
        # Setup mocks
        mock_image = MagicMock()
        mock_resized_image = MagicMock()
        mock_image.resize.return_value = mock_resized_image
        mock_image_cls.from_base64.return_value = mock_image
        
        # Mock image saving to b64
        mock_resized_image.to_base64.return_value = "mock_resized_base64_image"
        
        # Make API request - with size parameters
        response = api_client.post(
            "/images/resize",
            json={
                "image": sample_base64_image,
                "width": 200,
                "height": 150,
                "maintain_aspect_ratio": True
            }
        )
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["image"] == "mock_resized_base64_image"
        
        # Verify image was loaded from base64
        mock_image_cls.from_base64.assert_called_once_with(sample_base64_image)
        
        # Verify resize was called with correct parameters
        mock_image.resize.assert_called_once_with(
            (200, 150), maintain_aspect_ratio=True
        )
        
        # Reset mocks
        mock_image_cls.reset_mock()
        mock_image.reset_mock()
        mock_image.resize.return_value = mock_resized_image
        mock_image_cls.from_base64.return_value = mock_image
        
        # Make API request - with scale parameter
        response = api_client.post(
            "/images/resize",
            json={
                "image": sample_base64_image,
                "scale": 0.5
            }
        )
        
        # Check response
        assert response.status_code == 200
        
        # Verify resize was called with correct parameters
        mock_image.resize.assert_called_once_with(0.5)


def test_rotate_image_endpoint(api_client, sample_base64_image):
    """Test the rotate image endpoint."""
    # Mock the Image.rotate method
    with patch('llama_canvas.api.routes.images.Image') as mock_image_cls:
        
        # Setup mocks
        mock_image = MagicMock()
        mock_rotated_image = MagicMock()
        mock_image.rotate.return_value = mock_rotated_image
        mock_image_cls.from_base64.return_value = mock_image
        
        # Mock image saving to b64
        mock_rotated_image.to_base64.return_value = "mock_rotated_base64_image"
        
        # Make API request
        response = api_client.post(
            "/images/rotate",
            json={
                "image": sample_base64_image,
                "angle": 90,
                "expand": True
            }
        )
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["image"] == "mock_rotated_base64_image"
        
        # Verify image was loaded from base64
        mock_image_cls.from_base64.assert_called_once_with(sample_base64_image)
        
        # Verify rotate was called with correct parameters
        mock_image.rotate.assert_called_once_with(90, expand=True)


def test_create_animation_endpoint(api_client, sample_base64_image):
    """Test the create animation endpoint."""
    # Mock the Video class
    with patch('llama_canvas.api.routes.videos.Video') as mock_video_cls, \
         patch('llama_canvas.api.routes.videos.Image') as mock_image_cls:
        
        # Setup mocks
        mock_video = MagicMock()
        mock_image = MagicMock()
        mock_video_cls.return_value = mock_video
        mock_image_cls.from_base64.return_value = mock_image
        
        # Prepare frames
        frames_base64 = [sample_base64_image] * 5
        
        # Mock create gif method
        with patch('os.path') as mock_path, \
             patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            
            mock_temp = MagicMock()
            mock_temp.__enter__.return_value.name = "/tmp/temp_video.gif"
            mock_temp_file.return_value = mock_temp
            mock_path.exists.return_value = True
            
            # Mock reading the file as bytes
            with patch('builtins.open', MagicMock()) as mock_open:
                mock_file = MagicMock()
                mock_file.__enter__.return_value.read.return_value = b"mock_gif_data"
                mock_open.return_value = mock_file
                
                # Make API request
                response = api_client.post(
                    "/videos/create",
                    json={
                        "frames": frames_base64,
                        "fps": 24,
                        "loop": 0
                    }
                )
                
                # Check response
                assert response.status_code == 200
                assert response.headers["content-type"] == "image/gif"
                
                # Verify frames were loaded from base64
                assert mock_image_cls.from_base64.call_count == 5
                
                # Verify video was created with correct parameters
                mock_video_cls.assert_called_once()
                mock_video.create_gif.assert_called_once()


if __name__ == "__main__":
    pytest.main() 