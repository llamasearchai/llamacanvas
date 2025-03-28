"""
Tests for the CLI functionality in LlamaCanvas.

This module contains tests for the command-line interface and its commands.
"""

import os
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from llama_canvas.cli import cli
from llama_canvas.core.image import Image
from llama_canvas.core.canvas import Canvas
from llama_canvas.core.video import Video


@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def temp_image_path(temp_dir):
    """Create a temporary file path for images."""
    return os.path.join(temp_dir, "test_image.png")


@pytest.fixture
def temp_output_path(temp_dir):
    """Create a temporary file path for output files."""
    return os.path.join(temp_dir, "output.png")


def test_cli_version(cli_runner):
    """Test the version command."""
    with patch('llama_canvas.cli.__version__', '0.1.0'):
        result = cli_runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert '0.1.0' in result.output


def test_cli_help(cli_runner):
    """Test the help command."""
    result = cli_runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'Usage:' in result.output
    assert 'Options:' in result.output
    assert 'Commands:' in result.output


def test_generate_command(cli_runner, temp_output_path):
    """Test the generate command."""
    # Mock the Canvas.generate_from_text method
    mock_image = MagicMock(spec=Image)
    
    with patch('llama_canvas.cli.Canvas') as mock_canvas_cls:
        mock_canvas = MagicMock()
        mock_canvas.generate_from_text.return_value = mock_image
        mock_canvas_cls.return_value = mock_canvas
        
        # Run the command
        result = cli_runner.invoke(cli, [
            'generate', 
            'A mountain landscape', 
            '--output', temp_output_path,
            '--width', '256',
            '--height', '256',
            '--model', 'stable-diffusion'
        ])
        
        # Check results
        assert result.exit_code == 0
        
        # Verify canvas was created with correct size
        mock_canvas_cls.assert_called_once_with(width=256, height=256, background_color=(255, 255, 255))
        
        # Verify generation was called with correct parameters
        mock_canvas.generate_from_text.assert_called_once_with('A mountain landscape', model='stable-diffusion')
        
        # Verify save was called
        mock_image.save.assert_called_once_with(temp_output_path)


def test_style_command(cli_runner, temp_image_path, temp_output_path):
    """Test the style command."""
    # Create a mock image file
    with open(temp_image_path, 'w') as f:
        f.write('mock image data')
    
    # Mock the Canvas.apply_style method
    mock_styled_image = MagicMock(spec=Image)
    
    with patch('llama_canvas.cli.Image') as mock_image_cls, \
         patch('llama_canvas.cli.Canvas') as mock_canvas_cls:
        
        mock_image = MagicMock(spec=Image)
        mock_image_cls.return_value = mock_image
        
        mock_canvas = MagicMock()
        mock_canvas.apply_style.return_value = mock_styled_image
        mock_canvas_cls.return_value = mock_canvas
        
        # Run the command
        result = cli_runner.invoke(cli, [
            'style',
            temp_image_path,
            'van-gogh',
            '--output', temp_output_path,
            '--strength', '0.7',
            '--use-claude'
        ])
        
        # Check results
        assert result.exit_code == 0
        
        # Verify image was loaded
        mock_image_cls.assert_called_once_with(temp_image_path)
        
        # Verify style was applied with correct parameters
        mock_canvas.apply_style.assert_called_once_with(
            mock_image, 'van-gogh', strength=0.7, use_claude=True
        )
        
        # Verify result was saved
        mock_styled_image.save.assert_called_once_with(temp_output_path)


def test_enhance_command(cli_runner, temp_image_path, temp_output_path):
    """Test the enhance command."""
    # Create a mock image file
    with open(temp_image_path, 'w') as f:
        f.write('mock image data')
    
    # Mock the Canvas.enhance_resolution method
    mock_enhanced_image = MagicMock(spec=Image)
    
    with patch('llama_canvas.cli.Image') as mock_image_cls, \
         patch('llama_canvas.cli.Canvas') as mock_canvas_cls:
        
        mock_image = MagicMock(spec=Image)
        mock_image_cls.return_value = mock_image
        
        mock_canvas = MagicMock()
        mock_canvas.enhance_resolution.return_value = mock_enhanced_image
        mock_canvas_cls.return_value = mock_canvas
        
        # Run the command
        result = cli_runner.invoke(cli, [
            'enhance',
            temp_image_path,
            '--output', temp_output_path,
            '--scale', '2'
        ])
        
        # Check results
        assert result.exit_code == 0
        
        # Verify image was loaded
        mock_image_cls.assert_called_once_with(temp_image_path)
        
        # Verify enhance was called with correct parameters
        mock_canvas.enhance_resolution.assert_called_once_with(mock_image, scale=2)
        
        # Verify result was saved
        mock_enhanced_image.save.assert_called_once_with(temp_output_path)


def test_blend_command(cli_runner, temp_dir, temp_output_path):
    """Test the blend command."""
    # Create two mock image files
    image1_path = os.path.join(temp_dir, "image1.png")
    image2_path = os.path.join(temp_dir, "image2.png")
    
    with open(image1_path, 'w') as f:
        f.write('mock image1 data')
    
    with open(image2_path, 'w') as f:
        f.write('mock image2 data')
    
    # Mock the Canvas.blend_images method
    mock_blended_image = MagicMock(spec=Image)
    
    with patch('llama_canvas.cli.Image') as mock_image_cls, \
         patch('llama_canvas.cli.Canvas') as mock_canvas_cls:
        
        mock_image1 = MagicMock(spec=Image)
        mock_image2 = MagicMock(spec=Image)
        mock_image_cls.side_effect = [mock_image1, mock_image2]
        
        mock_canvas = MagicMock()
        mock_canvas.blend_images.return_value = mock_blended_image
        mock_canvas_cls.return_value = mock_canvas
        
        # Run the command
        result = cli_runner.invoke(cli, [
            'blend',
            image1_path,
            image2_path,
            '--output', temp_output_path,
            '--alpha', '0.3',
            '--mode', 'overlay'
        ])
        
        # Check results
        assert result.exit_code == 0
        
        # Verify images were loaded
        assert mock_image_cls.call_count == 2
        
        # Verify blend was called with correct parameters
        mock_canvas.blend_images.assert_called_once_with(
            mock_image1, mock_image2, alpha=0.3, mode='overlay'
        )
        
        # Verify result was saved
        mock_blended_image.save.assert_called_once_with(temp_output_path)


def test_animation_command(cli_runner, temp_dir, temp_output_path):
    """Test the animation command."""
    # Create a directory with mock frame files
    frames_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frames_dir)
    
    # Create 5 mock frame files
    frame_paths = []
    for i in range(5):
        frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
        frame_paths.append(frame_path)
        with open(frame_path, 'w') as f:
            f.write(f'mock frame {i} data')
    
    # Mock the Video.create_gif method
    mock_video = MagicMock(spec=Video)
    
    with patch('llama_canvas.cli.Image') as mock_image_cls, \
         patch('llama_canvas.cli.Canvas') as mock_canvas_cls, \
         patch('llama_canvas.cli.Video') as mock_video_cls, \
         patch('llama_canvas.cli.glob') as mock_glob:
        
        # Mock frame loading
        mock_frames = [MagicMock(spec=Image) for _ in range(5)]
        mock_image_cls.side_effect = mock_frames
        
        # Mock glob to return frame paths
        mock_glob.glob.return_value = frame_paths
        
        # Mock video creation
        mock_video_cls.return_value = mock_video
        
        # Run the command
        result = cli_runner.invoke(cli, [
            'animation',
            frames_dir,
            '--output', temp_output_path,
            '--fps', '24',
            '--pattern', '*.png'
        ])
        
        # Check results
        assert result.exit_code == 0
        
        # Verify frames were loaded
        assert mock_image_cls.call_count == 5
        
        # Verify video was created
        mock_video_cls.assert_called_once()
        
        # Verify gif was created
        assert mock_video.create_gif.called


def test_filter_command(cli_runner, temp_image_path, temp_output_path):
    """Test the filter command."""
    # Create a mock image file
    with open(temp_image_path, 'w') as f:
        f.write('mock image data')
    
    # Mock the Image.apply_filter method
    mock_filtered_image = MagicMock(spec=Image)
    
    with patch('llama_canvas.cli.Image') as mock_image_cls:
        mock_image = MagicMock(spec=Image)
        mock_image.apply_filter.return_value = mock_filtered_image
        mock_image_cls.return_value = mock_image
        
        # Run the command
        result = cli_runner.invoke(cli, [
            'filter',
            temp_image_path,
            'blur',
            '--output', temp_output_path
        ])
        
        # Check results
        assert result.exit_code == 0
        
        # Verify image was loaded
        mock_image_cls.assert_called_once_with(temp_image_path)
        
        # Verify filter was applied
        mock_image.apply_filter.assert_called_once_with('blur')
        
        # Verify result was saved
        mock_filtered_image.save.assert_called_once_with(temp_output_path)


def test_ui_command(cli_runner):
    """Test the UI command."""
    with patch('llama_canvas.cli.subprocess') as mock_subprocess, \
         patch('llama_canvas.cli.webbrowser') as mock_webbrowser:
        
        # Run the command with browser
        result = cli_runner.invoke(cli, ['ui', '--browse'])
        
        # Check results
        assert result.exit_code == 0
        
        # Verify subprocess was called to start server
        assert mock_subprocess.Popen.called
        
        # Verify browser was opened
        assert mock_webbrowser.open.called
        
        # Run without browser
        mock_subprocess.reset_mock()
        mock_webbrowser.reset_mock()
        
        result = cli_runner.invoke(cli, ['ui'])
        
        # Verify subprocess was called but browser wasn't
        assert mock_subprocess.Popen.called
        assert not mock_webbrowser.open.called


def test_rotate_command(cli_runner, temp_image_path, temp_output_path):
    """Test the rotate command."""
    # Create a mock image file
    with open(temp_image_path, 'w') as f:
        f.write('mock image data')
    
    # Mock the Image.rotate method
    mock_rotated_image = MagicMock(spec=Image)
    
    with patch('llama_canvas.cli.Image') as mock_image_cls:
        mock_image = MagicMock(spec=Image)
        mock_image.rotate.return_value = mock_rotated_image
        mock_image_cls.return_value = mock_image
        
        # Run the command
        result = cli_runner.invoke(cli, [
            'rotate',
            temp_image_path,
            '90',
            '--output', temp_output_path,
            '--expand'
        ])
        
        # Check results
        assert result.exit_code == 0
        
        # Verify image was loaded
        mock_image_cls.assert_called_once_with(temp_image_path)
        
        # Verify rotation was applied
        mock_image.rotate.assert_called_once_with(90, expand=True)
        
        # Verify result was saved
        mock_rotated_image.save.assert_called_once_with(temp_output_path)


def test_resize_command(cli_runner, temp_image_path, temp_output_path):
    """Test the resize command."""
    # Create a mock image file
    with open(temp_image_path, 'w') as f:
        f.write('mock image data')
    
    # Mock the Image.resize method
    mock_resized_image = MagicMock(spec=Image)
    
    with patch('llama_canvas.cli.Image') as mock_image_cls:
        mock_image = MagicMock(spec=Image)
        mock_image.resize.return_value = mock_resized_image
        mock_image_cls.return_value = mock_image
        
        # Run the command with specific dimensions
        result = cli_runner.invoke(cli, [
            'resize',
            temp_image_path,
            '--output', temp_output_path,
            '--width', '200',
            '--height', '150',
            '--maintain-aspect-ratio'
        ])
        
        # Check results
        assert result.exit_code == 0
        
        # Verify image was loaded
        mock_image_cls.assert_called_once_with(temp_image_path)
        
        # Verify resize was called with right parameters
        mock_image.resize.assert_called_once_with(
            (200, 150), maintain_aspect_ratio=True
        )
        
        # Verify result was saved
        mock_resized_image.save.assert_called_once_with(temp_output_path)
        
        # Test resize with scale factor
        mock_image_cls.reset_mock()
        mock_image.reset_mock()
        
        result = cli_runner.invoke(cli, [
            'resize',
            temp_image_path,
            '--output', temp_output_path,
            '--scale', '0.5'
        ])
        
        assert result.exit_code == 0
        mock_image.resize.assert_called_once_with(0.5)


if __name__ == "__main__":
    pytest.main() 