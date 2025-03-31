"""
Tests for the Video class in LlamaCanvas.

This module contains tests for video creation, editing, and export functionality.
"""

import os
import tempfile
from pathlib import Path
import pytest
import numpy as np
from PIL import Image as PILImage

from llama_canvas.core.image import Image
from llama_canvas.core.video import Video


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
    fd, path = tempfile.mkstemp(suffix='.mp4')
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


def test_video_init(sample_frames):
    """Test video initialization with different parameters."""
    # Default initialization
    video = Video(sample_frames)
    assert video.fps == 30
    assert len(video.frames) == len(sample_frames)
    assert video.width == sample_frames[0].width
    assert video.height == sample_frames[0].height
    assert video.duration == len(sample_frames) / 30  # Duration in seconds
    
    # Custom FPS
    video = Video(sample_frames, fps=60)
    assert video.fps == 60
    assert video.duration == len(sample_frames) / 60
    
    # Test with empty frames list
    with pytest.raises(ValueError):
        Video([])


def test_video_frame_access(sample_video):
    """Test frame access methods."""
    # Get by index
    frame = sample_video[0]
    assert isinstance(frame, Image)
    assert frame.width == 100
    assert frame.height == 100
    
    # Get by time
    middle_frame = sample_video.get_frame_at_time(sample_video.duration / 2)
    assert isinstance(middle_frame, Image)
    
    # Out of bounds index
    with pytest.raises(IndexError):
        sample_video[len(sample_video.frames)]
    
    # Out of bounds time
    with pytest.raises(ValueError):
        sample_video.get_frame_at_time(sample_video.duration + 1)


def test_add_frame(sample_video):
    """Test adding frames to a video."""
    initial_count = len(sample_video.frames)
    
    # Create a new frame
    new_frame = Image(np.zeros((100, 100, 3), dtype=np.uint8))
    
    # Add frame at the end
    sample_video.add_frame(new_frame)
    assert len(sample_video.frames) == initial_count + 1
    assert sample_video.frames[-1] is new_frame
    
    # Add frame at specific position
    sample_video.add_frame(new_frame.copy(), position=1)
    assert len(sample_video.frames) == initial_count + 2
    assert sample_video.frames[1] is not new_frame  # It's a copy


def test_remove_frame(sample_video):
    """Test removing frames from a video."""
    initial_count = len(sample_video.frames)
    
    # Remove by index
    removed = sample_video.remove_frame(0)
    assert len(sample_video.frames) == initial_count - 1
    assert isinstance(removed, Image)
    
    # Remove by time
    time = sample_video.duration / 2
    removed = sample_video.remove_frame_at_time(time)
    assert len(sample_video.frames) == initial_count - 2
    assert isinstance(removed, Image)
    
    # Test invalid index
    with pytest.raises(IndexError):
        sample_video.remove_frame(100)
    
    # Test invalid time
    with pytest.raises(ValueError):
        sample_video.remove_frame_at_time(sample_video.duration + 1)


def test_replace_frame(sample_video):
    """Test replacing frames in a video."""
    new_frame = Image(np.ones((100, 100, 3), dtype=np.uint8) * 255)
    
    # Replace by index
    replaced = sample_video.replace_frame(0, new_frame)
    assert sample_video.frames[0] is new_frame
    assert isinstance(replaced, Image)
    
    # Replace by time
    time = sample_video.duration / 2
    replaced = sample_video.replace_frame_at_time(time, new_frame.copy())
    assert isinstance(replaced, Image)
    
    # Test invalid index
    with pytest.raises(IndexError):
        sample_video.replace_frame(100, new_frame)
    
    # Test invalid time
    with pytest.raises(ValueError):
        sample_video.replace_frame_at_time(sample_video.duration + 1, new_frame)


def test_set_fps(sample_video):
    """Test changing the FPS of a video."""
    initial_duration = sample_video.duration
    initial_fps = sample_video.fps
    
    # Double the FPS
    sample_video.set_fps(initial_fps * 2)
    
    # Duration should be halved
    assert sample_video.fps == initial_fps * 2
    assert sample_video.duration == initial_duration / 2
    
    # Invalid FPS
    with pytest.raises(ValueError):
        sample_video.set_fps(0)
    with pytest.raises(ValueError):
        sample_video.set_fps(-10)


def test_trim(sample_video):
    """Test trimming a video."""
    initial_count = len(sample_video.frames)
    
    # Trim the first and last frames
    trimmed = sample_video.trim(start_time=1/sample_video.fps, end_time=sample_video.duration - 1/sample_video.fps)
    
    assert len(trimmed.frames) == initial_count - 2
    assert trimmed.fps == sample_video.fps
    
    # Test invalid times
    with pytest.raises(ValueError):
        sample_video.trim(start_time=-1)
    with pytest.raises(ValueError):
        sample_video.trim(start_time=0, end_time=sample_video.duration + 1)
    with pytest.raises(ValueError):
        sample_video.trim(start_time=5, end_time=2)  # Start after end


def test_resize(sample_video):
    """Test resizing a video."""
    # Resize to half the dimensions
    new_width = sample_video.width // 2
    new_height = sample_video.height // 2
    
    resized = sample_video.resize((new_width, new_height))
    
    assert resized.width == new_width
    assert resized.height == new_height
    assert len(resized.frames) == len(sample_video.frames)
    
    # Check that all frames were resized
    for frame in resized.frames:
        assert frame.width == new_width
        assert frame.height == new_height


def test_create_gif(sample_video, temp_video_path, mocker):
    """Test creating a GIF from the video."""
    # Update file extension
    gif_path = temp_video_path.replace('.mp4', '.gif')
    
    # Mock the save_gif method
    mocker.patch.object(Image, 'pil_image')
    
    # Mock the PIL's save method
    mock_save = mocker.MagicMock()
    mocker.patch('PIL.Image.save', mock_save)
    
    # Create GIF
    sample_video.create_gif(gif_path, loop=0, optimize=True)
    
    # File path should be updated with .gif extension if needed
    assert gif_path.endswith('.gif')


def test_apply_filter_to_all_frames(sample_video):
    """Test applying a filter to all frames."""
    # Store original frame data
    original_frames = [frame.array.copy() for frame in sample_video.frames]
    
    # Apply filter
    filtered = sample_video.apply_filter_to_all_frames("blur")
    
    assert len(filtered.frames) == len(sample_video.frames)
    assert filtered is not sample_video
    
    # Verify each frame was modified
    for i, frame in enumerate(filtered.frames):
        assert not np.array_equal(frame.array, original_frames[i])


def test_create_transition(sample_frames):
    """Test creating a transition between two videos."""
    # Create two short videos
    video1 = Video(sample_frames[:3], fps=30)
    video2 = Video(sample_frames[5:8], fps=30)
    
    # Create a crossfade transition
    transition = Video.create_transition(video1, video2, transition_frames=5, transition_type="crossfade")
    
    assert len(transition.frames) == len(video1.frames) + len(video2.frames) + 5 - 2
    assert transition.fps == video1.fps  # Should inherit from first video


def test_concatenate_videos(sample_frames):
    """Test concatenating multiple videos."""
    # Create two videos with different FPS
    video1 = Video(sample_frames[:5], fps=30)
    video2 = Video(sample_frames[5:], fps=60)
    
    # Concatenate with FPS adjustment
    result = Video.concatenate([video1, video2], target_fps=30)
    
    assert len(result.frames) == len(video1.frames) + len(video2.frames)
    assert result.fps == 30
    
    # Concatenate without adjustment
    result = Video.concatenate([video1, video2], adjust_fps=False)
    
    assert len(result.frames) == len(video1.frames) + len(video2.frames)
    assert result.fps == video1.fps  # Should inherit from first video


def test_apply_filter_to_frame_range(sample_video):
    """Test applying a filter to a range of frames."""
    # Store original frame data for the range
    start_idx = 2
    end_idx = 5
    original_frames = [frame.array.copy() for frame in sample_video.frames[start_idx:end_idx]]
    
    # Apply filter to range
    filtered = sample_video.apply_filter_to_frame_range("sharpen", start_frame=start_idx, end_frame=end_idx)
    
    # Verify frames in range were modified
    for i in range(start_idx, end_idx):
        assert not np.array_equal(filtered.frames[i].array, original_frames[i-start_idx])
    
    # Verify frames outside range were not modified
    for i in range(0, start_idx):
        assert np.array_equal(filtered.frames[i].array, sample_video.frames[i].array)
    
    for i in range(end_idx, len(sample_video.frames)):
        assert np.array_equal(filtered.frames[i].array, sample_video.frames[i].array)


def test_reverse(sample_video):
    """Test reversing a video."""
    original_frames = sample_video.frames.copy()
    
    # Reverse the video
    reversed_video = sample_video.reverse()
    
    assert len(reversed_video.frames) == len(original_frames)
    assert reversed_video.fps == sample_video.fps
    
    # Verify frames are in reverse order
    for i in range(len(original_frames)):
        assert np.array_equal(
            reversed_video.frames[i].array,
            original_frames[len(original_frames) - 1 - i].array
        )


def test_create_loop(sample_video):
    """Test creating a looping video."""
    initial_count = len(sample_video.frames)
    
    # Create a loop with 3 repetitions
    looped = sample_video.create_loop(repetitions=3)
    
    assert len(looped.frames) == initial_count * 3
    assert looped.fps == sample_video.fps
    
    # Test smooth looping
    smooth_loop = sample_video.create_loop(repetitions=2, smooth_transition=True, transition_frames=5)
    
    # Should include transition frames between repetitions
    assert len(smooth_loop.frames) > initial_count * 2
    assert smooth_loop.fps == sample_video.fps


if __name__ == "__main__":
    pytest.main() 