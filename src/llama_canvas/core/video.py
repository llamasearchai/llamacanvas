"""
Core Video class for LlamaCanvas.

The Video class provides capabilities for creating and manipulating
animated sequences of images, as well as basic video processing.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union, Any, Dict

from llama_canvas.core.image import Image
from llama_canvas.utils.logging import get_logger

logger = get_logger(__name__)

class Video:
    """
    Video represents an animated sequence of images.
    
    It provides capabilities for creating, manipulating, and exporting
    animations and videos.
    """
    
    def __init__(
        self,
        frames: Optional[List[Image]] = None,
        fps: int = 24,
        width: Optional[int] = None,
        height: Optional[int] = None
    ):
        """
        Initialize a Video object.
        
        Args:
            frames: Initial list of frames
            fps: Frames per second
            width: Video width (if not specified, uses width of first frame)
            height: Video height (if not specified, uses height of first frame)
        """
        self.frames = frames or []
        self.fps = fps
        
        if frames and not (width and height):
            if len(frames) > 0:
                width = width or frames[0].width
                height = height or frames[0].height
        
        self.width = width or 512
        self.height = height or 512
        
        # Ensure all frames are at the correct dimensions
        for i, frame in enumerate(self.frames):
            if frame.width != self.width or frame.height != self.height:
                self.frames[i] = frame.resize((self.width, self.height))
                
        logger.info(f"Video initialized with {len(self.frames)} frames at {fps} FPS")
    
    def add_frame(self, frame: Image, position: Optional[int] = None) -> None:
        """
        Add a frame to the video.
        
        Args:
            frame: Frame to add
            position: Position to insert frame (appends if not specified)
        """
        # Ensure frame matches video dimensions
        if frame.width != self.width or frame.height != self.height:
            frame = frame.resize((self.width, self.height))
        
        if position is None:
            self.frames.append(frame)
        else:
            self.frames.insert(position, frame)
            
        logger.debug(f"Added frame at position {position if position is not None else len(self.frames) - 1}")
    
    def remove_frame(self, position: int) -> None:
        """
        Remove a frame from the video.
        
        Args:
            position: Position of frame to remove
        """
        if 0 <= position < len(self.frames):
            self.frames.pop(position)
            logger.debug(f"Removed frame at position {position}")
        else:
            logger.warning(f"Invalid frame position: {position}")
    
    def get_frame(self, position: int) -> Optional[Image]:
        """
        Get a frame from the video.
        
        Args:
            position: Position of frame to get
            
        Returns:
            Frame at specified position or None if position is invalid
        """
        if 0 <= position < len(self.frames):
            return self.frames[position]
        return None
    
    def set_fps(self, fps: int) -> None:
        """
        Set the frames per second.
        
        Args:
            fps: New frames per second value
        """
        self.fps = fps
        logger.debug(f"Set FPS to {fps}")
    
    def save_gif(self, path: Union[str, Path], loop: int = 0, optimize: bool = True) -> None:
        """
        Save the video as an animated GIF.
        
        Args:
            path: Output file path
            loop: Number of times GIF should loop (0 = infinite)
            optimize: Whether to optimize the GIF
        """
        if not self.frames:
            logger.warning("No frames to save")
            return
        
        logger.info(f"Saving video as GIF to {path}")
        
        # Convert frames to PIL images
        pil_frames = [frame.image for frame in self.frames]
        
        # Calculate duration in milliseconds
        duration = int(1000 / self.fps)
        
        # Save as GIF
        pil_frames[0].save(
            path,
            format="GIF",
            append_images=pil_frames[1:],
            save_all=True,
            duration=duration,
            loop=loop,
            optimize=optimize
        )
    
    def save_mp4(self, path: Union[str, Path], codec: str = "libx264", bitrate: str = "1M") -> None:
        """
        Save the video as an MP4 file.
        
        Requires moviepy to be installed.
        
        Args:
            path: Output file path
            codec: Video codec to use
            bitrate: Video bitrate
        """
        try:
            import moviepy.editor as mpy
        except ImportError:
            logger.error("moviepy is required to save as MP4. Install it with 'pip install moviepy'")
            return
        
        if not self.frames:
            logger.warning("No frames to save")
            return
        
        logger.info(f"Saving video as MP4 to {path}")
        
        # Create a temporary directory for frame files
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_paths = []
            
            # Save each frame as a temporary image file
            for i, frame in enumerate(self.frames):
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                frame.save(frame_path)
                frame_paths.append(frame_path)
            
            # Create a clip from the images
            clip = mpy.ImageSequenceClip(frame_paths, fps=self.fps)
            
            # Write the clip to a file
            clip.write_videofile(str(path), codec=codec, bitrate=bitrate)
    
    def apply_to_all_frames(self, func: callable) -> 'Video':
        """
        Apply a function to all frames.
        
        Args:
            func: Function that takes an Image and returns an Image
            
        Returns:
            New Video with transformed frames
        """
        new_frames = [func(frame) for frame in self.frames]
        return Video(new_frames, fps=self.fps, width=self.width, height=self.height)
    
    def slice(self, start: int, end: Optional[int] = None) -> 'Video':
        """
        Create a new video from a slice of this video.
        
        Args:
            start: Start frame index
            end: End frame index (exclusive, None means until the end)
            
        Returns:
            New Video containing the slice
        """
        end = end or len(self.frames)
        return Video(self.frames[start:end], fps=self.fps, width=self.width, height=self.height)
    
    def concatenate(self, other: 'Video') -> 'Video':
        """
        Concatenate another video to this one.
        
        Args:
            other: Video to concatenate
            
        Returns:
            New Video containing both videos concatenated
        """
        # Ensure other video frames match dimensions
        other_frames = [
            frame.resize((self.width, self.height)) 
            if frame.width != self.width or frame.height != self.height else frame
            for frame in other.frames
        ]
        
        return Video(
            self.frames + other_frames,
            fps=self.fps,
            width=self.width,
            height=self.height
        )
    
    def __len__(self) -> int:
        """Return the number of frames."""
        return len(self.frames)
    
    @property
    def duration(self) -> float:
        """Return the duration in seconds."""
        return len(self.frames) / self.fps 