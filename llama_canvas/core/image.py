"""
Image class for LlamaCanvas.

Provides an enhanced Image class with various manipulation and processing capabilities.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
from PIL import Image as PILImage, ImageEnhance, ImageFilter, ImageOps

from llama_canvas.utils.logging import get_logger

logger = get_logger(__name__)


class Image:
    """
    Enhanced image class with additional processing capabilities.
    
    This class provides a wrapper around PIL Image with additional methods for
    common image manipulation tasks and integration with the LlamaCanvas ecosystem.
    """
    
    def __init__(self, image: Union[PILImage.Image, np.ndarray, str, Path, 'Image']):
        """
        Initialize an image from various sources.
        
        Args:
            image: Image source (PIL Image, numpy array, file path, or another Image)
        """
        if isinstance(image, Image):
            self._image = image._image.copy()
        elif isinstance(image, PILImage.Image):
            self._image = image
        elif isinstance(image, np.ndarray):
            self._image = PILImage.fromarray(
                image.astype(np.uint8) if image.dtype != np.uint8 else image
            )
        elif isinstance(image, (str, Path)):
            self._image = PILImage.open(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Convert to RGB if needed (unless it's RGBA which we want to preserve)
        if self._image.mode not in ["RGB", "RGBA"]:
            self._image = self._image.convert("RGB")
    
    @property
    def width(self) -> int:
        """Get image width."""
        return self._image.width
    
    @property
    def height(self) -> int:
        """Get image height."""
        return self._image.height
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get image shape as (height, width, channels)."""
        return (self.height, self.width, 3 if self._image.mode == "RGB" else 4)
    
    @property
    def array(self) -> np.ndarray:
        """Get image as numpy array."""
        return np.array(self._image)
    
    @property
    def pil_image(self) -> PILImage.Image:
        """Get underlying PIL image."""
        return self._image
    
    @property
    def mode(self) -> str:
        """Get image mode (RGB, RGBA, etc.)."""
        return self._image.mode
    
    def copy(self) -> 'Image':
        """Create a copy of the image."""
        return Image(self._image.copy())
    
    def resize(
        self, 
        size: Union[Tuple[int, int], int],
        resample: int = PILImage.LANCZOS,
        maintain_aspect_ratio: bool = False
    ) -> 'Image':
        """
        Resize the image.
        
        Args:
            size: New size as (width, height) or scaling factor if int
            resample: Resampling filter
            maintain_aspect_ratio: Whether to maintain aspect ratio when resizing
            
        Returns:
            Resized image
        """
        if isinstance(size, int):
            # Size is a scaling factor
            new_width = int(self.width * size)
            new_height = int(self.height * size)
            size = (new_width, new_height)
        elif maintain_aspect_ratio:
            # Calculate new dimensions while maintaining aspect ratio
            target_width, target_height = size
            aspect = self.width / self.height
            
            if target_width / target_height > aspect:
                # Width is the limiting factor
                new_width = int(target_height * aspect)
                new_height = target_height
                size = (new_width, new_height)
            else:
                # Height is the limiting factor
                new_width = target_width
                new_height = int(target_width / aspect)
                size = (new_width, new_height)
        
        return Image(self._image.resize(size, resample=resample))
    
    def crop(self, box: Tuple[int, int, int, int]) -> 'Image':
        """
        Crop the image.
        
        Args:
            box: Crop box as (left, upper, right, lower)
            
        Returns:
            Cropped image
        """
        return Image(self._image.crop(box))
    
    def adjust_brightness(self, factor: float) -> 'Image':
        """
        Adjust image brightness.
        
        Args:
            factor: Brightness adjustment factor (1.0 = original)
            
        Returns:
            Adjusted image
        """
        enhancer = ImageEnhance.Brightness(self._image)
        return Image(enhancer.enhance(factor))
    
    def adjust_contrast(self, factor: float) -> 'Image':
        """
        Adjust image contrast.
        
        Args:
            factor: Contrast adjustment factor (1.0 = original)
            
        Returns:
            Adjusted image
        """
        enhancer = ImageEnhance.Contrast(self._image)
        return Image(enhancer.enhance(factor))
    
    def apply_filter(self, filter_type: str) -> 'Image':
        """
        Apply a filter to the image.
        
        Args:
            filter_type: Filter name (blur, sharpen, contour, etc.)
            
        Returns:
            Filtered image
        """
        filters = {
            "blur": ImageFilter.BLUR,
            "sharpen": ImageFilter.SHARPEN,
            "contour": ImageFilter.CONTOUR,
            "detail": ImageFilter.DETAIL,
            "edge_enhance": ImageFilter.EDGE_ENHANCE,
            "find_edges": ImageFilter.FIND_EDGES,
            "smooth": ImageFilter.SMOOTH,
        }
        
        if filter_type not in filters:
            raise ValueError(f"Unsupported filter: {filter_type}")
        
        return Image(self._image.filter(filters[filter_type]))
    
    def rotate(self, angle: float, expand: bool = False) -> 'Image':
        """
        Rotate the image.
        
        Args:
            angle: Rotation angle in degrees
            expand: Whether to expand the output to fit the rotated image
            
        Returns:
            Rotated image
        """
        return Image(self._image.rotate(angle, expand=expand))
    
    def flip_horizontal(self) -> 'Image':
        """
        Flip the image horizontally.
        
        Returns:
            Flipped image
        """
        return Image(ImageOps.mirror(self._image))
    
    def flip_vertical(self) -> 'Image':
        """
        Flip the image vertically.
        
        Returns:
            Flipped image
        """
        return Image(ImageOps.flip(self._image))
    
    def convert(self, mode: str) -> 'Image':
        """
        Convert the image to a different mode (RGB, RGBA, L, etc.).
        
        Args:
            mode: Target mode
            
        Returns:
            Converted image
        """
        return Image(self._image.convert(mode))
    
    def compose(self, other: 'Image', position: Tuple[int, int] = (0, 0)) -> 'Image':
        """
        Compose this image with another image at specified position.
        
        Args:
            other: Image to compose with
            position: Position (x, y) to place the other image
            
        Returns:
            Composed image
        """
        if self._image.mode != "RGBA":
            base = self._image.convert("RGBA")
        else:
            base = self._image.copy()
        
        if other.mode != "RGBA":
            overlay = other.pil_image.convert("RGBA")
        else:
            overlay = other.pil_image
        
        result = base.copy()
        result.paste(overlay, position, overlay)
        return Image(result)
    
    def save(self, path: Union[str, Path], format: Optional[str] = None, **kwargs) -> None:
        """
        Save the image to a file.
        
        Args:
            path: Output file path
            format: Optional file format (deduced from extension if not provided)
            **kwargs: Additional save parameters
        """
        self._image.save(path, format=format, **kwargs)
    
    @staticmethod
    def blend(image1: 'Image', image2: 'Image', alpha: float) -> 'Image':
        """
        Blend two images.
        
        Args:
            image1: First image
            image2: Second image
            alpha: Blending factor (0.0 to 1.0)
            
        Returns:
            Blended image
        """
        # Ensure images are the same size
        if image1.width != image2.width or image1.height != image2.height:
            image2 = image2.resize((image1.width, image1.height))
            
        # Ensure both images are in RGB mode
        img1 = image1.pil_image
        img2 = image2.pil_image
        if img1.mode != "RGB":
            img1 = img1.convert("RGB")
        if img2.mode != "RGB":
            img2 = img2.convert("RGB")
            
        return Image(PILImage.blend(img1, img2, alpha)) 