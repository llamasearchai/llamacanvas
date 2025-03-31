# Image API Reference

The `Image` class in LlamaCanvas provides a comprehensive set of methods for working with images. This class handles loading, saving, manipulating, and analyzing image data.

## Class Overview

```python
from llama_canvas.core.image import Image
```

## Creating and Loading Images

### Static Methods

#### `Image.create()`

Creates a new blank image with the specified dimensions and background color.

```python
@staticmethod
def create(width: int, 
           height: int, 
           background_color: Union[str, Tuple[int, int, int, int]] = "transparent") -> "Image":
    """
    Create a new blank image.
    
    Args:
        width: Width of the image in pixels
        height: Height of the image in pixels
        background_color: Color to fill the image with. Can be a string name ('transparent', 'white', 'black', etc.) 
                          or an RGBA tuple (0-255 for each channel)
    
    Returns:
        A new Image instance
    """
```

Example:
```python
# Create a 512x512 blank white image
white_image = Image.create(512, 512, "white")

# Create a 1024x768 image with a semi-transparent blue background
blue_image = Image.create(1024, 768, (0, 0, 255, 128))
```

#### `Image.load()`

Loads an image from a file or URL.

```python
@staticmethod
def load(source: Union[str, Path, bytes]) -> "Image":
    """
    Load an image from a file path, URL, or bytes.
    
    Args:
        source: File path, URL, or bytes containing image data
    
    Returns:
        An Image instance with the loaded image
    
    Raises:
        ValueError: If the image cannot be loaded
    """
```

Example:
```python
# Load from file
img1 = Image.load("path/to/image.png")

# Load from URL
img2 = Image.load("https://example.com/image.jpg")

# Load from bytes
with open("image.png", "rb") as f:
    img3 = Image.load(f.read())
```

#### `Image.from_array()`

Creates an image from a NumPy array.

```python
@staticmethod
def from_array(array: np.ndarray) -> "Image":
    """
    Create an image from a NumPy array.
    
    Args:
        array: NumPy array in the format (height, width, channels)
              where channels is 1 (grayscale), 3 (RGB), or 4 (RGBA)
    
    Returns:
        An Image instance containing the array data
    
    Raises:
        ValueError: If the array format is invalid
    """
```

Example:
```python
import numpy as np

# Create a red square image
red_array = np.zeros((100, 100, 3), dtype=np.uint8)
red_array[:, :, 0] = 255  # Set red channel to maximum
img = Image.from_array(red_array)
```

## Basic Properties and Methods

### Properties

- `width`: Width of the image in pixels
- `height`: Height of the image in pixels
- `size`: Tuple of (width, height)
- `aspect_ratio`: Width divided by height
- `channels`: Number of color channels (1 for grayscale, 3 for RGB, 4 for RGBA)
- `mode`: Color mode ('L' for grayscale, 'RGB', 'RGBA', etc.)
- `format`: Image format if known ('PNG', 'JPEG', etc.)

### Methods

#### `save()`

Saves the image to a file.

```python
def save(self, 
         path: Union[str, Path], 
         format: Optional[str] = None, 
         quality: int = 95) -> None:
    """
    Save the image to a file.
    
    Args:
        path: Path where the image should be saved
        format: Image format ('PNG', 'JPEG', 'WEBP', etc.). If None, inferred from file extension
        quality: Quality for lossy formats (0-100)
    
    Raises:
        ValueError: If the image cannot be saved or the format is unsupported
    """
```

#### `copy()`

Creates a deep copy of the image.

```python
def copy(self) -> "Image":
    """
    Create a deep copy of the image.
    
    Returns:
        A new Image instance with the same content
    """
```

#### `to_array()`

Converts the image to a NumPy array.

```python
def to_array(self) -> np.ndarray:
    """
    Convert the image to a NumPy array.
    
    Returns:
        NumPy array in the format (height, width, channels)
    """
```

#### `to_base64()`

Converts the image to a base64-encoded string.

```python
def to_base64(self, 
              format: str = "PNG") -> str:
    """
    Convert the image to a base64-encoded string.
    
    Args:
        format: Image format to use ('PNG', 'JPEG', etc.)
    
    Returns:
        Base64-encoded string
    """
```

#### `to_bytes()`

Converts the image to bytes.

```python
def to_bytes(self, 
             format: str = "PNG", 
             quality: int = 95) -> bytes:
    """
    Convert the image to bytes.
    
    Args:
        format: Image format to use ('PNG', 'JPEG', etc.)
        quality: Quality for lossy formats (0-100)
    
    Returns:
        Image data as bytes
    """
```

## Transformation Methods

#### `resize()`

Resizes the image to the specified dimensions.

```python
def resize(self, 
           width: Optional[int] = None, 
           height: Optional[int] = None, 
           method: str = "lanczos") -> "Image":
    """
    Resize the image.
    
    Args:
        width: New width in pixels. If None, calculated from height to maintain aspect ratio
        height: New height in pixels. If None, calculated from width to maintain aspect ratio
        method: Resampling method ('nearest', 'box', 'bilinear', 'hamming', 'bicubic', 'lanczos')
    
    Returns:
        A new resized Image instance
    
    Raises:
        ValueError: If both width and height are None
    """
```

#### `crop()`

Crops the image to the specified region.

```python
def crop(self, 
         x: int, 
         y: int, 
         width: int, 
         height: int) -> "Image":
    """
    Crop the image to the specified region.
    
    Args:
        x: X-coordinate of the top-left corner
        y: Y-coordinate of the top-left corner
        width: Width of the cropped region
        height: Height of the cropped region
    
    Returns:
        A new cropped Image instance
    
    Raises:
        ValueError: If the crop region is outside the image bounds
    """
```

#### `rotate()`

Rotates the image by the specified angle.

```python
def rotate(self, 
           angle: float, 
           expand: bool = False, 
           fill_color: Union[str, Tuple[int, int, int, int]] = "transparent") -> "Image":
    """
    Rotate the image by the specified angle.
    
    Args:
        angle: Rotation angle in degrees (counter-clockwise)
        expand: Whether to expand the image to fit the rotated content
        fill_color: Color to fill new areas with
    
    Returns:
        A new rotated Image instance
    """
```

#### `flip()`

Flips the image horizontally or vertically.

```python
def flip(self, 
         direction: str = "horizontal") -> "Image":
    """
    Flip the image horizontally or vertically.
    
    Args:
        direction: Direction to flip ('horizontal' or 'vertical')
    
    Returns:
        A new flipped Image instance
    
    Raises:
        ValueError: If direction is not 'horizontal' or 'vertical'
    """
```

## Color Operations

#### `adjust_brightness()`

Adjusts the brightness of the image.

```python
def adjust_brightness(self, 
                      factor: float) -> "Image":
    """
    Adjust the brightness of the image.
    
    Args:
        factor: Brightness adjustment factor (0.0 = black, 1.0 = original, >1.0 = brighter)
    
    Returns:
        A new adjusted Image instance
    """
```

#### `adjust_contrast()`

Adjusts the contrast of the image.

```python
def adjust_contrast(self, 
                    factor: float) -> "Image":
    """
    Adjust the contrast of the image.
    
    Args:
        factor: Contrast adjustment factor (0.0 = gray, 1.0 = original, >1.0 = more contrast)
    
    Returns:
        A new adjusted Image instance
    """
```

#### `adjust_saturation()`

Adjusts the color saturation of the image.

```python
def adjust_saturation(self, 
                      factor: float) -> "Image":
    """
    Adjust the color saturation of the image.
    
    Args:
        factor: Saturation adjustment factor (0.0 = grayscale, 1.0 = original, >1.0 = more saturated)
    
    Returns:
        A new adjusted Image instance
    """
```

#### `convert_mode()`

Converts the image to a different color mode.

```python
def convert_mode(self, 
                 mode: str) -> "Image":
    """
    Convert the image to a different color mode.
    
    Args:
        mode: Target color mode ('L', 'RGB', 'RGBA', etc.)
    
    Returns:
        A new Image instance in the target mode
    
    Raises:
        ValueError: If the conversion is not supported
    """
```

## Filters and Effects

#### `apply_filter()`

Applies a predefined filter to the image.

```python
def apply_filter(self, 
                 filter_name: str, 
                 **parameters) -> "Image":
    """
    Apply a predefined filter to the image.
    
    Args:
        filter_name: Name of the filter to apply
                    ('blur', 'sharpen', 'edge_enhance', 'emboss', 'find_edges', etc.)
        **parameters: Filter-specific parameters
    
    Returns:
        A new filtered Image instance
    
    Raises:
        ValueError: If the filter is not supported
    """
```

#### `apply_effect()`

Applies a visual effect to the image.

```python
def apply_effect(self, 
                 effect_name: str, 
                 intensity: float = 1.0, 
                 **parameters) -> "Image":
    """
    Apply a visual effect to the image.
    
    Args:
        effect_name: Name of the effect to apply
                    ('vignette', 'grain', 'duotone', 'glitch', etc.)
        intensity: Effect intensity (0.0 to 1.0)
        **parameters: Effect-specific parameters
    
    Returns:
        A new Image instance with the effect applied
    
    Raises:
        ValueError: If the effect is not supported
    """
```

## Drawing and Compositing

#### `draw_rectangle()`

Draws a rectangle on the image.

```python
def draw_rectangle(self, 
                   x: int, 
                   y: int, 
                   width: int, 
                   height: int, 
                   color: Union[str, Tuple[int, int, int, int]], 
                   fill: bool = True, 
                   line_width: int = 1) -> "Image":
    """
    Draw a rectangle on the image.
    
    Args:
        x: X-coordinate of the top-left corner
        y: Y-coordinate of the top-left corner
        width: Width of the rectangle
        height: Height of the rectangle
        color: Color of the rectangle
        fill: Whether to fill the rectangle
        line_width: Width of the outline if not filled
    
    Returns:
        A new Image instance with the rectangle drawn
    """
```

#### `draw_ellipse()`

Draws an ellipse on the image.

```python
def draw_ellipse(self, 
                 x: int, 
                 y: int, 
                 width: int, 
                 height: int, 
                 color: Union[str, Tuple[int, int, int, int]], 
                 fill: bool = True, 
                 line_width: int = 1) -> "Image":
    """
    Draw an ellipse on the image.
    
    Args:
        x: X-coordinate of the top-left corner of the bounding box
        y: Y-coordinate of the top-left corner of the bounding box
        width: Width of the bounding box
        height: Height of the bounding box
        color: Color of the ellipse
        fill: Whether to fill the ellipse
        line_width: Width of the outline if not filled
    
    Returns:
        A new Image instance with the ellipse drawn
    """
```

#### `draw_line()`

Draws a line on the image.

```python
def draw_line(self, 
              x1: int, 
              y1: int, 
              x2: int, 
              y2: int, 
              color: Union[str, Tuple[int, int, int, int]], 
              line_width: int = 1) -> "Image":
    """
    Draw a line on the image.
    
    Args:
        x1: X-coordinate of the start point
        y1: Y-coordinate of the start point
        x2: X-coordinate of the end point
        y2: Y-coordinate of the end point
        color: Color of the line
        line_width: Width of the line
    
    Returns:
        A new Image instance with the line drawn
    """
```

#### `draw_text()`

Draws text on the image.

```python
def draw_text(self, 
              text: str, 
              x: int, 
              y: int, 
              font: Optional[str] = None, 
              font_size: int = 12, 
              color: Union[str, Tuple[int, int, int, int]] = "black", 
              align: str = "left") -> "Image":
    """
    Draw text on the image.
    
    Args:
        text: The text to draw
        x: X-coordinate of the text position
        y: Y-coordinate of the text position
        font: Font to use (path to .ttf file or font name)
        font_size: Size of the font in points
        color: Color of the text
        align: Text alignment ('left', 'center', 'right')
    
    Returns:
        A new Image instance with the text drawn
    """
```

#### `composite()`

Composites another image onto this image.

```python
def composite(self, 
              other: "Image", 
              x: int = 0, 
              y: int = 0, 
              opacity: float = 1.0, 
              blend_mode: str = "normal") -> "Image":
    """
    Composite another image onto this image.
    
    Args:
        other: The image to composite
        x: X-coordinate of the top-left corner
        y: Y-coordinate of the top-left corner
        opacity: Opacity of the composited image (0.0 to 1.0)
        blend_mode: Blending mode ('normal', 'multiply', 'screen', 'overlay', etc.)
    
    Returns:
        A new composited Image instance
    """
```

## Analysis Methods

#### `get_dominant_colors()`

Extracts the dominant colors from the image.

```python
def get_dominant_colors(self, 
                        count: int = 5) -> List[Tuple[int, int, int]]:
    """
    Extract the dominant colors from the image.
    
    Args:
        count: Number of colors to extract
    
    Returns:
        List of RGB color tuples ordered by dominance
    """
```

#### `get_average_color()`

Gets the average color of the image.

```python
def get_average_color(self) -> Tuple[int, int, int]:
    """
    Get the average color of the image.
    
    Returns:
        RGB color tuple representing the average color
    """
```

#### `get_histogram()`

Gets the color histogram of the image.

```python
def get_histogram(self, 
                  channel: Optional[int] = None) -> np.ndarray:
    """
    Get the color histogram of the image.
    
    Args:
        channel: Channel index to get histogram for (None for all channels)
    
    Returns:
        NumPy array containing the histogram data
    """
```

## Advanced Methods

#### `create_mask()`

Creates a mask image.

```python
@staticmethod
def create_mask(width: int, 
                height: int, 
                background: str = "black") -> "Image":
    """
    Create a mask image.
    
    Args:
        width: Width of the mask in pixels
        height: Height of the mask in pixels
        background: Background color ('black' for fully transparent, 'white' for fully opaque)
    
    Returns:
        A new Image instance with the mask
    """
```

#### `apply_mask()`

Applies a mask to the image.

```python
def apply_mask(self, 
               mask: "Image") -> "Image":
    """
    Apply a mask to the image.
    
    Args:
        mask: Mask image (grayscale or alpha channel)
              White areas in the mask keep the original image,
              Black areas become transparent.
    
    Returns:
        A new masked Image instance
    
    Raises:
        ValueError: If the mask dimensions don't match the image
    """
```

#### `enhance()`

Enhances the image resolution using AI upscaling.

```python
def enhance(self, 
            scale: float = 2.0, 
            method: str = "super_resolution") -> "Image":
    """
    Enhance the image resolution using AI upscaling.
    
    Args:
        scale: Scale factor for the enhancement
        method: Enhancement method to use ('super_resolution', 'esrgan', etc.)
    
    Returns:
        A new enhanced Image instance
    """
```

## Example Usage

```python
from llama_canvas.core.image import Image

# Load an image
img = Image.load("input.jpg")

# Resize to 512x512
resized = img.resize(512, 512)

# Adjust brightness and contrast
adjusted = resized.adjust_brightness(1.2).adjust_contrast(1.1)

# Apply a filter
filtered = adjusted.apply_filter("edge_enhance")

# Draw some elements
with_rectangle = filtered.draw_rectangle(100, 100, 300, 200, "red", fill=False, line_width=3)
with_text = with_rectangle.draw_text("Enhanced Image", 120, 120, font_size=24, color="white")

# Composite with another image
overlay = Image.load("overlay.png")
result = with_text.composite(overlay, x=200, y=200, opacity=0.7, blend_mode="overlay")

# Save the result
result.save("output.png")

# Get analysis
dominant_colors = result.get_dominant_colors(count=3)
print(f"Dominant colors: {dominant_colors}")
``` 