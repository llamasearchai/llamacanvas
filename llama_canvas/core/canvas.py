"""
Core Canvas class for LlamaCanvas.

The Canvas class represents the main workspace for image and video generation,
manipulation, and processing. It integrates with various generators, processors,
and the agent system to provide a unified interface for creative tasks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image as PILImage

from llama_canvas.core.image import Image
from llama_canvas.utils.config import settings
from llama_canvas.utils.logging import get_logger

logger = get_logger(__name__)


class Canvas:
    """
    Canvas represents the main workspace for multi-modal content generation and manipulation.

    It serves as the central coordinating class for the LlamaCanvas system,
    providing access to image generation, manipulation, agent dispatching,
    and integration with Claude API.
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        use_agents: bool = True,
    ):
        """
        Initialize a new Canvas.

        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            background_color: RGB color tuple for canvas background
            use_agents: Whether to initialize and use the agent system
        """
        self.width = width
        self.height = height
        self.background_color = background_color
        self.layers: List[Image] = []

        # Initialize with blank canvas
        blank = np.ones((height, width, 3), dtype=np.uint8) * np.array(
            background_color, dtype=np.uint8
        )
        self.base_image = Image(PILImage.fromarray(blank))

        # Initialize agent manager if enabled
        self.agents = None
        if use_agents:
            try:
                from llama_canvas.core.agent_manager import AgentManager

                self.agents = AgentManager(self)
            except ImportError:
                logger.warning("AgentManager not available, agents disabled")

        logger.info(f"Canvas initialized with dimensions {width}x{height}")

    def generate_from_text(
        self, prompt: str, model: str = "stable-diffusion-v2", **kwargs
    ) -> Image:
        """
        Generate an image from text prompt and add it to the canvas.

        Args:
            prompt: Text description of the image to generate
            model: Model to use for generation
            **kwargs: Additional parameters for the generator

        Returns:
            Generated image
        """
        logger.info(f"Generating image from text prompt using {model}")

        # If agents are enabled and model is not explicitly Claude, let the agent system handle it
        if self.agents and not model.startswith("claude-"):
            return self.agents.generate_image(prompt, model=model, **kwargs)

        # For demonstration purposes, create a simple gradient with text
        logger.warning("No generator available, creating placeholder image")

        # Create a simple gradient background
        gradient = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for y in range(self.height):
            color = [int(255 * y / self.height), 100, int(200 * (1 - y / self.height))]
            gradient[y, :] = color

        image = Image(gradient)

        self.layers.append(image)
        return image

    def apply_style(
        self,
        image: Image,
        style: Union[str, Image],
        strength: float = 0.8,
        use_claude: bool = False,
        **kwargs,
    ) -> Image:
        """
        Apply a style to an image.

        Args:
            image: Source image to stylize
            style: Style name or reference image
            strength: Style transfer strength (0.0 to 1.0)
            use_claude: Whether to use Claude for style transfer
            **kwargs: Additional style transfer parameters

        Returns:
            Stylized image
        """
        logger.info(f"Applying style {'with Claude' if use_claude else ''}")

        # If agents are enabled and use_claude is True, let the agent system handle it
        if self.agents and use_claude:
            return self.agents.apply_style(image, style, strength=strength, **kwargs)

        # Simple placeholder implementation for style transfer
        logger.warning("No style transfer processor available, applying simple filter")

        # Apply a simple filter as a placeholder
        styled_image = image.apply_filter("edge_enhance")

        return styled_image

    def enhance_resolution(self, image: Image, scale: int = 2, **kwargs) -> Image:
        """
        Enhance image resolution.

        Args:
            image: Image to enhance
            scale: Upscaling factor
            **kwargs: Additional parameters for super resolution

        Returns:
            Enhanced image
        """
        logger.info(f"Enhancing image resolution by {scale}x")

        # If agents are enabled, let the agent system handle it for better results
        if self.agents and kwargs.get("use_agents", True):
            return self.agents.enhance_resolution(image, scale=scale, **kwargs)

        # Simple implementation using resize
        enhanced = image.resize((image.width * scale, image.height * scale))

        # Apply some sharpening to simulate super-resolution
        enhanced = enhanced.apply_filter("sharpen")

        return enhanced

    def blend_images(
        self, image1: Image, image2: Image, alpha: float = 0.5, mode: str = "normal"
    ) -> Image:
        """
        Blend two images together.

        Args:
            image1: First image
            image2: Second image
            alpha: Blending factor (0.0 to 1.0)
            mode: Blending mode (normal, multiply, screen, overlay, etc.)

        Returns:
            Blended image
        """
        logger.info(f"Blending images using {mode} mode with alpha={alpha}")

        # Ensure images are the same size
        if image1.width != image2.width or image1.height != image2.height:
            image2 = image2.resize((image1.width, image1.height))

        if mode == "normal":
            blended = Image.blend(image1, image2, alpha)
        else:
            # Only normal mode is implemented for now
            logger.warning(f"Blend mode '{mode}' not implemented, using normal mode")
            blended = Image.blend(image1, image2, alpha)

        return blended

    def create_animation(self, frames: List[Image], fps: int = 24, **kwargs) -> "Video":
        """
        Create an animation from a list of images.

        Args:
            frames: List of frames
            fps: Frames per second
            **kwargs: Additional animation parameters

        Returns:
            Video object containing the animation
        """
        # Import here to avoid circular imports
        from llama_canvas.core.video import Video

        logger.info(f"Creating animation with {len(frames)} frames at {fps} FPS")

        video = Video(frames, fps=fps)
        return video

    def save(self, path: Union[str, Path], format: Optional[str] = None) -> None:
        """
        Save the current canvas state to a file.

        Args:
            path: Output file path
            format: Optional file format (deduced from extension if not provided)
        """
        logger.info(f"Saving canvas to {path}")

        if not self.layers:
            self.base_image.save(path, format=format)
            return

        # Composite all layers
        composite = self.layers[0].copy()
        for layer in self.layers[1:]:
            composite = self.blend_images(composite, layer)

        composite.save(path, format=format)

    def clear(self) -> None:
        """Clear all layers from the canvas."""
        logger.info("Clearing canvas")

        self.layers = []

        # Reset to blank canvas
        blank = np.ones((self.height, self.width, 3), dtype=np.uint8) * np.array(
            self.background_color, dtype=np.uint8
        )
        self.base_image = Image(PILImage.fromarray(blank))
