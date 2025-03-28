"""
LlamaCanvas - Advanced AI-driven multi-modal generation platform with Claude API integration.

This package provides tools for image and video generation, manipulation,
and enhancement using cutting-edge AI models and techniques.
"""

__version__ = "1.0.0"

from llama_canvas.core.canvas import Canvas
from llama_canvas.core.image import Image
from llama_canvas.core.video import Video

__all__ = ["Canvas", "Image", "Video"] 