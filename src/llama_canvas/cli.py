"""
Command Line Interface for LlamaCanvas.

This module provides the CLI for interacting with LlamaCanvas functionality
from the command line.
"""

import argparse
import sys
import os
import logging
import webbrowser
from pathlib import Path
from typing import List, Optional

from llama_canvas.core.canvas import Canvas
from llama_canvas.core.image import Image
from llama_canvas.utils.logging import setup_logging, get_logger
from llama_canvas.utils.config import settings

# Set up logging
setup_logging()
logger = get_logger(__name__)


def generate_image(args):
    """
    Generate an image from a text prompt.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Generating image from prompt: {args.prompt}")
    
    # Create canvas
    canvas = Canvas(
        width=args.width,
        height=args.height,
        use_agents=not args.no_agents
    )
    
    # Generate image
    image = canvas.generate_from_text(
        args.prompt,
        model=args.model,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps
    )
    
    # Save image
    output_path = Path(args.output)
    image.save(output_path)
    
    logger.info(f"Image saved to {output_path}")
    
    # Open the image if requested
    if args.open:
        try:
            from PIL import Image as PILImage
            PILImage.open(output_path).show()
        except Exception as e:
            logger.error(f"Error opening image: {e}")


def apply_style(args):
    """
    Apply a style to an image.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Applying style '{args.style}' to {args.input}")
    
    # Load input image
    try:
        image = Image(args.input)
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        sys.exit(1)
    
    # Create canvas
    canvas = Canvas(use_agents=not args.no_agents)
    
    # Apply style
    styled = canvas.apply_style(
        image,
        args.style,
        strength=args.strength,
        use_claude=args.use_claude
    )
    
    # Save styled image
    output_path = Path(args.output)
    styled.save(output_path)
    
    logger.info(f"Styled image saved to {output_path}")
    
    # Open the image if requested
    if args.open:
        try:
            from PIL import Image as PILImage
            PILImage.open(output_path).show()
        except Exception as e:
            logger.error(f"Error opening image: {e}")


def enhance_image(args):
    """
    Enhance an image.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Enhancing image {args.input}")
    
    # Load input image
    try:
        image = Image(args.input)
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        sys.exit(1)
    
    # Create canvas
    canvas = Canvas(use_agents=not args.no_agents)
    
    # Enhance image
    enhanced = canvas.enhance_resolution(
        image,
        scale=args.scale
    )
    
    # Save enhanced image
    output_path = Path(args.output)
    enhanced.save(output_path)
    
    logger.info(f"Enhanced image saved to {output_path}")
    
    # Open the image if requested
    if args.open:
        try:
            from PIL import Image as PILImage
            PILImage.open(output_path).show()
        except Exception as e:
            logger.error(f"Error opening image: {e}")


def run_ui(args):
    """
    Run the web UI.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Starting web UI on {args.host}:{args.port}")
    
    try:
        from llama_canvas.api.app import run_app
        
        # Open browser if requested
        if args.browse:
            url = f"http://{args.host}:{args.port}"
            webbrowser.open(url)
        
        # Run the FastAPI app
        run_app(host=args.host, port=args.port)
        
    except ImportError as e:
        logger.error(f"Error importing API app: {e}")
        logger.error("Make sure FastAPI and uvicorn are installed: pip install fastapi uvicorn")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running web UI: {e}")
        sys.exit(1)


def main(argv: Optional[List[str]] = None):
    """
    Main entry point for the CLI.
    
    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])
    """
    parser = argparse.ArgumentParser(
        description="LlamaCanvas - Advanced AI-driven multi-modal generation platform"
    )
    
    # Common arguments
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Command to run",
        required=True
    )
    
    # Generate image command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate an image from a text prompt"
    )
    generate_parser.add_argument(
        "prompt",
        help="Text prompt for image generation"
    )
    generate_parser.add_argument(
        "--output", "-o",
        default="output.png",
        help="Output file path"
    )
    generate_parser.add_argument(
        "--width", "-w",
        type=int,
        default=512,
        help="Image width in pixels"
    )
    generate_parser.add_argument(
        "--height", "-h",
        type=int,
        default=512,
        help="Image height in pixels"
    )
    generate_parser.add_argument(
        "--model", "-m",
        default="stable-diffusion-v2",
        help="Model to use for generation"
    )
    generate_parser.add_argument(
        "--guidance-scale", "-g",
        type=float,
        default=7.5,
        help="Guidance scale for generation"
    )
    generate_parser.add_argument(
        "--steps", "-s",
        type=int,
        default=50,
        help="Number of inference steps"
    )
    generate_parser.add_argument(
        "--no-agents",
        action="store_true",
        help="Disable agent system"
    )
    generate_parser.add_argument(
        "--open",
        action="store_true",
        help="Open the image after generation"
    )
    generate_parser.set_defaults(func=generate_image)
    
    # Apply style command
    style_parser = subparsers.add_parser(
        "style",
        help="Apply a style to an image"
    )
    style_parser.add_argument(
        "input",
        help="Input image path"
    )
    style_parser.add_argument(
        "style",
        help="Style name or reference image path"
    )
    style_parser.add_argument(
        "--output", "-o",
        default="styled_output.png",
        help="Output file path"
    )
    style_parser.add_argument(
        "--strength", "-s",
        type=float,
        default=0.8,
        help="Style transfer strength (0.0 to 1.0)"
    )
    style_parser.add_argument(
        "--use-claude",
        action="store_true",
        help="Use Claude for style transfer"
    )
    style_parser.add_argument(
        "--no-agents",
        action="store_true",
        help="Disable agent system"
    )
    style_parser.add_argument(
        "--open",
        action="store_true",
        help="Open the image after processing"
    )
    style_parser.set_defaults(func=apply_style)
    
    # Enhance image command
    enhance_parser = subparsers.add_parser(
        "enhance",
        help="Enhance an image"
    )
    enhance_parser.add_argument(
        "input",
        help="Input image path"
    )
    enhance_parser.add_argument(
        "--output", "-o",
        default="enhanced_output.png",
        help="Output file path"
    )
    enhance_parser.add_argument(
        "--scale", "-s",
        type=int,
        default=2,
        help="Upscaling factor"
    )
    enhance_parser.add_argument(
        "--no-agents",
        action="store_true",
        help="Disable agent system"
    )
    enhance_parser.add_argument(
        "--open",
        action="store_true",
        help="Open the image after processing"
    )
    enhance_parser.set_defaults(func=enhance_image)
    
    # UI command
    ui_parser = subparsers.add_parser(
        "ui",
        help="Run the web UI"
    )
    ui_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to"
    )
    ui_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to bind the server to"
    )
    ui_parser.add_argument(
        "--browse", "-b",
        action="store_true",
        help="Open a browser window"
    )
    ui_parser.set_defaults(func=run_ui)
    
    # Parse arguments
    args = parser.parse_args(argv)
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run the selected command
    args.func(args)


if __name__ == "__main__":
    main() 