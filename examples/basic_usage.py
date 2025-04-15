#!/usr/bin/env python3
"""
Basic Usage Examples for LlamaCanvas

This script demonstrates the basic functionality of LlamaCanvas for image generation,
style transfer, and enhancement.
"""

import argparse
import os
from pathlib import Path

from llama_canvas.core.canvas import Canvas
from llama_canvas.core.image import Image


def generate_image(canvas, prompt, output_path):
    """Generate an image from a text prompt."""
    print(f"Generating image from prompt: '{prompt}'")
    image = canvas.generate_from_text(prompt)
    image.save(output_path)
    print(f"Image saved to {output_path}")
    return image


def apply_style(canvas, image_path, style, output_path):
    """Apply a style to an existing image."""
    print(f"Applying style '{style}' to image {image_path}")
    original = Image.load(image_path)
    styled = canvas.apply_style(original, style)
    styled.save(output_path)
    print(f"Styled image saved to {output_path}")
    return styled


def enhance_image(canvas, image_path, scale, denoise, output_path):
    """Enhance an image with optional upscaling and denoising."""
    print(f"Enhancing image {image_path} (scale={scale}, denoise={denoise})")
    original = Image.load(image_path)
    enhanced = canvas.enhance(original, scale=scale, denoise=denoise)
    enhanced.save(output_path)
    print(f"Enhanced image saved to {output_path}")
    return enhanced


def run_pipeline(canvas, prompt, style, scale, output_dir):
    """Run a complete pipeline: generate -> style -> enhance."""
    os.makedirs(output_dir, exist_ok=True)

    # Generate base image
    base_path = os.path.join(output_dir, "generated.png")
    image = generate_image(canvas, prompt, base_path)

    # Apply style
    styled_path = os.path.join(output_dir, "styled.png")
    styled = apply_style(canvas, base_path, style, styled_path)

    # Enhance
    enhanced_path = os.path.join(output_dir, "enhanced.png")
    enhanced = enhance_image(canvas, styled_path, scale, True, enhanced_path)

    print(f"Pipeline complete. All images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="LlamaCanvas Basic Examples")
    parser.add_argument(
        "--prompt",
        default="A serene landscape with mountains and a lake at sunset",
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--style", default="Van Gogh", help="Style to apply to the generated image"
    )
    parser.add_argument(
        "--scale", type=float, default=1.5, help="Scaling factor for enhancement"
    )
    parser.add_argument(
        "--output-dir", default="output", help="Directory to save output images"
    )
    parser.add_argument(
        "--width", type=int, default=512, help="Width of generated image"
    )
    parser.add_argument(
        "--height", type=int, default=512, help="Height of generated image"
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize canvas
    canvas = Canvas(width=args.width, height=args.height)

    # Run the pipeline
    run_pipeline(
        canvas=canvas,
        prompt=args.prompt,
        style=args.style,
        scale=args.scale,
        output_dir=str(output_dir),
    )


if __name__ == "__main__":
    main()
