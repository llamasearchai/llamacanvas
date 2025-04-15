# Quick Start Guide

This guide provides a quick introduction to using LlamaCanvas for various AI-driven image and video tasks.

## Basic Usage

### Command Line Interface

LlamaCanvas provides a simple command-line interface for common operations.

#### Generate an image from text:

```bash
llama-canvas generate "A serene landscape with mountains and a lake at sunset" --output landscape.png
```

#### Apply a style to an image:

```bash
llama-canvas style input.png "Van Gogh" --output styled.png
```

#### Enhance image resolution:

```bash
llama-canvas enhance input.png --scale 2 --output enhanced.png
```

#### Launch the web UI:

```bash
llama-canvas ui --browse
```

### Python API

For more complex tasks, you can use the Python API:

#### Image Generation

```python
from llama_canvas.core.canvas import Canvas

# Create a canvas
canvas = Canvas(width=512, height=512)

# Generate an image from text
image = canvas.generate_from_text(
    "A serene landscape with mountains and a lake at sunset"
)

# Save the result
image.save("landscape.png")
```

#### Style Transfer

```python
from llama_canvas.core.canvas import Canvas
from llama_canvas.core.image import Image

# Create a canvas
canvas = Canvas()

# Load an image
original = Image.load("input.png")

# Apply a style
styled = canvas.apply_style(original, "Van Gogh")

# Save the result
styled.save("styled_output.png")
```

#### Image Enhancement

```python
from llama_canvas.core.canvas import Canvas
from llama_canvas.core.image import Image

# Create a canvas
canvas = Canvas()

# Load an image
original = Image.load("input.png")

# Enhance the image
enhanced = canvas.enhance(original, scale=2, denoise=True)

# Save the result
enhanced.save("enhanced_output.png")
```

#### Video Creation

```python
from llama_canvas.core.canvas import Canvas
from llama_canvas.core.video import Video

# Create a canvas
canvas = Canvas()

# Generate a video from a sequence of prompts
prompts = [
    "A serene landscape with mountains at dawn",
    "A serene landscape with mountains at noon",
    "A serene landscape with mountains at sunset",
    "A serene landscape with mountains at night"
]

# Create video with transitions between images
video = canvas.create_video_from_prompts(
    prompts, 
    duration=10,
    fps=30,
    transition="fade"
)

# Save the result
video.save("landscape_timelapse.mp4")
```

## Using the Web UI

LlamaCanvas includes a web UI for interactive use:

1. Start the server:
   ```bash
   llama-canvas ui --browse
   ```

2. This will open a browser window to the UI (usually at http://localhost:8000)

3. Use the interface to:
   - Generate images from text
   - Apply styles to uploaded images
   - Enhance images
   - Create videos
   - View your creation history

## Working with Claude

If you have installed LlamaCanvas with Claude integration:

```python
from llama_canvas.core.canvas import Canvas
from llama_canvas.utils.claude import ClaudeAgent

# Configure Claude
claude_agent = ClaudeAgent(api_key="your-api-key", model="claude-3-opus-20240229")

# Create a canvas with Claude agent
canvas = Canvas(agent=claude_agent)

# Generate an image with Claude
image = canvas.generate_from_text(
    "A robot artist painting a landscape with digital brushes"
)

# Save the result
image.save("claude_generated.png")
```

## Creating Pipelines

LlamaCanvas allows you to chain operations into pipelines:

```python
from llama_canvas.core.canvas import Canvas
from llama_canvas.core.image import Image
from llama_canvas.utils.pipeline import Pipeline

# Create a canvas
canvas = Canvas()

# Create a pipeline
pipeline = Pipeline(canvas)

# Add operations to the pipeline
pipeline.add_operation("generate", prompt="A mountain landscape")
pipeline.add_operation("apply_style", style="Van Gogh")
pipeline.add_operation("enhance", scale=1.5, denoise=True)

# Execute the pipeline
result = pipeline.execute()

# Save the result
result.save("pipeline_result.png")
```

## Next Steps

For more detailed information, explore:

- [Core Concepts](core_concepts.md) - Learn about the fundamental concepts of LlamaCanvas
- [Image Generation](image_generation.md) - Detailed guide on generating images
- [Style Transfer](style_transfer.md) - Learn about available styles and customization
- [Image Enhancement](image_enhancement.md) - Advanced image enhancement techniques
- [Video Processing](video_processing.md) - In-depth video creation and editing guide
- [API Reference](../api_reference/canvas.md) - Complete API documentation 