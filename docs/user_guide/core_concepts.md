# Core Concepts

This page introduces the fundamental concepts of LlamaCanvas that form the foundation of the library.

## Canvas

The `Canvas` is the central concept in LlamaCanvas, representing a workspace where AI-powered image and video generation happens. The Canvas manages resources, coordinates between different agents, and provides a consistent interface for all operations.

A Canvas can contain multiple layers, each potentially representing a different image or graphical element, and they can be manipulated together to create complex compositions.

```python
from llama_canvas.core.canvas import Canvas

# Create a canvas with specific dimensions
canvas = Canvas(width=1024, height=768)

# Generate content on the canvas
canvas.generate_from_text("A beautiful mountain landscape at sunset")
```

## Agents

Agents in LlamaCanvas are specialized components that handle specific AI tasks. Each agent integrates with a different AI model or service to provide capabilities like:

- Text-to-image generation
- Image-to-image transformation
- Style transfer
- Image enhancement
- Video generation

The most important agents in LlamaCanvas include:

- `ClaudeAgent`: Integrates with Anthropic's Claude API for multi-modal generation
- `StableDiffusionAgent`: Uses Stable Diffusion models for high-quality image generation
- `EnhancementAgent`: Specializes in image enhancement and upscaling
- `VideoAgent`: Handles video creation and processing

Agents are typically managed by the Canvas, but can also be used directly for specialized workflows.

```python
from llama_canvas.agents.claude import ClaudeAgent

# Create a Claude agent with a specific model
agent = ClaudeAgent(model="claude-3-opus-20240229")

# Use the agent directly
response = agent.generate(prompt="Create an image of a futuristic city")
```

## Images

The `Image` class represents image data within LlamaCanvas. It provides methods for manipulation, analysis, and export of images, including:

- Loading and saving images in various formats
- Basic transformations (resize, crop, rotate)
- Converting between different color spaces
- Applying filters and effects
- Extracting information from images (color palette, dominant colors)

```python
from llama_canvas.core.image import Image

# Load an image
img = Image.load("input.png")

# Resize the image
resized = img.resize(width=512, height=512)

# Apply a filter
filtered = resized.apply_filter("enhance_colors")

# Save the result
filtered.save("output.png")
```

## Layers

Layers allow for non-destructive editing and complex compositions. Each layer can contain an image or other graphical elements, and has properties like:

- Opacity
- Blend mode
- Position and transformation
- Masks and effects

Layers can be added, removed, reordered, and modified within a Canvas.

```python
from llama_canvas.core.canvas import Canvas
from llama_canvas.core.image import Image

canvas = Canvas(width=1024, height=768)

# Add a background layer
background = Image.load("background.png")
canvas.add_layer(background, name="Background")

# Add a foreground element
foreground = Image.load("foreground.png")
canvas.add_layer(foreground, name="Foreground", opacity=0.8, blend_mode="overlay")

# Adjust a layer
canvas.adjust_layer("Foreground", position=(100, 100), scale=0.5)
```

## Styles

Styles in LlamaCanvas define the visual appearance and characteristics that can be applied to images. Styles encompass:

- Artistic styles (like Impressionism, Cubism, etc.)
- Visual effects (like HDR, vignette, film grain)
- Color transformations (like sepia, black and white, color grade)

Styles can be applied using various agents and methods.

```python
from llama_canvas.core.canvas import Canvas

canvas = Canvas()
image = canvas.load_image("input.png")

# Apply an artistic style
styled = canvas.apply_style(image, "impressionist_painting")

# Apply a visual effect
with_effect = canvas.apply_effect(styled, "film_grain", intensity=0.5)
```

## Pipelines

Pipelines allow you to chain together multiple operations into a reusable workflow. A pipeline can include:

- Image generation
- Transformations and effects
- Style applications
- Export operations

Pipelines can be saved, loaded, and shared between projects.

```python
from llama_canvas.core.pipeline import Pipeline
from llama_canvas.core.canvas import Canvas

# Create a pipeline
pipeline = Pipeline("landscape_generator")

# Add steps to the pipeline
pipeline.add_step("generate", {
    "prompt": "A serene landscape with mountains and a lake at sunset",
    "width": 1024,
    "height": 768
})
pipeline.add_step("apply_style", {
    "style": "oil_painting"
})
pipeline.add_step("enhance", {
    "method": "super_resolution",
    "scale": 2
})

# Execute the pipeline
canvas = Canvas()
result = pipeline.execute(canvas)
result.save("landscape.png")
```

## Projects

A Project in LlamaCanvas represents a complete creative work, potentially containing multiple canvases, resources, and metadata. Projects can be saved to disk and loaded later to continue work.

```python
from llama_canvas.core.project import Project

# Create a new project
project = Project("my_artwork")

# Add a canvas to the project
canvas = project.create_canvas(width=1024, height=768)

# Work with the canvas
canvas.generate_from_text("A beautiful mountain landscape at sunset")

# Save the project
project.save("/path/to/projects/my_artwork.llca")
```

## Export Formats

LlamaCanvas supports multiple export formats for sharing and using your creations:

- Common image formats (PNG, JPEG, WebP, TIFF)
- Video formats (MP4, GIF)
- Vector formats (SVG)
- Project format (.llca) for saving the complete project state

Each format has specific options for quality, compression, and metadata.

```python
from llama_canvas.core.canvas import Canvas

canvas = Canvas()
# ... create content on the canvas ...

# Export in different formats
canvas.export("output.png", format="png")
canvas.export("output.jpg", format="jpeg", quality=95)
canvas.export("output.gif", format="gif", fps=10)
```

Understanding these core concepts will help you navigate the LlamaCanvas library and build powerful image and video generation workflows. 