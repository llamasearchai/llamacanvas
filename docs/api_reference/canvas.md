# Canvas API Reference

The `Canvas` class is the central component of LlamaCanvas, providing methods for image and video generation, manipulation, and enhancement.

## Canvas Class

```python
from llama_canvas.core.canvas import Canvas
```

### Constructor

```python
Canvas(width=512, height=512, agent=None, config=None)
```

Creates a new Canvas instance.

#### Parameters

- **width** (`int`, optional): The default width for generated images. Defaults to 512.
- **height** (`int`, optional): The default height for generated images. Defaults to 512.
- **agent** (`Agent`, optional): The AI agent to use for generation. If not provided, a default agent will be used.
- **config** (`dict`, optional): Configuration options for the canvas.

#### Returns

A new `Canvas` instance.

#### Example

```python
from llama_canvas.core.canvas import Canvas

# Create a canvas with default settings
canvas = Canvas()

# Create a canvas with custom dimensions
canvas = Canvas(width=1024, height=768)

# Create a canvas with a specific agent
from llama_canvas.utils.claude import ClaudeAgent
claude_agent = ClaudeAgent(api_key="your-api-key")
canvas = Canvas(agent=claude_agent)
```

### Methods

#### generate_from_text

```python
generate_from_text(prompt, width=None, height=None, options=None)
```

Generates an image from a text prompt.

##### Parameters

- **prompt** (`str`): The text prompt describing the image to generate.
- **width** (`int`, optional): The width of the generated image. If not provided, uses the canvas default.
- **height** (`int`, optional): The height of the generated image. If not provided, uses the canvas default.
- **options** (`dict`, optional): Additional options for the generation process.

##### Returns

An `Image` object representing the generated image.

##### Example

```python
from llama_canvas.core.canvas import Canvas

canvas = Canvas()
image = canvas.generate_from_text(
    "A serene landscape with mountains and a lake at sunset"
)
image.save("landscape.png")
```

#### apply_style

```python
apply_style(image, style, strength=1.0, options=None)
```

Applies a style to an image.

##### Parameters

- **image** (`Image`): The image to apply the style to.
- **style** (`str`): The name of the style to apply, or a text description of the style.
- **strength** (`float`, optional): The strength of the style application, between 0.0 and 1.0. Defaults to 1.0.
- **options** (`dict`, optional): Additional options for the style application.

##### Returns

An `Image` object representing the styled image.

##### Example

```python
from llama_canvas.core.canvas import Canvas
from llama_canvas.core.image import Image

canvas = Canvas()
original = Image.load("input.png")
styled = canvas.apply_style(original, "Van Gogh")
styled.save("styled_output.png")
```

#### enhance

```python
enhance(image, scale=1.0, denoise=False, sharpen=False, options=None)
```

Enhances an image with AI-powered upscaling, denoising, and sharpening.

##### Parameters

- **image** (`Image`): The image to enhance.
- **scale** (`float`, optional): The factor by which to scale the image. Defaults to 1.0.
- **denoise** (`bool`, optional): Whether to apply denoising. Defaults to False.
- **sharpen** (`bool`, optional): Whether to apply sharpening. Defaults to False.
- **options** (`dict`, optional): Additional options for the enhancement process.

##### Returns

An `Image` object representing the enhanced image.

##### Example

```python
from llama_canvas.core.canvas import Canvas
from llama_canvas.core.image import Image

canvas = Canvas()
original = Image.load("input.png")
enhanced = canvas.enhance(original, scale=2.0, denoise=True, sharpen=True)
enhanced.save("enhanced_output.png")
```

#### create_video_from_prompts

```python
create_video_from_prompts(prompts, duration=10, fps=30, transition="fade", options=None)
```

Creates a video from a sequence of text prompts.

##### Parameters

- **prompts** (`list[str]`): A list of text prompts to generate images from.
- **duration** (`float`, optional): The duration of the video in seconds. Defaults to 10.
- **fps** (`int`, optional): The frames per second of the video. Defaults to 30.
- **transition** (`str`, optional): The type of transition between images. Options include "fade", "dissolve", "slide", "zoom". Defaults to "fade".
- **options** (`dict`, optional): Additional options for the video creation process.

##### Returns

A `Video` object representing the created video.

##### Example

```python
from llama_canvas.core.canvas import Canvas

canvas = Canvas()
prompts = [
    "A serene landscape with mountains at dawn",
    "A serene landscape with mountains at noon",
    "A serene landscape with mountains at sunset",
    "A serene landscape with mountains at night"
]
video = canvas.create_video_from_prompts(prompts, duration=10, fps=30, transition="fade")
video.save("landscape_timelapse.mp4")
```

#### create_video_from_images

```python
create_video_from_images(images, duration=10, fps=30, transition="fade", options=None)
```

Creates a video from a sequence of images.

##### Parameters

- **images** (`list[Image]`): A list of Image objects to create a video from.
- **duration** (`float`, optional): The duration of the video in seconds. Defaults to 10.
- **fps** (`int`, optional): The frames per second of the video. Defaults to 30.
- **transition** (`str`, optional): The type of transition between images. Options include "fade", "dissolve", "slide", "zoom". Defaults to "fade".
- **options** (`dict`, optional): Additional options for the video creation process.

##### Returns

A `Video` object representing the created video.

##### Example

```python
from llama_canvas.core.canvas import Canvas
from llama_canvas.core.image import Image

canvas = Canvas()
images = [
    Image.load("dawn.png"),
    Image.load("noon.png"),
    Image.load("sunset.png"),
    Image.load("night.png")
]
video = canvas.create_video_from_images(images, duration=10, fps=30, transition="fade")
video.save("landscape_timelapse.mp4")
```

## Usage with Different Agents

### Claude Agent

```python
from llama_canvas.core.canvas import Canvas
from llama_canvas.utils.claude import ClaudeAgent

claude_agent = ClaudeAgent(api_key="your-api-key", model="claude-3-opus-20240229")
canvas = Canvas(agent=claude_agent)

image = canvas.generate_from_text("A robot artist painting a landscape")
image.save("claude_generated.png")
```

### Stable Diffusion Agent

```python
from llama_canvas.core.canvas import Canvas
from llama_canvas.utils.stable_diffusion import StableDiffusionAgent

sd_agent = StableDiffusionAgent(model="stabilityai/stable-diffusion-xl-base-1.0")
canvas = Canvas(agent=sd_agent)

image = canvas.generate_from_text("A robot artist painting a landscape")
image.save("sd_generated.png")
``` 