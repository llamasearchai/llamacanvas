# Image Generation

LlamaCanvas provides multiple methods for generating images using AI models. This guide covers the various techniques available for creating images from text prompts or other inputs.

## Text-to-Image Generation

The most common way to generate images with LlamaCanvas is from text descriptions. LlamaCanvas supports multiple AI models for this purpose, including Claude and Stable Diffusion.

### Using Claude for Image Generation

Claude's multimodal capabilities allow it to generate high-quality images from text descriptions:

```python
from llama_canvas.core.canvas import Canvas

# Create a canvas
canvas = Canvas()

# Generate an image using Claude
image = canvas.generate_from_text(
    prompt="A photorealistic image of a futuristic city with flying cars and neon lights",
    model="claude-3-opus-20240229",  # Optional, uses default model if not specified
    width=1024,
    height=1024
)

# Save the generated image
image.save("futuristic_city.png")
```

Using the command line interface:

```bash
llama-canvas generate "A photorealistic image of a futuristic city with flying cars and neon lights" \
    --model claude-3-opus-20240229 \
    --width 1024 \
    --height 1024 \
    --output futuristic_city.png
```

### Using Stable Diffusion

For more control over the generation process, LlamaCanvas also supports Stable Diffusion:

```python
from llama_canvas.core.canvas import Canvas
from llama_canvas.agents.stable_diffusion import StableDiffusionAgent

# Create a canvas
canvas = Canvas()

# Generate an image using Stable Diffusion
image = canvas.generate_from_text(
    prompt="A mystical forest with glowing mushrooms and fairy lights",
    agent="stable_diffusion",
    model="runwayml/stable-diffusion-v1-5",  # Optional, uses default model if not specified
    width=512,
    height=512,
    params={
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "negative_prompt": "blurry, low quality, distorted"
    }
)

# Save the generated image
image.save("mystical_forest.png")
```

Using the command line interface:

```bash
llama-canvas generate "A mystical forest with glowing mushrooms and fairy lights" \
    --agent stable_diffusion \
    --model runwayml/stable-diffusion-v1-5 \
    --width 512 \
    --height 512 \
    --params '{"num_inference_steps": 50, "guidance_scale": 7.5, "negative_prompt": "blurry, low quality, distorted"}' \
    --output mystical_forest.png
```

## Advanced Prompt Engineering

Crafting effective prompts is a key skill for generating high-quality images. LlamaCanvas provides utilities to help with prompt engineering:

```python
from llama_canvas.utils.prompts import enhance_prompt, analyze_prompt

# Enhance a basic prompt with more details
enhanced_prompt = enhance_prompt(
    "A cat in a garden",
    style="photorealistic",
    detail_level="high",
    aspect="wide angle"
)
print(enhanced_prompt)
# Output: "A photorealistic high-detail wide angle shot of a cat in a garden, 
#          with fine fur details, natural lighting, and vibrant colors."

# Analyze a prompt for potential improvements
analysis = analyze_prompt("robot")
print(analysis)
# Output: {
#   "specificity": "low",
#   "suggestions": ["Add details about the robot's appearance", 
#                  "Specify the environment", 
#                  "Add information about lighting and atmosphere"]
# }
```

## Batch Generation

For generating multiple variations or a series of related images:

```python
from llama_canvas.core.canvas import Canvas

canvas = Canvas()

# Generate multiple variations of the same prompt
variations = canvas.generate_variations(
    prompt="A serene landscape with mountains",
    count=4,
    variation_strength=0.3
)

# Save all variations
for i, image in enumerate(variations):
    image.save(f"landscape_variation_{i}.png")

# Generate a series of related images
prompts = [
    "A forest at dawn",
    "A forest at midday",
    "A forest at sunset",
    "A forest at night"
]

series = canvas.generate_batch(prompts)

# Save the series
for i, image in enumerate(series):
    image.save(f"forest_{i}.png")
```

## Image-to-Image Generation

LlamaCanvas also supports generating images based on existing images:

```python
from llama_canvas.core.canvas import Canvas
from llama_canvas.core.image import Image

canvas = Canvas()
input_image = Image.load("input.png")

# Generate a new image based on the input image
output_image = canvas.generate_from_image(
    image=input_image,
    prompt="Convert this into a watercolor painting",
    strength=0.7  # How much to preserve of the original image (0-1)
)

output_image.save("watercolor_version.png")
```

Using the command line interface:

```bash
llama-canvas img2img input.png "Convert this into a watercolor painting" \
    --strength 0.7 \
    --output watercolor_version.png
```

## Inpainting and Outpainting

LlamaCanvas supports inpainting (filling in parts of an image) and outpainting (extending an image beyond its original boundaries):

```python
from llama_canvas.core.canvas import Canvas
from llama_canvas.core.image import Image

canvas = Canvas()
base_image = Image.load("portrait.png")

# Create a mask (white areas will be repainted)
mask = Image.create_mask(
    width=base_image.width,
    height=base_image.height
)
mask.draw_rectangle(x=100, y=50, width=200, height=300, color="white")

# Inpaint the masked area
inpainted = canvas.inpaint(
    image=base_image,
    mask=mask,
    prompt="A beautiful flower bouquet"
)
inpainted.save("portrait_with_flowers.png")

# Outpaint to extend the image
outpainted = canvas.outpaint(
    image=base_image,
    direction="right",
    extend_by=256,
    prompt="Continue the scene with a beach and ocean"
)
outpainted.save("portrait_extended.png")
```

## Using Templates and References

For more controlled generation, you can use templates and reference images:

```python
from llama_canvas.core.canvas import Canvas
from llama_canvas.core.image import Image

canvas = Canvas()

# Generate with a reference image for style
reference = Image.load("reference_style.png")
styled_image = canvas.generate_from_text(
    prompt="A castle on a hill",
    reference_image=reference,
    reference_mode="style"  # Use the reference for style only
)
styled_image.save("castle_styled.png")

# Use a template (image with a transparent area to fill)
template = Image.load("frame_template.png")
filled_template = canvas.fill_template(
    template=template,
    prompt="A portrait of a young woman with flowers in her hair"
)
filled_template.save("portrait_in_frame.png")
```

## Controlling Generation Parameters

LlamaCanvas allows fine-grained control over the generation process:

```python
from llama_canvas.core.canvas import Canvas

canvas = Canvas()

# Generate with specific parameters
image = canvas.generate_from_text(
    prompt="A surreal landscape with floating islands",
    width=768,
    height=512,
    params={
        # Claude parameters
        "temperature": 0.7,
        "max_tokens": 4000,
        
        # Stable Diffusion parameters
        "num_inference_steps": 75,
        "guidance_scale": 8.5,
        "scheduler": "DPMSolverMultistep",
        "seed": 42,  # For reproducible results
        "negative_prompt": "blurry, distorted, low quality, ugly"
    }
)
image.save("surreal_landscape.png")
```

## Managing Generation Results

LlamaCanvas provides utilities for managing and organizing generated images:

```python
from llama_canvas.core.canvas import Canvas
from llama_canvas.utils.gallery import Gallery

# Create a gallery to manage generated images
gallery = Gallery("my_generations")

# Generate and store images
canvas = Canvas()

for prompt in ["mountain landscape", "ocean sunset", "forest path"]:
    image = canvas.generate_from_text(prompt)
    
    # Add to gallery with metadata
    gallery.add(
        image,
        metadata={
            "prompt": prompt,
            "model": "claude-3-opus-20240229",
            "date": "2024-05-01"
        }
    )

# Save the gallery
gallery.save("my_gallery.json")

# Export all images
gallery.export_all("output_folder")

# Search the gallery
forest_images = gallery.search("forest")
```

By mastering these image generation techniques, you can create a wide variety of visual content with LlamaCanvas. Experiment with different models, prompts, and parameters to achieve your desired results. 