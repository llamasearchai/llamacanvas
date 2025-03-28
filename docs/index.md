# LlamaCanvas Documentation

Welcome to the LlamaCanvas documentation. LlamaCanvas is an advanced AI-driven multi-modal generation platform with Claude API integration.

<div align="center">
  <img src="../resources/images/llamacanvas_logo.png" alt="LlamaCanvas Logo" width="300">
</div>

## What is LlamaCanvas?

LlamaCanvas provides tools for creating and transforming visual content using state-of-the-art AI models. It integrates with Claude API and offers a comprehensive platform for image and video generation, manipulation, and enhancement.

## Key Features

- **Image Generation**: Create images from text prompts using Stable Diffusion or Claude
- **Image Enhancement**: Upscale, denoise, and improve images with AI
- **Style Transfer**: Apply artistic styles to existing images
- **Video Processing**: Generate and manipulate video content
- **Extensible Agent System**: Integrate different AI models with a unified interface
- **Multiple Interfaces**: Web UI, CLI, and Python API options
- **Claude Integration**: Leverage Anthropic's Claude for multi-modal generation
- **Pipeline Architecture**: Chain operations for complex transformations

## Installation

```bash
# Basic installation
pip install llama-canvas

# With all features
pip install llama-canvas[full]
```

For more detailed installation options, see the [Installation Guide](user_guide/installation.md).

## Quick Start

### Generate an image from text

```python
from llama_canvas.core.canvas import Canvas

canvas = Canvas(width=512, height=512)
image = canvas.generate_from_text(
    "A serene landscape with mountains and a lake at sunset"
)
image.save("landscape.png")
```

### Apply style transfer

```python
from llama_canvas.core.canvas import Canvas
from llama_canvas.core.image import Image

canvas = Canvas()
original = Image.load("input.png")
styled = canvas.apply_style(original, "Van Gogh")
styled.save("styled_output.png")
```

For more examples, check out the [Quick Start Guide](user_guide/quickstart.md) or browse the [Examples](examples/basic_usage.md) section.

## Documentation Structure

- **User Guide**: Detailed explanations of LlamaCanvas concepts and features
- **API Reference**: Comprehensive documentation of the Python API
- **Examples**: Sample code and usage patterns
- **Contributing**: Guidelines for contributing to LlamaCanvas

## Support and Community

- GitHub Issues: [Report bugs or request features](https://github.com/llamasearch/llamacanvas/issues)
- GitHub Discussions: [Ask questions and share ideas](https://github.com/llamasearch/llamacanvas/discussions)

## License

LlamaCanvas is released under the MIT License. See the [LICENSE](https://github.com/llamasearch/llamacanvas/blob/main/LICENSE) file for details. 