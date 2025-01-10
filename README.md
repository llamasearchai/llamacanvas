# LlamaCanvas

Advanced AI-driven multi-modal generation platform with Claude API integration

[![PyPI version](https://badge.fury.io/py/llama-canvas.svg)](https://badge.fury.io/py/llama-canvas)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

LlamaCanvas is a comprehensive platform for AI-powered image and video generation, manipulation, and enhancement. It integrates with Claude API and provides tools for creating and transforming visual content using state-of-the-art AI models.

Key features:
- Image generation from text prompts using Stable Diffusion or Claude
- Image enhancement, style transfer, and manipulation
- Video creation and processing
- Extensible agent system for integrating different AI models
- Web UI and CLI interfaces
- Python API for seamless integration into your applications

## Installation

```bash
# Basic installation
pip install llama-canvas

# With API server dependencies
pip install llama-canvas[api]

# With Claude integration
pip install llama-canvas[claude]

# With Stable Diffusion support
pip install llama-canvas[diffusion]

# With video processing capabilities
pip install llama-canvas[video]

# Full installation with all features
pip install llama-canvas[full]
```

## Quick Start

### Command Line Interface

Generate an image from a text prompt:

```bash
llama-canvas generate "A serene landscape with mountains and a lake at sunset" --output landscape.png
```

Apply a style to an image:

```bash
llama-canvas style input.png "Van Gogh" --output styled.png
```

Enhance image resolution:

```bash
llama-canvas enhance input.png --scale 2 --output enhanced.png
```

Run the web UI:

```bash
llama-canvas ui --browse
```

### Python API

```python
from llama_canvas.core.canvas import Canvas
from llama_canvas.core.image import Image

# Create a canvas
canvas = Canvas(width=512, height=512)

# Generate an image
image = canvas.generate_from_text("A serene landscape with mountains and a lake at sunset")

# Apply a style
styled = canvas.apply_style(image, "Van Gogh")

# Save the result
styled.save("styled_landscape.png")
```

## Web UI

LlamaCanvas includes a web UI built with FastAPI, allowing you to generate and manipulate images through a browser interface. To start the web UI:

```bash
llama-canvas ui --browse
```

This will start the server and open a browser window to the UI.

## Configuration

LlamaCanvas can be configured via environment variables or a configuration file. Create a file at `~/.llama_canvas/config.json` with your settings:

```json
{
  "claude_api_key": "your-anthropic-api-key",
  "default_claude_model": "claude-3-opus-20240229",
  "default_image_width": 512,
  "default_image_height": 512
}
```

Alternatively, set environment variables:

```bash
export CLAUDE_API_KEY="your-anthropic-api-key"
export DEFAULT_CLAUDE_MODEL="claude-3-opus-20240229"
```

## Documentation

For detailed documentation, see [LlamaCanvas Documentation](https://llamasearch.ai

## Contributing

Contributions are welcome! Please check out our [contributing guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
# Updated in commit 1 - 2025-04-04 17:00:28

# Updated in commit 9 - 2025-04-04 17:00:29

# Updated in commit 17 - 2025-04-04 17:00:31

# Updated in commit 25 - 2025-04-04 17:00:32

# Updated in commit 1 - 2025-04-05 14:24:31

# Updated in commit 9 - 2025-04-05 14:24:31
