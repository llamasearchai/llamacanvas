# LlamaCanvas

Advanced AI-driven multi-modal generation platform with Claude API integration

[![PyPI version](https://badge.fury.io/py/llama-canvas.svg)](https://badge.fury.io/py/llama-canvas)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://llamasearchai.github.io/llamacanvas/)
[![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://pypi.org/project/llama-canvas/)

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

## Architecture

LlamaCanvas is built with a modular architecture that allows for flexibility and extensibility:

### Core Components

- **Canvas**: The central workspace where all operations take place
- **Agents**: Specialized components that handle specific AI tasks
- **Pipelines**: Chains of operations for complex workflows
- **Storage**: Manages saved images, projects, and galleries

```
┌─────────────────────────┐
│       Application       │
│  (CLI, Web UI, API)     │
└───────────┬─────────────┘
            │
┌───────────▼─────────────┐
│         Canvas          │
└───┬────────┬────────┬───┘
    │        │        │
┌───▼──┐ ┌───▼──┐ ┌───▼──┐
│Agents│ │Images│ │Video │
└──────┘ └──────┘ └──────┘
```

### Agent System

LlamaCanvas uses a flexible agent system to integrate with various AI models:

- **ClaudeAgent**: Handles Claude API integration for multimodal generation
- **StableDiffusionAgent**: Works with Stable Diffusion models
- **EnhancementAgent**: Specializes in image enhancement
- **VideoAgent**: Handles video generation and processing

Each agent implements a common interface but can have specialized parameters and capabilities.

## Roadmap

Our development roadmap for the coming months:

- **Q2 2024**:
  - Advanced style transfer algorithms
  - Improved image enhancement with fine-tuning options
  - Better integration with Claude 3 Opus for high-quality generations

- **Q3 2024**:
  - Video generation capabilities
  - Animation tools and transitions
  - Batch processing with advanced queuing

- **Q4 2024**:
  - Collaborative canvas features
  - Project sharing and galleries
  - Advanced pipeline designer in the UI

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

For detailed documentation, see [LlamaCanvas Documentation](https://llamasearchai.github.io/llamacanvas/).

## Contributing

We welcome contributions! Please check out our [contributing guidelines](CONTRIBUTING.md).

## Community

Join our community channels to get help, share your creations, and connect with other users:

- [Discord Server](https://discord.gg/llamacanvas)
- [Twitter/X](https://twitter.com/llamacanvas)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 