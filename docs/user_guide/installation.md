# Installation Guide

This guide will help you install LlamaCanvas and set up your environment.

## Requirements

LlamaCanvas requires the following:

- Python 3.8 or later
- pip (Python package installer)
- For video processing: ffmpeg
- For GPU acceleration: CUDA-compatible GPU (optional)

## Basic Installation

The simplest way to install LlamaCanvas is with pip:

```bash
pip install llama-canvas
```

This will install the core functionality of LlamaCanvas without optional dependencies.

## Installation with Options

LlamaCanvas offers several installation options for different use cases:

### API Server

To use the web UI and API server:

```bash
pip install llama-canvas[api]
```

### Claude Integration

To use Anthropic's Claude for multi-modal capabilities:

```bash
pip install llama-canvas[claude]
```

### Stable Diffusion Support

To use Stable Diffusion for image generation:

```bash
pip install llama-canvas[diffusion]
```

### Video Processing

To enable video creation and manipulation:

```bash
pip install llama-canvas[video]
```

### Development Environment

For development and contributing:

```bash
pip install llama-canvas[dev]
```

### Documentation

For building documentation locally:

```bash
pip install llama-canvas[docs]
```

### Full Installation

To install all features and dependencies:

```bash
pip install llama-canvas[full]
```

## Installation from Source

To install from source:

```bash
git clone https://github.com/llamasearch/llamacanvas.git
cd llamacanvas
pip install -e ".[full]"
```

## Verifying Installation

To verify that LlamaCanvas is installed correctly:

```bash
python -c "import llama_canvas; print(llama_canvas.__version__)"
```

You should see the version number printed out.

## Configuration

After installation, you may want to set up your configuration. Create a file at `~/.llama_canvas/config.json`:

```json
{
  "claude_api_key": "your-anthropic-api-key",
  "default_claude_model": "claude-3-opus-20240229",
  "default_image_width": 512,
  "default_image_height": 512
}
```

Alternatively, you can set environment variables:

```bash
export CLAUDE_API_KEY="your-anthropic-api-key"
export DEFAULT_CLAUDE_MODEL="claude-3-opus-20240229"
```

See the [Configuration Guide](configuration.md) for more details.

## Troubleshooting

### Common Issues

#### Missing dependencies

If you encounter errors about missing dependencies, try installing with the appropriate option:

```bash
pip install llama-canvas[full]
```

#### API key not found

If you get errors about missing API keys, make sure you've set them up in your configuration file or environment variables.

#### Video processing issues

For video processing issues, ensure ffmpeg is installed and available in your PATH:

```bash
# On Ubuntu/Debian
sudo apt-get install ffmpeg

# On macOS with Homebrew
brew install ffmpeg

# On Windows with Chocolatey
choco install ffmpeg
```

### Getting Help

If you encounter issues that aren't covered here:

1. Check the [GitHub Issues](https://github.com/llamasearch/llamacanvas/issues) to see if the problem has been reported
2. Search the [GitHub Discussions](https://github.com/llamasearch/llamacanvas/discussions) for similar problems
3. Open a new issue or discussion if needed 