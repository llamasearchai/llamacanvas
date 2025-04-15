# Configuration

LlamaCanvas offers multiple ways to configure the library to suit your needs. This page outlines the available configuration options and methods to apply them.

## Configuration Methods

LlamaCanvas provides several methods to configure the library, listed in order of precedence:

1. **Runtime Configuration**: Parameters passed directly to methods and classes
2. **Environment Variables**: System-wide environment variables
3. **Configuration Files**: JSON configuration files in specific locations
4. **Default Values**: Built-in fallback values

## Configuration File

The primary way to configure LlamaCanvas is through a JSON configuration file located at `~/.llama_canvas/config.json`. Here's an example configuration file:

```json
{
  "claude_api_key": "your-anthropic-api-key",
  "default_claude_model": "claude-3-opus-20240229",
  "default_image_width": 512,
  "default_image_height": 512,
  "default_output_dir": "~/llama_canvas_output",
  "log_level": "INFO",
  "agent_config": {
    "claude": {
      "timeout": 60,
      "max_retries": 3
    },
    "stable_diffusion": {
      "device": "cuda",
      "precision": "fp16"
    }
  }
}
```

## Environment Variables

You can also configure LlamaCanvas using environment variables. These will override values in the configuration file. The following environment variables are supported:

| Environment Variable | Description | Default Value |
|---------------------|-------------|---------------|
| `CLAUDE_API_KEY` | Your Anthropic API key | None |
| `DEFAULT_CLAUDE_MODEL` | Default Claude model to use | "claude-3-opus-20240229" |
| `DEFAULT_IMAGE_WIDTH` | Default image width | 512 |
| `DEFAULT_IMAGE_HEIGHT` | Default image height | 512 |
| `DEFAULT_OUTPUT_DIR` | Default directory for saving outputs | "~/llama_canvas_output" |
| `LLAMA_CANVAS_LOG_LEVEL` | Logging level | "INFO" |

Example of setting environment variables:

```bash
export CLAUDE_API_KEY="your-anthropic-api-key"
export DEFAULT_CLAUDE_MODEL="claude-3-opus-20240229"
```

## Configuration Options

Below is a detailed list of all configuration options available in LlamaCanvas:

### General Options

| Option | Description | Default Value |
|--------|-------------|---------------|
| `default_image_width` | Default width for generated images | 512 |
| `default_image_height` | Default height for generated images | 512 |
| `default_output_dir` | Default directory for saving outputs | "~/llama_canvas_output" |
| `log_level` | Logging level | "INFO" |

### Claude Integration

| Option | Description | Default Value |
|--------|-------------|---------------|
| `claude_api_key` | Your Anthropic API key | None |
| `default_claude_model` | Default Claude model to use | "claude-3-opus-20240229" |
| `claude_timeout` | Timeout for Claude API requests (seconds) | 60 |
| `claude_max_retries` | Maximum number of retries for Claude API requests | 3 |

### Stable Diffusion Options

| Option | Description | Default Value |
|--------|-------------|---------------|
| `sd_device` | Device to run Stable Diffusion on ("cuda", "cpu") | "cuda" if available, else "cpu" |
| `sd_precision` | Precision for Stable Diffusion models ("fp16", "fp32") | "fp16" |
| `sd_default_model` | Default Stable Diffusion model | "runwayml/stable-diffusion-v1-5" |

### Web UI Options

| Option | Description | Default Value |
|--------|-------------|---------------|
| `ui_host` | Host address for the web UI | "127.0.0.1" |
| `ui_port` | Port for the web UI | 8080 |
| `ui_theme` | Theme for the web UI ("light", "dark") | "light" |

## Programmatic Configuration

You can also configure LlamaCanvas programmatically:

```python
from llama_canvas.utils.config import Config

# Load the default configuration
config = Config()

# Update specific values
config.update({
    "default_image_width": 1024,
    "default_image_height": 1024,
    "claude_api_key": "your-anthropic-api-key"
})

# Apply the configuration to the current session
config.apply()
```

## Configuration Validation

LlamaCanvas automatically validates your configuration when it's loaded. If there are any issues, warnings will be logged, and default values will be used where possible. If critical configuration options are missing (like API keys when needed), appropriate exceptions will be raised. 