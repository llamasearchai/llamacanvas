"""
Configuration utilities for LlamaCanvas.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


class Settings:
    """Class to manage application settings."""

    def __init__(self):
        """Initialize settings from environment and config file."""
        self._settings: Dict[str, Any] = {}
        self._load_environment()
        self._load_config_file()

    def _load_environment(self) -> None:
        """Load settings from environment variables."""
        # Load .env file if present
        load_dotenv()

        # Map environment variables to settings
        env_mapping = {
            "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
            "LOG_LEVEL": "LOG_LEVEL",
            "MODEL_CACHE_DIR": "MODEL_CACHE_DIR",
            "DEFAULT_CLAUDE_MODEL": "DEFAULT_CLAUDE_MODEL",
        }

        for env_var, setting_name in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                self._settings[setting_name] = value

    def _load_config_file(self) -> None:
        """Load settings from config file if present."""
        config_paths = [
            Path.home() / ".llama_canvas" / "config.json",
            Path.home() / ".config" / "llama_canvas" / "config.json",
            Path("config.json"),
        ]

        for path in config_paths:
            if path.exists():
                try:
                    with open(path, "r") as f:
                        config = json.load(f)
                        self._settings.update(config)
                    break
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error loading config from {path}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value.

        Args:
            key: Setting key
            default: Default value if setting not found

        Returns:
            Setting value or default
        """
        return self._settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a setting value.

        Args:
            key: Setting key
            value: Setting value
        """
        self._settings[key] = value

    def save(self, path: Optional[Path] = None) -> None:
        """
        Save settings to config file.

        Args:
            path: Path to save to (default: ~/.llama_canvas/config.json)
        """
        if path is None:
            path = Path.home() / ".llama_canvas" / "config.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, "w") as f:
                json.dump(self._settings, f, indent=2)
        except IOError as e:
            print(f"Error saving config to {path}: {e}")

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to settings."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-style setting of values."""
        self.set(key, value)


# Create global settings instance
settings = Settings()
