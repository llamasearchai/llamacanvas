"""
Tests for utility functions in LlamaCanvas.

This module contains tests for various utility functions across the package.
"""

import os
import tempfile
import logging
import json
import io
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock, mock_open

from llama_canvas.utils.logging import get_logger, setup_logging
from llama_canvas.utils.config import settings, Settings
from llama_canvas.utils.validators import validate_image_size, validate_aspect_ratio
from llama_canvas.utils.file_utils import ensure_directory, get_file_extension, safe_filename


class TestLogging:
    """Tests for the logging utilities."""

    def test_get_logger(self):
        """Test getting a logger instance."""
        logger = get_logger("test_module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
        
        # Test with different log level
        with patch('llama_canvas.utils.logging.settings') as mock_settings:
            mock_settings.get.return_value = "DEBUG"
            logger = get_logger("test_debug")
            assert logger.level == logging.DEBUG

    def test_get_logger_with_handlers(self):
        """Test getting a logger that already has handlers."""
        # Create a logger with handlers
        logger_name = "test_with_handlers"
        logger = logging.getLogger(logger_name)
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        
        # Get the same logger through our function
        result = get_logger(logger_name)
        
        # Should return the same logger without adding handlers
        assert result is logger
        assert len(logger.handlers) == 1
        
        # Clean up
        logger.removeHandler(handler)

    def test_setup_logging(self):
        """Test setting up logging."""
        with patch('logging.basicConfig') as mock_basic_config:
            setup_logging(level="INFO")
            
            mock_basic_config.assert_called_once()
            args, kwargs = mock_basic_config.call_args
            assert kwargs["level"] == logging.INFO
            
            # Test custom format
            mock_basic_config.reset_mock()
            custom_format = "%(asctime)s - %(name)s - %(message)s"
            setup_logging(level="ERROR", log_format=custom_format)
            
            mock_basic_config.assert_called_once()
            args, kwargs = mock_basic_config.call_args
            assert kwargs["level"] == logging.ERROR
            assert kwargs["format"] == custom_format

    def test_setup_logging_with_file(self):
        """Test setting up logging with a log file."""
        with patch('logging.FileHandler') as mock_file_handler, \
             patch('logging.StreamHandler') as mock_stream_handler, \
             patch('logging.getLogger') as mock_get_logger:
            
            mock_root = MagicMock()
            mock_get_logger.return_value = mock_root
            
            # Setup logging with file
            setup_logging(log_file="/tmp/test.log", log_level="WARNING")
            
            # Check handlers added
            assert mock_stream_handler.called
            assert mock_file_handler.called
            assert mock_file_handler.call_args[0][0] == "/tmp/test.log"
            
            # Verify log levels
            handlers = mock_root.addHandler.call_args_list
            assert len(handlers) >= 2  # At least console and file handlers
            
            # Check that loggers were suppressed
            suppressed_calls = mock_get_logger.call_args_list
            suppressed_loggers = [args[0][0] for args in suppressed_calls if args[0][0] in ("PIL", "urllib3", "matplotlib")]
            assert len(suppressed_loggers) >= 3


class TestConfig:
    """Tests for the configuration utilities."""

    def test_settings_init(self):
        """Test Settings initialization."""
        with patch('llama_canvas.utils.config.Settings._load_environment') as mock_load_env, \
             patch('llama_canvas.utils.config.Settings._load_config_file') as mock_load_config:
            
            test_settings = Settings()
            
            # Should call both load methods
            assert mock_load_env.called
            assert mock_load_config.called
            
            # Should initialize empty settings dict
            assert test_settings._settings == {}

    def test_settings_load_environment(self):
        """Test loading settings from environment variables."""
        with patch.dict('os.environ', {
            'ANTHROPIC_API_KEY': 'test_api_key',
            'LOG_LEVEL': 'DEBUG',
            'MODEL_CACHE_DIR': '/tmp/cache',
            'DEFAULT_CLAUDE_MODEL': 'claude-3-opus'
        }), patch('llama_canvas.utils.config.load_dotenv'):
            
            test_settings = Settings()
            test_settings._load_environment()
            
            # Should load values from environment
            assert test_settings._settings['ANTHROPIC_API_KEY'] == 'test_api_key'
            assert test_settings._settings['LOG_LEVEL'] == 'DEBUG'
            assert test_settings._settings['MODEL_CACHE_DIR'] == '/tmp/cache'
            assert test_settings._settings['DEFAULT_CLAUDE_MODEL'] == 'claude-3-opus'

    def test_settings_load_config_file(self):
        """Test loading settings from config file."""
        test_config = {
            'default_image_width': 1024,
            'default_image_height': 768,
            'log_level': 'INFO'
        }
        
        # Mock path.exists to return True only for a specific path
        def mock_exists(path):
            return str(path) == str(Path.home() / ".config" / "llama_canvas" / "config.json")
        
        with patch('pathlib.Path.exists', side_effect=mock_exists), \
             patch('builtins.open', mock_open(read_data=json.dumps(test_config))):
            
            test_settings = Settings()
            test_settings._settings = {}  # Reset settings
            test_settings._load_config_file()
            
            # Should load values from config
            assert test_settings._settings['default_image_width'] == 1024
            assert test_settings._settings['default_image_height'] == 768
            assert test_settings._settings['log_level'] == 'INFO'

    def test_settings_load_config_file_error(self):
        """Test handling errors when loading config file."""
        # Mock path.exists to return True
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', side_effect=IOError("Test error")), \
             patch('builtins.print') as mock_print:
            
            test_settings = Settings()
            test_settings._settings = {}  # Reset settings
            test_settings._load_config_file()
            
            # Should handle error and print message
            assert mock_print.called
            assert "Error loading config" in mock_print.call_args[0][0]
            
            # Settings should remain empty
            assert test_settings._settings == {}

    def test_settings_get(self):
        """Test getting settings values."""
        test_settings = Settings()
        test_settings._settings = {'test_key': 'test_value'}
        
        # Get existing value
        assert test_settings.get('test_key') == 'test_value'
        
        # Get non-existent value
        assert test_settings.get('nonexistent_key') is None
        
        # Get non-existent value with default
        assert test_settings.get('nonexistent_key', 'default_value') == 'default_value'

    def test_settings_set(self):
        """Test setting settings values."""
        test_settings = Settings()
        test_settings._settings = {}
        
        # Set new value
        test_settings.set('new_key', 'new_value')
        assert test_settings._settings['new_key'] == 'new_value'
        
        # Update existing value
        test_settings.set('new_key', 'updated_value')
        assert test_settings._settings['new_key'] == 'updated_value'

    def test_settings_save(self):
        """Test saving settings to a file."""
        test_settings = Settings()
        test_settings._settings = {'test_key': 'test_value'}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "test_config.json"
            
            # Save settings
            test_settings.save(test_path)
            
            # Verify file was created
            assert test_path.exists()
            
            # Verify contents
            with open(test_path, 'r') as f:
                saved_config = json.load(f)
                assert saved_config == {'test_key': 'test_value'}

    def test_settings_save_error(self):
        """Test handling errors when saving settings."""
        test_settings = Settings()
        test_settings._settings = {'test_key': 'test_value'}
        
        # Mock open to raise IOError
        with patch('builtins.open', side_effect=IOError("Test error")), \
             patch('builtins.print') as mock_print, \
             patch('pathlib.Path.mkdir'):
            
            test_settings.save(Path("/nonexistent/path/config.json"))
            
            # Should handle error and print message
            assert mock_print.called
            assert "Error saving config" in mock_print.call_args[0][0]

    def test_settings_dict_access(self):
        """Test dictionary-style access to settings."""
        test_settings = Settings()
        test_settings._settings = {'test_key': 'test_value'}
        
        # Get using dictionary syntax
        assert test_settings['test_key'] == 'test_value'
        
        # Set using dictionary syntax
        test_settings['new_key'] = 'new_value'
        assert test_settings._settings['new_key'] == 'new_value'


def test_validate_image_size():
    """Test validating image dimensions."""
    # Valid size
    assert validate_image_size(512, 512) == (512, 512)
    
    # Below minimum
    with pytest.raises(ValueError):
        validate_image_size(10, 512)
    
    # Above maximum
    with pytest.raises(ValueError):
        validate_image_size(512, 10000)
    
    # Default values when None provided
    with patch('llama_canvas.utils.validators.settings') as mock_settings:
        mock_settings.default_image_width = 800
        mock_settings.default_image_height = 600
        
        assert validate_image_size(None, None) == (800, 600)
        assert validate_image_size(400, None) == (400, 600)
        assert validate_image_size(None, 400) == (800, 400)


def test_validate_aspect_ratio():
    """Test validating aspect ratio."""
    # Valid aspect ratio
    assert validate_aspect_ratio(800, 600) == (800, 600)
    
    # Extreme aspect ratio
    with pytest.raises(ValueError):
        validate_aspect_ratio(2000, 100)  # 20:1 ratio
    
    # Adjust ratio to fit within limits
    corrected_width, corrected_height = validate_aspect_ratio(800, 100, adjust=True)
    
    # Should adjust to be within the max ratio (probably 4:1 or similar)
    assert corrected_width < 800 or corrected_height > 100
    assert 0.25 <= corrected_width / corrected_height <= 4.0  # Common limit


def test_ensure_directory():
    """Test ensuring a directory exists."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with existing directory
        assert ensure_directory(temp_dir) == temp_dir
        
        # Test with new subdirectory
        new_dir = os.path.join(temp_dir, "new_subdir")
        assert ensure_directory(new_dir) == new_dir
        assert os.path.exists(new_dir)
        
        # Test with file path
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as temp_file:
            # Should return the directory containing the file
            assert ensure_directory(temp_file.name) == temp_dir


def test_get_file_extension():
    """Test getting file extension."""
    assert get_file_extension("image.png") == ".png"
    assert get_file_extension("path/to/document.pdf") == ".pdf"
    assert get_file_extension("file_without_extension") == ""
    assert get_file_extension("path/to/.hidden") == ".hidden"
    
    # With normalization
    assert get_file_extension("IMAGE.PNG", normalize=True) == ".png"
    
    # With default when no extension
    assert get_file_extension("no_extension", default=".txt") == ".txt"


def test_safe_filename():
    """Test creating safe filenames."""
    assert safe_filename("hello world") == "hello_world"
    assert safe_filename("file/with\\invalid:chars?") == "file_with_invalid_chars_"
    
    # With timestamp
    with patch('llama_canvas.utils.file_utils.datetime') as mock_datetime:
        mock_datetime.now.return_value.strftime.return_value = "20240320_123456"
        
        assert safe_filename("test", add_timestamp=True) == "test_20240320_123456"
    
    # With maximum length
    long_name = "a" * 100
    assert len(safe_filename(long_name, max_length=50)) <= 50


if __name__ == "__main__":
    pytest.main() 