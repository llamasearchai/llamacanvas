"""
Tests for accessibility features in LlamaCanvas.

This module contains tests for accessibility features such as
screen reader support, keyboard navigation, and color contrast utilities.
"""

import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np
from PIL import Image as PILImage

from llama_canvas.accessibility import (
    AccessibilityManager,
    ScreenReaderSupport,
    KeyboardNavigator,
    ColorContrastChecker,
    TextToSpeech,
    VoiceCommand,
    AccessibilitySettings,
    get_color_contrast_ratio,
    is_color_contrast_sufficient,
    generate_alt_text,
    make_accessible_component,
    register_keyboard_shortcut
)


class TestAccessibilityManager:
    """Tests for the AccessibilityManager class."""
    
    def test_init(self):
        """Test initialization of the AccessibilityManager."""
        manager = AccessibilityManager()
        
        # Test default properties
        assert manager.enabled is True
        assert manager.screen_reader is not None
        assert manager.keyboard_navigator is not None
        assert manager.settings is not None
    
    def test_toggle_accessibility(self):
        """Test toggling the accessibility features on and off."""
        manager = AccessibilityManager()
        
        # Test initial state
        assert manager.enabled is True
        
        # Test toggling off
        manager.toggle()
        assert manager.enabled is False
        
        # Test toggling back on
        manager.toggle()
        assert manager.enabled is True
        
        # Test direct setting
        manager.set_enabled(False)
        assert manager.enabled is False
    
    def test_register_component(self):
        """Test registering a UI component for accessibility."""
        manager = AccessibilityManager()
        component = MagicMock()
        
        manager.register_component(component)
        
        # Component should be registered
        assert component in manager.components
        
        # Test component setup
        assert component.setup_accessibility.called
    
    def test_announce(self):
        """Test screen reader announcements."""
        manager = AccessibilityManager()
        
        with patch.object(manager, 'screen_reader') as mock_screen_reader:
            manager.announce("Test announcement")
            
            # Should call screen reader
            assert mock_screen_reader.speak.called
            assert mock_screen_reader.speak.call_args[0][0] == "Test announcement"
            
            # Test announcement with priority
            manager.announce("Important message", priority="high")
            assert mock_screen_reader.speak.call_args[0][0] == "Important message"
            assert mock_screen_reader.speak.call_args[1]["priority"] == "high"


class TestScreenReader:
    """Tests for screen reader functionality."""
    
    def test_init(self):
        """Test initialization of the ScreenReaderSupport."""
        screen_reader = ScreenReaderSupport()
        
        # Test default properties
        assert screen_reader.enabled is True
        assert screen_reader.voice_rate == 1.0
        assert screen_reader.volume == 1.0
    
    def test_speak(self):
        """Test text-to-speech functionality."""
        screen_reader = ScreenReaderSupport()
        
        with patch('llama_canvas.accessibility.pyttsx3') as mock_pyttsx3:
            # Mock TTS engine
            mock_engine = MagicMock()
            mock_pyttsx3.init.return_value = mock_engine
            
            # Test speaking text
            screen_reader.speak("Hello world")
            
            # Should initialize TTS engine
            assert mock_pyttsx3.init.called
            
            # Should say text
            assert mock_engine.say.called
            assert mock_engine.say.call_args[0][0] == "Hello world"
            
            # Should run engine
            assert mock_engine.runAndWait.called
    
    def test_adjust_voice_settings(self):
        """Test adjusting voice settings."""
        screen_reader = ScreenReaderSupport()
        
        with patch('llama_canvas.accessibility.pyttsx3') as mock_pyttsx3:
            # Mock TTS engine
            mock_engine = MagicMock()
            mock_pyttsx3.init.return_value = mock_engine
            
            # Test changing rate
            screen_reader.set_rate(1.5)
            
            # Rate should be updated
            assert screen_reader.voice_rate == 1.5
            
            # Should set engine property
            assert mock_engine.setProperty.called
            assert mock_engine.setProperty.call_args_list[0][0] == ('rate', 1.5)
            
            # Test changing volume
            screen_reader.set_volume(0.8)
            
            # Volume should be updated
            assert screen_reader.volume == 0.8
            
            # Should set engine property
            assert mock_engine.setProperty.call_args_list[1][0] == ('volume', 0.8)
    
    def test_read_component(self):
        """Test reading UI component content."""
        screen_reader = ScreenReaderSupport()
        component = MagicMock()
        component.get_accessibility_text.return_value = "Button: Submit Form"
        
        with patch.object(screen_reader, 'speak') as mock_speak:
            screen_reader.read_component(component)
            
            # Should get accessibility text from component
            assert component.get_accessibility_text.called
            
            # Should speak the text
            assert mock_speak.called
            assert mock_speak.call_args[0][0] == "Button: Submit Form"


class TestKeyboardNavigation:
    """Tests for keyboard navigation functionality."""
    
    def test_init(self):
        """Test initialization of KeyboardNavigator."""
        navigator = KeyboardNavigator()
        
        # Test default properties
        assert navigator.enabled is True
        assert navigator.focus_index == -1
        assert navigator.focusable_elements == []
    
    def test_register_element(self):
        """Test registering focusable elements."""
        navigator = KeyboardNavigator()
        
        # Create focusable elements
        button1 = MagicMock()
        button2 = MagicMock()
        
        # Register elements
        navigator.register_element(button1)
        navigator.register_element(button2)
        
        # Elements should be registered
        assert len(navigator.focusable_elements) == 2
        assert navigator.focusable_elements[0] is button1
        assert navigator.focusable_elements[1] is button2
    
    def test_navigation(self):
        """Test keyboard navigation between elements."""
        navigator = KeyboardNavigator()
        
        # Create focusable elements
        button1 = MagicMock()
        button2 = MagicMock()
        button3 = MagicMock()
        
        # Register elements
        navigator.register_elements([button1, button2, button3])
        
        # Test focus next
        navigator.focus_next()
        
        # First element should be focused
        assert navigator.focus_index == 0
        assert button1.set_focus.called
        
        # Reset mocks
        button1.reset_mock()
        button2.reset_mock()
        button3.reset_mock()
        
        # Test focus next again
        navigator.focus_next()
        
        # Second element should be focused
        assert navigator.focus_index == 1
        assert button2.set_focus.called
        
        # Reset mocks
        button1.reset_mock()
        button2.reset_mock()
        button3.reset_mock()
        
        # Test focus previous
        navigator.focus_previous()
        
        # First element should be focused again
        assert navigator.focus_index == 0
        assert button1.set_focus.called
        
        # Test wrapping around
        navigator.focus_index = 2  # Last element
        
        # Reset mocks
        button1.reset_mock()
        button2.reset_mock()
        button3.reset_mock()
        
        # Test focus next (should wrap to first)
        navigator.focus_next()
        
        # First element should be focused
        assert navigator.focus_index == 0
        assert button1.set_focus.called
    
    def test_handle_key_press(self):
        """Test handling key press events."""
        navigator = KeyboardNavigator()
        
        # Create focusable elements
        button = MagicMock()
        
        # Register element
        navigator.register_element(button)
        navigator.focus_index = 0  # Focus the button
        
        # Test handling key press
        with patch.object(navigator, 'focus_next') as mock_focus_next, \
             patch.object(navigator, 'focus_previous') as mock_focus_previous:
            
            # Test Tab key
            navigator.handle_key_press({"key": "Tab"})
            
            # Should call focus_next
            assert mock_focus_next.called
            
            # Test Shift+Tab
            navigator.handle_key_press({"key": "Tab", "shift": True})
            
            # Should call focus_previous
            assert mock_focus_previous.called
            
            # Test Enter key on focused element
            navigator.handle_key_press({"key": "Enter"})
            
            # Should activate focused element
            assert button.activate.called


class TestColorContrast:
    """Tests for color contrast functionality."""
    
    def test_contrast_ratio_calculation(self):
        """Test calculation of color contrast ratios."""
        # Test with black and white (should be max contrast)
        ratio = get_color_contrast_ratio("#000000", "#FFFFFF")
        
        # Should be close to 21:1 (maximum contrast ratio)
        assert round(ratio, 1) == 21.0
        
        # Test with similar colors (low contrast)
        ratio = get_color_contrast_ratio("#CCCCCC", "#DDDDDD")
        
        # Should be low contrast
        assert ratio < 2.0
        
        # Test with RGB tuples instead of hex
        ratio = get_color_contrast_ratio((255, 0, 0), (0, 0, 255))
        
        # Should calculate correctly with tuple input
        assert ratio > 1.0
    
    def test_contrast_sufficiency(self):
        """Test determining if contrast is sufficient for accessibility."""
        # Test with high contrast (black/white)
        assert is_color_contrast_sufficient("#000000", "#FFFFFF") is True
        
        # Test with low contrast
        assert is_color_contrast_sufficient("#CCCCCC", "#DDDDDD") is False
        
        # Test with medium contrast and large text
        medium_contrast = is_color_contrast_sufficient("#0000AA", "#CCCCCC", is_large_text=True)
        
        # Large text can have less contrast
        assert medium_contrast is True
    
    def test_contrast_checker_class(self):
        """Test the ColorContrastChecker class."""
        checker = ColorContrastChecker()
        
        # Test checking a color combination
        result = checker.check_contrast("#000000", "#FFFFFF")
        
        # Should return results dictionary
        assert result["ratio"] > 20.0
        assert result["passes_AA"] is True
        assert result["passes_AAA"] is True
        
        # Test suggesting better contrast
        suggested_color = checker.suggest_better_color("#CCCCCC", "#DDDDDD")
        
        # Should suggest a color with better contrast
        assert suggested_color is not None
        
        # Test checking contrast in an image
        with patch('llama_canvas.accessibility.PILImage') as mock_pil:
            mock_image = MagicMock()
            mock_pil.open.return_value = mock_image
            
            checker.check_image_contrast("path/to/image.jpg")
            
            # Should open and analyze image
            assert mock_pil.open.called


class TestAltTextGeneration:
    """Tests for alt text generation functionality."""
    
    def test_generate_alt_text(self):
        """Test generating alt text for images."""
        # Create a sample image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[40:60, 40:60] = [255, 0, 0]  # Red square
        pil_img = PILImage.fromarray(img)
        
        with patch('llama_canvas.accessibility.image_captioning_model') as mock_model:
            # Mock image captioning model
            mock_model.generate_caption.return_value = "A simple image with a red square in the center"
            
            # Generate alt text
            alt_text = generate_alt_text(pil_img)
            
            # Should return caption from model
            assert alt_text == "A simple image with a red square in the center"
            
            # Test with additional context
            alt_text = generate_alt_text(pil_img, context="This is part of a tutorial")
            
            # Should include context in model input
            assert "tutorial" in mock_model.generate_caption.call_args[1]["context"]


class TestVoiceCommands:
    """Tests for voice command functionality."""
    
    def test_init(self):
        """Test initialization of VoiceCommand module."""
        voice_cmd = VoiceCommand()
        
        # Test default properties
        assert voice_cmd.enabled is True
        assert voice_cmd.commands == {}
        assert voice_cmd.listening is False
    
    def test_register_command(self):
        """Test registering voice commands."""
        voice_cmd = VoiceCommand()
        callback = MagicMock()
        
        # Register command
        voice_cmd.register_command("open file", callback)
        
        # Command should be registered
        assert "open file" in voice_cmd.commands
        assert voice_cmd.commands["open file"] is callback
        
        # Test registering multiple commands
        voice_cmd.register_commands({
            "save file": MagicMock(),
            "exit program": MagicMock()
        })
        
        # Commands should be registered
        assert "save file" in voice_cmd.commands
        assert "exit program" in voice_cmd.commands
    
    def test_start_listening(self):
        """Test starting voice recognition."""
        voice_cmd = VoiceCommand()
        
        with patch('llama_canvas.accessibility.speech_recognition') as mock_sr:
            # Mock speech recognition
            mock_recognizer = MagicMock()
            mock_sr.Recognizer.return_value = mock_recognizer
            
            # Start listening
            voice_cmd.start_listening()
            
            # Should be listening
            assert voice_cmd.listening is True
            
            # Should initialize recognizer
            assert mock_sr.Recognizer.called
            
            # Should start listening in background
            assert mock_recognizer.listen_in_background.called
    
    def test_process_command(self):
        """Test processing recognized voice commands."""
        voice_cmd = VoiceCommand()
        callback = MagicMock()
        
        # Register command
        voice_cmd.register_command("open file", callback)
        
        # Process command
        voice_cmd.process_command("open file")
        
        # Should call callback
        assert callback.called
        
        # Test with unregistered command
        with patch.object(voice_cmd, 'announce_error') as mock_announce:
            voice_cmd.process_command("unknown command")
            
            # Should announce error
            assert mock_announce.called


class TestAccessibilitySettings:
    """Tests for accessibility settings."""
    
    def test_init(self):
        """Test initialization of AccessibilitySettings."""
        settings = AccessibilitySettings()
        
        # Test default settings
        assert settings.screen_reader_enabled is True
        assert settings.keyboard_navigation_enabled is True
        assert settings.high_contrast_mode is False
        assert settings.text_size_multiplier == 1.0
    
    def test_save_load_settings(self):
        """Test saving and loading settings."""
        settings = AccessibilitySettings()
        
        # Change settings
        settings.high_contrast_mode = True
        settings.text_size_multiplier = 1.5
        
        with patch('llama_canvas.accessibility.json') as mock_json, \
             patch('llama_canvas.accessibility.open') as mock_open:
            
            # Save settings
            settings.save()
            
            # Should open file and save JSON
            assert mock_open.called
            assert mock_json.dump.called
            
            # Reset mocks
            mock_json.reset_mock()
            mock_open.reset_mock()
            
            # Mock settings file content
            mock_json.load.return_value = {
                "screen_reader_enabled": False,
                "keyboard_navigation_enabled": True,
                "high_contrast_mode": True,
                "text_size_multiplier": 2.0
            }
            
            # Load settings
            settings.load()
            
            # Should open file and load JSON
            assert mock_open.called
            assert mock_json.load.called
            
            # Settings should be updated
            assert settings.screen_reader_enabled is False
            assert settings.keyboard_navigation_enabled is True
            assert settings.high_contrast_mode is True
            assert settings.text_size_multiplier == 2.0
    
    def test_apply_settings(self):
        """Test applying settings to components."""
        settings = AccessibilitySettings()
        
        # Change settings
        settings.high_contrast_mode = True
        settings.text_size_multiplier = 1.5
        
        # Create mock components
        components = [MagicMock() for _ in range(3)]
        
        # Apply settings
        settings.apply_to_components(components)
        
        # Each component should have settings applied
        for component in components:
            assert component.apply_accessibility_settings.called
            assert component.apply_accessibility_settings.call_args[0][0] is settings


class TestHelperFunctions:
    """Tests for accessibility helper functions."""
    
    def test_make_accessible_component(self):
        """Test making components accessible."""
        # Create component
        component = MagicMock()
        
        # Make accessible
        make_accessible_component(component)
        
        # Component should be modified
        assert hasattr(component, "accessibility_props")
        assert hasattr(component, "get_accessibility_text")
    
    def test_register_keyboard_shortcut(self):
        """Test registering keyboard shortcuts."""
        # Create mock UI
        ui = MagicMock()
        callback = MagicMock()
        
        # Register shortcut
        register_keyboard_shortcut(ui, "Ctrl+S", callback, "Save file")
        
        # Should register with UI
        assert ui.register_shortcut.called
        assert ui.register_shortcut.call_args[0][0] == "Ctrl+S"
        assert ui.register_shortcut.call_args[0][1] is callback
        assert "Save file" in ui.register_shortcut.call_args[1]["description"]


if __name__ == "__main__":
    pytest.main() 