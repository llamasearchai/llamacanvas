"""
Tests for user interface functionality in LlamaCanvas.

This module contains tests for UI components such as 
GUI widgets, web interfaces, and interactive elements.
"""

import os
import tempfile
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call
from PIL import Image as PILImage

from llama_canvas.ui import (
    Window,
    Canvas,
    ToolBar,
    MenuBar,
    StatusBar,
    Button,
    Slider,
    ColorPicker,
    FileDialog,
    Panel,
    Tab,
    Dialog,
    ProgressBar,
    WebView,
    Tooltip,
    create_window,
    show_dialog,
    open_file_dialog,
    save_file_dialog,
    show_tooltip,
    get_screen_size
)


class TestWidgets:
    """Tests for basic UI widgets."""
    
    def test_button(self):
        """Test Button widget functionality."""
        # Test callback function
        callback = MagicMock()
        
        # Create button with callback
        button = Button("Test Button", callback=callback)
        
        # Test properties
        assert button.text == "Test Button"
        assert button.callback is callback
        
        # Test click event
        button.on_click()
        
        # Callback should be called
        assert callback.called
        
        # Test disable/enable
        button.enabled = False
        assert not button.enabled
        
        # Click should not trigger callback when disabled
        callback.reset_mock()
        button.on_click()
        assert not callback.called
        
        # Re-enable and test again
        button.enabled = True
        button.on_click()
        assert callback.called
    
    def test_slider(self):
        """Test Slider widget functionality."""
        # Test callback function
        callback = MagicMock()
        
        # Create slider
        slider = Slider(min_value=0, max_value=100, value=50, callback=callback)
        
        # Test initial properties
        assert slider.min_value == 0
        assert slider.max_value == 100
        assert slider.value == 50
        
        # Test value change
        slider.set_value(75)
        
        # Value should be updated
        assert slider.value == 75
        
        # Callback should be called with new value
        callback.assert_called_with(75)
        
        # Test value clamping
        slider.set_value(150)
        assert slider.value == 100  # Clamped to max
        
        slider.set_value(-10)
        assert slider.value == 0  # Clamped to min
    
    def test_color_picker(self):
        """Test ColorPicker widget functionality."""
        # Test callback function
        callback = MagicMock()
        
        # Create color picker with initial color
        color_picker = ColorPicker(initial_color="#FF0000", callback=callback)
        
        # Test initial properties
        assert color_picker.color == "#FF0000"
        
        # Test color change
        color_picker.set_color("#00FF00")
        
        # Color should be updated
        assert color_picker.color == "#00FF00"
        
        # Callback should be called with new color
        callback.assert_called_with("#00FF00")
        
        # Test color validation
        with pytest.raises(ValueError):
            color_picker.set_color("not_a_color")
    
    def test_file_dialog(self):
        """Test FileDialog widget functionality."""
        with patch('llama_canvas.ui.tk') as mock_tk:
            # Mock tkinter filedialog functions
            mock_tk.filedialog.askopenfilename.return_value = "/path/to/file.jpg"
            mock_tk.filedialog.asksaveasfilename.return_value = "/path/to/save.jpg"
            
            # Create file dialog for opening files
            file_dialog = FileDialog(mode="open", file_types=[("Images", "*.jpg")])
            
            # Test properties
            assert file_dialog.mode == "open"
            assert file_dialog.file_types == [("Images", "*.jpg")]
            
            # Test open dialog
            file_path = file_dialog.show()
            
            # Should call appropriate tkinter function
            assert mock_tk.filedialog.askopenfilename.called
            assert file_path == "/path/to/file.jpg"
            
            # Test save dialog
            file_dialog.mode = "save"
            file_path = file_dialog.show()
            
            # Should call appropriate tkinter function
            assert mock_tk.filedialog.asksaveasfilename.called
            assert file_path == "/path/to/save.jpg"
            
            # Test with default path
            file_dialog.default_path = "/my/default/path"
            file_dialog.show()
            
            # Should include default path
            assert mock_tk.filedialog.asksaveasfilename.call_args[1]["initialdir"] == "/my/default/path"
    
    def test_progress_bar(self):
        """Test ProgressBar widget functionality."""
        # Create progress bar
        progress_bar = ProgressBar(min_value=0, max_value=100, value=0)
        
        # Test initial properties
        assert progress_bar.min_value == 0
        assert progress_bar.max_value == 100
        assert progress_bar.value == 0
        
        # Test value update
        progress_bar.set_value(50)
        assert progress_bar.value == 50
        
        # Test percentage calculation
        assert progress_bar.get_percentage() == 50
        
        # Test value clamping
        progress_bar.set_value(150)
        assert progress_bar.value == 100  # Clamped to max
        
        progress_bar.set_value(-10)
        assert progress_bar.value == 0  # Clamped to min
        
        # Test increment
        progress_bar.set_value(50)
        progress_bar.increment(10)
        assert progress_bar.value == 60
        
        # Test complete
        progress_bar.complete()
        assert progress_bar.value == progress_bar.max_value


class TestContainers:
    """Tests for UI container components."""
    
    def test_panel(self):
        """Test Panel container functionality."""
        # Create panel
        panel = Panel(title="Test Panel")
        
        # Test initial properties
        assert panel.title == "Test Panel"
        assert panel.children == []
        
        # Create widgets to add
        button = Button("Test Button")
        slider = Slider(0, 100, 50)
        
        # Add widgets to panel
        panel.add_child(button)
        
        # Test child management
        assert len(panel.children) == 1
        assert panel.children[0] is button
        
        # Add multiple children
        panel.add_children([slider])
        
        # Test multiple children
        assert len(panel.children) == 2
        assert panel.children[1] is slider
        
        # Test remove child
        panel.remove_child(button)
        
        # Should remove child
        assert len(panel.children) == 1
        assert panel.children[0] is slider
        
        # Test clear children
        panel.clear_children()
        
        # Should remove all children
        assert len(panel.children) == 0
    
    def test_tab(self):
        """Test Tab container functionality."""
        # Create tabs
        tab1 = Tab(title="Tab 1")
        tab2 = Tab(title="Tab 2")
        
        # Test initial properties
        assert tab1.title == "Tab 1"
        assert tab1.children == []
        
        # Create widgets to add
        button = Button("Test Button")
        slider = Slider(0, 100, 50)
        
        # Add widgets to tabs
        tab1.add_child(button)
        tab2.add_child(slider)
        
        # Test child management
        assert len(tab1.children) == 1
        assert tab1.children[0] is button
        assert len(tab2.children) == 1
        assert tab2.children[0] is slider
        
        # Test tab selection
        tab1.select()
        assert tab1.selected
        
        tab2.select()
        assert tab2.selected
        assert not tab1.selected  # Previous tab should be deselected
    
    def test_dialog(self):
        """Test Dialog functionality."""
        # Create callback functions
        on_ok = MagicMock()
        on_cancel = MagicMock()
        
        # Create dialog
        dialog = Dialog(
            title="Test Dialog",
            message="This is a test dialog",
            on_ok=on_ok,
            on_cancel=on_cancel
        )
        
        # Test initial properties
        assert dialog.title == "Test Dialog"
        assert dialog.message == "This is a test dialog"
        assert dialog.on_ok is on_ok
        assert dialog.on_cancel is on_cancel
        
        # Test dialog actions
        dialog.ok()
        assert on_ok.called
        
        dialog.cancel()
        assert on_cancel.called


class TestWindows:
    """Tests for window management."""
    
    def test_window(self):
        """Test Window functionality."""
        # Create window
        window = Window(title="Test Window", width=800, height=600)
        
        # Test initial properties
        assert window.title == "Test Window"
        assert window.width == 800
        assert window.height == 600
        assert not window.visible
        
        # Test show/hide
        window.show()
        assert window.visible
        
        window.hide()
        assert not window.visible
        
        # Test resize
        window.resize(1024, 768)
        assert window.width == 1024
        assert window.height == 768
        
        # Test adding widgets
        button = Button("Test Button")
        panel = Panel("Test Panel")
        
        window.add_widget(button)
        
        # Test widget management
        assert len(window.widgets) == 1
        assert window.widgets[0] is button
        
        # Add multiple widgets
        window.add_widgets([panel])
        
        # Test multiple widgets
        assert len(window.widgets) == 2
        assert window.widgets[1] is panel
        
        # Test remove widget
        window.remove_widget(button)
        
        # Should remove widget
        assert len(window.widgets) == 1
        assert window.widgets[0] is panel
    
    def test_tool_bar(self):
        """Test ToolBar functionality."""
        # Create tool bar
        tool_bar = ToolBar()
        
        # Test initial properties
        assert tool_bar.buttons == []
        
        # Create tool buttons
        button1 = Button("Tool 1")
        button2 = Button("Tool 2")
        
        # Add buttons to toolbar
        tool_bar.add_button(button1)
        
        # Test button management
        assert len(tool_bar.buttons) == 1
        assert tool_bar.buttons[0] is button1
        
        # Add multiple buttons
        tool_bar.add_buttons([button2])
        
        # Test multiple buttons
        assert len(tool_bar.buttons) == 2
        assert tool_bar.buttons[1] is button2
    
    def test_menu_bar(self):
        """Test MenuBar functionality."""
        # Create menu bar
        menu_bar = MenuBar()
        
        # Test initial properties
        assert menu_bar.menus == {}
        
        # Create menu callback
        callback = MagicMock()
        
        # Add menu
        menu_bar.add_menu("File")
        
        # Test menu management
        assert "File" in menu_bar.menus
        assert menu_bar.menus["File"] == []
        
        # Add menu items
        menu_bar.add_menu_item("File", "Open", callback)
        menu_bar.add_menu_item("File", "Save", callback)
        
        # Test menu item management
        assert len(menu_bar.menus["File"]) == 2
        assert menu_bar.menus["File"][0]["label"] == "Open"
        assert menu_bar.menus["File"][0]["callback"] is callback
        assert menu_bar.menus["File"][1]["label"] == "Save"
        
        # Test menu item triggering
        menu_bar.trigger_menu_item("File", "Open")
        
        # Callback should be called
        assert callback.called
    
    def test_status_bar(self):
        """Test StatusBar functionality."""
        # Create status bar
        status_bar = StatusBar()
        
        # Test initial properties
        assert status_bar.text == ""
        
        # Set status message
        status_bar.set_text("Ready")
        
        # Text should be updated
        assert status_bar.text == "Ready"
        
        # Test temporary message
        with patch('llama_canvas.ui.threading') as mock_threading:
            # Mock timer
            mock_timer = MagicMock()
            mock_threading.Timer.return_value = mock_timer
            
            status_bar.set_temporary_text("Processing...", duration=5)
            
            # Text should be updated
            assert status_bar.text == "Processing..."
            
            # Timer should be created and started
            assert mock_threading.Timer.called
            assert mock_threading.Timer.call_args[0][0] == 5
            assert mock_timer.start.called


class TestCanvas:
    """Tests for Canvas widget."""
    
    def test_canvas(self):
        """Test Canvas functionality."""
        # Create canvas
        canvas = Canvas(width=400, height=300)
        
        # Test initial properties
        assert canvas.width == 400
        assert canvas.height == 300
        assert canvas.background_color == "#FFFFFF"  # Default white
        
        # Test background color
        canvas.set_background_color("#000000")
        assert canvas.background_color == "#000000"
        
        # Create a sample image for testing drawing
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[40:60, 40:60] = [255, 0, 0]  # Red square
        pil_img = PILImage.fromarray(img)
        
        # Test drawing image
        with patch.object(canvas, '_draw_image') as mock_draw:
            canvas.draw_image(pil_img, x=50, y=50)
            
            # Should call internal draw method
            assert mock_draw.called
            assert mock_draw.call_args[0][0] is pil_img
            assert mock_draw.call_args[0][1] == 50
            assert mock_draw.call_args[0][2] == 50
        
        # Test drawing shapes
        with patch.object(canvas, '_draw_line') as mock_line, \
             patch.object(canvas, '_draw_rectangle') as mock_rect, \
             patch.object(canvas, '_draw_ellipse') as mock_ellipse:
            
            # Draw line
            canvas.draw_line(0, 0, 100, 100, color="#FF0000", width=2)
            
            # Should call line drawing
            assert mock_line.called
            assert mock_line.call_args[0] == (0, 0, 100, 100)
            assert mock_line.call_args[1]["color"] == "#FF0000"
            assert mock_line.call_args[1]["width"] == 2
            
            # Draw rectangle
            canvas.draw_rectangle(50, 50, 100, 100, fill_color="#00FF00", outline_color="#000000")
            
            # Should call rectangle drawing
            assert mock_rect.called
            assert mock_rect.call_args[0] == (50, 50, 100, 100)
            assert mock_rect.call_args[1]["fill_color"] == "#00FF00"
            assert mock_rect.call_args[1]["outline_color"] == "#000000"
            
            # Draw ellipse
            canvas.draw_ellipse(100, 100, 50, 30, fill_color="#0000FF")
            
            # Should call ellipse drawing
            assert mock_ellipse.called
            assert mock_ellipse.call_args[0] == (100, 100, 50, 30)
            assert mock_ellipse.call_args[1]["fill_color"] == "#0000FF"
        
        # Test clear
        with patch.object(canvas, '_clear') as mock_clear:
            canvas.clear()
            
            # Should call clear method
            assert mock_clear.called
        
        # Test save
        with patch('llama_canvas.ui.PILImage') as mock_pil:
            mock_image = MagicMock()
            mock_pil.new.return_value = mock_image
            
            canvas.save("test.png")
            
            # Should create image and save
            assert mock_pil.new.called
            assert mock_image.save.called
            assert mock_image.save.call_args[0][0] == "test.png"


class TestWebComponents:
    """Tests for web interface components."""
    
    def test_web_view(self):
        """Test WebView functionality."""
        with patch('llama_canvas.ui.webview') as mock_webview:
            # Create web view
            web_view = WebView(width=800, height=600)
            
            # Test initial properties
            assert web_view.width == 800
            assert web_view.height == 600
            assert web_view.url is None
            
            # Test loading URL
            web_view.load_url("https://example.com")
            
            # URL should be updated
            assert web_view.url == "https://example.com"
            
            # Test showing web view
            web_view.show()
            
            # Should call webview create/show
            assert mock_webview.create_window.called
            assert mock_webview.start.called
            
            # Test loading HTML
            web_view.load_html("<html><body>Test</body></html>")
            
            # Test evaluating JavaScript
            web_view.eval_js("document.title = 'Test';")
            
            # Should call webview evaluate
            assert mock_webview.evaluate_js.called


class TestInteractions:
    """Tests for UI interactions."""
    
    def test_tooltip(self):
        """Test Tooltip functionality."""
        # Create tooltip
        tooltip = Tooltip(text="This is a tooltip")
        
        # Test initial properties
        assert tooltip.text == "This is a tooltip"
        
        # Test show/hide
        with patch.object(tooltip, '_show') as mock_show, \
             patch.object(tooltip, '_hide') as mock_hide:
            
            tooltip.show(x=100, y=200)
            
            # Should call show method
            assert mock_show.called
            assert mock_show.call_args[0] == (100, 200)
            
            tooltip.hide()
            
            # Should call hide method
            assert mock_hide.called
    
    def test_event_handling(self):
        """Test event handling in UI components."""
        # Create button with event handling
        button = Button("Test Button")
        
        # Test event callbacks
        click_callback = MagicMock()
        hover_callback = MagicMock()
        
        # Register event handlers
        button.on_click_event = click_callback
        button.on_hover_event = hover_callback
        
        # Simulate events
        button.trigger_click()
        button.trigger_hover()
        
        # Callbacks should be called
        assert click_callback.called
        assert hover_callback.called


class TestHelperFunctions:
    """Tests for UI helper functions."""
    
    def test_create_window(self):
        """Test create_window helper function."""
        with patch('llama_canvas.ui.Window') as mock_window:
            # Mock window instance
            window_instance = MagicMock()
            mock_window.return_value = window_instance
            
            # Call helper function
            window = create_window("Test Window", 800, 600)
            
            # Should create Window
            assert mock_window.called
            assert mock_window.call_args[1]["title"] == "Test Window"
            assert mock_window.call_args[1]["width"] == 800
            assert mock_window.call_args[1]["height"] == 600
            
            # Should return window instance
            assert window is window_instance
    
    def test_show_dialog(self):
        """Test show_dialog helper function."""
        with patch('llama_canvas.ui.Dialog') as mock_dialog:
            # Mock dialog instance
            dialog_instance = MagicMock()
            mock_dialog.return_value = dialog_instance
            
            # Call helper function
            result = show_dialog(
                title="Test Dialog",
                message="This is a test dialog",
                dialog_type="info"
            )
            
            # Should create Dialog
            assert mock_dialog.called
            assert mock_dialog.call_args[1]["title"] == "Test Dialog"
            assert mock_dialog.call_args[1]["message"] == "This is a test dialog"
            assert mock_dialog.call_args[1]["dialog_type"] == "info"
            
            # Should show dialog and return result
            assert dialog_instance.show.called
            assert result is dialog_instance.show.return_value
    
    def test_file_dialogs(self):
        """Test file dialog helper functions."""
        with patch('llama_canvas.ui.FileDialog') as mock_file_dialog:
            # Mock file dialog instance
            dialog_instance = MagicMock()
            dialog_instance.show.return_value = "/path/to/file.txt"
            mock_file_dialog.return_value = dialog_instance
            
            # Test open file dialog
            file_path = open_file_dialog(
                title="Open File",
                file_types=[("Text Files", "*.txt")]
            )
            
            # Should create FileDialog with open mode
            assert mock_file_dialog.called
            assert mock_file_dialog.call_args[1]["title"] == "Open File"
            assert mock_file_dialog.call_args[1]["mode"] == "open"
            assert mock_file_dialog.call_args[1]["file_types"] == [("Text Files", "*.txt")]
            
            # Should show dialog and return result
            assert dialog_instance.show.called
            assert file_path == "/path/to/file.txt"
            
            # Reset mock
            mock_file_dialog.reset_mock()
            
            # Test save file dialog
            file_path = save_file_dialog(
                title="Save File",
                default_file="document.txt",
                file_types=[("Text Files", "*.txt")]
            )
            
            # Should create FileDialog with save mode
            assert mock_file_dialog.called
            assert mock_file_dialog.call_args[1]["title"] == "Save File"
            assert mock_file_dialog.call_args[1]["mode"] == "save"
            assert mock_file_dialog.call_args[1]["default_file"] == "document.txt"
    
    def test_get_screen_size(self):
        """Test get_screen_size helper function."""
        with patch('llama_canvas.ui.tk') as mock_tk:
            # Mock tkinter root and screen dimensions
            mock_root = MagicMock()
            mock_tk.Tk.return_value = mock_root
            mock_root.winfo_screenwidth.return_value = 1920
            mock_root.winfo_screenheight.return_value = 1080
            
            # Call helper function
            width, height = get_screen_size()
            
            # Should query screen dimensions
            assert mock_root.winfo_screenwidth.called
            assert mock_root.winfo_screenheight.called
            
            # Should return dimensions
            assert width == 1920
            assert height == 1080
            
            # Should destroy temporary root
            assert mock_root.destroy.called


class TestIntegration:
    """Integration tests for UI components."""
    
    def test_window_with_components(self):
        """Test window with multiple UI components."""
        # Create window
        window = Window("Test Application", 800, 600)
        
        # Create menu bar
        menu_bar = MenuBar()
        menu_bar.add_menu("File")
        menu_bar.add_menu_item("File", "Open", lambda: None)
        menu_bar.add_menu_item("File", "Save", lambda: None)
        menu_bar.add_menu_item("File", "Exit", lambda: None)
        
        # Create tool bar
        tool_bar = ToolBar()
        tool_bar.add_button(Button("New"))
        tool_bar.add_button(Button("Open"))
        tool_bar.add_button(Button("Save"))
        
        # Create status bar
        status_bar = StatusBar()
        status_bar.set_text("Ready")
        
        # Create canvas
        canvas = Canvas(width=600, height=400)
        
        # Create side panel
        panel = Panel("Tools")
        panel.add_child(Button("Select"))
        panel.add_child(Button("Draw"))
        panel.add_child(ColorPicker("#000000"))
        
        # Add components to window
        window.set_menu_bar(menu_bar)
        window.set_tool_bar(tool_bar)
        window.set_status_bar(status_bar)
        window.add_widget(canvas)
        window.add_widget(panel)
        
        # Test component integration
        assert window.menu_bar is menu_bar
        assert window.tool_bar is tool_bar
        assert window.status_bar is status_bar
        assert len(window.widgets) == 2
        assert canvas in window.widgets
        assert panel in window.widgets
        
        # Test window layout update
        with patch.object(window, '_update_layout') as mock_update:
            window.update_layout()
            
            # Should call layout update
            assert mock_update.called
        
        # Test window event handling
        with patch.object(window, 'on_resize') as mock_resize:
            window.resize(1024, 768)
            
            # Should trigger resize event
            assert mock_resize.called
            assert mock_resize.call_args[0] == (1024, 768)


if __name__ == "__main__":
    pytest.main() 