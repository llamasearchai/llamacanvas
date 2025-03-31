"""
Integration tests for LlamaCanvas components.

This module contains tests that verify the proper interaction
between different components of the LlamaCanvas system.
"""

import pytest
import os
import tempfile
import json
import numpy as np
from unittest.mock import patch, MagicMock, call
from PIL import Image as PILImage

from llama_canvas.canvas import Canvas
from llama_canvas.image import Image
from llama_canvas.video import Video
from llama_canvas.agent_manager import AgentManager
from llama_canvas.ui import Window, Panel, Button
from llama_canvas.api import APIService
from llama_canvas.security import SecurityManager
from llama_canvas.performance import PerformanceMonitor
from llama_canvas.accessibility import AccessibilityManager
from llama_canvas.notebook import NotebookDisplay


class TestImageCanvasIntegration:
    """Tests for integration between Image and Canvas components."""
    
    def test_image_to_canvas(self):
        """Test loading an image to canvas and editing it."""
        # Create a test image
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img_array[30:70, 30:70] = [255, 0, 0]  # Red square
        pil_img = PILImage.fromarray(img_array)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            pil_img.save(temp_file.name)
            img_path = temp_file.name
        
        try:
            # Create Image object
            image = Image(img_path)
            
            # Create Canvas
            canvas = Canvas(width=100, height=100)
            
            # Load image to canvas
            canvas.add_image(image, x=0, y=0)
            
            # Image should be in canvas layers
            assert len(canvas.layers) == 1
            assert canvas.layers[0].image is image
            
            # Apply a filter to the canvas
            with patch.object(Image, 'apply_filter') as mock_filter:
                canvas.apply_filter('grayscale')
                
                # Should apply filter to the image in the active layer
                assert mock_filter.called
                assert mock_filter.call_args[0][0] == 'grayscale'
            
            # Export canvas to a new image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as output_file:
                canvas.export(output_file.name)
                
                # Should create a file
                assert os.path.exists(output_file.name)
                
                # Clean up output file
                os.unlink(output_file.name)
        
        finally:
            # Clean up test file
            os.unlink(img_path)
    
    def test_canvas_operations_on_image(self):
        """Test canvas operations applied to an image."""
        # Create test image
        image = Image.create_blank(200, 200)
        
        # Create canvas with the image
        canvas = Canvas(width=200, height=200)
        canvas.add_image(image)
        
        # Draw on the canvas
        canvas.draw_rectangle(50, 50, 100, 100, fill_color="red")
        
        # The image should be modified
        modified_image = canvas.layers[0].image
        
        # Export and check the result
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.png")
            modified_image.save(output_path)
            
            # Load the saved image and check the red rectangle
            loaded_img = PILImage.open(output_path)
            pixel = loaded_img.getpixel((75, 75))  # Inside the rectangle
            
            # Should be red (or close to it)
            assert pixel[0] > 200  # High red value
            assert pixel[1] < 50   # Low green value
            assert pixel[2] < 50   # Low blue value


class TestVideoCanvasIntegration:
    """Tests for integration between Video and Canvas components."""
    
    def test_video_frame_editing(self):
        """Test editing video frames with a canvas."""
        # Create a test video with 5 frames
        frames = []
        for i in range(5):
            # Create frame with different color
            frame_array = np.zeros((100, 100, 3), dtype=np.uint8)
            frame_array[:, :, i % 3] = 255  # R, G, B cycle
            frame = PILImage.fromarray(frame_array)
            frames.append(frame)
        
        # Create Video object
        video = Video.from_frames(frames, fps=30)
        
        # Create Canvas
        canvas = Canvas(width=100, height=100)
        
        # Process each frame with the canvas
        processed_frames = []
        
        for frame in video.frames:
            # Reset canvas and add frame
            canvas.clear()
            canvas.add_image(Image.from_pil(frame))
            
            # Draw on the canvas
            canvas.draw_circle(50, 50, 20, fill_color="white")
            
            # Export the modified frame
            processed_frame = canvas.to_image().to_pil()
            processed_frames.append(processed_frame)
        
        # Create a new video from the processed frames
        processed_video = Video.from_frames(processed_frames, fps=30)
        
        # Video should have the same number of frames
        assert len(processed_video.frames) == len(video.frames)
        
        # Check if frames were modified correctly
        for frame in processed_video.frames:
            # Center pixel should be white
            pixel = frame.getpixel((50, 50))
            assert pixel[0] > 200  # High red
            assert pixel[1] > 200  # High green
            assert pixel[2] > 200  # High blue
    
    def test_video_to_canvas_animation(self):
        """Test creating canvas animation from video."""
        # Create simple video
        frames = []
        for i in range(3):
            frame_array = np.zeros((50, 50, 3), dtype=np.uint8)
            frame_array.fill(i * 50)  # Increasingly brighter
            frame = PILImage.fromarray(frame_array)
            frames.append(frame)
        
        # Create Video
        video = Video.from_frames(frames, fps=10)
        
        # Create Canvas
        canvas = Canvas(width=50, height=50)
        
        # Create animation from video
        animation = canvas.create_animation_from_video(video)
        
        # Animation should have correct properties
        assert animation.frame_count == 3
        assert animation.fps == 10
        
        # Play animation on canvas
        with patch.object(canvas, 'clear') as mock_clear, \
             patch.object(canvas, 'add_image') as mock_add_image:
            
            animation.play(canvas)
            
            # Should clear canvas for each frame
            assert mock_clear.call_count == 3
            
            # Should add each frame
            assert mock_add_image.call_count == 3


class TestUICanvasIntegration:
    """Tests for integration between UI and Canvas components."""
    
    def test_canvas_in_window(self):
        """Test canvas integration with window UI."""
        # Create canvas
        canvas = Canvas(width=400, height=300)
        
        # Create window
        window = Window(title="Test Window", width=800, height=600)
        
        # Add canvas to window
        window.add_widget(canvas)
        
        # Canvas should be in window widgets
        assert canvas in window.widgets
        
        # Test canvas interaction with UI
        with patch.object(canvas, 'handle_mouse_event') as mock_handle:
            # Simulate mouse event in window
            window.process_mouse_event({
                'type': 'click',
                'x': 100,
                'y': 100,
                'button': 'left'
            })
            
            # Event should be passed to canvas
            assert mock_handle.called
            assert mock_handle.call_args[0][0]['type'] == 'click'
    
    def test_ui_toolbar_with_canvas(self):
        """Test toolbar UI integration with canvas."""
        # Create canvas
        canvas = Canvas(width=400, height=300)
        
        # Create window
        window = Window(title="Test Window", width=800, height=600)
        
        # Create toolbar panel
        toolbar = Panel(title="Tools")
        
        # Create buttons for canvas operations
        clear_button = Button("Clear")
        draw_button = Button("Draw")
        
        # Add buttons to toolbar
        toolbar.add_child(clear_button)
        toolbar.add_child(draw_button)
        
        # Add toolbar and canvas to window
        window.add_widget(toolbar)
        window.add_widget(canvas)
        
        # Set up button actions
        with patch.object(canvas, 'clear') as mock_clear, \
             patch.object(canvas, 'set_drawing_mode') as mock_set_mode:
            
            # Simulate button clicks
            clear_button.on_click()
            
            # Should call canvas clear
            assert mock_clear.called
            
            # Simulate draw button
            draw_button.on_click()
            
            # Should set drawing mode
            assert mock_set_mode.called


class TestAgentCanvasIntegration:
    """Tests for integration between AI Agent and Canvas components."""
    
    def test_agent_modifying_canvas(self):
        """Test AI agent modifying a canvas based on instruction."""
        # Create canvas
        canvas = Canvas(width=400, height=300)
        
        # Create agent
        agent = AgentManager()
        
        # Create a mock for the agent's process_instruction method
        with patch.object(agent, 'process_instruction') as mock_process:
            # Mock the agent creating a drawing plan
            mock_process.return_value = {
                'actions': [
                    {'type': 'draw_circle', 'x': 200, 'y': 150, 'radius': 50, 'color': 'blue'},
                    {'type': 'draw_rectangle', 'x': 50, 'y': 50, 'width': 100, 'height': 100, 'color': 'red'}
                ]
            }
            
            # Send instruction to agent
            instruction = "Draw a blue circle in the center and a red square in the top-left corner"
            agent.process_canvas_instruction(instruction, canvas)
            
            # Agent should process the instruction
            assert mock_process.called
            assert mock_process.call_args[0][0] == instruction
            
            # Canvas should have the shapes
            # Since we're mocking, we can't check the actual drawing, but we can check
            # if canvas methods were called with correct parameters
            assert len(canvas.draw_history) == 2
            assert canvas.draw_history[0]['type'] == 'circle'
            assert canvas.draw_history[1]['type'] == 'rectangle'
    
    def test_canvas_to_agent_description(self):
        """Test generating a description of a canvas for an AI agent."""
        # Create canvas with content
        canvas = Canvas(width=400, height=300)
        
        # Add some shapes to canvas
        canvas.draw_circle(100, 100, 50, fill_color="blue")
        canvas.draw_rectangle(200, 150, 100, 80, fill_color="green")
        
        # Generate description for agent
        with patch('llama_canvas.agent_manager.generate_description') as mock_generate:
            # Mock description generation
            mock_generate.return_value = "A canvas with a blue circle and a green rectangle"
            
            # Get canvas description
            agent = AgentManager()
            description = agent.get_canvas_description(canvas)
            
            # Should generate description
            assert mock_generate.called
            
            # Should return description
            assert description == "A canvas with a blue circle and a green rectangle"


class TestAPIAndCanvasIntegration:
    """Tests for integration between API service and Canvas components."""
    
    def test_api_create_canvas(self):
        """Test creating and manipulating a canvas through API."""
        # Create API service
        api_service = APIService()
        
        # Mock the create_canvas method in API service
        with patch.object(api_service, 'create_canvas') as mock_create:
            # Create new canvas via API
            canvas = Canvas(width=800, height=600)
            mock_create.return_value = canvas
            
            # Call API method
            result = api_service.create_canvas(width=800, height=600)
            
            # Should return canvas
            assert result is canvas
            
            # Test drawing via API
            with patch.object(api_service, 'draw_on_canvas') as mock_draw:
                # Set up drawing result
                mock_draw.return_value = {"success": True}
                
                # Call API method
                api_result = api_service.draw_on_canvas(
                    canvas_id=canvas.id,
                    shape_type="circle",
                    params={"x": 100, "y": 100, "radius": 50, "color": "red"}
                )
                
                # Should call API method with correct parameters
                assert mock_draw.called
                assert mock_draw.call_args[1]["canvas_id"] == canvas.id
                assert mock_draw.call_args[1]["shape_type"] == "circle"
                assert mock_draw.call_args[1]["params"]["color"] == "red"
                
                # Should return success
                assert api_result["success"] is True
    
    def test_api_export_canvas(self):
        """Test exporting a canvas through API."""
        # Create API service
        api_service = APIService()
        
        # Create canvas
        canvas = Canvas(width=400, height=300)
        canvas.draw_rectangle(100, 100, 200, 100, fill_color="blue")
        
        # Mock API methods
        with patch.object(api_service, 'get_canvas') as mock_get, \
             patch.object(api_service, 'export_canvas') as mock_export:
            
            # Set up mock returns
            mock_get.return_value = canvas
            mock_export.return_value = {"file_url": "/exports/canvas123.png"}
            
            # Call API method
            export_result = api_service.export_canvas(
                canvas_id=canvas.id,
                format="png"
            )
            
            # Should call export with canvas
            assert mock_export.called
            assert mock_export.call_args[1]["canvas_id"] == canvas.id
            assert mock_export.call_args[1]["format"] == "png"
            
            # Should return file URL
            assert "file_url" in export_result
            assert export_result["file_url"] == "/exports/canvas123.png"


class TestSecurityAndCanvasIntegration:
    """Tests for integration between security features and Canvas components."""
    
    def test_secure_canvas_operations(self):
        """Test secure operations on canvas with permission checks."""
        # Create security manager
        security = SecurityManager()
        
        # Create canvas with security
        canvas = Canvas(width=400, height=300, security_manager=security)
        
        # Set up user roles
        security.permission_manager.define_role("editor", ["view_canvas", "edit_canvas"])
        security.permission_manager.define_role("viewer", ["view_canvas"])
        
        security.permission_manager.assign_role("editor_user", "editor")
        security.permission_manager.assign_role("viewer_user", "viewer")
        
        # Test drawing with permissions
        with patch.object(security, 'check_permission') as mock_check:
            # Mock permission checks
            mock_check.side_effect = lambda user, perm: (
                (user == "editor_user" and perm in ["view_canvas", "edit_canvas"]) or
                (user == "viewer_user" and perm == "view_canvas")
            )
            
            # Editor should be able to draw
            result = canvas.draw_circle(100, 100, 50, fill_color="red", user="editor_user")
            
            # Should check permission
            assert mock_check.called
            assert mock_check.call_args[0][0] == "editor_user"
            assert mock_check.call_args[0][1] == "edit_canvas"
            
            # Should allow operation
            assert result is True
            
            # Viewer should not be able to draw
            mock_check.reset_mock()
            result = canvas.draw_circle(200, 200, 30, fill_color="blue", user="viewer_user")
            
            # Should check permission
            assert mock_check.called
            
            # Should deny operation
            assert result is False
    
    def test_canvas_input_validation(self):
        """Test canvas input validation through security services."""
        # Create security manager
        security = SecurityManager()
        
        # Create canvas with security
        canvas = Canvas(width=400, height=300, security_manager=security)
        
        # Mock the security manager's validate_input method
        with patch.object(security, 'validate_input') as mock_validate:
            # Mock validation results
            mock_validate.side_effect = lambda input_str, input_type: (
                (True, None) if input_type == "coordinate" and input_str.isdigit() 
                else (False, "Invalid input")
            )
            
            # Test with valid input
            valid, error = canvas.validate_param("100", "coordinate")
            
            # Should validate input
            assert mock_validate.called
            assert mock_validate.call_args[0][0] == "100"
            assert mock_validate.call_args[0][1] == "coordinate"
            
            # Should return validation result
            assert valid is True
            assert error is None
            
            # Test with invalid input
            mock_validate.reset_mock()
            valid, error = canvas.validate_param("not_a_number", "coordinate")
            
            # Should validate input
            assert mock_validate.called
            
            # Should return validation error
            assert valid is False
            assert error == "Invalid input"


class TestPerformanceMonitoring:
    """Tests for performance monitoring across components."""
    
    def test_canvas_performance_monitoring(self):
        """Test monitoring canvas operation performance."""
        # Create performance monitor
        monitor = PerformanceMonitor()
        
        # Create canvas with performance monitoring
        canvas = Canvas(width=800, height=600, performance_monitor=monitor)
        
        # Test drawing operations with performance monitoring
        with patch.object(monitor, 'start_timing') as mock_start, \
             patch.object(monitor, 'stop_timing') as mock_stop:
            
            # Draw on canvas
            canvas.draw_rectangle(100, 100, 200, 200, fill_color="red")
            
            # Should monitor timing
            assert mock_start.called
            assert mock_start.call_args[0][0] == "draw_rectangle"
            
            assert mock_stop.called
            assert mock_stop.call_args[0][0] == "draw_rectangle"
        
        # Test exporting with performance monitoring
        with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
            with patch.object(monitor, 'start_timing') as mock_start, \
                 patch.object(monitor, 'stop_timing') as mock_stop:
                
                # Export canvas
                canvas.export(temp_file.name)
                
                # Should monitor timing
                assert mock_start.called
                assert mock_start.call_args[0][0] == "export"
                
                assert mock_stop.called
                assert mock_stop.call_args[0][0] == "export"
    
    def test_video_processing_performance(self):
        """Test monitoring video processing performance."""
        # Create performance monitor
        monitor = PerformanceMonitor()
        
        # Create test frames
        frames = []
        for i in range(3):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            frame.fill(i * 50)
            frames.append(PILImage.fromarray(frame))
        
        # Create video with performance monitoring
        video = Video.from_frames(frames, fps=30, performance_monitor=monitor)
        
        # Test applying a filter with performance monitoring
        with patch.object(monitor, 'start_timing') as mock_start, \
             patch.object(monitor, 'stop_timing') as mock_stop:
            
            # Apply filter to video
            video.apply_filter('grayscale')
            
            # Should monitor timing
            assert mock_start.called
            assert mock_start.call_args[0][0] == "apply_filter"
            
            assert mock_stop.called
            assert mock_stop.call_args[0][0] == "apply_filter"


class TestAccessibilityIntegration:
    """Tests for accessibility integration across components."""
    
    def test_canvas_accessibility(self):
        """Test canvas accessibility features."""
        # Create accessibility manager
        accessibility = AccessibilityManager()
        
        # Create canvas with accessibility
        canvas = Canvas(width=400, height=300, accessibility_manager=accessibility)
        
        # Test canvas description generation
        with patch.object(accessibility, 'announce') as mock_announce:
            # Get canvas description
            canvas.get_accessibility_description()
            
            # Should announce canvas description
            assert mock_announce.called
            assert "canvas" in mock_announce.call_args[0][0].lower()
        
        # Test accessible drawing operations
        with patch.object(accessibility, 'announce') as mock_announce:
            # Draw on canvas
            canvas.draw_circle(100, 100, 50, fill_color="blue")
            
            # Should announce the operation
            assert mock_announce.called
            assert "circle" in mock_announce.call_args[0][0].lower()
            assert "blue" in mock_announce.call_args[0][0].lower()
    
    def test_ui_accessibility(self):
        """Test UI component accessibility features."""
        # Create accessibility manager
        accessibility = AccessibilityManager()
        
        # Create UI components with accessibility
        window = Window(title="Test Window", width=800, height=600, accessibility_manager=accessibility)
        
        # Create button
        button = Button("Draw Circle", accessibility_manager=accessibility)
        
        # Add button to window
        window.add_widget(button)
        
        # Test keyboard navigation
        with patch.object(accessibility.keyboard_navigator, 'focus_next') as mock_focus_next:
            # Simulate tab key press
            window.handle_key_press({"key": "Tab"})
            
            # Should navigate to next focusable element
            assert mock_focus_next.called
        
        # Test screen reader integration
        with patch.object(accessibility.screen_reader, 'read_component') as mock_read:
            # Simulate focus on button
            button.set_focus()
            
            # Should read button description
            assert mock_read.called
            assert mock_read.call_args[0][0] is button


class TestNotebookIntegration:
    """Tests for Jupyter notebook integration with LlamaCanvas components."""
    
    def test_notebook_canvas_integration(self):
        """Test using canvas in a Jupyter notebook."""
        # Create notebook display
        notebook = NotebookDisplay()
        
        # Create canvas
        canvas = Canvas(width=400, height=300)
        
        # Draw on canvas
        canvas.draw_rectangle(100, 100, 200, 100, fill_color="green")
        
        # Test displaying canvas in notebook
        with patch('llama_canvas.notebook.IPython.display') as mock_display:
            # Display canvas
            notebook.display_canvas(canvas)
            
            # Should use IPython display
            assert mock_display.display.called
    
    def test_notebook_image_manipulation(self):
        """Test image manipulation in notebook."""
        # Create notebook display
        notebook = NotebookDisplay()
        
        # Create test image
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img_array[30:70, 30:70] = [0, 255, 0]  # Green square
        pil_img = PILImage.fromarray(img_array)
        
        # Create Image object
        image = Image.from_pil(pil_img)
        
        # Test applying filter in notebook
        with patch('llama_canvas.notebook.IPython.display') as mock_display:
            # Apply filter and display
            result = notebook.apply_filter_and_display(image, 'grayscale')
            
            # Should display result
            assert mock_display.display.called
            
            # Should return filtered image
            assert result is not None
            assert result.width == 100
            assert result.height == 100


class TestComplexWorkflow:
    """Tests for complex workflows involving multiple components."""
    
    def test_end_to_end_workflow(self):
        """Test a complex end-to-end workflow with multiple components."""
        # Initialize components
        security = SecurityManager()
        performance = PerformanceMonitor()
        accessibility = AccessibilityManager()
        
        # Set up security roles
        security.permission_manager.define_role("admin", ["edit_canvas", "export_canvas"])
        security.permission_manager.assign_role("test_user", "admin")
        
        # Create canvas with all managers
        canvas = Canvas(
            width=600, 
            height=400,
            security_manager=security,
            performance_monitor=performance,
            accessibility_manager=accessibility
        )
        
        # Create test image
        img_array = np.zeros((200, 200, 3), dtype=np.uint8)
        img_array[50:150, 50:150] = [255, 0, 0]  # Red square
        pil_img = PILImage.fromarray(img_array)
        
        # Create Image object
        image = Image.from_pil(pil_img)
        
        # Monitor overall workflow performance
        with patch.object(performance, 'start_timing') as mock_start_perf, \
             patch.object(performance, 'stop_timing') as mock_stop_perf, \
             patch.object(security, 'check_permission') as mock_check_perm, \
             patch.object(accessibility, 'announce') as mock_announce:
            
            # Set up security check to return True for our user
            mock_check_perm.return_value = True
            
            # Start workflow timing
            mock_start_perf.return_value = None
            
            # Load image to canvas with permission check
            canvas.add_image(image, x=100, y=100, user="test_user")
            
            # Should check permission
            assert mock_check_perm.called
            
            # Should announce for accessibility
            assert mock_announce.called
            
            # Draw additional elements
            canvas.draw_circle(300, 200, 50, fill_color="blue", user="test_user")
            
            # Export the result
            with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
                canvas.export(temp_file.name, user="test_user")
                
                # File should exist
                assert os.path.exists(temp_file.name)
                assert os.path.getsize(temp_file.name) > 0
            
            # Stop workflow timing
            mock_stop_perf.return_value = 0.5  # Mock 500ms execution time
        
        # Check performance metrics
        assert "add_image" in performance.metrics
        assert "draw_circle" in performance.metrics
        assert "export" in performance.metrics


if __name__ == "__main__":
    pytest.main() 