"""
Tests for notebook functionality in LlamaCanvas.

This module contains tests for Jupyter notebook integration and functionality.
"""

import os
import tempfile
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from llama_canvas.notebook import (
    init_notebook,
    display_image,
    create_canvas,
    create_interactive_canvas,
    create_grid,
    export_notebook,
    execute_cell,
    run_pipeline,
    register_magics,
    capture_output
)


class TestNotebookDisplay:
    """Tests for notebook display functions."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        # Create a gradient image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        for y in range(100):
            for x in range(100):
                img[y, x, 0] = int(255 * x / 100)  # Red increases from left to right
                img[y, x, 1] = int(255 * y / 100)  # Green increases from top to bottom
                img[y, x, 2] = 128                 # Blue is constant
        return img
    
    @pytest.fixture
    def mock_display(self):
        """Create a mock for IPython display."""
        with patch('llama_canvas.notebook.display') as mock_display:
            yield mock_display
    
    def test_init_notebook(self, mock_display):
        """Test initializing notebook environment."""
        with patch('llama_canvas.notebook.HTML') as mock_html:
            init_notebook()
            
            # Should call HTML with CSS
            assert mock_html.called
            # Should call display with HTML
            assert mock_display.called
            
            # The HTML should contain CSS styles
            css_content = mock_html.call_args[0][0]
            assert "css" in css_content.lower()
            assert ".llama-canvas" in css_content
    
    def test_display_image(self, sample_image, mock_display):
        """Test displaying images in notebook."""
        # Test with numpy array
        with patch('llama_canvas.notebook.Image') as mock_image:
            display_image(sample_image, title="Test Image")
            
            # Should convert image to HTML
            assert mock_image.called
            # Should display the image
            assert mock_display.called
            
            # Test with caption
            display_image(sample_image, title="Test Image", caption="This is a test")
            assert mock_display.call_count == 2
    
    def test_create_canvas(self, mock_display):
        """Test creating a canvas in notebook."""
        with patch('llama_canvas.notebook.Canvas') as MockCanvas, \
             patch('llama_canvas.notebook.HTML') as mock_html:
            
            # Mock Canvas instance
            mock_canvas = MagicMock()
            MockCanvas.return_value = mock_canvas
            mock_canvas.to_html.return_value = "<canvas></canvas>"
            
            # Create canvas
            canvas = create_canvas(width=500, height=300)
            
            # Should create Canvas instance with correct dimensions
            assert MockCanvas.called
            assert MockCanvas.call_args[1]['width'] == 500
            assert MockCanvas.call_args[1]['height'] == 300
            
            # Should create HTML from canvas
            assert mock_html.called
            
            # Should display canvas
            assert mock_display.called
            
            # Should return Canvas instance
            assert canvas is mock_canvas
    
    def test_create_interactive_canvas(self, mock_display):
        """Test creating an interactive canvas."""
        with patch('llama_canvas.notebook.widgets') as mock_widgets, \
             patch('llama_canvas.notebook.Canvas') as MockCanvas:
            
            # Mock Canvas instance
            mock_canvas = MagicMock()
            MockCanvas.return_value = mock_canvas
            
            # Mock interactive output
            mock_output = MagicMock()
            mock_widgets.Output.return_value = mock_output
            
            # Create interactive canvas
            canvas, controls = create_interactive_canvas(width=500, height=300)
            
            # Should create Canvas with correct dimensions
            assert MockCanvas.called
            assert MockCanvas.call_args[1]['width'] == 500
            assert MockCanvas.call_args[1]['height'] == 300
            
            # Should create widgets for controls
            assert mock_widgets.VBox.called  # Layout container
            assert mock_widgets.HBox.called  # Control grouping
            assert mock_widgets.Button.called  # At least one button
            assert mock_widgets.Dropdown.called  # At least one dropdown
            
            # Should create output widget
            assert mock_widgets.Output.called
            
            # Should display controls
            assert mock_display.called
            
            # Should return both canvas and controls
            assert canvas is mock_canvas
            assert controls is not None
    
    def test_create_grid(self, sample_image, mock_display):
        """Test creating a grid of images."""
        images = [sample_image] * 4
        titles = ["Image 1", "Image 2", "Image 3", "Image 4"]
        
        with patch('llama_canvas.notebook.Image') as mock_image, \
             patch('llama_canvas.notebook.HTML') as mock_html:
            
            # Mock Image to return HTML
            mock_image.return_value = "<img>"
            
            # Create grid
            grid = create_grid(images, titles=titles, cols=2)
            
            # Should create Image for each image
            assert mock_image.call_count == 4
            
            # Should create HTML with grid layout
            assert mock_html.called
            html_content = mock_html.call_args[0][0]
            
            # HTML should contain grid CSS
            assert "grid" in html_content.lower()
            assert "display: grid" in html_content.lower()
            
            # Should have the right number of columns
            assert "grid-template-columns" in html_content.lower()
            assert "repeat(2, 1fr)" in html_content.lower()
            
            # Should display HTML
            assert mock_display.called
            
            # Should return HTML
            assert grid is mock_html.return_value


class TestNotebookInteraction:
    """Tests for notebook interaction functions."""
    
    @pytest.fixture
    def mock_ipython(self):
        """Create a mock for IPython."""
        with patch('llama_canvas.notebook.get_ipython') as mock_get_ipython:
            mock_ipython = MagicMock()
            mock_get_ipython.return_value = mock_ipython
            yield mock_ipython
    
    def test_execute_cell(self, mock_ipython):
        """Test executing a cell."""
        # Mock execute result
        mock_result = MagicMock()
        mock_result.success = True
        mock_ipython.run_cell.return_value = mock_result
        
        # Execute cell
        result = execute_cell('print("Hello, world!")')
        
        # Should call run_cell
        assert mock_ipython.run_cell.called
        assert mock_ipython.run_cell.call_args[0][0] == 'print("Hello, world!")'
        
        # Should return result
        assert result is mock_result
        
        # Test with error
        mock_result.success = False
        mock_result.error_in_exec = ValueError("Test error")
        
        # Should raise error
        with pytest.raises(ValueError):
            execute_cell('raise ValueError("Test error")', raise_errors=True)
        
        # Should not raise error if raise_errors is False
        result = execute_cell('raise ValueError("Test error")', raise_errors=False)
        assert result is mock_result
    
    def test_run_pipeline(self, mock_ipython):
        """Test running a pipeline of cells."""
        # Mock execute results
        mock_results = [MagicMock() for _ in range(3)]
        for result in mock_results:
            result.success = True
        mock_ipython.run_cell.side_effect = mock_results
        
        # Create pipeline
        cells = [
            'import numpy as np',
            'data = np.zeros((10, 10))',
            'result = data.sum()'
        ]
        
        # Run pipeline
        results = run_pipeline(cells)
        
        # Should call run_cell for each cell
        assert mock_ipython.run_cell.call_count == 3
        
        # Should return all results
        assert results == mock_results
        
        # Test with error
        mock_results[1].success = False
        mock_results[1].error_in_exec = ValueError("Test error")
        mock_ipython.run_cell.side_effect = mock_results
        
        # Should stop on error
        with pytest.raises(ValueError):
            run_pipeline(cells, stop_on_error=True)
        
        # Should have run only the first cell
        assert mock_ipython.run_cell.call_count == 1
    
    def test_register_magics(self, mock_ipython):
        """Test registering magic commands."""
        # Register magics
        register_magics()
        
        # Should register magics with IPython
        assert mock_ipython.register_magics.called
    
    def test_capture_output(self, mock_ipython):
        """Test capturing output from a cell."""
        # Mock output
        mock_output = MagicMock()
        mock_output.stdout = "Hello, world!"
        mock_ipython.run_cell.return_value = mock_output
        
        # Capture output
        with patch('llama_canvas.notebook.CaptureOutput') as MockCapture:
            mock_capture = MagicMock()
            MockCapture.return_value = mock_capture
            mock_capture.__enter__.return_value = mock_capture
            mock_capture.stdout = "Hello, world!"
            
            with capture_output() as output:
                print("Hello, world!")
            
            # Should create CaptureOutput
            assert MockCapture.called
            
            # Should capture output
            assert output.stdout == "Hello, world!"


class TestNotebookExport:
    """Tests for notebook export functions."""
    
    @pytest.fixture
    def sample_notebook(self):
        """Create a sample notebook for testing."""
        return {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["# LlamaCanvas Test Notebook"]
                },
                {
                    "cell_type": "code",
                    "source": ["import numpy as np\n", "import matplotlib.pyplot as plt"],
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "source": ["plt.plot([1, 2, 3, 4])\n", "plt.show()"],
                    "outputs": [{"output_type": "display_data", "data": {}}]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "name": "python3",
                    "display_name": "Python 3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
    
    def test_export_notebook(self, sample_notebook, tmp_path):
        """Test exporting a notebook."""
        with patch('llama_canvas.notebook.get_ipython') as mock_get_ipython:
            # Mock get_notebook_path
            mock_ipython = MagicMock()
            mock_get_ipython.return_value = mock_ipython
            mock_ipython.kernel.path = str(tmp_path)
            
            # Mock nbformat
            with patch('llama_canvas.notebook.nbformat') as mock_nbformat:
                mock_nbformat.reads.return_value = sample_notebook
                mock_nbformat.writes.return_value = json.dumps(sample_notebook)
                
                # Test exporting as .py
                py_path = export_notebook(format="python", output_path=tmp_path / "test.py")
                
                # Should create .py file
                assert py_path.endswith(".py")
                assert os.path.exists(py_path)
                
                # Test exporting as .html
                html_path = export_notebook(format="html", output_path=tmp_path / "test.html")
                
                # Should create .html file
                assert html_path.endswith(".html")
                assert os.path.exists(html_path)
                
                # Test exporting as .pdf
                with patch('llama_canvas.notebook.subprocess.run') as mock_run:
                    mock_run.return_value.returncode = 0
                    
                    pdf_path = export_notebook(format="pdf", output_path=tmp_path / "test.pdf")
                    
                    # Should create .pdf file
                    assert pdf_path.endswith(".pdf")
                    
                    # Should call nbconvert
                    assert mock_run.called
                    args = mock_run.call_args[0][0]
                    assert "nbconvert" in args
                    assert "--to pdf" in " ".join(args)
                
                # Test with invalid format
                with pytest.raises(ValueError):
                    export_notebook(format="invalid")


if __name__ == "__main__":
    pytest.main() 