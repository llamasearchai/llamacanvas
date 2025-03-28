"""
Tests for file utility functions in LlamaCanvas.

This module contains comprehensive tests for file handling and utility functions.
"""

import os
import tempfile
import datetime
import shutil
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from llama_canvas.utils.file_utils import (
    ensure_directory, 
    get_file_extension, 
    safe_filename,
    is_valid_image_format,
    is_valid_video_format,
    get_mime_type,
    create_temp_file,
    create_temp_directory,
    get_file_size,
    human_readable_size,
    copy_file,
    move_file,
    list_files_by_extension
)


class TestDirectoryOperations:
    """Tests for directory handling functions."""
    
    def test_ensure_directory(self):
        """Test ensuring a directory exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with existing directory
            assert ensure_directory(temp_dir) == temp_dir
            
            # Test with new subdirectory
            new_dir = os.path.join(temp_dir, "new_subdir")
            assert ensure_directory(new_dir) == new_dir
            assert os.path.exists(new_dir)
            assert os.path.isdir(new_dir)
            
            # Test with nested directories
            nested_dir = os.path.join(temp_dir, "parent", "child", "grandchild")
            assert ensure_directory(nested_dir) == nested_dir
            assert os.path.exists(nested_dir)
            assert os.path.isdir(nested_dir)
            
            # Test with file path
            with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as temp_file:
                # Should return the directory containing the file
                assert ensure_directory(temp_file.name) == temp_dir
    
    def test_create_temp_directory(self):
        """Test creating a temporary directory."""
        # Create temp directory
        temp_dir = create_temp_directory()
        
        try:
            # Verify directory exists
            assert os.path.exists(temp_dir)
            assert os.path.isdir(temp_dir)
            
            # Test with prefix
            prefix_dir = create_temp_directory(prefix="llama_test_")
            try:
                assert os.path.exists(prefix_dir)
                assert os.path.basename(prefix_dir).startswith("llama_test_")
            finally:
                if os.path.exists(prefix_dir):
                    shutil.rmtree(prefix_dir)
            
            # Test with parent directory
            with tempfile.TemporaryDirectory() as parent_dir:
                child_dir = create_temp_directory(parent_dir=parent_dir)
                assert os.path.exists(child_dir)
                assert parent_dir in child_dir
        finally:
            # Clean up
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def test_list_files_by_extension(self):
        """Test listing files by extension."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files
            files = {
                "test1.png": "test content",
                "test2.jpg": "test content",
                "test3.png": "test content",
                "test4.txt": "test content",
                "test5.PNG": "test content",  # Test case sensitivity
            }
            
            for filename, content in files.items():
                with open(os.path.join(temp_dir, filename), "w") as f:
                    f.write(content)
            
            # List PNG files
            png_files = list_files_by_extension(temp_dir, ".png")
            assert len(png_files) == 2
            assert os.path.join(temp_dir, "test1.png") in png_files
            assert os.path.join(temp_dir, "test3.png") in png_files
            
            # List PNG files with case insensitivity
            png_files = list_files_by_extension(temp_dir, ".png", case_sensitive=False)
            assert len(png_files) == 3
            assert os.path.join(temp_dir, "test5.PNG") in png_files
            
            # List multiple extensions
            image_files = list_files_by_extension(temp_dir, [".png", ".jpg"])
            assert len(image_files) == 3
            
            # Test recursive
            # Create a subdirectory with more files
            subdir = os.path.join(temp_dir, "subdir")
            os.makedirs(subdir)
            with open(os.path.join(subdir, "sub1.png"), "w") as f:
                f.write("test content")
            
            # Non-recursive shouldn't find the subdirectory file
            png_files = list_files_by_extension(temp_dir, ".png")
            assert os.path.join(subdir, "sub1.png") not in png_files
            
            # Recursive should find it
            png_files = list_files_by_extension(temp_dir, ".png", recursive=True)
            assert os.path.join(subdir, "sub1.png") in png_files


class TestFileOperations:
    """Tests for file handling functions."""
    
    def test_copy_file(self):
        """Test copying a file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a source file
            source_path = os.path.join(temp_dir, "source.txt")
            with open(source_path, "w") as f:
                f.write("test content")
            
            # Copy to destination
            dest_path = os.path.join(temp_dir, "dest.txt")
            copy_file(source_path, dest_path)
            
            # Verify destination exists
            assert os.path.exists(dest_path)
            
            # Verify content was copied
            with open(dest_path, "r") as f:
                assert f.read() == "test content"
            
            # Test copying to nonexistent directory
            dest_dir = os.path.join(temp_dir, "new_dir")
            dest_path2 = os.path.join(dest_dir, "dest2.txt")
            
            copy_file(source_path, dest_path2, create_dirs=True)
            assert os.path.exists(dest_path2)
            
            # Test overwrite protection
            with open(dest_path, "w") as f:
                f.write("modified content")
            
            # Should not overwrite by default
            with pytest.raises(FileExistsError):
                copy_file(source_path, dest_path, overwrite=False)
            
            # Should overwrite with flag
            copy_file(source_path, dest_path, overwrite=True)
            with open(dest_path, "r") as f:
                assert f.read() == "test content"
    
    def test_move_file(self):
        """Test moving a file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a source file
            source_path = os.path.join(temp_dir, "source.txt")
            with open(source_path, "w") as f:
                f.write("test content")
            
            # Move to destination
            dest_path = os.path.join(temp_dir, "dest.txt")
            move_file(source_path, dest_path)
            
            # Verify source no longer exists
            assert not os.path.exists(source_path)
            
            # Verify destination exists
            assert os.path.exists(dest_path)
            
            # Verify content was moved
            with open(dest_path, "r") as f:
                assert f.read() == "test content"
            
            # Test moving to nonexistent directory
            source_path = os.path.join(temp_dir, "source2.txt")
            with open(source_path, "w") as f:
                f.write("test content 2")
            
            dest_dir = os.path.join(temp_dir, "new_dir")
            dest_path2 = os.path.join(dest_dir, "dest2.txt")
            
            move_file(source_path, dest_path2, create_dirs=True)
            assert not os.path.exists(source_path)
            assert os.path.exists(dest_path2)
    
    def test_create_temp_file(self):
        """Test creating a temporary file."""
        # Create temp file with no content
        temp_file = create_temp_file()
        
        try:
            # Verify file exists
            assert os.path.exists(temp_file)
            assert os.path.isfile(temp_file)
            
            # Test with content
            content_file = create_temp_file(content="test content")
            try:
                assert os.path.exists(content_file)
                with open(content_file, "r") as f:
                    assert f.read() == "test content"
            finally:
                if os.path.exists(content_file):
                    os.remove(content_file)
            
            # Test with suffix
            suffix_file = create_temp_file(suffix=".png")
            try:
                assert os.path.exists(suffix_file)
                assert suffix_file.endswith(".png")
            finally:
                if os.path.exists(suffix_file):
                    os.remove(suffix_file)
            
            # Test with directory
            with tempfile.TemporaryDirectory() as temp_dir:
                dir_file = create_temp_file(directory=temp_dir)
                assert os.path.exists(dir_file)
                assert temp_dir in dir_file
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_get_file_size(self):
        """Test getting file size."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Write some content
            content = "x" * 1024  # 1KB
            temp_file.write(content.encode())
            temp_file.flush()
            
            # Test getting size in bytes
            size = get_file_size(temp_file.name)
            assert size == 1024
            
            # Test with nonexistent file
            with pytest.raises(FileNotFoundError):
                get_file_size("/nonexistent/file.txt")
        
        # Clean up
        os.remove(temp_file.name)
    
    def test_human_readable_size(self):
        """Test converting size to human-readable format."""
        assert human_readable_size(0) == "0 B"
        assert human_readable_size(1023) == "1023 B"
        assert human_readable_size(1024) == "1.0 KB"
        assert human_readable_size(1048576) == "1.0 MB"
        assert human_readable_size(1073741824) == "1.0 GB"
        assert human_readable_size(1099511627776) == "1.0 TB"
        assert human_readable_size(1.5 * 1024) == "1.5 KB"
        
        # Test with custom precision
        assert human_readable_size(1.5 * 1024, precision=0) == "2 KB"
        assert human_readable_size(1.5 * 1024, precision=3) == "1.500 KB"


class TestFilePathOperations:
    """Tests for file path handling functions."""
    
    def test_get_file_extension(self):
        """Test getting file extension."""
        assert get_file_extension("image.png") == ".png"
        assert get_file_extension("path/to/document.pdf") == ".pdf"
        assert get_file_extension("file_without_extension") == ""
        assert get_file_extension("path/to/.hidden") == ".hidden"
        assert get_file_extension("multiple.dots.in.name.txt") == ".txt"
        
        # Test with Path object
        assert get_file_extension(Path("image.png")) == ".png"
        
        # With normalization
        assert get_file_extension("IMAGE.PNG", normalize=True) == ".png"
        assert get_file_extension("Mixed.Case.JPG", normalize=True) == ".jpg"
        
        # With default when no extension
        assert get_file_extension("no_extension", default=".txt") == ".txt"
        assert get_file_extension("", default=".default") == ".default"
    
    def test_safe_filename(self):
        """Test creating safe filenames."""
        assert safe_filename("hello world") == "hello_world"
        assert safe_filename("file/with\\invalid:chars?") == "file_with_invalid_chars_"
        assert safe_filename("..") == "__"  # Handle special cases
        assert safe_filename("a.b.c") == "a.b.c"  # Dots should be preserved
        assert safe_filename("  leading trailing  ") == "leading_trailing"  # Trim spaces
        
        # With timestamp
        with patch('datetime.datetime') as mock_datetime:
            mock_now = MagicMock()
            mock_now.strftime.return_value = "20240320_123456"
            mock_datetime.now.return_value = mock_now
            
            assert safe_filename("test", add_timestamp=True) == "test_20240320_123456"
            assert safe_filename("test.", add_timestamp=True) == "test_20240320_123456"
        
        # With maximum length
        long_name = "a" * 100
        assert len(safe_filename(long_name, max_length=50)) <= 50
        
        # With timestamp and max length
        with patch('datetime.datetime') as mock_datetime:
            mock_now = MagicMock()
            mock_now.strftime.return_value = "20240320_123456"
            mock_datetime.now.return_value = mock_now
            
            safe_name = safe_filename("test", add_timestamp=True, max_length=20)
            assert len(safe_name) <= 20
            assert "test" in safe_name
            
        # Empty input
        assert safe_filename("") == "untitled"
        assert safe_filename("", add_timestamp=True).startswith("untitled_")


class TestFileFormatValidation:
    """Tests for file format validation functions."""
    
    def test_is_valid_image_format(self):
        """Test validating image formats."""
        # Test common image formats
        assert is_valid_image_format("test.png") is True
        assert is_valid_image_format("test.jpg") is True
        assert is_valid_image_format("test.jpeg") is True
        assert is_valid_image_format("test.gif") is True
        assert is_valid_image_format("test.bmp") is True
        assert is_valid_image_format("test.webp") is True
        
        # Test case sensitivity
        assert is_valid_image_format("test.PNG") is True
        
        # Test invalid formats
        assert is_valid_image_format("test.txt") is False
        assert is_valid_image_format("test.mp4") is False
        assert is_valid_image_format("test") is False
        
        # Test with Path object
        assert is_valid_image_format(Path("test.png")) is True
        
        # Test with custom allowed formats
        assert is_valid_image_format("test.svg", allowed_formats=[".svg", ".png"]) is True
        assert is_valid_image_format("test.jpg", allowed_formats=[".svg", ".png"]) is False
    
    def test_is_valid_video_format(self):
        """Test validating video formats."""
        # Test common video formats
        assert is_valid_video_format("test.mp4") is True
        assert is_valid_video_format("test.avi") is True
        assert is_valid_video_format("test.mov") is True
        assert is_valid_video_format("test.mkv") is True
        assert is_valid_video_format("test.webm") is True
        
        # Test case sensitivity
        assert is_valid_video_format("test.MP4") is True
        
        # Test invalid formats
        assert is_valid_video_format("test.txt") is False
        assert is_valid_video_format("test.png") is False
        assert is_valid_video_format("test") is False
        
        # Test with Path object
        assert is_valid_video_format(Path("test.mp4")) is True
        
        # Test with custom allowed formats
        assert is_valid_video_format("test.flv", allowed_formats=[".flv", ".mp4"]) is True
        assert is_valid_video_format("test.mov", allowed_formats=[".flv", ".3gp"]) is False
    
    def test_get_mime_type(self):
        """Test getting MIME type for files."""
        # Test common types
        assert get_mime_type("test.png").startswith("image/png")
        assert get_mime_type("test.jpg").startswith("image/jpeg")
        assert get_mime_type("test.mp4").startswith("video/mp4")
        assert get_mime_type("test.txt").startswith("text/plain")
        assert get_mime_type("test.json").startswith("application/json")
        
        # Test with Path object
        assert get_mime_type(Path("test.png")).startswith("image/png")
        
        # Test nonexistent file (should guess based on extension)
        assert get_mime_type("/nonexistent/test.png").startswith("image/png")
        
        # Test with actual file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(b"test content")
            temp_file.flush()
            assert get_mime_type(temp_file.name).startswith("text/plain")
            os.remove(temp_file.name)


if __name__ == "__main__":
    pytest.main() 