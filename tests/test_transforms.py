"""
Tests for transform functions in LlamaCanvas.

This module contains comprehensive tests for image and video transformation utilities.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage

from llama_canvas.utils.transforms import (
    add_audio_to_video,
    add_watermark,
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    apply_filter,
    blend_images,
    combine_frames,
    convert_image_format,
    create_thumbnail,
    crop_image,
    crop_video,
    extract_frame,
    flip_image,
    resize_image,
    resize_video,
    rotate_image,
    trim_video,
)


class TestImageTransforms:
    """Tests for image transformation functions."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image for testing."""
        # Create a gradient image
        width, height = 100, 100
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                img[y, x, 0] = int(255 * x / width)  # Red increases from left to right
                img[y, x, 1] = int(
                    255 * y / height
                )  # Green increases from top to bottom
                img[y, x, 2] = 128  # Blue is constant

        return img

    @pytest.fixture
    def sample_pil_image(self, sample_image):
        """Convert numpy image to PIL Image."""
        return PILImage.fromarray(sample_image)

    def test_resize_image(self, sample_image):
        """Test resizing images."""
        # Test resizing to specific dimensions
        resized = resize_image(sample_image, width=50, height=50)
        assert resized.shape == (50, 50, 3)

        # Test preserving aspect ratio with only width
        resized = resize_image(sample_image, width=50)
        assert resized.shape == (50, 50, 3)  # Should maintain square shape

        # Test preserving aspect ratio with only height
        resized = resize_image(sample_image, height=75)
        assert resized.shape == (75, 75, 3)

        # Test different interpolation methods
        resized_nearest = resize_image(
            sample_image, width=50, height=50, interpolation="nearest"
        )
        resized_bilinear = resize_image(
            sample_image, width=50, height=50, interpolation="bilinear"
        )

        # Different interpolation should produce different results
        assert not np.array_equal(resized_nearest, resized_bilinear)

        # Test with PIL image
        pil_img = PILImage.fromarray(sample_image)
        resized_pil = resize_image(pil_img, width=40, height=40)
        assert isinstance(resized_pil, np.ndarray)
        assert resized_pil.shape == (40, 40, 3)

    def test_crop_image(self, sample_image):
        """Test cropping images."""
        # Test standard crop
        cropped = crop_image(sample_image, x=25, y=25, width=50, height=50)
        assert cropped.shape == (50, 50, 3)

        # Top-left pixel of cropped should match the specified crop position
        assert np.array_equal(cropped[0, 0], sample_image[25, 25])

        # Test crop with coordinates outside the image
        with pytest.raises(ValueError):
            crop_image(sample_image, x=80, y=80, width=50, height=50)

        # Test crop with PIL image
        pil_img = PILImage.fromarray(sample_image)
        cropped_pil = crop_image(pil_img, x=10, y=10, width=30, height=40)
        assert isinstance(cropped_pil, np.ndarray)
        assert cropped_pil.shape == (40, 30, 3)

    def test_rotate_image(self, sample_image):
        """Test rotating images."""
        # Test 90 degree rotation
        rotated = rotate_image(sample_image, angle=90)
        assert rotated.shape == (
            100,
            100,
            3,
        )  # Shape should be the same for 90 degree rotation

        # Test that the pixels have been correctly rotated
        # After 90 degree rotation, the top-left pixel should be at bottom-left
        assert np.array_equal(rotated[-1, 0], sample_image[0, 0])

        # Test with expansion (shape changes for non-90 degree rotations)
        rotated_exp = rotate_image(sample_image, angle=45, expand=True)
        assert rotated_exp.shape[0] > 100 and rotated_exp.shape[1] > 100

        # Test with fill color
        rotated_fill = rotate_image(
            sample_image, angle=45, expand=True, fill_color=(255, 0, 0)
        )
        # The corners should be filled with the specified color
        assert np.array_equal(rotated_fill[0, 0], [255, 0, 0])

        # Test with PIL image
        pil_img = PILImage.fromarray(sample_image)
        rotated_pil = rotate_image(pil_img, angle=180)
        assert isinstance(rotated_pil, np.ndarray)
        assert rotated_pil.shape == (100, 100, 3)

    def test_flip_image(self, sample_image):
        """Test flipping images."""
        # Test horizontal flip
        flipped_h = flip_image(sample_image, horizontal=True, vertical=False)
        assert flipped_h.shape == sample_image.shape

        # First pixel of original should be last pixel of horizontally flipped
        assert np.array_equal(flipped_h[0, -1], sample_image[0, 0])

        # Test vertical flip
        flipped_v = flip_image(sample_image, horizontal=False, vertical=True)
        assert flipped_v.shape == sample_image.shape

        # First pixel of original should be at bottom row after vertical flip
        assert np.array_equal(flipped_v[-1, 0], sample_image[0, 0])

        # Test both horizontal and vertical flip
        flipped_both = flip_image(sample_image, horizontal=True, vertical=True)

        # Bottom-right pixel of flipped_both should be top-left pixel of original
        assert np.array_equal(flipped_both[-1, -1], sample_image[0, 0])

        # Test with PIL image
        pil_img = PILImage.fromarray(sample_image)
        flipped_pil = flip_image(pil_img, horizontal=True)
        assert isinstance(flipped_pil, np.ndarray)
        assert flipped_pil.shape == (100, 100, 3)

    def test_adjust_brightness(self, sample_image):
        """Test adjusting image brightness."""
        # Increase brightness
        brightened = adjust_brightness(sample_image, factor=1.5)

        # Image shape should remain the same
        assert brightened.shape == sample_image.shape

        # Pixels should be brighter
        assert np.mean(brightened) > np.mean(sample_image)

        # Decrease brightness
        darkened = adjust_brightness(sample_image, factor=0.5)

        # Pixels should be darker
        assert np.mean(darkened) < np.mean(sample_image)

        # Test with PIL image
        pil_img = PILImage.fromarray(sample_image)
        brightened_pil = adjust_brightness(pil_img, factor=1.2)
        assert isinstance(brightened_pil, np.ndarray)
        assert brightened_pil.shape == (100, 100, 3)

    def test_adjust_contrast(self, sample_image):
        """Test adjusting image contrast."""
        # Increase contrast
        high_contrast = adjust_contrast(sample_image, factor=2.0)

        # Image shape should remain the same
        assert high_contrast.shape == sample_image.shape

        # Variance should be higher with increased contrast
        assert np.var(high_contrast) > np.var(sample_image)

        # Decrease contrast
        low_contrast = adjust_contrast(sample_image, factor=0.5)

        # Variance should be lower with decreased contrast
        assert np.var(low_contrast) < np.var(sample_image)

        # Test with PIL image
        pil_img = PILImage.fromarray(sample_image)
        contrast_pil = adjust_contrast(pil_img, factor=1.5)
        assert isinstance(contrast_pil, np.ndarray)
        assert contrast_pil.shape == (100, 100, 3)

    def test_adjust_saturation(self, sample_image):
        """Test adjusting image saturation."""
        # Increase saturation
        saturated = adjust_saturation(sample_image, factor=2.0)

        # Image shape should remain the same
        assert saturated.shape == sample_image.shape

        # Decrease saturation (towards grayscale)
        desaturated = adjust_saturation(sample_image, factor=0.0)

        # Shape should remain the same
        assert desaturated.shape == sample_image.shape

        # With saturation=0, each pixel should have equal R, G, B (grayscale)
        for y in range(desaturated.shape[0]):
            for x in range(desaturated.shape[1]):
                pixel = desaturated[y, x]
                # The R, G, B values should be approximately equal
                assert abs(int(pixel[0]) - int(pixel[1])) <= 1
                assert abs(int(pixel[1]) - int(pixel[2])) <= 1

        # Test with PIL image
        pil_img = PILImage.fromarray(sample_image)
        sat_pil = adjust_saturation(pil_img, factor=1.5)
        assert isinstance(sat_pil, np.ndarray)
        assert sat_pil.shape == (100, 100, 3)

    def test_apply_filter(self, sample_image):
        """Test applying image filters."""
        # Test blur filter
        blurred = apply_filter(sample_image, filter_type="blur", radius=2)
        assert blurred.shape == sample_image.shape

        # Test sharpen filter
        sharpened = apply_filter(sample_image, filter_type="sharpen")
        assert sharpened.shape == sample_image.shape

        # Test edge detection filter
        edges = apply_filter(sample_image, filter_type="edge_enhance")
        assert edges.shape == sample_image.shape

        # Test emboss filter
        embossed = apply_filter(sample_image, filter_type="emboss")
        assert embossed.shape == sample_image.shape

        # Test invalid filter
        with pytest.raises(ValueError):
            apply_filter(sample_image, filter_type="invalid_filter")

        # Test with PIL image
        pil_img = PILImage.fromarray(sample_image)
        filtered_pil = apply_filter(pil_img, filter_type="blur")
        assert isinstance(filtered_pil, np.ndarray)
        assert filtered_pil.shape == (100, 100, 3)

    def test_blend_images(self, sample_image):
        """Test blending two images."""
        # Create a second image with different pattern
        second_img = np.zeros_like(sample_image)
        height, width = second_img.shape[:2]
        for y in range(height):
            for x in range(width):
                second_img[y, x, 0] = 128
                second_img[y, x, 1] = int(255 * x / width)
                second_img[y, x, 2] = int(255 * y / height)

        # Test blending with equal weights
        blended = blend_images(sample_image, second_img, alpha=0.5)
        assert blended.shape == sample_image.shape

        # Blend with 0.0 alpha should equal the base image
        blended_base = blend_images(sample_image, second_img, alpha=0.0)
        assert np.array_equal(blended_base, sample_image)

        # Blend with 1.0 alpha should equal the second image
        blended_second = blend_images(sample_image, second_img, alpha=1.0)
        assert np.array_equal(blended_second, second_img)

        # Test with different sized images
        smaller_img = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            blend_images(sample_image, smaller_img)

        # Test with PIL images
        pil_img1 = PILImage.fromarray(sample_image)
        pil_img2 = PILImage.fromarray(second_img)
        blended_pil = blend_images(pil_img1, pil_img2, alpha=0.3)
        assert isinstance(blended_pil, np.ndarray)
        assert blended_pil.shape == (100, 100, 3)

    def test_add_watermark(self, sample_image, sample_pil_image):
        """Test adding watermark to images."""
        # Create small watermark image
        watermark = np.ones((20, 60, 4), dtype=np.uint8) * 255
        watermark[:, :, 3] = 128  # Set alpha to 50%

        # Test adding watermark to numpy image
        watermarked = add_watermark(sample_image, watermark, position="bottom-right")
        assert watermarked.shape == sample_image.shape

        # Test with PIL images
        wm_pil = PILImage.fromarray(watermark)
        watermarked_pil = add_watermark(sample_pil_image, wm_pil, position="top-left")
        assert isinstance(watermarked_pil, np.ndarray)
        assert watermarked_pil.shape == (100, 100, 3)

        # Test invalid position
        with pytest.raises(ValueError):
            add_watermark(sample_image, watermark, position="invalid-position")

        # Test with text watermark
        text_watermarked = add_watermark(
            sample_image,
            text="LlamaCanvas",
            text_color=(255, 255, 255),
            text_size=12,
            position="center",
        )
        assert text_watermarked.shape == sample_image.shape

    def test_convert_image_format(self, sample_image, tmp_path):
        """Test converting image formats."""
        # Save sample image as PNG
        png_path = os.path.join(tmp_path, "test.png")
        pil_img = PILImage.fromarray(sample_image)
        pil_img.save(png_path)

        # Convert PNG to JPEG
        jpg_path = convert_image_format(png_path, output_format="JPEG")
        assert jpg_path.endswith(".jpg")
        assert os.path.exists(jpg_path)

        # Convert PNG to WebP
        webp_path = convert_image_format(png_path, output_format="WEBP")
        assert webp_path.endswith(".webp")
        assert os.path.exists(webp_path)

        # Convert with custom output path
        custom_path = os.path.join(tmp_path, "custom.gif")
        gif_path = convert_image_format(
            png_path, output_format="GIF", output_path=custom_path
        )
        assert gif_path == custom_path
        assert os.path.exists(gif_path)

        # Test with quality parameter (JPEG)
        jpg_low_quality = convert_image_format(
            png_path, output_format="JPEG", quality=10
        )
        jpg_high_quality = convert_image_format(
            png_path, output_format="JPEG", quality=95
        )

        # Low quality JPEG should be smaller in file size
        assert os.path.getsize(jpg_low_quality) < os.path.getsize(jpg_high_quality)

        # Test invalid format
        with pytest.raises(ValueError):
            convert_image_format(png_path, output_format="INVALID")

    def test_create_thumbnail(self, sample_image, tmp_path):
        """Test creating image thumbnails."""
        # Save sample image
        img_path = os.path.join(tmp_path, "test.png")
        pil_img = PILImage.fromarray(sample_image)
        pil_img.save(img_path)

        # Create thumbnail with default settings
        thumb_path = create_thumbnail(img_path)
        assert os.path.exists(thumb_path)

        # Thumbnail should be smaller than original
        thumb_img = PILImage.open(thumb_path)
        assert thumb_img.width <= 200 and thumb_img.height <= 200

        # Create thumbnail with custom size
        custom_thumb = create_thumbnail(img_path, max_size=(50, 50))
        thumb_img = PILImage.open(custom_thumb)
        assert thumb_img.width <= 50 and thumb_img.height <= 50

        # Test with custom output path
        output_path = os.path.join(tmp_path, "custom_thumb.jpg")
        thumb_path = create_thumbnail(img_path, output_path=output_path)
        assert thumb_path == output_path
        assert os.path.exists(thumb_path)

        # Test with PIL image input
        pil_thumb = create_thumbnail(pil_img, return_image=True)
        assert isinstance(pil_thumb, PILImage.Image)
        assert pil_thumb.width <= 200 and pil_thumb.height <= 200


class TestVideoTransforms:
    """Tests for video transformation functions."""

    @pytest.fixture
    def sample_video_path(self, tmp_path):
        """Create a mock video file path."""
        return os.path.join(tmp_path, "sample.mp4")

    def test_resize_video(self, sample_video_path):
        """Test resizing videos."""
        with patch("llama_canvas.utils.transforms.subprocess.run") as mock_run:
            # Mock successful process
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_run.return_value = mock_process

            # Test basic resize
            output_path = resize_video(sample_video_path, width=640, height=480)

            # Check ffmpeg was called with correct parameters
            assert mock_run.called
            args = mock_run.call_args[0][0]
            assert "-vf" in args
            assert "scale=640:480" in args[args.index("-vf") + 1]
            assert os.path.basename(output_path).endswith("_resized.mp4")

            # Test with custom output path
            custom_path = os.path.join(os.path.dirname(sample_video_path), "custom.mp4")
            output_path = resize_video(
                sample_video_path, width=320, height=240, output_path=custom_path
            )
            assert output_path == custom_path

            # Test with keeping aspect ratio
            _ = resize_video(sample_video_path, width=720, keep_aspect_ratio=True)
            args = mock_run.call_args[0][0]
            assert "-vf" in args
            assert "scale=720:-1" in args[args.index("-vf") + 1]

            # Test with error
            mock_process.returncode = 1
            with pytest.raises(RuntimeError):
                resize_video(sample_video_path, width=640, height=480)

    def test_crop_video(self, sample_video_path):
        """Test cropping videos."""
        with patch("llama_canvas.utils.transforms.subprocess.run") as mock_run:
            # Mock successful process
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_run.return_value = mock_process

            # Test basic crop
            output_path = crop_video(
                sample_video_path, x=100, y=100, width=400, height=300
            )

            # Check ffmpeg was called with correct parameters
            assert mock_run.called
            args = mock_run.call_args[0][0]
            assert "-vf" in args
            assert "crop=400:300:100:100" in args[args.index("-vf") + 1]
            assert os.path.basename(output_path).endswith("_cropped.mp4")

            # Test with custom output path
            custom_path = os.path.join(os.path.dirname(sample_video_path), "custom.mp4")
            output_path = crop_video(
                sample_video_path,
                x=50,
                y=50,
                width=200,
                height=200,
                output_path=custom_path,
            )
            assert output_path == custom_path

            # Test with error
            mock_process.returncode = 1
            with pytest.raises(RuntimeError):
                crop_video(sample_video_path, x=100, y=100, width=400, height=300)

    def test_trim_video(self, sample_video_path):
        """Test trimming videos."""
        with patch("llama_canvas.utils.transforms.subprocess.run") as mock_run:
            # Mock successful process
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_run.return_value = mock_process

            # Test basic trim
            output_path = trim_video(
                sample_video_path, start_time="00:00:10", end_time="00:00:20"
            )

            # Check ffmpeg was called with correct parameters
            assert mock_run.called
            args = mock_run.call_args[0][0]
            assert "-ss" in args
            assert "00:00:10" in args[args.index("-ss") + 1]
            assert "-to" in args
            assert "00:00:20" in args[args.index("-to") + 1]
            assert os.path.basename(output_path).endswith("_trimmed.mp4")

            # Test with seconds instead of time string
            _ = trim_video(sample_video_path, start_time=30, duration=15)
            args = mock_run.call_args[0][0]
            assert "-ss" in args
            assert "30" in args[args.index("-ss") + 1]
            assert "-t" in args
            assert "15" in args[args.index("-t") + 1]

            # Test with custom output path
            custom_path = os.path.join(os.path.dirname(sample_video_path), "custom.mp4")
            output_path = trim_video(
                sample_video_path, start_time=5, end_time=10, output_path=custom_path
            )
            assert output_path == custom_path

            # Test with error
            mock_process.returncode = 1
            with pytest.raises(RuntimeError):
                trim_video(sample_video_path, start_time=5, end_time=10)

    def test_extract_frame(self, sample_video_path):
        """Test extracting frames from videos."""
        with patch("llama_canvas.utils.transforms.subprocess.run") as mock_run:
            # Mock successful process
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_run.return_value = mock_process

            # Test extracting a single frame
            output_path = extract_frame(sample_video_path, time="00:00:10")

            # Check ffmpeg was called with correct parameters
            assert mock_run.called
            args = mock_run.call_args[0][0]
            assert "-ss" in args
            assert "00:00:10" in args[args.index("-ss") + 1]
            assert "-frames:v" in args
            assert "1" in args[args.index("-frames:v") + 1]
            assert output_path.endswith(".jpg")

            # Test with seconds instead of time string
            _ = extract_frame(sample_video_path, time=30)
            args = mock_run.call_args[0][0]
            assert "-ss" in args
            assert "30" in args[args.index("-ss") + 1]

            # Test with custom output format
            _ = extract_frame(sample_video_path, time=10, output_format="png")
            assert mock_run.call_args[0][0][-1].endswith(".png")

            # Test with custom output path
            custom_path = os.path.join(os.path.dirname(sample_video_path), "custom.jpg")
            output_path = extract_frame(
                sample_video_path, time=5, output_path=custom_path
            )
            assert output_path == custom_path

            # Test with error
            mock_process.returncode = 1
            with pytest.raises(RuntimeError):
                extract_frame(sample_video_path, time=5)

    def test_combine_frames(self, tmp_path):
        """Test combining frames into a video."""
        # Create some dummy frame files
        frame_paths = []
        for i in range(5):
            frame_path = os.path.join(tmp_path, f"frame_{i:04d}.jpg")
            # Create empty file
            with open(frame_path, "w") as f:
                f.write("")
            frame_paths.append(frame_path)

        with patch("llama_canvas.utils.transforms.subprocess.run") as mock_run:
            # Mock successful process
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_run.return_value = mock_process

            # Test basic frame combination
            output_path = combine_frames(frame_paths, fps=24)

            # Check ffmpeg was called with correct parameters
            assert mock_run.called
            args = mock_run.call_args[0][0]
            assert "-framerate" in args
            assert "24" in args[args.index("-framerate") + 1]
            assert os.path.basename(output_path).endswith(".mp4")

            # Test with pattern instead of list
            frame_pattern = os.path.join(tmp_path, "frame_%04d.jpg")
            _ = combine_frames(frame_pattern, fps=30)
            args = mock_run.call_args[0][0]
            assert "-framerate" in args
            assert "30" in args[args.index("-framerate") + 1]
            assert frame_pattern in args

            # Test with custom output path and format
            custom_path = os.path.join(tmp_path, "custom.avi")
            output_path = combine_frames(frame_paths, fps=15, output_path=custom_path)
            assert output_path == custom_path

            # Test with error
            mock_process.returncode = 1
            with pytest.raises(RuntimeError):
                combine_frames(frame_paths, fps=24)

    def test_add_audio_to_video(self, sample_video_path, tmp_path):
        """Test adding audio to a video."""
        # Create a dummy audio file
        audio_path = os.path.join(tmp_path, "audio.mp3")
        with open(audio_path, "w") as f:
            f.write("")

        with patch("llama_canvas.utils.transforms.subprocess.run") as mock_run:
            # Mock successful process
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_run.return_value = mock_process

            # Test basic audio addition
            output_path = add_audio_to_video(sample_video_path, audio_path)

            # Check ffmpeg was called with correct parameters
            assert mock_run.called
            args = mock_run.call_args[0][0]
            assert "-i" in args
            video_index = args.index("-i")
            audio_index = args.index("-i", video_index + 1)
            assert sample_video_path in args[video_index + 1]
            assert audio_path in args[audio_index + 1]
            assert "-c:v" in args
            assert "copy" in args[args.index("-c:v") + 1]
            assert os.path.basename(output_path).endswith("_with_audio.mp4")

            # Test with custom volume
            _ = add_audio_to_video(sample_video_path, audio_path, volume=0.5)
            args = mock_run.call_args[0][0]
            assert "-filter:a" in args
            assert "volume=0.5" in args[args.index("-filter:a") + 1]

            # Test with custom output path
            custom_path = os.path.join(tmp_path, "custom.mp4")
            output_path = add_audio_to_video(
                sample_video_path, audio_path, output_path=custom_path
            )
            assert output_path == custom_path

            # Test replacing original audio
            _ = add_audio_to_video(sample_video_path, audio_path, replace_audio=True)
            args = mock_run.call_args[0][0]
            assert "-map" in args
            assert "0:v:0" in args[args.index("-map") + 1]
            assert "-map" in args[args.index("-map") + 2 :]
            assert "1:a:0" in args[args.index("-map", args.index("-map") + 1) + 1]

            # Test with error
            mock_process.returncode = 1
            with pytest.raises(RuntimeError):
                add_audio_to_video(sample_video_path, audio_path)


if __name__ == "__main__":
    pytest.main()
