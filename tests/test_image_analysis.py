"""
Tests for image analysis functionality in LlamaCanvas.

This module contains tests for image analysis features such as
feature detection, object recognition, and image comparison.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage

from llama_canvas.analysis import (
    analyze_colors,
    calculate_histogram,
    classify_image,
    compare_images,
    detect_faces,
    detect_features,
    detect_objects,
    detect_text,
    extract_metadata,
    get_dominant_colors,
    get_image_stats,
    segment_image,
)


class TestFeatureDetection:
    """Tests for feature detection utilities."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        # Create a gradient image with some "features"
        width, height = 200, 200
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Add gradient background
        for y in range(height):
            for x in range(width):
                img[y, x, 0] = int(255 * x / width)  # Red increases from left to right
                img[y, x, 1] = int(
                    255 * y / height
                )  # Green increases from top to bottom
                img[y, x, 2] = 128  # Blue is constant

        # Add a "corner" feature - a white square
        img[20:40, 20:40] = [255, 255, 255]

        # Add an "edge" feature - a vertical line
        img[:, 100:105] = [0, 0, 0]

        return img

    @pytest.fixture
    def sample_image_path(self, sample_image, tmp_path):
        """Save the sample image to a temporary file and return the path."""
        img_path = os.path.join(tmp_path, "sample.jpg")
        PILImage.fromarray(sample_image).save(img_path)
        return img_path

    def test_detect_features(self, sample_image):
        """Test detecting features in an image."""
        # Test with default parameters
        features = detect_features(sample_image)

        # Should return a dictionary of features
        assert isinstance(features, dict)
        assert "keypoints" in features
        assert "descriptors" in features

        # Should detect some features
        assert len(features["keypoints"]) > 0

        # Test with different algorithm
        features_orb = detect_features(sample_image, method="orb")
        assert isinstance(features_orb, dict)
        assert len(features_orb["keypoints"]) > 0

        # Test with invalid algorithm
        with pytest.raises(ValueError, match="method"):
            detect_features(sample_image, method="invalid")

        # Test with PIL image
        pil_img = PILImage.fromarray(sample_image)
        features_pil = detect_features(pil_img)
        assert isinstance(features_pil, dict)
        assert len(features_pil["keypoints"]) > 0

        # Test with mask
        mask = np.zeros((sample_image.shape[0], sample_image.shape[1]), dtype=np.uint8)
        mask[50:150, 50:150] = 255  # Only detect features in center region
        features_masked = detect_features(sample_image, mask=mask)

        # Should detect fewer features with mask
        assert len(features_masked["keypoints"]) < len(features["keypoints"])

    def test_detect_edges(self, sample_image):
        """Test detecting edges in an image."""
        with patch("llama_canvas.analysis.cv2") as mock_cv2:
            # Mock the Canny edge detector
            mock_cv2.Canny.return_value = np.zeros_like(sample_image[:, :, 0])

            # Call the function (assuming it's using cv2.Canny internally)
            edges = detect_features(sample_image, feature_type="edges")

            # Should call Canny
            assert mock_cv2.Canny.called

            # Should return edges
            assert isinstance(edges, np.ndarray)

    def test_detect_corners(self, sample_image):
        """Test detecting corners in an image."""
        with patch("llama_canvas.analysis.cv2") as mock_cv2:
            # Mock the Harris corner detector
            mock_corners = np.zeros_like(sample_image[:, :, 0], dtype=np.float32)
            mock_corners[30, 30] = 0.5  # One corner
            mock_cv2.cornerHarris.return_value = mock_corners

            # Call the function (assuming it's using cv2.cornerHarris internally)
            corners = detect_features(sample_image, feature_type="corners")

            # Should call cornerHarris
            assert mock_cv2.cornerHarris.called

            # Should return corners
            assert isinstance(corners, np.ndarray)


class TestObjectDetection:
    """Tests for object detection utilities."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock object detection model."""
        mock = MagicMock()
        mock.detect.return_value = [
            {"label": "person", "confidence": 0.92, "bbox": [10, 10, 100, 200]},
            {"label": "dog", "confidence": 0.85, "bbox": [150, 50, 250, 150]},
        ]
        return mock

    def test_detect_objects(self, sample_image, mock_model):
        """Test detecting objects in an image."""
        with patch("llama_canvas.analysis.get_model", return_value=mock_model):
            # Test with default parameters
            objects = detect_objects(sample_image)

            # Should return a list of objects
            assert isinstance(objects, list)
            assert len(objects) == 2

            # Check object properties
            person = objects[0]
            assert person["label"] == "person"
            assert person["confidence"] > 0.9
            assert "bbox" in person

            # Test with minimum confidence
            objects_high_conf = detect_objects(sample_image, min_confidence=0.9)
            assert len(objects_high_conf) == 1  # Only person should be detected

            # Test with specific classes
            objects_filtered = detect_objects(sample_image, classes=["dog"])
            assert len(objects_filtered) == 1
            assert objects_filtered[0]["label"] == "dog"

            # Test with PIL image
            pil_img = PILImage.fromarray(sample_image)
            objects_pil = detect_objects(pil_img)
            assert len(objects_pil) == 2

    def test_detect_faces(self, sample_image):
        """Test detecting faces in an image."""
        with patch("llama_canvas.analysis.cv2") as mock_cv2:
            # Mock face detector
            mock_detector = MagicMock()
            mock_cv2.CascadeClassifier.return_value = mock_detector

            # Mock detection result
            mock_detector.detectMultiScale.return_value = np.array(
                [[50, 50, 100, 100], [200, 200, 80, 80]]  # x, y, width, height
            )

            # Detect faces
            faces = detect_faces(sample_image)

            # Should call detector
            assert mock_detector.detectMultiScale.called

            # Should return a list of faces
            assert isinstance(faces, list)
            assert len(faces) == 2

            # Check face properties
            face = faces[0]
            assert "bbox" in face
            assert "confidence" in face

            # Test with minimum size
            mock_detector.detectMultiScale.reset_mock()
            faces_min_size = detect_faces(sample_image, min_size=(50, 50))

            # Should call detector with min size
            assert mock_detector.detectMultiScale.called
            call_kwargs = mock_detector.detectMultiScale.call_args[1]
            assert call_kwargs["minSize"] == (50, 50)

    def test_detect_text(self, sample_image_path):
        """Test detecting text in an image."""
        with patch("llama_canvas.analysis.pytesseract") as mock_tesseract:
            # Mock OCR result
            mock_tesseract.image_to_data.return_value = """
            level page_num block_num par_num line_num word_num left top width height conf text
            1 1 0 0 0 0 0 0 100 100 -1 
            2 1 1 0 0 0 10 10 80 30 -1 
            3 1 1 1 0 0 10 10 80 30 -1 
            4 1 1 1 1 0 10 10 80 30 -1 
            5 1 1 1 1 1 10 10 80 30 95 Hello
            5 1 1 1 1 2 100 10 80 30 90 World
            """

            # Detect text
            text = detect_text(sample_image_path)

            # Should call OCR
            assert mock_tesseract.image_to_data.called

            # Should return a dictionary with text
            assert isinstance(text, dict)
            assert "text" in text
            assert "Hello World" in text["text"]

            # Should include words with confidence
            assert "words" in text
            assert len(text["words"]) == 2
            assert text["words"][0]["text"] == "Hello"
            assert text["words"][0]["confidence"] > 90

            # Test with numpy array input
            mock_tesseract.image_to_data.reset_mock()
            text_np = detect_text(np.array(PILImage.open(sample_image_path)))

            # Should still call OCR
            assert mock_tesseract.image_to_data.called

            # Should return text
            assert isinstance(text_np, dict)
            assert "text" in text_np

            # Test with language option
            mock_tesseract.image_to_data.reset_mock()
            text_lang = detect_text(sample_image_path, lang="fra")

            # Should call OCR with language
            assert mock_tesseract.image_to_data.called
            assert mock_tesseract.image_to_data.call_args[1]["lang"] == "fra"


class TestImageComparison:
    """Tests for image comparison utilities."""

    @pytest.fixture
    def similar_images(self):
        """Create a pair of similar images."""
        # Base image
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        for y in range(100):
            for x in range(100):
                img1[y, x] = [int(255 * x / 100), int(255 * y / 100), 128]

        # Similar image with slight differences
        img2 = img1.copy()
        img2[40:60, 40:60] = [200, 200, 200]  # Add a small region with different color

        return img1, img2

    @pytest.fixture
    def different_images(self):
        """Create a pair of different images."""
        # First image - color gradient
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        for y in range(100):
            for x in range(100):
                img1[y, x] = [int(255 * x / 100), int(255 * y / 100), 128]

        # Second image - checkerboard pattern
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        for y in range(100):
            for x in range(100):
                if (x // 10 + y // 10) % 2 == 0:
                    img2[y, x] = [255, 255, 255]
                else:
                    img2[y, x] = [0, 0, 0]

        return img1, img2

    def test_compare_images(self, similar_images, different_images):
        """Test comparing images."""
        img1_similar, img2_similar = similar_images
        img1_different, img2_different = different_images

        # Test with similar images
        result_similar = compare_images(img1_similar, img2_similar)

        # Should return a dictionary with similarity metrics
        assert isinstance(result_similar, dict)
        assert "mse" in result_similar  # Mean Squared Error
        assert "ssim" in result_similar  # Structural Similarity Index
        assert "histogram_correlation" in result_similar

        # Similar images should have high SSIM and low MSE
        assert result_similar["ssim"] > 0.8  # SSIM close to 1 means similar
        assert result_similar["mse"] < 1000  # MSE close to 0 means similar

        # Test with different images
        result_different = compare_images(img1_different, img2_different)

        # Different images should have low SSIM and high MSE
        assert result_different["ssim"] < 0.5
        assert result_different["mse"] > result_similar["mse"]

        # Test with PIL images
        pil1 = PILImage.fromarray(img1_similar)
        pil2 = PILImage.fromarray(img2_similar)
        result_pil = compare_images(pil1, pil2)

        # Should return similarity metrics
        assert isinstance(result_pil, dict)
        assert "ssim" in result_pil

        # Test with different comparison method
        result_hist = compare_images(img1_similar, img2_similar, method="histogram")

        # Should return histogram comparison
        assert isinstance(result_hist, dict)
        assert "histogram_correlation" in result_hist

        # Test with images of different sizes
        img_small = img1_similar[0:50, 0:50]
        with pytest.raises(ValueError):
            compare_images(img1_similar, img_small)

    def test_calculate_histogram(self, similar_images):
        """Test calculating image histograms."""
        img = similar_images[0]

        # Test with default parameters
        hist = calculate_histogram(img)

        # Should return a dictionary with histograms
        assert isinstance(hist, dict)
        assert "red" in hist
        assert "green" in hist
        assert "blue" in hist

        # Histograms should be numpy arrays
        assert isinstance(hist["red"], np.ndarray)

        # Should have the right number of bins
        assert len(hist["red"]) == 256  # Default bins

        # Test with different number of bins
        hist_bins = calculate_histogram(img, bins=32)
        assert len(hist_bins["red"]) == 32

        # Test with PIL image
        pil_img = PILImage.fromarray(img)
        hist_pil = calculate_histogram(pil_img)
        assert isinstance(hist_pil, dict)
        assert "red" in hist_pil

        # Test with grayscale image
        gray_img = np.mean(img, axis=2).astype(np.uint8)
        hist_gray = calculate_histogram(gray_img)
        assert isinstance(hist_gray, dict)
        assert "gray" in hist_gray


class TestColorAnalysis:
    """Tests for color analysis utilities."""

    @pytest.fixture
    def color_image(self):
        """Create an image with various colors for testing."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Red region (top-left)
        img[0:50, 0:50] = [255, 0, 0]

        # Green region (top-right)
        img[0:50, 50:100] = [0, 255, 0]

        # Blue region (bottom-left)
        img[50:100, 0:50] = [0, 0, 255]

        # Yellow region (bottom-right)
        img[50:100, 50:100] = [255, 255, 0]

        return img

    def test_analyze_colors(self, color_image):
        """Test analyzing colors in an image."""
        # Test with default parameters
        color_analysis = analyze_colors(color_image)

        # Should return a dictionary with color info
        assert isinstance(color_analysis, dict)
        assert "average_color" in color_analysis
        assert "dominant_colors" in color_analysis
        assert "color_histogram" in color_analysis

        # Average color should be the average of all regions
        avg_color = color_analysis["average_color"]
        assert isinstance(avg_color, tuple)
        assert len(avg_color) == 3
        assert avg_color[0] > 120  # Red component (average of regions)
        assert avg_color[1] > 120  # Green component (average of regions)
        assert avg_color[2] > 60  # Blue component (average of regions)

        # Dominant colors should include the main colors
        dom_colors = color_analysis["dominant_colors"]
        assert isinstance(dom_colors, list)
        assert len(dom_colors) >= 4  # Should find at least our 4 main colors

        # Red, green, blue, and yellow should be among dominant colors
        colors_found = [False, False, False, False]  # Red, Green, Blue, Yellow
        for color in dom_colors:
            color_rgb = color["color"]
            # Check for red region
            if color_rgb[0] > 200 and color_rgb[1] < 50 and color_rgb[2] < 50:
                colors_found[0] = True
            # Check for green region
            elif color_rgb[0] < 50 and color_rgb[1] > 200 and color_rgb[2] < 50:
                colors_found[1] = True
            # Check for blue region
            elif color_rgb[0] < 50 and color_rgb[1] < 50 and color_rgb[2] > 200:
                colors_found[2] = True
            # Check for yellow region
            elif color_rgb[0] > 200 and color_rgb[1] > 200 and color_rgb[2] < 50:
                colors_found[3] = True

        # All colors should be found
        assert all(colors_found)

        # Test with PIL image
        pil_img = PILImage.fromarray(color_image)
        color_analysis_pil = analyze_colors(pil_img)
        assert isinstance(color_analysis_pil, dict)

        # Test with more dominant colors
        color_analysis_more = analyze_colors(color_image, num_colors=10)
        assert len(color_analysis_more["dominant_colors"]) == 10

    def test_get_dominant_colors(self, color_image):
        """Test getting dominant colors."""
        # Test with default parameters
        dominant_colors = get_dominant_colors(color_image)

        # Should return a list of colors
        assert isinstance(dominant_colors, list)
        assert len(dominant_colors) == 5  # Default is 5 colors

        # Each entry should have color and percentage
        for color_entry in dominant_colors:
            assert "color" in color_entry
            assert "percentage" in color_entry
            assert isinstance(color_entry["color"], tuple)
            assert len(color_entry["color"]) == 3
            assert 0 <= color_entry["percentage"] <= 1

        # Test with different number of colors
        dominant_colors_8 = get_dominant_colors(color_image, n=8)
        assert len(dominant_colors_8) == 8

        # Test with PIL image
        pil_img = PILImage.fromarray(color_image)
        dominant_colors_pil = get_dominant_colors(pil_img)
        assert isinstance(dominant_colors_pil, list)
        assert len(dominant_colors_pil) == 5


class TestImageMetadata:
    """Tests for image metadata utilities."""

    @pytest.fixture
    def sample_image_with_metadata(self, tmp_path):
        """Create a sample image with metadata."""
        # Create a simple image
        img = PILImage.new("RGB", (100, 100), color=(73, 109, 137))

        # Add some EXIF data
        from PIL import ExifTags

        exif_data = img.getexif()

        # Set some EXIF tags
        # 271 = Make, 272 = Model, 306 = DateTime
        exif_data[271] = "LlamaCanvas"
        exif_data[272] = "TestModel"
        exif_data[306] = "2023:04:01 12:00:00"

        # Save the image with EXIF data
        img_path = os.path.join(tmp_path, "metadata.jpg")
        img.save(img_path, exif=exif_data)

        return img_path

    def test_extract_metadata(self, sample_image_with_metadata):
        """Test extracting metadata from an image."""
        # Extract metadata
        metadata = extract_metadata(sample_image_with_metadata)

        # Should return a dictionary with metadata
        assert isinstance(metadata, dict)
        assert "exif" in metadata
        assert "image" in metadata

        # Basic image info should be present
        assert metadata["image"]["width"] == 100
        assert metadata["image"]["height"] == 100
        assert metadata["image"]["format"].lower() in ["jpeg", "jpg"]

        # EXIF data should be present
        assert "Make" in metadata["exif"]
        assert metadata["exif"]["Make"] == "LlamaCanvas"
        assert "Model" in metadata["exif"]
        assert metadata["exif"]["Model"] == "TestModel"
        assert "DateTime" in metadata["exif"]
        assert metadata["exif"]["DateTime"] == "2023:04:01 12:00:00"

        # Test with PIL image input
        pil_img = PILImage.open(sample_image_with_metadata)
        metadata_pil = extract_metadata(pil_img)

        # Should still extract metadata
        assert isinstance(metadata_pil, dict)
        assert "exif" in metadata_pil

        # Test with image without metadata
        with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
            img = PILImage.new("RGB", (100, 100), color=(73, 109, 137))
            img.save(temp_file.name)

            # Extract metadata
            metadata_empty = extract_metadata(temp_file.name)

            # Should return a dictionary with basic info but minimal EXIF
            assert isinstance(metadata_empty, dict)
            assert "image" in metadata_empty
            assert "exif" in metadata_empty
            assert len(metadata_empty["exif"]) == 0  # No EXIF data

    def test_get_image_stats(self, color_image):
        """Test getting image statistics."""
        # Get image stats
        stats = get_image_stats(color_image)

        # Should return a dictionary with statistics
        assert isinstance(stats, dict)
        assert "dimensions" in stats
        assert "channels" in stats
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "histogram" in stats

        # Check dimensions
        assert stats["dimensions"] == (100, 100)

        # Check channels (R, G, B)
        assert stats["channels"] == 3

        # Check min, max, mean, std (should have values for each channel)
        assert len(stats["min"]) == 3
        assert len(stats["max"]) == 3
        assert len(stats["mean"]) == 3
        assert len(stats["std"]) == 3

        # Test with PIL image
        pil_img = PILImage.fromarray(color_image)
        stats_pil = get_image_stats(pil_img)
        assert isinstance(stats_pil, dict)
        assert "dimensions" in stats_pil

        # Test with grayscale image
        gray_img = np.mean(color_image, axis=2).astype(np.uint8)
        stats_gray = get_image_stats(gray_img)
        assert stats_gray["channels"] == 1


class TestImageClassification:
    """Tests for image classification utilities."""

    @pytest.fixture
    def mock_classifier(self):
        """Create a mock image classifier."""
        mock = MagicMock()
        mock.predict.return_value = [
            {"label": "cat", "confidence": 0.85},
            {"label": "dog", "confidence": 0.10},
            {"label": "bird", "confidence": 0.05},
        ]
        return mock

    def test_classify_image(self, color_image, mock_classifier):
        """Test classifying an image."""
        with patch("llama_canvas.analysis.get_model", return_value=mock_classifier):
            # Test with default parameters
            classification = classify_image(color_image)

            # Should return a list of class predictions
            assert isinstance(classification, list)
            assert len(classification) == 3

            # Check prediction properties
            top_pred = classification[0]
            assert top_pred["label"] == "cat"
            assert top_pred["confidence"] > 0.8

            # Test with top_k parameter
            classification_top1 = classify_image(color_image, top_k=1)
            assert len(classification_top1) == 1

            # Test with PIL image
            pil_img = PILImage.fromarray(color_image)
            classification_pil = classify_image(pil_img)
            assert isinstance(classification_pil, list)
            assert len(classification_pil) == 3

            # Test with different model
            with patch("llama_canvas.analysis.get_model", return_value=mock_classifier):
                classification_model = classify_image(color_image, model="resnet50")
                assert isinstance(classification_model, list)

    def test_segment_image(self, color_image):
        """Test segmenting an image."""
        with patch("llama_canvas.analysis.cv2") as mock_cv2:
            # Mock segmentation result
            mock_ret, mock_labels, mock_stats, mock_centroids = (
                True,
                np.zeros((5, 100, 100), dtype=np.int32),
                np.array([[0, 0, 0, 0, 0], [10, 10, 20, 20, 400]]),
                np.array([[0, 0], [20, 20]]),
            )
            mock_cv2.connectedComponentsWithStats.return_value = (
                mock_ret,
                mock_labels,
                mock_stats,
                mock_centroids,
            )

            # Segment image
            segments = segment_image(color_image)

            # Should call segmentation
            assert mock_cv2.connectedComponentsWithStats.called

            # Should return segments
            assert isinstance(segments, dict)
            assert "num_segments" in segments
            assert segments["num_segments"] == 4  # 4 regions in our test image
            assert "labels" in segments
            assert "stats" in segments
            assert "centroids" in segments

            # Test with different method
            mock_cv2.reset_mock()
            mock_cv2.watershed.return_value = np.zeros((100, 100), dtype=np.int32)

            segments_watershed = segment_image(color_image, method="watershed")

            # Should call watershed
            assert mock_cv2.watershed.called

            # Should return segments
            assert isinstance(segments_watershed, dict)

            # Test with invalid method
            with pytest.raises(ValueError):
                segment_image(color_image, method="invalid")


if __name__ == "__main__":
    pytest.main()
