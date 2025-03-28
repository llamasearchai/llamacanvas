"""
Tests for document processing functionality in LlamaCanvas.

This module contains tests for document features such as 
creating, manipulating, and rendering documents in various formats.
"""

import os
import tempfile
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image as PILImage

from llama_canvas.document import (
    Document,
    DocumentGenerator,
    TextElement,
    ImageElement,
    TableElement,
    ChartElement,
    Page,
    Section,
    Template,
    load_document,
    save_document,
    export_to_pdf,
    export_to_html,
    export_to_docx,
    export_to_markdown,
    render_document
)


class TestDocumentElements:
    """Tests for document elements."""
    
    def test_text_element(self):
        """Test creation and properties of text elements."""
        # Test with basic text
        text = TextElement("Hello World")
        
        # Should store text content
        assert text.content == "Hello World"
        
        # Should have default properties
        assert text.font_size is not None
        assert text.font_family is not None
        assert text.color is not None
        
        # Test with custom properties
        text_custom = TextElement(
            "Custom Text",
            font_size=16,
            font_family="Arial",
            color="#FF0000",
            bold=True,
            italic=False,
            alignment="center"
        )
        
        # Should store custom properties
        assert text_custom.font_size == 16
        assert text_custom.font_family == "Arial"
        assert text_custom.color == "#FF0000"
        assert text_custom.bold is True
        assert text_custom.italic is False
        assert text_custom.alignment == "center"
        
        # Test rendering
        with patch('llama_canvas.document.TextRenderer') as mock_renderer:
            mock_renderer.return_value.render.return_value = "rendered_text"
            
            rendered = text.render()
            
            # Should call renderer
            assert mock_renderer.called
            assert mock_renderer.return_value.render.called
            
            # Should return rendered content
            assert rendered == "rendered_text"
    
    def test_image_element(self):
        """Test creation and properties of image elements."""
        # Create a sample image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[40:60, 40:60] = [255, 0, 0]  # Red square
        pil_img = PILImage.fromarray(img)
        
        # Test with PIL image
        image = ImageElement(pil_img)
        
        # Should store image
        assert image.image is pil_img
        
        # Should have default properties
        assert image.width is not None
        assert image.height is not None
        
        # Test with numpy array
        image_np = ImageElement(img)
        
        # Should convert to PIL Image
        assert isinstance(image_np.image, PILImage.Image)
        
        # Test with custom properties
        image_custom = ImageElement(
            pil_img,
            width=200,
            height=150,
            caption="Test Image",
            border=True,
            alignment="right"
        )
        
        # Should store custom properties
        assert image_custom.width == 200
        assert image_custom.height == 150
        assert image_custom.caption == "Test Image"
        assert image_custom.border is True
        assert image_custom.alignment == "right"
        
        # Test with path
        with patch('llama_canvas.document.PILImage') as mock_pil:
            mock_pil.open.return_value = pil_img
            
            image_path = ImageElement("path/to/image.jpg")
            
            # Should call PIL.Image.open
            assert mock_pil.open.called
            assert mock_pil.open.call_args[0][0] == "path/to/image.jpg"
        
        # Test rendering
        with patch('llama_canvas.document.ImageRenderer') as mock_renderer:
            mock_renderer.return_value.render.return_value = "rendered_image"
            
            rendered = image.render()
            
            # Should call renderer
            assert mock_renderer.called
            assert mock_renderer.return_value.render.called
            
            # Should return rendered content
            assert rendered == "rendered_image"
    
    def test_table_element(self):
        """Test creation and properties of table elements."""
        # Test with basic data
        data = [
            ["Name", "Age", "City"],
            ["Alice", 25, "New York"],
            ["Bob", 30, "Los Angeles"],
            ["Charlie", 35, "Chicago"]
        ]
        
        table = TableElement(data)
        
        # Should store data
        assert table.data == data
        
        # Should have default properties
        assert table.headers is None  # Inferred from first row
        assert table.width is not None
        assert table.cell_padding is not None
        
        # Test with custom properties
        headers = ["Full Name", "Age (years)", "Location"]
        table_custom = TableElement(
            data,
            headers=headers,
            width=500,
            cell_padding=8,
            border_width=2,
            header_background="#EEEEEE",
            alignment="center"
        )
        
        # Should store custom properties
        assert table_custom.headers == headers
        assert table_custom.width == 500
        assert table_custom.cell_padding == 8
        assert table_custom.border_width == 2
        assert table_custom.header_background == "#EEEEEE"
        assert table_custom.alignment == "center"
        
        # Test rendering
        with patch('llama_canvas.document.TableRenderer') as mock_renderer:
            mock_renderer.return_value.render.return_value = "rendered_table"
            
            rendered = table.render()
            
            # Should call renderer
            assert mock_renderer.called
            assert mock_renderer.return_value.render.called
            
            # Should return rendered content
            assert rendered == "rendered_table"
    
    def test_chart_element(self):
        """Test creation and properties of chart elements."""
        # Test with basic data
        data = {
            'x': [1, 2, 3, 4, 5],
            'y': [10, 15, 13, 17, 20]
        }
        
        chart = ChartElement(data, chart_type="line")
        
        # Should store data and chart type
        assert chart.data == data
        assert chart.chart_type == "line"
        
        # Should have default properties
        assert chart.width is not None
        assert chart.height is not None
        assert chart.title is None
        
        # Test with custom properties
        chart_custom = ChartElement(
            data,
            chart_type="bar",
            width=500,
            height=300,
            title="Sample Chart",
            x_label="X Axis",
            y_label="Y Axis",
            colors=["#FF0000", "#00FF00", "#0000FF"],
            grid=True
        )
        
        # Should store custom properties
        assert chart_custom.chart_type == "bar"
        assert chart_custom.width == 500
        assert chart_custom.height == 300
        assert chart_custom.title == "Sample Chart"
        assert chart_custom.x_label == "X Axis"
        assert chart_custom.y_label == "Y Axis"
        assert chart_custom.colors == ["#FF0000", "#00FF00", "#0000FF"]
        assert chart_custom.grid is True
        
        # Test rendering
        with patch('llama_canvas.document.ChartRenderer') as mock_renderer:
            mock_renderer.return_value.render.return_value = "rendered_chart"
            
            rendered = chart.render()
            
            # Should call renderer
            assert mock_renderer.called
            assert mock_renderer.return_value.render.called
            
            # Should return rendered content
            assert rendered == "rendered_chart"


class TestDocumentStructure:
    """Tests for document structure components."""
    
    def test_page(self):
        """Test creation and properties of document pages."""
        # Test with default properties
        page = Page()
        
        # Should have default properties
        assert page.width is not None
        assert page.height is not None
        assert page.margin is not None
        assert page.elements == []
        
        # Test with custom properties
        page_custom = Page(
            width=800,
            height=1000,
            margin=50,
            background_color="#EEEEEE",
            header_text="Test Header",
            footer_text="Test Footer"
        )
        
        # Should store custom properties
        assert page_custom.width == 800
        assert page_custom.height == 1000
        assert page_custom.margin == 50
        assert page_custom.background_color == "#EEEEEE"
        assert page_custom.header_text == "Test Header"
        assert page_custom.footer_text == "Test Footer"
        
        # Test adding elements
        text = TextElement("Hello World")
        page.add_element(text)
        
        # Should add element to page
        assert len(page.elements) == 1
        assert page.elements[0] is text
        
        # Test adding multiple elements
        image = ImageElement(np.zeros((10, 10, 3), dtype=np.uint8))
        table = TableElement([["A", "B"], [1, 2]])
        page.add_elements([image, table])
        
        # Should add all elements
        assert len(page.elements) == 3
        assert page.elements[1] is image
        assert page.elements[2] is table
        
        # Test rendering
        with patch('llama_canvas.document.PageRenderer') as mock_renderer:
            mock_renderer.return_value.render.return_value = "rendered_page"
            
            rendered = page.render()
            
            # Should call renderer
            assert mock_renderer.called
            assert mock_renderer.return_value.render.called
            
            # Should return rendered content
            assert rendered == "rendered_page"
    
    def test_section(self):
        """Test creation and properties of document sections."""
        # Test with default properties
        section = Section("Test Section")
        
        # Should store title
        assert section.title == "Test Section"
        
        # Should have default properties
        assert section.pages == []
        
        # Test with custom properties
        section_custom = Section(
            "Custom Section",
            level=2,
            title_font_size=18,
            title_alignment="center"
        )
        
        # Should store custom properties
        assert section_custom.title == "Custom Section"
        assert section_custom.level == 2
        assert section_custom.title_font_size == 18
        assert section_custom.title_alignment == "center"
        
        # Test adding pages
        page = Page()
        section.add_page(page)
        
        # Should add page to section
        assert len(section.pages) == 1
        assert section.pages[0] is page
        
        # Test adding multiple pages
        page2 = Page()
        page3 = Page()
        section.add_pages([page2, page3])
        
        # Should add all pages
        assert len(section.pages) == 3
        assert section.pages[1] is page2
        assert section.pages[2] is page3
        
        # Test rendering
        with patch('llama_canvas.document.SectionRenderer') as mock_renderer:
            mock_renderer.return_value.render.return_value = "rendered_section"
            
            rendered = section.render()
            
            # Should call renderer
            assert mock_renderer.called
            assert mock_renderer.return_value.render.called
            
            # Should return rendered content
            assert rendered == "rendered_section"


class TestDocument:
    """Tests for the Document class."""
    
    def test_init(self):
        """Test Document initialization."""
        # Test with default properties
        doc = Document("Test Document")
        
        # Should store title
        assert doc.title == "Test Document"
        
        # Should have default properties
        assert doc.author is None
        assert doc.description is None
        assert doc.created_date is not None
        assert doc.sections == []
        
        # Test with custom properties
        doc_custom = Document(
            "Custom Document",
            author="Test Author",
            description="Test Description",
            created_date="2023-04-01",
            stylesheet={"body": {"font-family": "Arial"}}
        )
        
        # Should store custom properties
        assert doc_custom.title == "Custom Document"
        assert doc_custom.author == "Test Author"
        assert doc_custom.description == "Test Description"
        assert doc_custom.created_date == "2023-04-01"
        assert doc_custom.stylesheet == {"body": {"font-family": "Arial"}}
    
    def test_add_section(self):
        """Test adding sections to documents."""
        doc = Document("Test Document")
        
        # Test adding a section
        section = Section("Test Section")
        doc.add_section(section)
        
        # Should add section to document
        assert len(doc.sections) == 1
        assert doc.sections[0] is section
        
        # Test adding multiple sections
        section2 = Section("Section 2")
        section3 = Section("Section 3")
        doc.add_sections([section2, section3])
        
        # Should add all sections
        assert len(doc.sections) == 3
        assert doc.sections[1] is section2
        assert doc.sections[2] is section3
    
    def test_get_metadata(self):
        """Test retrieving document metadata."""
        doc = Document(
            "Test Document",
            author="Test Author",
            description="Test Description",
            created_date="2023-04-01"
        )
        
        # Get metadata
        metadata = doc.get_metadata()
        
        # Should return metadata dictionary
        assert isinstance(metadata, dict)
        assert metadata["title"] == "Test Document"
        assert metadata["author"] == "Test Author"
        assert metadata["description"] == "Test Description"
        assert metadata["created_date"] == "2023-04-01"
    
    def test_render(self):
        """Test document rendering."""
        doc = Document("Test Document")
        
        # Add a section with a page
        section = Section("Test Section")
        page = Page()
        page.add_element(TextElement("Hello World"))
        section.add_page(page)
        doc.add_section(section)
        
        with patch('llama_canvas.document.DocumentRenderer') as mock_renderer:
            mock_renderer.return_value.render.return_value = "rendered_document"
            
            # Test rendering with default format
            rendered = doc.render()
            
            # Should call renderer
            assert mock_renderer.called
            assert mock_renderer.return_value.render.called
            
            # Should return rendered content
            assert rendered == "rendered_document"
            
            # Test rendering with specific format
            mock_renderer.reset_mock()
            rendered_html = doc.render(format="html")
            
            # Should call renderer with format
            assert mock_renderer.called
            assert mock_renderer.return_value.render.called
            assert mock_renderer.return_value.render.call_args[1]["format"] == "html"


class TestDocumentIO:
    """Tests for document I/O operations."""
    
    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing."""
        doc = Document("Sample Document")
        
        # Add a section with a page
        section = Section("Sample Section")
        page = Page()
        page.add_element(TextElement("Hello World"))
        section.add_page(page)
        doc.add_section(section)
        
        return doc
    
    def test_save_load_document(self, sample_document, tmp_path):
        """Test saving and loading documents."""
        # Save document
        file_path = os.path.join(tmp_path, "document.json")
        
        with patch('llama_canvas.document.json') as mock_json:
            save_document(sample_document, file_path)
            
            # Should call json.dump
            assert mock_json.dump.called
        
        # Load document
        with patch('llama_canvas.document.json') as mock_json, \
             patch('llama_canvas.document.Document') as mock_document:
            
            mock_json.load.return_value = {"title": "Sample Document"}
            mock_document.from_dict.return_value = sample_document
            
            loaded_doc = load_document(file_path)
            
            # Should call json.load
            assert mock_json.load.called
            
            # Should call Document.from_dict
            assert mock_document.from_dict.called
            
            # Should return a Document
            assert loaded_doc is sample_document
    
    def test_export_to_pdf(self, sample_document, tmp_path):
        """Test exporting documents to PDF."""
        # Export document
        file_path = os.path.join(tmp_path, "document.pdf")
        
        with patch('llama_canvas.document.PDFExporter') as mock_exporter:
            mock_exporter.return_value.export.return_value = file_path
            
            result = export_to_pdf(sample_document, file_path)
            
            # Should call PDFExporter
            assert mock_exporter.called
            assert mock_exporter.return_value.export.called
            
            # Should return file path
            assert result == file_path
    
    def test_export_to_html(self, sample_document, tmp_path):
        """Test exporting documents to HTML."""
        # Export document
        file_path = os.path.join(tmp_path, "document.html")
        
        with patch('llama_canvas.document.HTMLExporter') as mock_exporter:
            mock_exporter.return_value.export.return_value = file_path
            
            result = export_to_html(sample_document, file_path)
            
            # Should call HTMLExporter
            assert mock_exporter.called
            assert mock_exporter.return_value.export.called
            
            # Should return file path
            assert result == file_path
    
    def test_export_to_docx(self, sample_document, tmp_path):
        """Test exporting documents to DOCX."""
        # Export document
        file_path = os.path.join(tmp_path, "document.docx")
        
        with patch('llama_canvas.document.DOCXExporter') as mock_exporter:
            mock_exporter.return_value.export.return_value = file_path
            
            result = export_to_docx(sample_document, file_path)
            
            # Should call DOCXExporter
            assert mock_exporter.called
            assert mock_exporter.return_value.export.called
            
            # Should return file path
            assert result == file_path
    
    def test_export_to_markdown(self, sample_document, tmp_path):
        """Test exporting documents to Markdown."""
        # Export document
        file_path = os.path.join(tmp_path, "document.md")
        
        with patch('llama_canvas.document.MarkdownExporter') as mock_exporter:
            mock_exporter.return_value.export.return_value = file_path
            
            result = export_to_markdown(sample_document, file_path)
            
            # Should call MarkdownExporter
            assert mock_exporter.called
            assert mock_exporter.return_value.export.called
            
            # Should return file path
            assert result == file_path


class TestDocumentGenerator:
    """Tests for document generation."""
    
    def test_init(self):
        """Test DocumentGenerator initialization."""
        # Test with default properties
        generator = DocumentGenerator()
        
        # Should have default properties
        assert generator.template is None
        assert generator.stylesheet is not None
        
        # Test with custom properties
        template = Template("path/to/template.json")
        stylesheet = {"body": {"font-family": "Arial"}}
        
        generator_custom = DocumentGenerator(
            template=template,
            stylesheet=stylesheet
        )
        
        # Should store custom properties
        assert generator_custom.template is template
        assert generator_custom.stylesheet is stylesheet
    
    def test_generate_from_data(self):
        """Test generating documents from data."""
        generator = DocumentGenerator()
        
        # Test data
        data = {
            "title": "Generated Document",
            "sections": [
                {
                    "title": "Section 1",
                    "content": "Section 1 content"
                },
                {
                    "title": "Section 2",
                    "content": "Section 2 content"
                }
            ]
        }
        
        with patch('llama_canvas.document.Document') as mock_document:
            mock_doc = MagicMock()
            mock_document.return_value = mock_doc
            
            # Generate document
            doc = generator.generate_from_data(data)
            
            # Should create Document
            assert mock_document.called
            assert mock_document.call_args[0][0] == "Generated Document"
            
            # Should add sections
            assert mock_doc.add_section.call_count == 2
            
            # Should return the document
            assert doc is mock_doc
    
    def test_generate_from_template(self):
        """Test generating documents from templates."""
        # Mock template
        template = MagicMock()
        template.apply.return_value = Document("Templated Document")
        
        generator = DocumentGenerator(template=template)
        
        # Test data
        data = {
            "title": "Document Title",
            "content": "Document content"
        }
        
        # Generate document
        doc = generator.generate_from_template(data)
        
        # Should call template.apply
        assert template.apply.called
        assert template.apply.call_args[0][0] == data
        
        # Should return the document
        assert isinstance(doc, Document)
        assert doc.title == "Templated Document"


class TestTemplate:
    """Tests for document templates."""
    
    def test_init(self):
        """Test Template initialization."""
        # Test with path
        with patch('llama_canvas.document.json') as mock_json:
            mock_json.load.return_value = {"title": "Template Title"}
            
            template = Template("path/to/template.json")
            
            # Should call json.load
            assert mock_json.load.called
        
        # Test with dict
        template_dict = {
            "title": "Template Title",
            "sections": [
                {
                    "title": "Section Template",
                    "elements": ["text"]
                }
            ]
        }
        
        template = Template(template_dict)
        
        # Should store template data
        assert template.data == template_dict
    
    def test_apply(self):
        """Test applying templates to data."""
        # Create template
        template_dict = {
            "title": "{{title}}",
            "author": "{{author}}",
            "sections": [
                {
                    "title": "{{section_title}}",
                    "content": "{{section_content}}"
                }
            ]
        }
        
        template = Template(template_dict)
        
        # Test data
        data = {
            "title": "Document Title",
            "author": "Test Author",
            "section_title": "Section Title",
            "section_content": "Section content text"
        }
        
        with patch('llama_canvas.document.Document') as mock_document, \
             patch('llama_canvas.document.Section') as mock_section, \
             patch('llama_canvas.document.TextElement') as mock_text_element:
            
            mock_doc = MagicMock()
            mock_document.return_value = mock_doc
            
            mock_section_obj = MagicMock()
            mock_section.return_value = mock_section_obj
            
            # Apply template
            doc = template.apply(data)
            
            # Should create Document with substituted title and author
            assert mock_document.called
            assert mock_document.call_args[0][0] == "Document Title"
            assert mock_document.call_args[1]["author"] == "Test Author"
            
            # Should create Section with substituted title
            assert mock_section.called
            assert mock_section.call_args[0][0] == "Section Title"
            
            # Should add section to document
            assert mock_doc.add_section.called
            assert mock_doc.add_section.call_args[0][0] is mock_section_obj
            
            # Should return the document
            assert doc is mock_doc


def test_render_document():
    """Test document rendering function."""
    # Create document
    doc = Document("Test Document")
    
    # Mock rendering
    with patch.object(doc, 'render') as mock_render:
        mock_render.return_value = "rendered_content"
        
        # Test rendering with default format
        rendered = render_document(doc)
        
        # Should call document's render method
        assert mock_render.called
        
        # Should return rendered content
        assert rendered == "rendered_content"
        
        # Test rendering with specific format
        mock_render.reset_mock()
        rendered_html = render_document(doc, format="html")
        
        # Should call render with format
        assert mock_render.called
        assert mock_render.call_args[1]["format"] == "html"


if __name__ == "__main__":
    pytest.main() 