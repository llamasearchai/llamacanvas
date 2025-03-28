"""
Tests for data integration functionality in LlamaCanvas.

This module contains tests for data integration features such as 
loading data from various sources, data visualization, and integration
with data processing pipelines.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from PIL import Image as PILImage

from llama_canvas.data import (
    load_csv,
    load_json,
    load_excel,
    load_database,
    DataLoader,
    DataProcessor,
    DataVisualizer,
    export_data,
    plot_data,
    create_chart,
    create_heatmap,
    create_scatter_plot,
    create_bar_chart,
    create_line_chart,
    overlay_data_on_image,
    generate_data_grid,
    convert_data_format
)


class TestDataLoading:
    """Tests for data loading functionality."""
    
    @pytest.fixture
    def sample_csv_path(self, tmp_path):
        """Create a sample CSV file and return its path."""
        csv_path = os.path.join(tmp_path, "sample.csv")
        data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        data.to_csv(csv_path, index=False)
        return csv_path
    
    @pytest.fixture
    def sample_json_path(self, tmp_path):
        """Create a sample JSON file and return its path."""
        json_path = os.path.join(tmp_path, "sample.json")
        data = {
            'records': [
                {'id': 1, 'name': 'Alice', 'score': 95.5},
                {'id': 2, 'name': 'Bob', 'score': 82.0},
                {'id': 3, 'name': 'Charlie', 'score': 88.7}
            ],
            'metadata': {
                'source': 'test',
                'created': '2023-04-01'
            }
        }
        pd.DataFrame(data['records']).to_json(json_path, orient='records')
        return json_path
    
    @pytest.fixture
    def sample_excel_path(self, tmp_path):
        """Create a sample Excel file and return its path."""
        excel_path = os.path.join(tmp_path, "sample.xlsx")
        data = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'City': ['New York', 'Los Angeles', 'Chicago']
        })
        data.to_excel(excel_path, index=False)
        return excel_path
    
    def test_load_csv(self, sample_csv_path):
        """Test loading data from CSV file."""
        # Test with default parameters
        data = load_csv(sample_csv_path)
        
        # Should return a pandas DataFrame
        assert isinstance(data, pd.DataFrame)
        
        # Should have the correct shape
        assert data.shape == (5, 3)
        
        # Should have the correct column names
        assert list(data.columns) == ['A', 'B', 'C']
        
        # Test with custom parameters
        data_custom = load_csv(sample_csv_path, usecols=['A', 'C'], skiprows=1)
        
        # Should have only selected columns
        assert data_custom.shape[1] == 2
        assert list(data_custom.columns) == ['A', 'C']
        
        # Test with error handling
        with pytest.raises(FileNotFoundError):
            load_csv("nonexistent.csv")
    
    def test_load_json(self, sample_json_path):
        """Test loading data from JSON file."""
        # Test with default parameters
        data = load_json(sample_json_path)
        
        # Should return a pandas DataFrame
        assert isinstance(data, pd.DataFrame)
        
        # Should have the correct shape and columns
        assert data.shape[0] == 3  # 3 records
        assert 'id' in data.columns
        assert 'name' in data.columns
        assert 'score' in data.columns
        
        # Test with custom parameters
        data_custom = load_json(sample_json_path, orient='records')
        
        # Should still be a DataFrame
        assert isinstance(data_custom, pd.DataFrame)
        
        # Test with error handling
        with pytest.raises(FileNotFoundError):
            load_json("nonexistent.json")
    
    def test_load_excel(self, sample_excel_path):
        """Test loading data from Excel file."""
        # Test with default parameters
        data = load_excel(sample_excel_path)
        
        # Should return a pandas DataFrame
        assert isinstance(data, pd.DataFrame)
        
        # Should have the correct shape
        assert data.shape == (3, 3)
        
        # Should have the correct column names
        assert list(data.columns) == ['Name', 'Age', 'City']
        
        # Test with custom parameters
        data_custom = load_excel(sample_excel_path, usecols=[0, 1], skiprows=1)
        
        # Should have only selected columns
        assert data_custom.shape[1] == 2
        
        # Test with error handling
        with pytest.raises(FileNotFoundError):
            load_excel("nonexistent.xlsx")
    
    def test_load_database(self):
        """Test loading data from a database."""
        with patch('llama_canvas.data.create_engine') as mock_create_engine:
            # Mock the database engine and query
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            
            # Mock the pandas read_sql function
            with patch('llama_canvas.data.pd.read_sql') as mock_read_sql:
                mock_df = pd.DataFrame({
                    'id': [1, 2, 3],
                    'name': ['Alice', 'Bob', 'Charlie']
                })
                mock_read_sql.return_value = mock_df
                
                # Test loading with a query
                data = load_database("SELECT * FROM users", "sqlite:///test.db")
                
                # Should call engine creation
                assert mock_create_engine.called
                assert mock_create_engine.call_args[0][0] == "sqlite:///test.db"
                
                # Should call read_sql
                assert mock_read_sql.called
                assert mock_read_sql.call_args[0][0] == "SELECT * FROM users"
                assert mock_read_sql.call_args[0][1] == mock_engine
                
                # Should return a DataFrame
                assert isinstance(data, pd.DataFrame)
                assert data.equals(mock_df)
                
                # Test with table name
                mock_read_sql.reset_mock()
                data_table = load_database(table="users", connection_string="sqlite:///test.db")
                
                # Should call read_sql with table
                assert mock_read_sql.called
                assert mock_read_sql.call_args[0][0] == "users"
                
                # Test with error
                mock_read_sql.side_effect = Exception("Database error")
                with pytest.raises(Exception):
                    load_database("SELECT * FROM users", "sqlite:///test.db")


class TestDataProcessor:
    """Tests for data processing functionality."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'category': ['A', 'B', 'A', 'C', 'B'],
            'date': pd.date_range(start='2023-01-01', periods=5),
            'missing': [10, 20, None, 40, None]
        })
    
    def test_init(self, sample_dataframe):
        """Test DataProcessor initialization."""
        processor = DataProcessor(sample_dataframe)
        
        # Should store the DataFrame
        assert processor.data is sample_dataframe
        
        # Should initialize with default parameters
        assert processor.inplace is False
    
    def test_clean_data(self, sample_dataframe):
        """Test cleaning data."""
        processor = DataProcessor(sample_dataframe)
        
        # Test dropping NA values
        cleaned = processor.clean_data(drop_na=True)
        
        # Should return a DataFrame
        assert isinstance(cleaned, pd.DataFrame)
        
        # Should drop rows with NA
        assert cleaned.shape[0] == 3  # 2 rows with NA removed
        
        # Test filling NA values
        filled = processor.clean_data(fill_na={'missing': 0})
        
        # Should fill NA with specified value
        assert filled['missing'].isna().sum() == 0
        assert (filled['missing'].values == [10, 20, 0, 40, 0]).all()
        
        # Test with inplace=True
        processor_inplace = DataProcessor(sample_dataframe.copy(), inplace=True)
        result = processor_inplace.clean_data(drop_na=True)
        
        # Should modify the original DataFrame
        assert result is processor_inplace.data
        assert processor_inplace.data.shape[0] == 3
    
    def test_filter_data(self, sample_dataframe):
        """Test filtering data."""
        processor = DataProcessor(sample_dataframe)
        
        # Test filtering with a condition
        filtered = processor.filter_data(lambda df: df['numeric'] > 2)
        
        # Should return a DataFrame
        assert isinstance(filtered, pd.DataFrame)
        
        # Should only include rows matching the condition
        assert filtered.shape[0] == 3  # Only rows with numeric > 2
        assert (filtered['numeric'].values > 2).all()
        
        # Test with column selection
        filtered_cols = processor.filter_data(None, columns=['numeric', 'category'])
        
        # Should only include selected columns
        assert filtered_cols.shape[1] == 2
        assert list(filtered_cols.columns) == ['numeric', 'category']
        
        # Test with both condition and columns
        filtered_both = processor.filter_data(
            lambda df: df['category'] == 'A',
            columns=['category', 'numeric']
        )
        
        # Should apply both filters
        assert filtered_both.shape == (2, 2)  # Only category='A' rows, and 2 columns
        assert (filtered_both['category'] == 'A').all()
    
    def test_transform_data(self, sample_dataframe):
        """Test transforming data."""
        processor = DataProcessor(sample_dataframe)
        
        # Test applying a transformation
        transformed = processor.transform_data(lambda df: df.assign(doubled=df['numeric'] * 2))
        
        # Should return a DataFrame
        assert isinstance(transformed, pd.DataFrame)
        
        # Should include the new column
        assert 'doubled' in transformed.columns
        assert (transformed['doubled'] == transformed['numeric'] * 2).all()
        
        # Test with column-specific transformations
        transforms = {
            'numeric': lambda x: x + 10,
            'category': lambda x: x.str.lower()
        }
        
        transformed_cols = processor.transform_data(None, column_transforms=transforms)
        
        # Should apply transformations to specified columns
        assert (transformed_cols['numeric'] == sample_dataframe['numeric'] + 10).all()
        assert (transformed_cols['category'] == sample_dataframe['category'].str.lower()).all()
    
    def test_aggregate_data(self, sample_dataframe):
        """Test aggregating data."""
        processor = DataProcessor(sample_dataframe)
        
        # Test grouping and aggregating
        aggregated = processor.aggregate_data(
            group_by='category',
            aggregations={'numeric': ['mean', 'sum']}
        )
        
        # Should return a DataFrame
        assert isinstance(aggregated, pd.DataFrame)
        
        # Should have the correct shape
        assert aggregated.shape == (3, 2)  # 3 unique categories, 2 aggregations
        
        # Should have correct column names
        assert 'numeric_mean' in aggregated.columns
        assert 'numeric_sum' in aggregated.columns
        
        # Test with custom naming
        custom_agg = processor.aggregate_data(
            group_by='category',
            aggregations={'numeric': [('avg', 'mean'), ('total', 'sum')]}
        )
        
        # Should use custom names
        assert 'numeric_avg' in custom_agg.columns
        assert 'numeric_total' in custom_agg.columns


class TestDataVisualizer:
    """Tests for data visualization functionality."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 15, 13, 17, 20],
            'category': ['A', 'B', 'A', 'B', 'C']
        })
    
    def test_init(self, sample_dataframe):
        """Test DataVisualizer initialization."""
        visualizer = DataVisualizer(sample_dataframe)
        
        # Should store the DataFrame
        assert visualizer.data is sample_dataframe
    
    def test_plot_data(self, sample_dataframe):
        """Test plotting data."""
        visualizer = DataVisualizer(sample_dataframe)
        
        with patch('llama_canvas.data.plt') as mock_plt:
            # Test basic plotting
            result = visualizer.plot_data(x='x', y='y')
            
            # Should call matplotlib
            assert mock_plt.figure.called
            
            # Should return a figure
            assert result is mock_plt.gcf.return_value
            
            # Test with additional parameters
            mock_plt.reset_mock()
            visualizer.plot_data(
                x='x', 
                y='y', 
                kind='scatter',
                color='red',
                title='Test Plot',
                xlabel='X Axis',
                ylabel='Y Axis'
            )
            
            # Should set title and labels
            assert mock_plt.title.called
            assert mock_plt.xlabel.called
            assert mock_plt.ylabel.called
    
    def test_create_chart(self, sample_dataframe):
        """Test creating different chart types."""
        visualizer = DataVisualizer(sample_dataframe)
        
        with patch('llama_canvas.data.plt') as mock_plt, \
             patch('llama_canvas.data.sns') as mock_sns:
            
            # Test bar chart
            visualizer.create_chart(
                x='category', 
                y='y', 
                chart_type='bar'
            )
            
            # Should call seaborn
            assert mock_sns.barplot.called
            
            # Test with different chart type
            mock_sns.reset_mock()
            visualizer.create_chart(
                x='x',
                y='y',
                chart_type='line'
            )
            
            # Should call appropriate seaborn function
            assert mock_sns.lineplot.called
            
            # Test with invalid chart type
            with pytest.raises(ValueError):
                visualizer.create_chart(x='x', y='y', chart_type='invalid')
    
    def test_create_heatmap(self, sample_dataframe):
        """Test creating heatmaps."""
        # Create correlation matrix
        corr_matrix = sample_dataframe[['x', 'y']].corr()
        
        visualizer = DataVisualizer(sample_dataframe)
        
        with patch('llama_canvas.data.plt') as mock_plt, \
             patch('llama_canvas.data.sns') as mock_sns:
            
            # Test creating heatmap
            result = visualizer.create_heatmap(data=corr_matrix)
            
            # Should call seaborn
            assert mock_sns.heatmap.called
            
            # Should return a figure
            assert result is mock_plt.gcf.return_value
    
    def test_overlay_data_on_image(self):
        """Test overlaying data on images."""
        # Create a sample image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[40:60, 40:60] = [255, 0, 0]  # Red square
        
        # Create sample data points
        data_points = pd.DataFrame({
            'x': [25, 50, 75],
            'y': [25, 50, 75],
            'value': [1, 2, 3]
        })
        
        with patch('llama_canvas.data.plt') as mock_plt, \
             patch('llama_canvas.data.PILImage') as mock_pil:
            
            # Mock PIL Image
            mock_pil.fromarray.return_value = MagicMock()
            
            # Test overlaying data
            result = overlay_data_on_image(
                image=img,
                data=data_points,
                x_col='x',
                y_col='y',
                value_col='value'
            )
            
            # Should use matplotlib
            assert mock_plt.figure.called
            assert mock_plt.imshow.called
            assert mock_plt.scatter.called
            
            # Should return an image
            assert result is not None
            
            # Test with heatmap overlay
            mock_plt.reset_mock()
            result_heatmap = overlay_data_on_image(
                image=img,
                data=data_points,
                x_col='x',
                y_col='y',
                value_col='value',
                overlay_type='heatmap'
            )
            
            # Should create heatmap
            assert mock_plt.figure.called
            assert mock_plt.imshow.called


class TestDataExport:
    """Tests for data export functionality."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'score': [95, 87, 92]
        })
    
    def test_export_csv(self, sample_dataframe, tmp_path):
        """Test exporting data to CSV."""
        output_path = os.path.join(tmp_path, "output.csv")
        
        # Test exporting
        result = export_data(sample_dataframe, output_path, format='csv')
        
        # Should return the output path
        assert result == output_path
        
        # File should exist
        assert os.path.exists(output_path)
        
        # Should be able to read back the same data
        loaded = pd.read_csv(output_path)
        assert loaded.shape == sample_dataframe.shape
        assert list(loaded.columns) == list(sample_dataframe.columns)
    
    def test_export_json(self, sample_dataframe, tmp_path):
        """Test exporting data to JSON."""
        output_path = os.path.join(tmp_path, "output.json")
        
        # Test exporting
        result = export_data(sample_dataframe, output_path, format='json')
        
        # Should return the output path
        assert result == output_path
        
        # File should exist
        assert os.path.exists(output_path)
        
        # Should be able to read back the same data
        loaded = pd.read_json(output_path)
        assert loaded.shape == sample_dataframe.shape
        assert list(loaded.columns) == list(sample_dataframe.columns)
    
    def test_export_excel(self, sample_dataframe, tmp_path):
        """Test exporting data to Excel."""
        output_path = os.path.join(tmp_path, "output.xlsx")
        
        # Test exporting
        result = export_data(sample_dataframe, output_path, format='excel')
        
        # Should return the output path
        assert result == output_path
        
        # File should exist
        assert os.path.exists(output_path)
        
        # Should be able to read back the same data
        loaded = pd.read_excel(output_path)
        assert loaded.shape == sample_dataframe.shape
        assert list(loaded.columns) == list(sample_dataframe.columns)
    
    def test_export_image(self, sample_dataframe, tmp_path):
        """Test exporting data visualization as image."""
        output_path = os.path.join(tmp_path, "output.png")
        
        with patch('llama_canvas.data.plt') as mock_plt:
            # Mock figure and saving
            mock_fig = MagicMock()
            mock_plt.figure.return_value = mock_fig
            
            # Test exporting with visualization
            result = export_data(
                sample_dataframe, 
                output_path, 
                format='image',
                visualization=lambda df: plt.scatter(df['id'], df['score'])
            )
            
            # Should call visualization function
            assert mock_plt.figure.called
            assert mock_fig.savefig.called
            
            # Should return the output path
            assert result == output_path
    
    def test_export_unsupported_format(self, sample_dataframe, tmp_path):
        """Test exporting with unsupported format."""
        output_path = os.path.join(tmp_path, "output.xyz")
        
        # Should raise an error
        with pytest.raises(ValueError):
            export_data(sample_dataframe, output_path, format='unsupported')


class TestDataIntegration:
    """Tests for data integration with other components."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'x': [100, 200, 300, 400],
            'y': [100, 200, 300, 400],
            'size': [10, 20, 30, 40],
            'color': ['red', 'green', 'blue', 'yellow']
        })
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        img[100:400, 100:400] = [200, 200, 200]  # Gray square
        return img
    
    def test_generate_data_grid(self, sample_dataframe):
        """Test generating a grid of data visualizations."""
        with patch('llama_canvas.data.plt') as mock_plt, \
             patch('llama_canvas.data.sns') as mock_sns:
            
            # Test generating grid
            grid = generate_data_grid(
                data=sample_dataframe,
                visualizations=[
                    {'x': 'x', 'y': 'y', 'kind': 'scatter'},
                    {'column': 'size', 'kind': 'hist'},
                    {'column': 'color', 'kind': 'count'}
                ],
                grid_size=(2, 2)
            )
            
            # Should create a figure with subplots
            assert mock_plt.figure.called
            assert mock_plt.subplots.called
            
            # Should return a figure
            assert grid is not None
    
    def test_convert_data_format(self, sample_dataframe):
        """Test converting data between formats."""
        # Test conversion to numpy array
        array = convert_data_format(sample_dataframe, target_format='numpy')
        
        # Should return a numpy array
        assert isinstance(array, np.ndarray)
        assert array.shape[0] == sample_dataframe.shape[0]
        
        # Test conversion to dict
        data_dict = convert_data_format(sample_dataframe, target_format='dict')
        
        # Should return a dictionary
        assert isinstance(data_dict, dict)
        assert list(data_dict.keys()) == list(sample_dataframe.columns)
        
        # Test conversion to list
        data_list = convert_data_format(sample_dataframe, target_format='records')
        
        # Should return a list of records
        assert isinstance(data_list, list)
        assert len(data_list) == sample_dataframe.shape[0]
        
        # Test with unsupported format
        with pytest.raises(ValueError):
            convert_data_format(sample_dataframe, target_format='unsupported')


if __name__ == "__main__":
    pytest.main() 