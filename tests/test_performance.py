"""
Tests for performance monitoring and benchmarking in LlamaCanvas.

This module contains tests for performance monitoring, profiling,
benchmarking, and optimization features.
"""

import pytest
import time
import cProfile
import pstats
import io
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock, call
from PIL import Image as PILImage

from llama_canvas.performance import (
    PerformanceMonitor,
    Profiler,
    Benchmark,
    MemoryTracker,
    ResourceUsage,
    optimized_transform,
    cache_result,
    measure_execution_time,
    get_memory_usage,
    profile_function,
    benchmark_function
)


class TestPerformanceMonitor:
    """Tests for the PerformanceMonitor class."""
    
    def test_init(self):
        """Test initialization of PerformanceMonitor."""
        monitor = PerformanceMonitor()
        
        # Test default properties
        assert monitor.enabled is True
        assert monitor.metrics == {}
        assert monitor.start_time is not None
    
    def test_start_stop_timing(self):
        """Test timing operations."""
        monitor = PerformanceMonitor()
        
        # Start timing an operation
        monitor.start_timing("test_operation")
        
        # Sleep for a short time
        time.sleep(0.01)
        
        # Stop timing
        duration = monitor.stop_timing("test_operation")
        
        # Duration should be positive
        assert duration > 0
        
        # Metric should be recorded
        assert "test_operation" in monitor.metrics
        assert monitor.metrics["test_operation"]["count"] == 1
        assert monitor.metrics["test_operation"]["total_time"] > 0
        assert monitor.metrics["test_operation"]["last_time"] > 0
    
    def test_record_metric(self):
        """Test recording arbitrary metrics."""
        monitor = PerformanceMonitor()
        
        # Record a metric
        monitor.record_metric("memory_usage", 1024)
        
        # Metric should be recorded
        assert "memory_usage" in monitor.metrics
        assert monitor.metrics["memory_usage"]["value"] == 1024
        assert monitor.metrics["memory_usage"]["count"] == 1
        
        # Record again with a different value
        monitor.record_metric("memory_usage", 2048)
        
        # Metric should be updated
        assert monitor.metrics["memory_usage"]["value"] == 2048
        assert monitor.metrics["memory_usage"]["count"] == 2
        assert "avg_value" in monitor.metrics["memory_usage"]
        assert monitor.metrics["memory_usage"]["avg_value"] == 1536  # Average of 1024 and 2048
    
    def test_get_report(self):
        """Test generating a performance report."""
        monitor = PerformanceMonitor()
        
        # Record some metrics
        monitor.start_timing("operation1")
        time.sleep(0.01)
        monitor.stop_timing("operation1")
        
        monitor.start_timing("operation2")
        time.sleep(0.02)
        monitor.stop_timing("operation2")
        
        monitor.record_metric("memory", 1024)
        
        # Generate report
        report = monitor.get_report()
        
        # Report should include all metrics
        assert "operation1" in report
        assert "operation2" in report
        assert "memory" in report
        
        # Report should include total execution time
        assert "total_execution_time" in report
        
        # Check format
        assert isinstance(report, dict)
    
    def test_reset(self):
        """Test resetting the performance monitor."""
        monitor = PerformanceMonitor()
        
        # Record some metrics
        monitor.start_timing("operation")
        time.sleep(0.01)
        monitor.stop_timing("operation")
        
        monitor.record_metric("memory", 1024)
        
        # Reset monitor
        old_start_time = monitor.start_time
        monitor.reset()
        
        # Metrics should be cleared
        assert monitor.metrics == {}
        
        # Start time should be updated
        assert monitor.start_time > old_start_time


class TestProfiler:
    """Tests for the Profiler class."""
    
    def test_init(self):
        """Test initialization of Profiler."""
        profiler = Profiler()
        
        # Test default properties
        assert profiler.enabled is True
        assert profiler.profile_stats is None
    
    def test_start_stop_profiling(self):
        """Test profiling code execution."""
        profiler = Profiler()
        
        with patch('llama_canvas.performance.cProfile.Profile') as mock_profile:
            # Mock profiler
            mock_profiler = MagicMock()
            mock_profile.return_value = mock_profiler
            
            # Start profiling
            profiler.start()
            
            # Should create profiler
            assert mock_profile.called
            
            # Should enable profiler
            assert mock_profiler.enable.called
            
            # Execute some code
            for i in range(1000):
                _ = i * i
            
            # Stop profiling
            profiler.stop()
            
            # Should disable profiler
            assert mock_profiler.disable.called
            
            # Should create stats
            assert mock_profiler.create_stats.called
    
    def test_get_stats(self):
        """Test getting profile statistics."""
        profiler = Profiler()
        
        # Create mock stats
        mock_stats = MagicMock()
        profiler.profile_stats = mock_stats
        
        # Get stats as string
        with patch('llama_canvas.performance.io.StringIO') as mock_stringio:
            # Mock string buffer
            mock_buffer = MagicMock()
            mock_stringio.return_value = mock_buffer
            
            profiler.get_stats_string()
            
            # Should create string buffer
            assert mock_stringio.called
            
            # Should sort stats
            assert mock_stats.sort_stats.called
            
            # Should print stats to buffer
            assert mock_stats.print_stats.called
            
            # Should get buffer value
            assert mock_buffer.getvalue.called
    
    def test_profile_decorator(self):
        """Test profiling with decorator."""
        # Define a function to profile
        def function_to_profile():
            result = 0
            for i in range(1000):
                result += i
            return result
        
        # Apply profiling decorator
        decorated_function = profile_function(function_to_profile)
        
        with patch('llama_canvas.performance.cProfile.Profile') as mock_profile:
            # Mock profiler
            mock_profiler = MagicMock()
            mock_profile.return_value = mock_profiler
            
            # Call decorated function
            result = decorated_function()
            
            # Should profile function execution
            assert mock_profile.called
            assert mock_profiler.enable.called
            assert mock_profiler.disable.called
            
            # Should return correct result
            assert result == sum(range(1000))


class TestBenchmark:
    """Tests for the Benchmark class."""
    
    def test_init(self):
        """Test initialization of Benchmark."""
        benchmark = Benchmark("Test Benchmark")
        
        # Test default properties
        assert benchmark.name == "Test Benchmark"
        assert benchmark.results == {}
    
    def test_run_benchmark(self):
        """Test running benchmarks."""
        benchmark = Benchmark("Test Benchmark")
        
        # Define test functions
        def fast_function():
            return sum(range(1000))
        
        def slow_function():
            time.sleep(0.01)
            return sum(range(1000))
        
        # Run benchmark
        benchmark.add_function("fast", fast_function)
        benchmark.add_function("slow", slow_function)
        
        benchmark.run(iterations=3)
        
        # Results should be recorded
        assert "fast" in benchmark.results
        assert "slow" in benchmark.results
        
        # Each function should have multiple timing results
        assert len(benchmark.results["fast"]["times"]) == 3
        assert len(benchmark.results["slow"]["times"]) == 3
        
        # Slow function should take longer
        assert benchmark.results["slow"]["avg_time"] > benchmark.results["fast"]["avg_time"]
    
    def test_compare_results(self):
        """Test comparing benchmark results."""
        benchmark = Benchmark("Comparison")
        
        # Create sample results
        benchmark.results = {
            "function1": {
                "avg_time": 0.001,
                "min_time": 0.0005,
                "max_time": 0.0015,
                "times": [0.001, 0.0005, 0.0015]
            },
            "function2": {
                "avg_time": 0.002,
                "min_time": 0.0015,
                "max_time": 0.0025,
                "times": [0.002, 0.0015, 0.0025]
            }
        }
        
        # Compare results
        comparison = benchmark.compare()
        
        # Should compare all functions
        assert "function1" in comparison
        assert "function2" in comparison
        
        # Should calculate speedup
        assert comparison["function1"]["vs_function2"] < 1.0  # function1 is faster
        assert comparison["function2"]["vs_function1"] > 1.0  # function2 is slower
    
    def test_benchmark_decorator(self):
        """Test benchmarking with decorator."""
        # Define a function to benchmark
        def function_to_benchmark(iterations):
            result = 0
            for i in range(iterations):
                result += i
            return result
        
        # Apply benchmarking decorator
        decorated_function = benchmark_function(function_to_benchmark)
        
        # Call decorated function
        with patch('llama_canvas.performance.time.time') as mock_time:
            # Mock time to return increasing values
            mock_time.side_effect = [0.0, 0.1]  # 100ms difference
            
            result = decorated_function(1000)
            
            # Should measure time
            assert mock_time.call_count == 2
            
            # Should return correct result
            assert result == sum(range(1000))


class TestMemoryTracking:
    """Tests for memory tracking functionality."""
    
    def test_memory_tracker_init(self):
        """Test initialization of MemoryTracker."""
        tracker = MemoryTracker()
        
        # Test default properties
        assert tracker.enabled is True
        assert tracker.snapshots == []
    
    def test_take_snapshot(self):
        """Test taking memory snapshots."""
        tracker = MemoryTracker()
        
        with patch('llama_canvas.performance.psutil.Process') as mock_process:
            # Mock process and memory info
            mock_proc = MagicMock()
            mock_process.return_value = mock_proc
            mock_proc.memory_info.return_value = MagicMock(rss=1024*1024, vms=2*1024*1024)  # 1MB RSS, 2MB VMS
            
            # Take snapshot
            snapshot = tracker.take_snapshot("Initial")
            
            # Should record memory info
            assert snapshot["name"] == "Initial"
            assert snapshot["rss"] == 1024*1024
            assert snapshot["vms"] == 2*1024*1024
            assert "timestamp" in snapshot
            
            # Should add to snapshots list
            assert len(tracker.snapshots) == 1
            assert tracker.snapshots[0] is snapshot
    
    def test_compare_snapshots(self):
        """Test comparing memory snapshots."""
        tracker = MemoryTracker()
        
        # Create mock snapshots
        tracker.snapshots = [
            {"name": "Initial", "rss": 1024*1024, "vms": 2*1024*1024, "timestamp": time.time()},
            {"name": "After Operation", "rss": 1.5*1024*1024, "vms": 2.5*1024*1024, "timestamp": time.time() + 10}
        ]
        
        # Compare snapshots
        comparison = tracker.compare_snapshots(0, 1)
        
        # Should calculate differences
        assert comparison["rss_diff"] == 0.5*1024*1024  # 0.5MB increase
        assert comparison["vms_diff"] == 0.5*1024*1024  # 0.5MB increase
        assert comparison["rss_diff_pct"] == 50.0  # 50% increase
        assert comparison["vms_diff_pct"] == 25.0  # 25% increase
    
    def test_get_memory_usage(self):
        """Test get_memory_usage function."""
        with patch('llama_canvas.performance.psutil.Process') as mock_process:
            # Mock process and memory info
            mock_proc = MagicMock()
            mock_process.return_value = mock_proc
            mock_proc.memory_info.return_value = MagicMock(rss=1024*1024)  # 1MB
            
            # Get memory usage
            memory = get_memory_usage()
            
            # Should return memory in MB
            assert memory == 1.0


class TestOptimizations:
    """Tests for optimization utilities."""
    
    def test_optimized_transform(self):
        """Test optimized transform function."""
        # Create sample image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        pil_img = PILImage.fromarray(img)
        
        # Define transform function
        def transform_func(image):
            # Simple grayscale conversion
            return image.convert("L")
        
        # Apply optimized transform
        with patch('llama_canvas.performance.time.time') as mock_time:
            # Mock time to return increasing values
            mock_time.side_effect = [0.0, 0.1]  # 100ms difference
            
            result = optimized_transform(pil_img, transform_func)
            
            # Should return transformed image
            assert result.mode == "L"  # Grayscale mode
            
            # Should measure time
            assert mock_time.call_count == 2
    
    def test_cache_result(self):
        """Test result caching decorator."""
        # Define a function with cache
        call_count = 0
        
        @cache_result
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * x
        
        # Call function multiple times with same argument
        result1 = expensive_function(5)
        result2 = expensive_function(5)
        
        # Results should be identical
        assert result1 == result2
        
        # Function should only be called once
        assert call_count == 1
        
        # Call with different argument
        result3 = expensive_function(10)
        
        # Should call function again
        assert call_count == 2
        assert result3 == 100
    
    def test_measure_execution_time(self):
        """Test execution time measurement decorator."""
        # Define a function to measure
        @measure_execution_time
        def test_function():
            time.sleep(0.01)
            return 42
        
        with patch('llama_canvas.performance.time.time') as mock_time:
            # Mock time to return increasing values
            mock_time.side_effect = [0.0, 0.1]  # 100ms difference
            
            with patch('llama_canvas.performance.logging') as mock_logging:
                # Call function
                result = test_function()
                
                # Should return correct result
                assert result == 42
                
                # Should log execution time
                assert mock_logging.info.called
                assert "execution time" in mock_logging.info.call_args[0][0]
                assert "0.1" in mock_logging.info.call_args[0][0]


class TestResourceUsage:
    """Tests for the ResourceUsage class."""
    
    def test_init(self):
        """Test initialization of ResourceUsage."""
        usage = ResourceUsage()
        
        # Test default properties
        assert usage.enabled is True
        assert usage.interval == 1.0
        assert usage.history == []
        assert usage.running is False
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping resource monitoring."""
        usage = ResourceUsage()
        
        with patch('llama_canvas.performance.threading.Thread') as mock_thread:
            # Mock thread
            thread_instance = MagicMock()
            mock_thread.return_value = thread_instance
            
            # Start monitoring
            usage.start()
            
            # Should create and start thread
            assert mock_thread.called
            assert thread_instance.start.called
            assert usage.running is True
            
            # Stop monitoring
            usage.stop()
            
            # Should set running to False
            assert usage.running is False
    
    def test_collect_data(self):
        """Test collecting resource usage data."""
        usage = ResourceUsage()
        
        with patch('llama_canvas.performance.psutil') as mock_psutil:
            # Mock psutil functions
            mock_psutil.cpu_percent.return_value = 25.0
            mock_psutil.virtual_memory.return_value = MagicMock(percent=50.0)
            mock_psutil.disk_usage.return_value = MagicMock(percent=30.0)
            
            # Collect data
            data = usage._collect_data()
            
            # Should collect CPU, memory, and disk usage
            assert data["cpu_percent"] == 25.0
            assert data["memory_percent"] == 50.0
            assert data["disk_percent"] == 30.0
            assert "timestamp" in data
    
    def test_get_statistics(self):
        """Test getting resource usage statistics."""
        usage = ResourceUsage()
        
        # Create mock history
        usage.history = [
            {"cpu_percent": 20, "memory_percent": 40, "disk_percent": 30, "timestamp": time.time()},
            {"cpu_percent": 30, "memory_percent": 50, "disk_percent": 30, "timestamp": time.time() + 1},
            {"cpu_percent": 40, "memory_percent": 60, "disk_percent": 30, "timestamp": time.time() + 2}
        ]
        
        # Get statistics
        stats = usage.get_statistics()
        
        # Should calculate averages
        assert stats["avg_cpu_percent"] == 30.0
        assert stats["avg_memory_percent"] == 50.0
        assert stats["avg_disk_percent"] == 30.0
        
        # Should calculate max values
        assert stats["max_cpu_percent"] == 40.0
        assert stats["max_memory_percent"] == 60.0
        assert stats["max_disk_percent"] == 30.0
        
        # Should include data points count
        assert stats["data_points"] == 3


class TestIntegrationTests:
    """Integration tests for performance utilities."""
    
    def test_monitor_image_processing(self):
        """Test monitoring performance of image processing."""
        # Create test image
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        pil_img = PILImage.fromarray(img)
        
        # Create performance monitor
        monitor = PerformanceMonitor()
        
        # Create memory tracker
        memory_tracker = MemoryTracker()
        memory_tracker.take_snapshot("Before Processing")
        
        # Monitor image operations
        monitor.start_timing("resize")
        resized_img = pil_img.resize((500, 500))
        monitor.stop_timing("resize")
        
        monitor.start_timing("rotate")
        rotated_img = resized_img.rotate(45)
        monitor.stop_timing("rotate")
        
        monitor.start_timing("filter")
        filtered_img = rotated_img.filter(PILImage.BLUR)
        monitor.stop_timing("filter")
        
        # Take memory snapshot after processing
        memory_tracker.take_snapshot("After Processing")
        
        # Get performance report
        report = monitor.get_report()
        
        # Check that all operations were timed
        assert "resize" in report
        assert "rotate" in report
        assert "filter" in report
        
        # Compare memory usage before and after
        memory_comparison = memory_tracker.compare_snapshots(0, 1)
        
        # There should be some memory difference (positive or negative)
        assert "rss_diff" in memory_comparison
        
        # Test cleanup
        del pil_img
        del resized_img
        del rotated_img
        del filtered_img


if __name__ == "__main__":
    pytest.main() 