"""
Performance tests for AudioProcessor.
"""
import pytest
import time
import tempfile
import os
import psutil
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.extractors import discover_extractors
from tests.fixtures.audio_fixtures import AudioFixtures


@pytest.mark.performance
class TestExtractionPerformance:
    """Performance tests for audio extraction."""
    
    @pytest.fixture
    def performance_audio_files(self):
        """Create audio files of different sizes for performance testing."""
        files = {}
        
        # Create files of different durations
        durations = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        
        for duration in durations:
            audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=duration)
            audio = AudioFixtures.normalize_audio(audio)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, audio, 22050)
                files[f"{duration}s"] = tmp.name
        
        yield files
        
        # Cleanup
        for file_path in files.values():
            os.unlink(file_path)
    
    def test_single_extractor_performance(self, performance_audio_files):
        """Test performance of individual extractors."""
        from src.extractors.mfcc_extractor import MFCCExtractor
        
        extractor = MFCCExtractor()
        results = {}
        
        for name, file_path in performance_audio_files.items():
            with tempfile.TemporaryDirectory() as tmp_dir:
                start_time = time.time()
                result = extractor.run(file_path, tmp_dir)
                end_time = time.time()
                
                processing_time = end_time - start_time
                results[name] = {
                    'duration': float(name.replace('s', '')),
                    'processing_time': processing_time,
                    'success': result.success
                }
                
                assert result.success, f"Extraction failed for {name}"
        
        # Verify performance requirements
        for name, data in results.items():
            duration = data['duration']
            processing_time = data['processing_time']
            
            # Processing time should be less than 3x real-time
            max_processing_time = duration * 3
            assert processing_time < max_processing_time, \
                f"Processing time {processing_time:.2f}s exceeds limit {max_processing_time:.2f}s for {name}"
            
            # Processing time should be less than 30 seconds for any file
            assert processing_time < 30, \
                f"Processing time {processing_time:.2f}s exceeds 30s limit for {name}"
    
    def test_all_extractors_performance(self, performance_audio_files):
        """Test performance of all extractors."""
        extractors = discover_extractors()
        test_file = performance_audio_files["5s"]  # Use 5-second file
        
        results = {}
        
        for extractor in extractors:
            with tempfile.TemporaryDirectory() as tmp_dir:
                start_time = time.time()
                result = extractor.run(test_file, tmp_dir)
                end_time = time.time()
                
                processing_time = end_time - start_time
                results[extractor.name] = {
                    'processing_time': processing_time,
                    'success': result.success
                }
                
                assert result.success, f"Extractor {extractor.name} failed"
        
        # Verify all extractors complete within reasonable time
        for name, data in results.items():
            processing_time = data['processing_time']
            
            # Each extractor should complete within 20 seconds
            assert processing_time < 20, \
                f"Extractor {name} took {processing_time:.2f}s, exceeds 20s limit"
    
    def test_concurrent_extraction_performance(self, performance_audio_files):
        """Test performance with concurrent extractions."""
        from src.extractors.mfcc_extractor import MFCCExtractor
        
        extractor = MFCCExtractor()
        test_file = performance_audio_files["2s"]  # Use 2-second file
        num_concurrent = 5
        
        def run_extraction():
            with tempfile.TemporaryDirectory() as tmp_dir:
                start_time = time.time()
                result = extractor.run(test_file, tmp_dir)
                end_time = time.time()
                return {
                    'processing_time': end_time - start_time,
                    'success': result.success
                }
        
        # Run concurrent extractions
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(run_extraction) for _ in range(num_concurrent)]
            results = [future.result() for future in as_completed(futures)]
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Verify all extractions succeeded
        for result in results:
            assert result['success'], "Concurrent extraction failed"
        
        # Verify total time is reasonable (should be less than sequential time)
        sequential_time = sum(result['processing_time'] for result in results)
        assert total_time < sequential_time, \
            f"Concurrent time {total_time:.2f}s not better than sequential {sequential_time:.2f}s"
        
        # Verify total time is less than 30 seconds
        assert total_time < 30, \
            f"Total concurrent time {total_time:.2f}s exceeds 30s limit"
    
    def test_memory_usage_performance(self, performance_audio_files):
        """Test memory usage during extraction."""
        from src.extractors.mfcc_extractor import MFCCExtractor
        
        extractor = MFCCExtractor()
        test_file = performance_audio_files["10s"]  # Use 10-second file
        
        process = psutil.Process(os.getpid())
        
        # Get initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run(test_file, tmp_dir)
            
            # Get peak memory usage
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            assert result.success, "Extraction failed"
            
            # Memory increase should be less than 200MB
            assert memory_increase < 200, \
                f"Memory increase {memory_increase:.2f}MB exceeds 200MB limit"
    
    def test_large_file_performance(self):
        """Test performance with large audio files."""
        from src.extractors.mfcc_extractor import MFCCExtractor
        
        extractor = MFCCExtractor()
        
        # Create a large audio file (60 seconds)
        audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=60.0)
        audio = AudioFixtures.normalize_audio(audio)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, 22050)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                start_time = time.time()
                result = extractor.run(tmp.name, tmp_dir)
                end_time = time.time()
                
                processing_time = end_time - start_time
                
                assert result.success, "Large file extraction failed"
                
                # Processing time should be less than 2x real-time for large files
                max_processing_time = 60 * 2  # 2x real-time
                assert processing_time < max_processing_time, \
                    f"Processing time {processing_time:.2f}s exceeds limit {max_processing_time:.2f}s"
            
            os.unlink(tmp.name)


@pytest.mark.performance
class TestAPIPerformance:
    """Performance tests for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from src.main import app
        return TestClient(app)
    
    def test_api_response_time(self, client):
        """Test API response times."""
        endpoints = [
            ("/", "GET"),
            ("/health", "GET"),
            ("/extractors", "GET"),
            ("/metrics", "GET"),
            ("/docs", "GET"),
            ("/redoc", "GET"),
            ("/openapi.json", "GET")
        ]
        
        for endpoint, method in endpoints:
            start_time = time.time()
            
            if method == "GET":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            assert response.status_code == 200, f"Endpoint {endpoint} failed"
            
            # Response time should be less than 1 second
            assert response_time < 1.0, \
                f"Endpoint {endpoint} response time {response_time:.3f}s exceeds 1s limit"
    
    def test_concurrent_api_requests(self, client):
        """Test API performance with concurrent requests."""
        def make_request():
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            return {
                'status_code': response.status_code,
                'response_time': end_time - start_time
            }
        
        num_requests = 20
        
        # Make concurrent requests
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Verify all requests succeeded
        for result in results:
            assert result['status_code'] == 200, "API request failed"
        
        # Verify total time is reasonable
        assert total_time < 10, \
            f"Total time for {num_requests} requests {total_time:.2f}s exceeds 10s limit"
        
        # Verify average response time
        avg_response_time = sum(r['response_time'] for r in results) / len(results)
        assert avg_response_time < 0.5, \
            f"Average response time {avg_response_time:.3f}s exceeds 0.5s limit"
    
    def test_api_memory_usage(self, client):
        """Test API memory usage under load."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make many requests
        for _ in range(100):
            response = client.get("/health")
            assert response.status_code == 200
        
        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be less than 50MB
        assert memory_increase < 50, \
            f"Memory increase {memory_increase:.2f}MB exceeds 50MB limit"


@pytest.mark.performance
class TestSystemPerformance:
    """System-level performance tests."""
    
    def test_cpu_usage_performance(self):
        """Test CPU usage during extraction."""
        from src.extractors.mfcc_extractor import MFCCExtractor
        
        extractor = MFCCExtractor()
        
        # Create test audio
        audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=5.0)
        audio = AudioFixtures.normalize_audio(audio)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, 22050)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Monitor CPU usage
                process = psutil.Process(os.getpid())
                
                start_time = time.time()
                result = extractor.run(tmp.name, tmp_dir)
                end_time = time.time()
                
                processing_time = end_time - start_time
                
                assert result.success, "Extraction failed"
                
                # CPU usage should be reasonable (less than 100% for 5 seconds)
                cpu_percent = process.cpu_percent()
                assert cpu_percent < 100, \
                    f"CPU usage {cpu_percent}% exceeds 100% limit"
            
            os.unlink(tmp.name)
    
    def test_disk_io_performance(self):
        """Test disk I/O performance during extraction."""
        from src.extractors.mfcc_extractor import MFCCExtractor
        
        extractor = MFCCExtractor()
        
        # Create test audio
        audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=2.0)
        audio = AudioFixtures.normalize_audio(audio)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, 22050)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                start_time = time.time()
                result = extractor.run(tmp.name, tmp_dir)
                end_time = time.time()
                
                processing_time = end_time - start_time
                
                assert result.success, "Extraction failed"
                
                # Processing time should be reasonable
                assert processing_time < 10, \
                    f"Processing time {processing_time:.2f}s exceeds 10s limit"
            
            os.unlink(tmp.name)
    
    def test_network_performance(self):
        """Test network performance (if applicable)."""
        # This test would be relevant if the system makes network calls
        # For now, we'll skip it
        pytest.skip("No network calls in current implementation")
    
    def test_database_performance(self):
        """Test database performance (if applicable)."""
        # This test would be relevant if the system uses a database
        # For now, we'll skip it
        pytest.skip("No database in current implementation")


@pytest.mark.performance
class TestScalabilityPerformance:
    """Scalability performance tests."""
    
    def test_extractor_scalability(self):
        """Test extractor scalability with different workloads."""
        from src.extractors.mfcc_extractor import MFCCExtractor
        
        extractor = MFCCExtractor()
        
        # Test different file sizes
        file_sizes = [0.5, 1.0, 2.0, 5.0, 10.0]
        results = {}
        
        for size in file_sizes:
            audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=size)
            audio = AudioFixtures.normalize_audio(audio)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, audio, 22050)
                
                with tempfile.TemporaryDirectory() as tmp_dir:
                    start_time = time.time()
                    result = extractor.run(tmp.name, tmp_dir)
                    end_time = time.time()
                    
                    processing_time = end_time - start_time
                    results[size] = {
                        'processing_time': processing_time,
                        'throughput': size / processing_time,  # seconds of audio per second of processing
                        'success': result.success
                    }
                    
                    assert result.success, f"Extraction failed for {size}s file"
                
                os.unlink(tmp.name)
        
        # Verify scalability
        for size, data in results.items():
            throughput = data['throughput']
            
            # Throughput should be at least 0.5x real-time
            assert throughput >= 0.5, \
                f"Throughput {throughput:.2f}x for {size}s file below 0.5x limit"
    
    def test_concurrent_workload_scalability(self):
        """Test scalability with concurrent workloads."""
        from src.extractors.mfcc_extractor import MFCCExtractor
        
        extractor = MFCCExtractor()
        
        # Test different levels of concurrency
        concurrency_levels = [1, 2, 4, 8]
        test_duration = 2.0  # 2-second test files
        
        results = {}
        
        for concurrency in concurrency_levels:
            # Create test files
            test_files = []
            for i in range(concurrency):
                audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=test_duration)
                audio = AudioFixtures.normalize_audio(audio)
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    sf.write(tmp.name, audio, 22050)
                    test_files.append(tmp.name)
            
            def run_extraction(file_path):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    start_time = time.time()
                    result = extractor.run(file_path, tmp_dir)
                    end_time = time.time()
                    return {
                        'processing_time': end_time - start_time,
                        'success': result.success
                    }
            
            # Run concurrent extractions
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(run_extraction, file_path) for file_path in test_files]
                extraction_results = [future.result() for future in as_completed(futures)]
            end_time = time.time()
            
            total_time = end_time - start_time
            
            # Verify all extractions succeeded
            for result in extraction_results:
                assert result['success'], f"Extraction failed with concurrency {concurrency}"
            
            results[concurrency] = {
                'total_time': total_time,
                'throughput': (concurrency * test_duration) / total_time,
                'avg_processing_time': sum(r['processing_time'] for r in extraction_results) / len(extraction_results)
            }
            
            # Cleanup
            for file_path in test_files:
                os.unlink(file_path)
        
        # Verify scalability
        for concurrency, data in results.items():
            throughput = data['throughput']
            
            # Throughput should increase with concurrency (up to a point)
            if concurrency > 1:
                # At least some improvement with concurrency
                assert throughput >= 0.5, \
                    f"Throughput {throughput:.2f}x for concurrency {concurrency} below 0.5x limit"
    
    def test_memory_scalability(self):
        """Test memory scalability with different workloads."""
        from src.extractors.mfcc_extractor import MFCCExtractor
        
        extractor = MFCCExtractor()
        
        # Test different file sizes
        file_sizes = [1.0, 5.0, 10.0, 30.0]
        results = {}
        
        process = psutil.Process(os.getpid())
        
        for size in file_sizes:
            audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=size)
            audio = AudioFixtures.normalize_audio(audio)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, audio, 22050)
                
                # Measure memory usage
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                with tempfile.TemporaryDirectory() as tmp_dir:
                    result = extractor.run(tmp.name, tmp_dir)
                    
                    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_increase = peak_memory - initial_memory
                    
                    results[size] = {
                        'memory_increase': memory_increase,
                        'memory_per_second': memory_increase / size,
                        'success': result.success
                    }
                    
                    assert result.success, f"Extraction failed for {size}s file"
                
                os.unlink(tmp.name)
        
        # Verify memory scalability
        for size, data in results.items():
            memory_per_second = data['memory_per_second']
            
            # Memory usage per second should be reasonable (less than 20MB per second)
            assert memory_per_second < 20, \
                f"Memory usage {memory_per_second:.2f}MB/s for {size}s file exceeds 20MB/s limit"


@pytest.mark.performance
class TestBenchmarkPerformance:
    """Benchmark performance tests."""
    
    def test_benchmark_extraction_speed(self):
        """Benchmark extraction speed against known baselines."""
        from src.extractors.mfcc_extractor import MFCCExtractor
        
        extractor = MFCCExtractor()
        
        # Create benchmark audio (1 second)
        audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0)
        audio = AudioFixtures.normalize_audio(audio)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, 22050)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Run multiple times for accurate measurement
                times = []
                for _ in range(5):
                    start_time = time.time()
                    result = extractor.run(tmp.name, tmp_dir)
                    end_time = time.time()
                    
                    assert result.success, "Benchmark extraction failed"
                    times.append(end_time - start_time)
                
                avg_time = sum(times) / len(times)
                
                # Benchmark: should process 1 second of audio in less than 3 seconds
                assert avg_time < 3.0, \
                    f"Average processing time {avg_time:.3f}s exceeds 3s benchmark"
            
            os.unlink(tmp.name)
    
    def test_benchmark_memory_efficiency(self):
        """Benchmark memory efficiency."""
        from src.extractors.mfcc_extractor import MFCCExtractor
        
        extractor = MFCCExtractor()
        
        # Create benchmark audio (5 seconds)
        audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=5.0)
        audio = AudioFixtures.normalize_audio(audio)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, 22050)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                process = psutil.Process(os.getpid())
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                result = extractor.run(tmp.name, tmp_dir)
                
                peak_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = peak_memory - initial_memory
                
                assert result.success, "Benchmark extraction failed"
                
                # Benchmark: should use less than 100MB for 5 seconds of audio
                assert memory_increase < 100, \
                    f"Memory increase {memory_increase:.2f}MB exceeds 100MB benchmark"
            
            os.unlink(tmp.name)
    
    def test_benchmark_concurrent_throughput(self):
        """Benchmark concurrent throughput."""
        from src.extractors.mfcc_extractor import MFCCExtractor
        
        extractor = MFCCExtractor()
        
        # Create benchmark audio (1 second)
        audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0)
        audio = AudioFixtures.normalize_audio(audio)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, 22050)
            
            def run_extraction():
                with tempfile.TemporaryDirectory() as tmp_dir:
                    start_time = time.time()
                    result = extractor.run(tmp.name, tmp_dir)
                    end_time = time.time()
                    return {
                        'processing_time': end_time - start_time,
                        'success': result.success
                    }
            
            # Run 10 concurrent extractions
            num_concurrent = 10
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                futures = [executor.submit(run_extraction) for _ in range(num_concurrent)]
                results = [future.result() for future in as_completed(futures)]
            end_time = time.time()
            
            total_time = end_time - start_time
            
            # Verify all extractions succeeded
            for result in results:
                assert result['success'], "Benchmark extraction failed"
            
            # Benchmark: should process 10 seconds of audio in less than 15 seconds
            assert total_time < 15, \
                f"Total time {total_time:.2f}s exceeds 15s benchmark for 10 concurrent extractions"
            
            # Calculate throughput
            throughput = (num_concurrent * 1.0) / total_time  # seconds of audio per second of processing
            
            # Benchmark: should achieve at least 0.5x real-time throughput
            assert throughput >= 0.5, \
                f"Throughput {throughput:.2f}x below 0.5x benchmark"
            
            os.unlink(tmp.name)
