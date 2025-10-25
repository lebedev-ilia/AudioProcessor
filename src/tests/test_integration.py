"""
Integration tests for AudioProcessor.
"""
import pytest
import time
import tempfile
import os
from fastapi.testclient import TestClient
from src.main import app
from src.extractors import discover_extractors
from src.storage.s3_client import S3Client
from src.celery_app import celery_app
from tests.fixtures.audio_fixtures import AudioFixtures


@pytest.mark.integration
class TestFullPipelineIntegration:
    """Integration tests for the full processing pipeline."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_audio_file(self):
        """Create a sample audio file for integration testing."""
        audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=2.0)
        audio = AudioFixtures.normalize_audio(audio)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio, 22050)
            yield tmp.name
        
        os.unlink(tmp.name)
    
    def test_full_extraction_pipeline(self, sample_audio_file):
        """Test the complete extraction pipeline with real extractors."""
        # Get all extractors
        extractors = discover_extractors()
        assert len(extractors) == 6
        
        # Create temporary directory for results
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = []
            
            # Run all extractors
            for extractor in extractors:
                result = extractor.run(sample_audio_file, tmp_dir)
                results.append(result)
                
                # Verify result
                assert result.success, f"Extractor {extractor.name} failed: {result.error}"
                assert result.payload is not None, f"Extractor {extractor.name} payload is None"
                assert len(result.payload) > 0, f"Extractor {extractor.name} payload is empty"
            
            # Verify we have results from all extractors
            assert len(results) == 6
            
            # Verify all extractors have different names
            extractor_names = [result.name for result in results]
            assert len(set(extractor_names)) == 6, "Duplicate extractor names found"
            
            # Verify all extractors succeeded
            successful_results = [r for r in results if r.success]
            assert len(successful_results) == 6, "Not all extractors succeeded"
    
    def test_api_to_celery_integration(self, client, sample_audio_file):
        """Test API to Celery integration."""
        # This test requires a real Celery broker
        pytest.skip("Requires real Celery broker")
        
        # Submit processing request
        request_data = {
            "video_id": "integration_test_video",
            "audio_uri": sample_audio_file,
            "dataset": "integration_test"
        }
        
        response = client.post("/process", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["accepted"] is True
        assert "celery_task_id" in data
        
        task_id = data["celery_task_id"]
        
        # Wait for task completion (with timeout)
        max_wait_time = 60  # 60 seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_response = client.get(f"/task/{task_id}")
            assert status_response.status_code == 200
            
            status_data = status_response.json()
            
            if status_data["status"] == "SUCCESS":
                assert status_data["result"] is not None
                assert status_data["result"]["status"] == "completed"
                assert status_data["result"]["video_id"] == "integration_test_video"
                break
            elif status_data["status"] == "FAILURE":
                pytest.fail(f"Task failed: {status_data.get('error', 'Unknown error')}")
            
            time.sleep(1)
        else:
            pytest.fail("Task did not complete within timeout")
    
    def test_s3_integration(self):
        """Test S3 integration (requires real S3/MinIO)."""
        pytest.skip("Requires real S3/MinIO instance")
        
        # This test would require a real S3/MinIO instance
        # It would test:
        # 1. File upload
        # 2. File download
        # 3. Manifest upload
        # 4. File existence checks
        # 5. File listing
        # 6. File deletion
    
    def test_celery_worker_integration(self):
        """Test Celery worker integration (requires real broker)."""
        pytest.skip("Requires real Celery broker")
        
        # This test would require a real Celery broker and worker
        # It would test:
        # 1. Task submission
        # 2. Task execution
        # 3. Task status updates
        # 4. Task result retrieval
        # 5. Error handling
        # 6. Retry mechanisms


@pytest.mark.integration
class TestExtractorIntegration:
    """Integration tests for extractors."""
    
    @pytest.fixture
    def sample_audio_file(self):
        """Create a sample audio file for extractor testing."""
        audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0)
        audio = AudioFixtures.normalize_audio(audio)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio, 22050)
            yield tmp.name
        
        os.unlink(tmp.name)
    
    def test_mfcc_extractor_integration(self, sample_audio_file):
        """Test MFCC extractor with real audio processing."""
        from src.extractors.mfcc_extractor import MFCCExtractor
        
        extractor = MFCCExtractor()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run(sample_audio_file, tmp_dir)
            
            assert result.success, f"MFCC extraction failed: {result.error}"
            assert result.payload is not None
            
            # Verify specific MFCC features
            expected_features = [
                'mfcc_0_mean', 'mfcc_1_mean', 'mfcc_2_mean',
                'mfcc_0_std', 'mfcc_1_std', 'mfcc_2_std',
                'mfcc_delta_0_mean', 'mfcc_delta_1_mean',
                'mfcc_mean', 'mfcc_std'
            ]
            
            for feature in expected_features:
                assert feature in result.payload, f"Missing MFCC feature: {feature}"
                assert result.payload[feature] is not None, f"MFCC feature {feature} is None"
    
    def test_mel_extractor_integration(self, sample_audio_file):
        """Test Mel extractor with real audio processing."""
        from src.extractors.mel_extractor import MelExtractor
        
        extractor = MelExtractor()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run(sample_audio_file, tmp_dir)
            
            assert result.success, f"Mel extraction failed: {result.error}"
            assert result.payload is not None
            
            # Verify specific Mel features
            expected_features = [
                'mel64_mean_0', 'mel64_mean_1', 'mel64_mean_2',
                'mel64_std_0', 'mel64_std_1', 'mel64_std_2',
                'mel64_mean', 'mel64_std'
            ]
            
            for feature in expected_features:
                assert feature in result.payload, f"Missing Mel feature: {feature}"
                assert result.payload[feature] is not None, f"Mel feature {feature} is None"
    
    def test_chroma_extractor_integration(self, sample_audio_file):
        """Test Chroma extractor with real audio processing."""
        from src.extractors.chroma_extractor import ChromaExtractor
        
        extractor = ChromaExtractor()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run(sample_audio_file, tmp_dir)
            
            assert result.success, f"Chroma extraction failed: {result.error}"
            assert result.payload is not None
            
            # Verify specific Chroma features
            expected_features = [
                'chroma_0_mean', 'chroma_1_mean', 'chroma_2_mean',
                'chroma_0_std', 'chroma_1_std', 'chroma_2_std',
                'chroma_mean', 'chroma_std'
            ]
            
            for feature in expected_features:
                assert feature in result.payload, f"Missing Chroma feature: {feature}"
                assert result.payload[feature] is not None, f"Chroma feature {feature} is None"
    
    def test_loudness_extractor_integration(self, sample_audio_file):
        """Test Loudness extractor with real audio processing."""
        from src.extractors.loudness_extractor import LoudnessExtractor
        
        extractor = LoudnessExtractor()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run(sample_audio_file, tmp_dir)
            
            assert result.success, f"Loudness extraction failed: {result.error}"
            assert result.payload is not None
            
            # Verify specific Loudness features
            expected_features = [
                'rms_mean', 'rms_std', 'rms_min', 'rms_max',
                'loudness_lufs', 'peak_amplitude', 'clip_fraction'
            ]
            
            for feature in expected_features:
                assert feature in result.payload, f"Missing Loudness feature: {feature}"
                assert result.payload[feature] is not None, f"Loudness feature {feature} is None"
    
    def test_vad_extractor_integration(self, sample_audio_file):
        """Test VAD extractor with real audio processing."""
        from src.extractors.vad_extractor import VADExtractor
        
        extractor = VADExtractor()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run(sample_audio_file, tmp_dir)
            
            assert result.success, f"VAD extraction failed: {result.error}"
            assert result.payload is not None
            
            # Verify specific VAD features
            expected_features = [
                'voiced_fraction', 'f0_mean', 'f0_std', 'f0_min', 'f0_max'
            ]
            
            for feature in expected_features:
                assert feature in result.payload, f"Missing VAD feature: {feature}"
                assert result.payload[feature] is not None, f"VAD feature {feature} is None"
    
    def test_clap_extractor_integration(self, sample_audio_file):
        """Test CLAP extractor with real audio processing."""
        from src.extractors.clap_extractor import CLAPExtractor
        
        extractor = CLAPExtractor()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run(sample_audio_file, tmp_dir)
            
            assert result.success, f"CLAP extraction failed: {result.error}"
            assert result.payload is not None
            
            # Verify specific CLAP features
            expected_features = [
                'clap_0', 'clap_1', 'clap_2',
                'clap_embedding', 'clap_mean', 'clap_std'
            ]
            
            for feature in expected_features:
                assert feature in result.payload, f"Missing CLAP feature: {feature}"
                assert result.payload[feature] is not None, f"CLAP feature {feature} is None"
    
    def test_extractor_error_handling(self):
        """Test extractor error handling with invalid input."""
        from src.extractors.mfcc_extractor import MFCCExtractor
        
        extractor = MFCCExtractor()
        
        # Test with non-existent file
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run("/non/existent/file.wav", tmp_dir)
            
            assert not result.success
            assert result.error is not None
            assert result.payload is None
        
        # Test with invalid file format
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"This is not audio data")
            tmp.flush()
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                result = extractor.run(tmp.name, tmp_dir)
                
                assert not result.success
                assert result.error is not None
                assert result.payload is None
            
            os.unlink(tmp.name)


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_api_health_integration(self, client):
        """Test API health endpoint integration."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "uptime" in data
        assert "memory_usage" in data
        assert "cpu_usage" in data
        assert "disk_usage" in data
    
    def test_api_extractors_integration(self, client):
        """Test API extractors endpoint integration."""
        response = client.get("/extractors")
        assert response.status_code == 200
        
        data = response.json()
        assert "extractors" in data
        assert len(data["extractors"]) == 6
        
        # Verify all extractors are available
        for extractor in data["extractors"]:
            assert extractor["status"] == "available"
            assert "name" in extractor
            assert "version" in extractor
            assert "description" in extractor
    
    def test_api_metrics_integration(self, client):
        """Test API metrics endpoint integration."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        
        # Verify metrics content
        content = response.text
        assert "audio_processor_" in content
        assert "tasks_total" in content
        assert "extractors_total" in content
    
    def test_api_docs_integration(self, client):
        """Test API documentation endpoints integration."""
        # Test OpenAPI docs
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        # Test OpenAPI JSON
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
        
        openapi_data = response.json()
        assert "openapi" in openapi_data
        assert "info" in openapi_data
        assert "paths" in openapi_data


@pytest.mark.integration
class TestPerformanceIntegration:
    """Integration tests for performance."""
    
    def test_extraction_performance(self):
        """Test extraction performance with different audio lengths."""
        from src.extractors.mfcc_extractor import MFCCExtractor
        
        extractor = MFCCExtractor()
        
        # Test different durations
        durations = [0.5, 1.0, 2.0, 5.0, 10.0]
        
        for duration in durations:
            # Generate audio
            audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=duration)
            audio = AudioFixtures.normalize_audio(audio)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                import soundfile as sf
                sf.write(tmp.name, audio, 22050)
                
                with tempfile.TemporaryDirectory() as tmp_dir:
                    start_time = time.time()
                    result = extractor.run(tmp.name, tmp_dir)
                    end_time = time.time()
                    
                    processing_time = end_time - start_time
                    
                    assert result.success, f"Extraction failed for duration {duration}"
                    
                    # Verify processing time is reasonable (less than 30 seconds for 10-second audio)
                    max_processing_time = duration * 3  # 3x real-time
                    assert processing_time < max_processing_time, \
                        f"Processing time {processing_time:.2f}s exceeds limit {max_processing_time:.2f}s for duration {duration}s"
                
                os.unlink(tmp.name)
    
    def test_memory_usage_integration(self):
        """Test memory usage during extraction."""
        import psutil
        import os
        
        from src.extractors.mfcc_extractor import MFCCExtractor
        
        extractor = MFCCExtractor()
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate long audio
        audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=5.0)
        audio = AudioFixtures.normalize_audio(audio)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio, 22050)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                result = extractor.run(tmp.name, tmp_dir)
                
                # Get memory usage after processing
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = final_memory - initial_memory
                
                assert result.success, "Extraction failed"
                
                # Verify memory increase is reasonable (less than 100MB)
                assert memory_increase < 100, \
                    f"Memory increase {memory_increase:.2f}MB exceeds limit 100MB"
            
            os.unlink(tmp.name)


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Integration tests for error handling."""
    
    def test_invalid_audio_file_handling(self):
        """Test handling of invalid audio files."""
        from src.extractors.mfcc_extractor import MFCCExtractor
        
        extractor = MFCCExtractor()
        
        # Test with corrupted file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b"This is not valid audio data")
            tmp.flush()
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                result = extractor.run(tmp.name, tmp_dir)
                
                assert not result.success
                assert result.error is not None
                assert result.payload is None
            
            os.unlink(tmp.name)
        
        # Test with empty file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b"")
            tmp.flush()
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                result = extractor.run(tmp.name, tmp_dir)
                
                assert not result.success
                assert result.error is not None
                assert result.payload is None
            
            os.unlink(tmp.name)
    
    def test_missing_dependencies_handling(self):
        """Test handling of missing dependencies."""
        # This test would require temporarily removing dependencies
        # and testing error handling
        pytest.skip("Requires dependency manipulation")
    
    def test_disk_space_handling(self):
        """Test handling of disk space issues."""
        # This test would require simulating disk space issues
        pytest.skip("Requires disk space simulation")
