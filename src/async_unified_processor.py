"""
Async Unified Audio Processor with parallel extractor execution.

This module provides an async version of UnifiedAudioProcessor that can:
1. Run extractors in parallel (CPU and GPU)
2. Process segments in parallel
3. Handle batch processing asynchronously
4. Manage resources with semaphores and backpressure
"""

import os
import json
import asyncio
import numpy as np
import platform
import time
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

from .segment_config import SegmentConfig, get_default_config
from .segment_pipeline import SegmentPipeline
from .extractors import discover_extractors
from .extractors.video_audio_extractor import VideoAudioExtractor
from .schemas.models import ExtractorResult, ManifestModel
from .smart_gpu_detector import get_smart_detector, get_smart_config, is_gpu_available
from .utils.logging import get_logger

logger = get_logger(__name__)


class AsyncUnifiedAudioProcessor:
    """Async processor that can extract both aggregated and segment features in parallel."""
    
    def __init__(self, config: Optional[SegmentConfig] = None, 
                 max_cpu_workers: Optional[int] = None,
                 max_gpu_workers: Optional[int] = None,
                 max_io_workers: Optional[int] = None,
                 gpu_batch_size: Optional[int] = None,
                 use_smart_detection: bool = True):
        """
        Initialize async processor with smart GPU detection.
        
        Args:
            config: Configuration for segment processing
            max_cpu_workers: Maximum CPU workers (auto-detected if None)
            max_gpu_workers: Maximum GPU workers (auto-detected if None)
            max_io_workers: Maximum I/O workers (auto-detected if None)
            gpu_batch_size: Batch size for GPU processing (auto-detected if None)
            use_smart_detection: Whether to use smart GPU detection
        """
        self.config = config or get_default_config()
        self.segment_pipeline = SegmentPipeline(self.config)
        
        # Initialize smart GPU detection
        if use_smart_detection:
            self.smart_detector = get_smart_detector()
            self.smart_config = self.smart_detector.get_smart_config()
            logger.info("🧠 Using smart GPU detection")
        else:
            self.smart_detector = None
            self.smart_config = None
            logger.info("⚙️ Using manual configuration")
        
        # Get available extractors
        self.extractors = discover_extractors()
        
        # Resource management - use smart config if available
        if self.smart_config:
            self.max_cpu_workers = max_cpu_workers or self.smart_config.max_cpu_workers
            self.max_gpu_workers = max_gpu_workers or self.smart_config.max_gpu_workers
            self.max_io_workers = max_io_workers or self.smart_config.max_io_workers
            self.gpu_batch_size = gpu_batch_size or self.smart_config.gpu_batch_size
        else:
            self.max_cpu_workers = max_cpu_workers or 8
            self.max_gpu_workers = max_gpu_workers or 2
            self.max_io_workers = max_io_workers or 16
            self.gpu_batch_size = gpu_batch_size or 8
        
        # Semaphores for resource control
        self.cpu_semaphore = asyncio.Semaphore(self.max_cpu_workers)
        
        # GPU semaphore - use smart detection
        if self.smart_config and self.smart_config.gpu_semaphore_enabled:
            self.gpu_semaphore = asyncio.Semaphore(self.max_gpu_workers)
            logger.info(f"🔒 GPU semaphore enabled with {self.max_gpu_workers} workers")
        else:
            self.gpu_semaphore = None
            logger.info("🔓 GPU semaphore disabled - CPU-only mode")
            
        self.io_semaphore = asyncio.Semaphore(self.max_io_workers)
        
        # Categorize extractors
        self.cpu_extractors = []
        self.gpu_extractors = []
        self._categorize_extractors()
        
        logger.info(f"Initialized AsyncUnifiedAudioProcessor with {len(self.extractors)} extractors")
        logger.info(f"CPU extractors: {len(self.cpu_extractors)}, GPU extractors: {len(self.gpu_extractors)}")
    
    def _categorize_extractors(self):
        """Categorize extractors by resource requirements using smart detection."""
        if self.smart_config:
            # Use smart categorization
            cpu_extractor_names = set(self.smart_config.cpu_extractors)
            gpu_extractor_names = set(self.smart_config.gpu_extractors)
            hybrid_extractor_names = set(self.smart_config.hybrid_extractors)
            
            for extractor in self.extractors:
                if extractor.name in gpu_extractor_names and self.smart_config.gpu_available:
                    self.gpu_extractors.append(extractor)
                elif extractor.name in hybrid_extractor_names:
                    # Hybrid extractors - use GPU if available, otherwise CPU
                    if self.smart_config.gpu_available:
                        self.gpu_extractors.append(extractor)
                    else:
                        self.cpu_extractors.append(extractor)
                elif extractor.name in cpu_extractor_names:
                    self.cpu_extractors.append(extractor)
                else:
                    # Default to CPU for unknown extractors
                    self.cpu_extractors.append(extractor)
        else:
            # Fallback to manual categorization
            for extractor in self.extractors:
                if extractor.category == "advanced" or extractor.name in ["advanced_embeddings", "asr_extractor"]:
                    # These are truly GPU-dependent
                    import torch
                    if torch.cuda.is_available():
                        self.gpu_extractors.append(extractor)
                    else:
                        logger.warning(f"Skipping GPU-dependent extractor {extractor.name} - no GPU available")
                elif extractor.name == "clap_extractor":
                    # CLAP can work on CPU, but is better on GPU
                    import torch
                    if torch.cuda.is_available():
                        self.gpu_extractors.append(extractor)
                    else:
                        self.cpu_extractors.append(extractor)
                else:
                    self.cpu_extractors.append(extractor)
        
        logger.info(f"🧠 Smart categorization: CPU extractors: {len(self.cpu_extractors)}, GPU extractors: {len(self.gpu_extractors)}")
    
    def _get_segment_extractors(self) -> List[str]:
        """Return list of extractors for per-segment processing."""
        return [
            "clap_extractor",
            "advanced_embeddings", 
            "loudness_extractor",
            "vad_extractor",
            "spectral_extractor",
            "tempo_extractor",
            "onset_extractor",
            "source_separation_extractor",
            "emotion_recognition_extractor",
            "quality_extractor",
            "asr_extractor",
            "mel_extractor",
            "mfcc_extractor"
        ]
    
    def _is_macos(self) -> bool:
        """Check if running on macOS."""
        return platform.system() == "Darwin"
    
    def _is_video_file(self, file_path: str) -> bool:
        """Check if file is a video file."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        return any(file_path.lower().endswith(ext) for ext in video_extensions)
    
    async def _extract_audio_from_video(self, video_path: str, output_dir: str) -> str:
        """Extract audio from video file asynchronously."""
        logger.info(f"Extracting audio from video: {video_path}")
        
        # Create temporary directory for audio
        audio_dir = os.path.join(output_dir, "extracted_audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        # Use VideoAudioExtractor in thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            video_extractor = VideoAudioExtractor()
            result = await loop.run_in_executor(
                executor, 
                video_extractor.run, 
                video_path, 
                audio_dir
            )
        
        if result.get("success", False):
            audio_path = result.get("extracted_audio_path")
            logger.info(f"Successfully extracted audio to: {audio_path}")
            return audio_path
        else:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"Failed to extract audio from video: {error_msg}")
            raise Exception(f"Audio extraction failed: {error_msg}")
    
    def _get_safe_extractors(self) -> List[str]:
        """Return list of safe extractors (exclude problematic ones on macOS)."""
        all_extractors = [extractor.name for extractor in self.extractors]
        
        if self._is_macos():
            safe_extractors = [ext for ext in all_extractors if ext != "speaker_diarization"]
            if len(safe_extractors) < len(all_extractors):
                logger.warning(
                    "🍎 macOS detected: speaker_diarization extractor excluded due to multiprocessing issues. "
                    f"Using {len(safe_extractors)} extractors instead of {len(all_extractors)}."
                )
            return safe_extractors
        else:
            return all_extractors
    
    async def _run_cpu_extractor(self, extractor, input_uri: str, tmp_path: str) -> ExtractorResult:
        """Run CPU extractor with semaphore control."""
        async with self.cpu_semaphore:
            logger.info(f"Running CPU extractor: {extractor.name}")
            start_time = time.time()
            
            try:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    extractor.run, 
                    input_uri, 
                    tmp_path
                )
                
                processing_time = time.time() - start_time
                
                # Update processing_time if it's None
                if result.processing_time is None:
                    result.processing_time = processing_time
                
                if result.success:
                    logger.info(f"✅ {extractor.name} completed successfully in {processing_time:.2f}s")
                else:
                    logger.warning(f"⚠️ {extractor.name} failed: {result.error}")
                
                return result
                
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"❌ {extractor.name} failed with exception: {e}")
                
                return ExtractorResult(
                    name=extractor.name,
                    version=extractor.version,
                    success=False,
                    error=str(e),
                    processing_time=processing_time
                )
    
    async def _run_gpu_extractor(self, extractor, input_uri: str, tmp_path: str) -> ExtractorResult:
        """Run GPU extractor with semaphore control."""
        # Use semaphore only if available (GPU present)
        if self.gpu_semaphore is not None:
            async with self.gpu_semaphore:
                return await self._run_extractor_impl(extractor, input_uri, tmp_path, "GPU")
        else:
            # No GPU semaphore, run directly
            return await self._run_extractor_impl(extractor, input_uri, tmp_path, "CPU (GPU fallback)")
    
    async def _run_extractor_impl(self, extractor, input_uri: str, tmp_path: str, extractor_type: str) -> ExtractorResult:
        """Implementation of extractor execution."""
        logger.info(f"Running {extractor_type} extractor: {extractor.name}")
        start_time = time.time()
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                extractor.run, 
                input_uri, 
                tmp_path
            )
            
            processing_time = time.time() - start_time
            
            # Update processing_time if it's None
            if result.processing_time is None:
                result.processing_time = processing_time
            
            if result.success:
                logger.info(f"✅ {extractor.name} completed successfully in {processing_time:.2f}s")
            else:
                logger.warning(f"⚠️ {extractor.name} failed: {result.error}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ {extractor.name} failed with exception: {e}")
            
            return ExtractorResult(
                name=extractor.name,
                version=extractor.version,
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    async def _extract_aggregated_features_async(
        self,
        input_uri: str,
        video_id: str,
        extractor_names: Optional[List[str]] = None
    ) -> List[ExtractorResult]:
        """Extract aggregated features using parallel extractors."""
        logger.info(f"Extracting aggregated features for {video_id} in parallel")
        
        # Filter extractors if names provided
        if extractor_names:
            available_names = {extractor.name for extractor in self.extractors}
            requested_names = set(extractor_names)
            unknown_names = requested_names - available_names
            
            if unknown_names:
                logger.warning(f"Unknown extractors: {unknown_names}")
            
            cpu_extractors_to_use = [
                extractor for extractor in self.cpu_extractors 
                if extractor.name in requested_names
            ]
            gpu_extractors_to_use = [
                extractor for extractor in self.gpu_extractors 
                if extractor.name in requested_names
            ]
        else:
            # Use safe extractors by default
            safe_extractor_names = self._get_safe_extractors()
            cpu_extractors_to_use = [
                extractor for extractor in self.cpu_extractors 
                if extractor.name in safe_extractor_names
            ]
            gpu_extractors_to_use = [
                extractor for extractor in self.gpu_extractors 
                if extractor.name in safe_extractor_names
            ]
        
        logger.info(f"Using {len(cpu_extractors_to_use)} CPU extractors and {len(gpu_extractors_to_use)} GPU extractors")
        
        # Create tasks for parallel execution
        tasks = []
        
        # Add CPU extractor tasks
        for extractor in cpu_extractors_to_use:
            task = asyncio.create_task(
                self._run_cpu_extractor(extractor, input_uri, "/tmp")
            )
            tasks.append(task)
        
        # Add GPU extractor tasks
        for extractor in gpu_extractors_to_use:
            task = asyncio.create_task(
                self._run_gpu_extractor(extractor, input_uri, "/tmp")
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        extractor_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(extractor_results):
            if isinstance(result, Exception):
                extractor = (cpu_extractors_to_use + gpu_extractors_to_use)[i]
                logger.error(f"Extractor {extractor.name} failed with exception: {result}")
                failed_result = ExtractorResult(
                    name=extractor.name,
                    version=extractor.version,
                    success=False,
                    error=str(result),
                    processing_time=0.0
                )
                final_results.append(failed_result)
            else:
                final_results.append(result)
        
        return final_results
    
    def _create_manifest(
        self,
        video_id: str,
        input_uri: str,
        extractor_results: List[ExtractorResult]
    ) -> ManifestModel:
        """Create manifest from extractor results."""
        # Calculate total processing time
        total_processing_time = sum(
            result.processing_time or 0.0 
            for result in extractor_results 
            if result.processing_time is not None
        )
        
        return ManifestModel(
            video_id=video_id,
            task_id=f"async_unified_{video_id}",
            dataset="async_unified",
            timestamp=datetime.utcnow().isoformat() + "Z",
            extractors=extractor_results,
            schema_version="audio_manifest_v1",
            total_processing_time=total_processing_time if total_processing_time > 0 else None
        )
    
    async def _extract_segment_features_async(
        self,
        video_id: str,
        extractor_results: List[ExtractorResult],
        manifest: ManifestModel,
        output_dir: str
    ) -> Dict[str, Any]:
        """Extract per-segment features from aggregated results asynchronously."""
        logger.info(f"Extracting segment features for {video_id} in parallel")
        
        # Convert extractor results to expected format
        extractor_outputs = {}
        for result in extractor_results:
            if result.success:
                extractor_outputs[result.name] = {
                    "success": True,
                    "payload": result.payload
                }
            else:
                extractor_outputs[result.name] = {
                    "success": False,
                    "error": result.error
                }
        
        # Get duration from manifest or extractor results
        duration = self._extract_duration(extractor_outputs)
        
        # Process with segment pipeline in thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            segment_result = await loop.run_in_executor(
                executor,
                self.segment_pipeline.process_single_video,
                video_id,
                extractor_outputs,
                duration,
                True  # save_features
            )
        
        if segment_result["success"]:
            # Move files to unified output directory
            segment_files = self._move_segment_files(
                video_id=video_id,
                segment_result=segment_result,
                output_dir=output_dir
            )
            
            return {
                "segment_files": segment_files,
                "num_segments": segment_result["num_segments"],
                "num_selected_segments": segment_result["num_selected_segments"],
                "feature_shape": segment_result["feature_shape"]
            }
        else:
            return {
                "segment_files": {},
                "error": segment_result["error"]
            }
    
    def _extract_duration(self, extractor_outputs: Dict[str, Any]) -> float:
        """Extract audio duration from extractor outputs."""
        # Try to get duration from various sources
        for extractor_name, result in extractor_outputs.items():
            if result.get("success") and result.get("payload"):
                payload = result["payload"]
                
                # Check for duration in payload
                if "duration" in payload:
                    return float(payload["duration"])
                
                # Check for duration in metadata
                if "metadata" in payload and "duration" in payload["metadata"]:
                    return float(payload["metadata"]["duration"])
        
        # Default duration if not found
        logger.warning("Duration not found in extractor outputs, using default 30.0s")
        return 30.0
    
    def _move_segment_files(
        self,
        video_id: str,
        segment_result: Dict[str, Any],
        output_dir: str
    ) -> Dict[str, str]:
        """Move segment files to unified output directory."""
        segment_files = {}
        
        if "file_paths" in segment_result:
            for file_type, file_path in segment_result["file_paths"].items():
                if os.path.exists(file_path):
                    # Create new filename in output directory
                    filename = os.path.basename(file_path)
                    new_path = os.path.join(output_dir, filename)
                    
                    # Move file
                    os.rename(file_path, new_path)
                    segment_files[file_type] = new_path
                    
                    logger.info(f"Moved {file_type}: {new_path}")
        
        return segment_files
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """Update segment configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
    
    async def process_audio_async(
        self,
        input_uri: str,
        video_id: str,
        aggregates_only: bool = False,
        segment_config: Optional[Dict[str, Any]] = None,
        extractor_names: Optional[List[str]] = None,
        output_dir: str = "async_unified_output"
    ) -> Dict[str, Any]:
        """
        Process audio file asynchronously and extract features.
        
        Args:
            input_uri: Path to audio/video file
            video_id: Unique video identifier
            aggregates_only: If True, extract only aggregated features
            segment_config: Configuration for segment processing
            extractor_names: List of extractor names to use
            output_dir: Output directory for results
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Async processing {video_id}: aggregates_only={aggregates_only}")
        
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Step 0: Extract audio from video if needed
            actual_input_uri = input_uri
            if self._is_video_file(input_uri):
                logger.info(f"Video file detected, extracting audio first...")
                actual_input_uri = await self._extract_audio_from_video(input_uri, output_dir)
                logger.info(f"Using extracted audio: {actual_input_uri}")
            
            # Step 1: Extract aggregated features using parallel extractors
            extractor_results = await self._extract_aggregated_features_async(
                input_uri=actual_input_uri,
                video_id=video_id,
                extractor_names=extractor_names  # Use provided extractor_names
            )
            
            # Create manifest with aggregated results
            manifest = self._create_manifest(
                video_id=video_id,
                input_uri=input_uri,
                extractor_results=extractor_results
            )
            
            # Save manifest
            manifest_path = os.path.join(output_dir, f"{video_id}_manifest.json")
            with open(manifest_path, 'w') as f:
                json.dump(manifest.dict(), f, indent=2)
            
            # Calculate total processing time
            total_processing_time = sum(
                result.processing_time or 0.0 
                for result in extractor_results 
                if result.processing_time is not None
            )
            
            result = {
                "video_id": video_id,
                "success": True,
                "aggregates_extracted": True,
                "manifest_path": manifest_path,
                "segments_extracted": False,
                "segment_files": {},
                "processing_time": total_processing_time
            }
            
            # Step 2: Extract segment features if requested
            if not aggregates_only:
                logger.info("Extracting per-segment features...")
                
                # Update config if provided
                if segment_config:
                    self._update_config_from_dict(segment_config)
                
                # Extract segment features using only segment-relevant extractors
                segment_extractor_names = self._get_segment_extractors()
                segment_extractor_results = await self._extract_aggregated_features_async(
                    input_uri=actual_input_uri,
                    video_id=video_id,
                    extractor_names=segment_extractor_names
                )
                
                segment_result = await self._extract_segment_features_async(
                    video_id=video_id,
                    extractor_results=segment_extractor_results,
                    manifest=manifest,
                    output_dir=output_dir
                )
                
                result.update(segment_result)
                result["segments_extracted"] = True
            
            logger.info(f"Successfully processed {video_id} asynchronously")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process {video_id}: {e}")
            return {
                "video_id": video_id,
                "success": False,
                "error": str(e),
                "aggregates_extracted": False,
                "segments_extracted": False
            }
    
    async def process_batch_async(
        self,
        video_data: List[Dict[str, Any]],
        aggregates_only: bool = False,
        segment_config: Optional[Dict[str, Any]] = None,
        extractor_names: Optional[List[str]] = None,
        output_dir: str = "async_unified_batch_output",
        max_concurrent_videos: int = 4
    ) -> Dict[str, Any]:
        """
        Process batch of videos asynchronously.
        
        Args:
            video_data: List of video data with keys: video_id, input_uri
            aggregates_only: If True, extract only aggregated features
            segment_config: Configuration for segment processing
            extractor_names: List of extractor names to use
            output_dir: Output directory for results
            max_concurrent_videos: Maximum concurrent video processing
            
        Returns:
            Dictionary with batch processing results
        """
        logger.info(f"Async processing batch of {len(video_data)} videos")
        
        batch_results = {
            "total_videos": len(video_data),
            "successful": 0,
            "failed": 0,
            "aggregates_only": aggregates_only,
            "results": []
        }
        
        # Create semaphore for concurrent video processing
        video_semaphore = asyncio.Semaphore(max_concurrent_videos)
        
        async def process_single_video(video_info):
            async with video_semaphore:
                video_id = video_info["video_id"]
                # Support both input_uri and audio_uri/video_uri for compatibility
                input_uri = video_info.get("input_uri") or video_info.get("audio_uri") or video_info.get("video_uri")
                
                # For batch: all extractors for aggregates, limited for segments
                batch_extractor_names = None if aggregates_only else self._get_segment_extractors()
                
                result = await self.process_audio_async(
                    input_uri=input_uri,
                    video_id=video_id,
                    aggregates_only=aggregates_only,
                    segment_config=segment_config,
                    extractor_names=batch_extractor_names,
                    output_dir=output_dir
                )
                
                return result
        
        # Process all videos concurrently
        tasks = [process_single_video(video_info) for video_info in video_data]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                video_id = video_data[i]["video_id"]
                logger.error(f"Video {video_id} failed with exception: {result}")
                failed_result = {
                    "video_id": video_id,
                    "success": False,
                    "error": str(result),
                    "aggregates_extracted": False,
                    "segments_extracted": False
                }
                batch_results["results"].append(failed_result)
                batch_results["failed"] += 1
            else:
                batch_results["results"].append(result)
                if result["success"]:
                    batch_results["successful"] += 1
                else:
                    batch_results["failed"] += 1
        
        logger.info(f"Async batch processing completed: {batch_results['successful']} successful, "
                   f"{batch_results['failed']} failed")
        
        return batch_results


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_async_processor():
        # Test async processor
        processor = AsyncUnifiedAudioProcessor()
        
        # Test with aggregates only
        result1 = await processor.process_audio_async(
            input_uri="test_audio.wav",
            video_id="test_async_001",
            aggregates_only=True
        )
        
        print("Async aggregates only result:", result1["success"])
        
        # Test with segments
        result2 = await processor.process_audio_async(
            input_uri="test_audio.wav",
            video_id="test_async_002",
            aggregates_only=False,
            segment_config={
                "segment_len": 3.0,
                "hop": 1.5,
                "max_seq_len": 32
            }
        )
        
        print("Async with segments result:", result2["success"])
        print("Segment files:", result2.get("segment_files", {}))
    
    # Run test
    asyncio.run(test_async_processor())
