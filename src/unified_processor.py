"""
Unified Audio Processor that can extract both aggregated features and per-segment sequences.

This module provides a single API that can:
1. Extract only aggregated features (traditional AudioProcessor behavior)
2. Extract both aggregated features AND per-segment sequences in one pass
3. Return results in a unified format
"""

import os
import json
import numpy as np
import platform
import time
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime

from .segment_config import SegmentConfig, get_default_config
from .segment_pipeline import SegmentPipeline
from .extractors import discover_extractors
from .extractors.video_audio_extractor import VideoAudioExtractor
from .schemas.models import ExtractorResult, ManifestModel

logger = logging.getLogger(__name__)


class UnifiedAudioProcessor:
    """Unified processor that can extract both aggregated and segment features."""
    
    def __init__(self, config: Optional[SegmentConfig] = None):
        """
        Initialize unified processor.
        
        Args:
            config: Configuration for segment processing (uses default if None)
        """
        self.config = config or get_default_config()
        self.segment_pipeline = SegmentPipeline(self.config)
        
        # Get available extractors
        self.extractors = discover_extractors()
        
        logger.info(f"Initialized UnifiedAudioProcessor with {len(self.extractors)} extractors")
    
    def _get_segment_extractors(self) -> List[str]:
        """
        Возвращает список extractors для per-segment обработки.
        Это ограниченный набор extractors согласно PLAN.md.
        """
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
        """
        Определяет, запущено ли приложение на macOS.
        """
        return platform.system() == "Darwin"
    
    def _is_video_file(self, file_path: str) -> bool:
        """
        Определяет, является ли файл видео файлом.
        """
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        return any(file_path.lower().endswith(ext) for ext in video_extensions)
    
    def _extract_audio_from_video(self, video_path: str, output_dir: str) -> str:
        """
        Извлекает аудио из видео файла.
        
        Args:
            video_path: Путь к видео файлу
            output_dir: Директория для сохранения аудио
            
        Returns:
            Путь к извлеченному аудио файлу
        """
        logger.info(f"Extracting audio from video: {video_path}")
        
        # Создаем временную директорию для аудио
        audio_dir = os.path.join(output_dir, "extracted_audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        # Используем VideoAudioExtractor
        video_extractor = VideoAudioExtractor()
        result = video_extractor.run(video_path, audio_dir)
        
        if result.get("success", False):
            audio_path = result.get("extracted_audio_path")  # Исправляем ключ
            logger.info(f"Successfully extracted audio to: {audio_path}")
            return audio_path
        else:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"Failed to extract audio from video: {error_msg}")
            raise Exception(f"Audio extraction failed: {error_msg}")
    
    def _get_safe_extractors(self) -> List[str]:
        """
        Возвращает список безопасных extractors.
        На macOS исключает speaker_diarization из-за проблем с multiprocessing.
        """
        all_extractors = [extractor.name for extractor in self.extractors]
        
        if self._is_macos():
            # Исключаем speaker_diarization на macOS из-за проблем с multiprocessing
            safe_extractors = [ext for ext in all_extractors if ext != "speaker_diarization"]
            if len(safe_extractors) < len(all_extractors):
                logger.warning(
                    "🍎 macOS detected: speaker_diarization extractor excluded due to multiprocessing issues. "
                    f"Using {len(safe_extractors)} extractors instead of {len(all_extractors)}."
                )
            return safe_extractors
        else:
            return all_extractors
    
    def process_audio(
        self,
        input_uri: str,
        video_id: str,
        aggregates_only: bool = False,
        segment_config: Optional[Dict[str, Any]] = None,
        extractor_names: Optional[List[str]] = None,
        output_dir: str = "unified_output"
    ) -> Dict[str, Any]:
        """
        Process audio file and extract features based on parameters.
        
        Args:
            input_uri: Path to audio/video file
            video_id: Unique video identifier
            aggregates_only: If True, extract only aggregated features
            segment_config: Configuration for segment processing (if not aggregates_only)
            extractor_names: List of extractor names to use (None for all)
            output_dir: Output directory for results
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing {video_id}: aggregates_only={aggregates_only}")
        
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Step 0: Extract audio from video if needed
            actual_input_uri = input_uri
            if self._is_video_file(input_uri):
                logger.info(f"Video file detected, extracting audio first...")
                actual_input_uri = self._extract_audio_from_video(input_uri, output_dir)
                logger.info(f"Using extracted audio: {actual_input_uri}")
            
            # Step 1: Extract aggregated features using AudioProcessor extractors
            # Используем переданные extractor_names или все extractors по умолчанию
            extractor_results = self._extract_aggregated_features(
                input_uri=actual_input_uri,  # Используем извлеченное аудио
                video_id=video_id,
                extractor_names=extractor_names  # Используем переданные extractor_names
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
            
            result = {
                "video_id": video_id,
                "success": True,
                "aggregates_extracted": True,
                "manifest_path": manifest_path,
                "segments_extracted": False,
                "segment_files": {}
            }
            
            # Step 2: Extract segment features if requested
            if not aggregates_only:
                logger.info("Extracting per-segment features...")
                
                # Update config if provided
                if segment_config:
                    self._update_config_from_dict(segment_config)
                
                # Extract segment features using only segment-relevant extractors
                segment_extractor_names = self._get_segment_extractors()
                segment_extractor_results = self._extract_aggregated_features(
                    input_uri=actual_input_uri,  # Используем извлеченное аудио
                    video_id=video_id,
                    extractor_names=segment_extractor_names
                )
                
                segment_result = self._extract_segment_features(
                    video_id=video_id,
                    extractor_results=segment_extractor_results,  # Используем только segment extractors
                    manifest=manifest,
                    output_dir=output_dir
                )
                
                result.update(segment_result)
                result["segments_extracted"] = True
            
            logger.info(f"Successfully processed {video_id}")
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
    
    def _extract_aggregated_features(
        self,
        input_uri: str,
        video_id: str,
        extractor_names: Optional[List[str]] = None
    ) -> List[ExtractorResult]:
        """Extract aggregated features using AudioProcessor extractors."""
        logger.info(f"Extracting aggregated features for {video_id}")
        
        # Filter extractors if names provided
        if extractor_names:
            available_names = {extractor.name for extractor in self.extractors}
            requested_names = set(extractor_names)
            unknown_names = requested_names - available_names
            
            if unknown_names:
                logger.warning(f"Unknown extractors: {unknown_names}")
            
            extractors_to_use = [
                extractor for extractor in self.extractors 
                if extractor.name in requested_names
            ]
        else:
            # Use safe extractors by default (exclude speaker_diarization for macOS compatibility)
            safe_extractor_names = self._get_safe_extractors()
            extractors_to_use = [
                extractor for extractor in self.extractors 
                if extractor.name in safe_extractor_names
            ]
        
        logger.info(f"Using {len(extractors_to_use)} extractors")
        
        # Run extractors
        extractor_results = []
        for extractor in extractors_to_use:
            try:
                logger.info(f"Running {extractor.name}...")
                start_time = time.time()
                result = extractor.run(input_uri, "/tmp")  # Use temp directory
                processing_time = time.time() - start_time
                
                # Update processing_time if it's None
                if result.processing_time is None:
                    result.processing_time = processing_time
                
                extractor_results.append(result)
                
                if result.success:
                    logger.info(f"✅ {extractor.name} completed successfully in {processing_time:.2f}s")
                else:
                    logger.warning(f"⚠️ {extractor.name} failed: {result.error}")
                    
            except Exception as e:
                processing_time = time.time() - start_time if 'start_time' in locals() else 0.0
                logger.error(f"❌ {extractor.name} failed with exception: {e}")
                # Create failed result
                failed_result = ExtractorResult(
                    name=extractor.name,
                    version=extractor.version,
                    success=False,
                    error=str(e),
                    processing_time=processing_time
                )
                extractor_results.append(failed_result)
        
        return extractor_results
    
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
            task_id=f"unified_{video_id}",
            dataset="unified",
            timestamp=datetime.utcnow().isoformat() + "Z",
            extractors=extractor_results,
            schema_version="audio_manifest_v1",
            total_processing_time=total_processing_time if total_processing_time > 0 else None
        )
    
    def _extract_segment_features(
        self,
        video_id: str,
        extractor_results: List[ExtractorResult],
        manifest: ManifestModel,
        output_dir: str
    ) -> Dict[str, Any]:
        """Extract per-segment features from aggregated results."""
        logger.info(f"Extracting segment features for {video_id}")
        
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
        
        # Process with segment pipeline
        segment_result = self.segment_pipeline.process_single_video(
            video_id=video_id,
            extractor_outputs=extractor_outputs,
            duration=duration,
            save_features=True
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
    
    def process_batch(
        self,
        video_data: List[Dict[str, Any]],
        aggregates_only: bool = False,
        segment_config: Optional[Dict[str, Any]] = None,
        extractor_names: Optional[List[str]] = None,
        output_dir: str = "unified_batch_output"
    ) -> Dict[str, Any]:
        """
        Process batch of videos.
        
        Args:
            video_data: List of video data with keys: video_id, input_uri
            aggregates_only: If True, extract only aggregated features
            segment_config: Configuration for segment processing
            extractor_names: List of extractor names to use
            output_dir: Output directory for results
            
        Returns:
            Dictionary with batch processing results
        """
        logger.info(f"Processing batch of {len(video_data)} videos")
        
        batch_results = {
            "total_videos": len(video_data),
            "successful": 0,
            "failed": 0,
            "aggregates_only": aggregates_only,
            "results": []
        }
        
        for video_info in video_data:
            video_id = video_info["video_id"]
            # Support both input_uri and audio_uri/video_uri for compatibility
            input_uri = video_info.get("input_uri") or video_info.get("audio_uri") or video_info.get("video_uri")
            
            # Для batch: все extractors для агрегатов, ограниченные для сегментов
            batch_extractor_names = None if aggregates_only else self._get_segment_extractors()
            
            result = self.process_audio(
                input_uri=input_uri,
                video_id=video_id,
                aggregates_only=aggregates_only,
                segment_config=segment_config,
                extractor_names=batch_extractor_names,
                output_dir=output_dir
            )
            
            batch_results["results"].append(result)
            
            if result["success"]:
                batch_results["successful"] += 1
            else:
                batch_results["failed"] += 1
        
        logger.info(f"Batch processing completed: {batch_results['successful']} successful, "
                   f"{batch_results['failed']} failed")
        
        return batch_results


def create_unified_request(
    video_id: str,
    input_uri: str,
    aggregates_only: bool = False,
    segment_config: Optional[Dict[str, Any]] = None,
    extractor_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create a unified processing request.
    
    Args:
        video_id: Video identifier
        input_uri: Path to audio/video file
        aggregates_only: If True, extract only aggregated features
        segment_config: Configuration for segment processing
        extractor_names: List of extractor names to use
        
    Returns:
        Request dictionary
    """
    request = {
        "video_id": video_id,
        "input_uri": input_uri,
        "aggregates_only": aggregates_only
    }
    
    if not aggregates_only and segment_config:
        request["segment_config"] = segment_config
    
    if extractor_names:
        request["extractor_names"] = extractor_names
    
    return request


# Example usage and testing
if __name__ == "__main__":
    # Test unified processor
    processor = UnifiedAudioProcessor()
    
    # Test with aggregates only
    result1 = processor.process_audio(
        input_uri="test_audio.wav",
        video_id="test_001",
        aggregates_only=True
    )
    
    print("Aggregates only result:", result1["success"])
    
    # Test with segments
    result2 = processor.process_audio(
        input_uri="test_audio.wav",
        video_id="test_002",
        aggregates_only=False,
        segment_config={
            "segment_len": 3.0,
            "hop": 1.5,
            "max_seq_len": 32
        }
    )
    
    print("With segments result:", result2["success"])
    print("Segment files:", result2.get("segment_files", {}))
