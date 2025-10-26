"""
Utility functions for per-segment audio feature extraction.

This module contains helper functions for:
- Creating time segments from audio duration
- Slicing arrays based on time ranges
- Basic segment operations
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
import logging

logger = logging.getLogger(__name__)


def make_segments(duration: float, segment_len: float = 3.0, hop: float = 1.5) -> List[Tuple[float, float]]:
    """
    Generate segment boundaries for audio of given duration.
    
    Args:
        duration: Total audio duration in seconds
        segment_len: Length of each segment in seconds
        hop: Hop length between segments in seconds
        
    Returns:
        List of (start_time, end_time) tuples for each segment
        
    Examples:
        >>> make_segments(10.0, 3.0, 1.5)
        [(0.0, 3.0), (1.5, 4.5), (3.0, 6.0), (4.5, 7.5), (6.0, 9.0), (7.5, 10.0)]
        
        >>> make_segments(2.0, 3.0, 1.5)  # Short audio
        [(0.0, 2.0)]
    """
    if duration <= 0:
        raise ValueError(f"Duration must be positive, got {duration}")
    
    if segment_len <= 0:
        raise ValueError(f"Segment length must be positive, got {segment_len}")
    
    if hop <= 0:
        raise ValueError(f"Hop length must be positive, got {hop}")
    
    # For very short audio, create single segment
    if duration < segment_len:
        return [(0.0, float(duration))]
    
    # Generate segment start times
    starts = np.arange(0, max(0, duration - segment_len) + 1e-6, hop)
    
    # Create segments
    segments = []
    for start in starts:
        end = min(start + segment_len, duration)
        segments.append((float(start), float(end)))
    
    # Ensure we have at least one segment
    if not segments:
        segments = [(0.0, float(duration))]
    
    logger.debug(f"Created {len(segments)} segments for duration {duration}s")
    return segments


def get_time_sliced_array(
    times: np.ndarray, 
    array: np.ndarray, 
    start: float, 
    end: float
) -> Optional[np.ndarray]:
    """
    Extract array elements that fall within a time range.
    
    Args:
        times: Array of timestamps aligned to array frames
        array: Array to slice (can be 1D or 2D)
        start: Start time in seconds
        end: End time in seconds
        
    Returns:
        Sliced array or None if no elements in range
        
    Examples:
        >>> times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        >>> array = np.array([1, 2, 3, 4, 5])
        >>> get_time_sliced_array(times, array, 0.5, 1.5)
        array([2, 3])
    """
    if times is None or array is None:
        return None
    
    if len(times) == 0 or len(array) == 0:
        return None
    
    if len(times) != len(array):
        logger.warning(f"Time array length ({len(times)}) != data array length ({len(array)})")
        return None
    
    # Find indices within time range
    mask = (times >= start) & (times < end)
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        return None
    
    # Slice array
    if array.ndim == 1:
        return array[indices]
    else:
        return array[indices, :]


def get_time_sliced_embeddings(
    times: np.ndarray,
    embeddings: np.ndarray,
    start: float,
    end: float
) -> Optional[np.ndarray]:
    """
    Extract embedding vectors that fall within a time range.
    
    Args:
        times: Array of timestamps for each embedding
        embeddings: 2D array of embeddings (N, D)
        start: Start time in seconds
        end: End time in seconds
        
    Returns:
        Sliced embeddings array or None if no embeddings in range
    """
    if times is None or embeddings is None:
        return None
    
    if len(times) == 0 or len(embeddings) == 0:
        return None
    
    if len(times) != embeddings.shape[0]:
        logger.warning(f"Time array length ({len(times)}) != embeddings length ({embeddings.shape[0]})")
        return None
    
    # Find indices within time range
    mask = (times >= start) & (times < end)
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        return None
    
    return embeddings[indices, :]


def aggregate_array_features(
    array: Optional[np.ndarray],
    aggregation_type: str = "mean"
) -> Optional[Union[float, np.ndarray]]:
    """
    Aggregate array features using specified method.
    
    Args:
        array: Input array (1D or 2D)
        aggregation_type: Type of aggregation ("mean", "std", "max", "min", "median")
        
    Returns:
        Aggregated value or None if array is None/empty
    """
    if array is None or len(array) == 0:
        return None
    
    try:
        if aggregation_type == "mean":
            return np.mean(array, axis=0) if array.ndim > 1 else float(np.mean(array))
        elif aggregation_type == "std":
            return np.std(array, axis=0) if array.ndim > 1 else float(np.std(array))
        elif aggregation_type == "max":
            return np.max(array, axis=0) if array.ndim > 1 else float(np.max(array))
        elif aggregation_type == "min":
            return np.min(array, axis=0) if array.ndim > 1 else float(np.min(array))
        elif aggregation_type == "median":
            return np.median(array, axis=0) if array.ndim > 1 else float(np.median(array))
        else:
            raise ValueError(f"Unknown aggregation type: {aggregation_type}")
    except Exception as e:
        logger.warning(f"Aggregation failed: {e}")
        return None


def create_time_array(
    duration: float,
    hop_length: int,
    sample_rate: int
) -> np.ndarray:
    """
    Create time array for frame-based features.
    
    Args:
        duration: Audio duration in seconds
        hop_length: Hop length in samples
        sample_rate: Sample rate in Hz
        
    Returns:
        Array of timestamps for each frame
    """
    num_frames = int(duration * sample_rate / hop_length)
    times = np.arange(num_frames) * hop_length / sample_rate
    return times


def interpolate_missing_values(
    array: np.ndarray,
    method: str = "linear"
) -> np.ndarray:
    """
    Interpolate missing values in array.
    
    Args:
        array: Input array with potential NaN values
        method: Interpolation method ("linear", "nearest", "zero")
        
    Returns:
        Array with interpolated values
    """
    if array is None or len(array) == 0:
        return array
    
    # Check if there are any NaN values
    if not np.any(np.isnan(array)):
        return array
    
    try:
        from scipy import interpolate
        
        if array.ndim == 1:
            # 1D interpolation
            valid_mask = ~np.isnan(array)
            if np.sum(valid_mask) < 2:
                # Not enough valid points for interpolation
                return np.nan_to_num(array, nan=0.0)
            
            valid_indices = np.where(valid_mask)[0]
            valid_values = array[valid_mask]
            
            # Create interpolation function
            f = interpolate.interp1d(
                valid_indices, 
                valid_values, 
                kind=method,
                bounds_error=False,
                fill_value="extrapolate"
            )
            
            # Interpolate all indices
            all_indices = np.arange(len(array))
            interpolated = f(all_indices)
            
            return interpolated
        else:
            # 2D interpolation - interpolate along first axis
            result = np.zeros_like(array)
            for i in range(array.shape[1]):
                result[:, i] = interpolate_missing_values(array[:, i], method)
            return result
            
    except ImportError:
        logger.warning("scipy not available, using simple NaN replacement")
        return np.nan_to_num(array, nan=0.0)
    except Exception as e:
        logger.warning(f"Interpolation failed: {e}")
        return np.nan_to_num(array, nan=0.0)


def validate_segment_bounds(
    segments: List[Tuple[float, float]],
    duration: float,
    tolerance: float = 1e-6
) -> bool:
    """
    Validate that segment bounds are reasonable.
    
    Args:
        segments: List of (start, end) tuples
        duration: Total audio duration
        tolerance: Tolerance for floating point comparisons
        
    Returns:
        True if segments are valid, False otherwise
    """
    if not segments:
        return False
    
    for i, (start, end) in enumerate(segments):
        # Check basic bounds
        if start < 0 or end < 0:
            logger.error(f"Segment {i}: negative time bounds ({start}, {end})")
            return False
        
        if start >= end:
            logger.error(f"Segment {i}: start >= end ({start}, {end})")
            return False
        
        if end > duration + tolerance:
            logger.error(f"Segment {i}: end > duration ({end} > {duration})")
            return False
    
    # Check for overlaps (optional - segments can overlap)
    # This is just a warning, not an error
    for i in range(len(segments) - 1):
        start1, end1 = segments[i]
        start2, end2 = segments[i + 1]
        if end1 > start2 + tolerance:
            logger.debug(f"Overlapping segments: {i} and {i+1}")
    
    return True


def get_segment_metadata(
    segments: List[Tuple[float, float]],
    video_id: str,
    duration: float
) -> Dict[str, Any]:
    """
    Create metadata for segments.
    
    Args:
        segments: List of (start, end) tuples
        video_id: Video identifier
        duration: Total audio duration
        
    Returns:
        Dictionary with segment metadata
    """
    return {
        "video_id": video_id,
        "duration": duration,
        "num_segments": len(segments),
        "segment_times": segments,
        "total_coverage": sum(end - start for start, end in segments),
        "coverage_ratio": sum(end - start for start, end in segments) / duration if duration > 0 else 0.0
    }
