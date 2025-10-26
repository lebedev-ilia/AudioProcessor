"""
Unit tests for segment utilities.
"""

import unittest
import numpy as np
from segment_utils import (
    make_segments,
    get_time_sliced_array,
    get_time_sliced_embeddings,
    aggregate_array_features,
    create_time_array,
    validate_segment_bounds,
    get_segment_metadata
)


class TestSegmentUtils(unittest.TestCase):
    """Test cases for segment utility functions."""
    
    def test_make_segments_normal(self):
        """Test segment creation for normal duration."""
        segments = make_segments(duration=10.0, segment_len=3.0, hop=1.5)
        
        expected = [(0.0, 3.0), (1.5, 4.5), (3.0, 6.0), (4.5, 7.5), (6.0, 9.0), (7.5, 10.0)]
        
        self.assertEqual(len(segments), 6)
        self.assertEqual(segments, expected)
    
    def test_make_segments_short(self):
        """Test segment creation for short audio."""
        segments = make_segments(duration=2.0, segment_len=3.0, hop=1.5)
        
        expected = [(0.0, 2.0)]
        
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments, expected)
    
    def test_make_segments_edge_cases(self):
        """Test segment creation edge cases."""
        # Zero duration
        with self.assertRaises(ValueError):
            make_segments(duration=0.0, segment_len=3.0, hop=1.5)
        
        # Negative duration
        with self.assertRaises(ValueError):
            make_segments(duration=-1.0, segment_len=3.0, hop=1.5)
        
        # Zero segment length
        with self.assertRaises(ValueError):
            make_segments(duration=10.0, segment_len=0.0, hop=1.5)
        
        # Zero hop
        with self.assertRaises(ValueError):
            make_segments(duration=10.0, segment_len=3.0, hop=0.0)
    
    def test_get_time_sliced_array(self):
        """Test time-based array slicing."""
        times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        array = np.array([1, 2, 3, 4, 5, 6, 7])
        
        # Normal slice
        result = get_time_sliced_array(times, array, 0.5, 2.5)
        expected = np.array([2, 3, 4, 5])
        np.testing.assert_array_equal(result, expected)
        
        # No overlap
        result = get_time_sliced_array(times, array, 4.0, 5.0)
        self.assertIsNone(result)
        
        # Empty arrays
        result = get_time_sliced_array(np.array([]), np.array([]), 0.0, 1.0)
        self.assertIsNone(result)
        
        # Mismatched lengths
        result = get_time_sliced_array(times, np.array([1, 2, 3]), 0.0, 1.0)
        self.assertIsNone(result)
    
    def test_get_time_sliced_embeddings(self):
        """Test time-based embedding slicing."""
        times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        embeddings = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15]
        ])
        
        result = get_time_sliced_embeddings(times, embeddings, 0.5, 2.0)
        expected = np.array([
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])
        np.testing.assert_array_equal(result, expected)
    
    def test_aggregate_array_features(self):
        """Test array feature aggregation."""
        array = np.array([1, 2, 3, 4, 5])
        
        # Mean aggregation
        result = aggregate_array_features(array, "mean")
        self.assertEqual(result, 3.0)
        
        # Std aggregation
        result = aggregate_array_features(array, "std")
        self.assertAlmostEqual(result, 1.581, places=2)
        
        # Max aggregation
        result = aggregate_array_features(array, "max")
        self.assertEqual(result, 5.0)
        
        # Min aggregation
        result = aggregate_array_features(array, "min")
        self.assertEqual(result, 1.0)
        
        # Median aggregation
        result = aggregate_array_features(array, "median")
        self.assertEqual(result, 3.0)
        
        # None input
        result = aggregate_array_features(None, "mean")
        self.assertIsNone(result)
        
        # Empty array
        result = aggregate_array_features(np.array([]), "mean")
        self.assertIsNone(result)
    
    def test_create_time_array(self):
        """Test time array creation."""
        times = create_time_array(duration=2.0, hop_length=512, sample_rate=22050)
        
        expected_length = int(2.0 * 22050 / 512)
        self.assertEqual(len(times), expected_length)
        
        # Check first few values
        expected_times = np.arange(expected_length) * 512 / 22050
        np.testing.assert_array_almost_equal(times, expected_times)
    
    def test_validate_segment_bounds(self):
        """Test segment bounds validation."""
        # Valid segments
        segments = [(0.0, 3.0), (1.5, 4.5), (3.0, 6.0)]
        self.assertTrue(validate_segment_bounds(segments, 10.0))
        
        # Empty segments
        self.assertFalse(validate_segment_bounds([], 10.0))
        
        # Negative start time
        segments = [(-1.0, 3.0)]
        self.assertFalse(validate_segment_bounds(segments, 10.0))
        
        # Start >= end
        segments = [(3.0, 3.0)]
        self.assertFalse(validate_segment_bounds(segments, 10.0))
        
        # End > duration
        segments = [(0.0, 15.0)]
        self.assertFalse(validate_segment_bounds(segments, 10.0))
    
    def test_get_segment_metadata(self):
        """Test segment metadata creation."""
        segments = [(0.0, 3.0), (1.5, 4.5), (3.0, 6.0)]
        video_id = "test_video"
        duration = 10.0
        
        metadata = get_segment_metadata(segments, video_id, duration)
        
        self.assertEqual(metadata["video_id"], video_id)
        self.assertEqual(metadata["duration"], duration)
        self.assertEqual(metadata["num_segments"], 3)
        self.assertEqual(metadata["segment_times"], segments)
        self.assertAlmostEqual(metadata["total_coverage"], 9.0)
        self.assertAlmostEqual(metadata["coverage_ratio"], 0.9)


if __name__ == "__main__":
    unittest.main()
