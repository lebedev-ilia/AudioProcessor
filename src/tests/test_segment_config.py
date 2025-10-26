"""
Unit tests for segment configuration.
"""

import unittest
import tempfile
import os
from segment_config import SegmentConfig, create_config, get_default_config


class TestSegmentConfig(unittest.TestCase):
    """Test cases for segment configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = get_default_config()
        
        self.assertEqual(config.segment_len, 3.0)
        self.assertEqual(config.hop, 1.5)
        self.assertEqual(config.max_seq_len, 128)
        self.assertEqual(config.k_start, 16)
        self.assertEqual(config.k_end, 16)
        self.assertEqual(config.scaler_type, "StandardScaler")
        self.assertEqual(config.storage_format, "npy+json")
        self.assertEqual(config.dtype, "float32")
    
    def test_custom_config(self):
        """Test custom configuration creation."""
        config = create_config(
            segment_len=5.0,
            hop=2.0,
            max_seq_len=64,
            k_start=8,
            k_end=8,
            scaler_type="RobustScaler",
            storage_format="tfrecord"
        )
        
        self.assertEqual(config.segment_len, 5.0)
        self.assertEqual(config.hop, 2.0)
        self.assertEqual(config.max_seq_len, 64)
        self.assertEqual(config.k_start, 8)
        self.assertEqual(config.k_end, 8)
        self.assertEqual(config.scaler_type, "RobustScaler")
        self.assertEqual(config.storage_format, "tfrecord")
    
    def test_importance_weights(self):
        """Test importance weights initialization."""
        config = SegmentConfig()
        
        self.assertIn("rms", config.importance_weights)
        self.assertIn("voiced_fraction", config.importance_weights)
        self.assertEqual(config.importance_weights["rms"], 0.6)
        self.assertEqual(config.importance_weights["voiced_fraction"], 0.4)
    
    def test_pca_dims(self):
        """Test PCA dimensions initialization."""
        config = SegmentConfig()
        
        self.assertIn("clap", config.pca_dims)
        self.assertIn("wav2vec", config.pca_dims)
        self.assertIn("yamnet", config.pca_dims)
        self.assertEqual(config.pca_dims["clap"], 128)
        self.assertEqual(config.pca_dims["wav2vec"], 64)
        self.assertEqual(config.pca_dims["yamnet"], 128)
    
    def test_feature_mapping(self):
        """Test feature mapping generation."""
        config = SegmentConfig()
        mapping = config.get_feature_mapping()
        
        self.assertIn("clap_extractor.clap_embedding", mapping)
        self.assertIn("loudness_extractor.rms_mean", mapping)
        self.assertIn("vad_extractor.f0_mean", mapping)
        
        self.assertEqual(mapping["clap_extractor.clap_embedding"], "clap_pca")
        self.assertEqual(mapping["loudness_extractor.rms_mean"], "rms_mean")
        self.assertEqual(mapping["vad_extractor.f0_mean"], "f0_mean")
    
    def test_array_fields(self):
        """Test array fields mapping."""
        config = SegmentConfig()
        array_fields = config.get_array_fields()
        
        self.assertIn("clap_extractor.clap_embedding", array_fields)
        self.assertIn("loudness_extractor.rms_array", array_fields)
        self.assertIn("vad_extractor.f0_array", array_fields)
        
        self.assertEqual(array_fields["clap_extractor.clap_embedding"], "clap_times")
        self.assertEqual(array_fields["loudness_extractor.rms_array"], "rms_times")
        self.assertEqual(array_fields["vad_extractor.f0_array"], "f0_times")
    
    def test_scalar_fields(self):
        """Test scalar fields list."""
        config = SegmentConfig()
        scalar_fields = config.get_scalar_fields()
        
        self.assertIn("loudness_extractor.rms_mean", scalar_fields)
        self.assertIn("vad_extractor.f0_mean", scalar_fields)
        self.assertIn("tempo_extractor.tempo_bpm", scalar_fields)
    
    def test_artifact_paths(self):
        """Test artifact paths generation."""
        config = SegmentConfig()
        paths = config.get_artifact_paths()
        
        self.assertIn("pca_clap", paths)
        self.assertIn("pca_wav2vec", paths)
        self.assertIn("scaler", paths)
        self.assertIn("config", paths)
        
        self.assertTrue(paths["pca_clap"].endswith("pca_clap.joblib"))
        self.assertTrue(paths["scaler"].endswith("scaler.joblib"))
        self.assertTrue(paths["config"].endswith("segment_config.json"))
    
    def test_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = SegmentConfig()
        config_dict = config.to_dict()
        
        self.assertIn("segment_len", config_dict)
        self.assertIn("hop", config_dict)
        self.assertIn("max_seq_len", config_dict)
        self.assertIn("importance_weights", config_dict)
        self.assertIn("pca_dims", config_dict)
        
        self.assertEqual(config_dict["segment_len"], 3.0)
        self.assertEqual(config_dict["hop"], 1.5)
        self.assertEqual(config_dict["max_seq_len"], 128)
    
    def test_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "segment_len": 5.0,
            "hop": 2.0,
            "max_seq_len": 64,
            "k_start": 8,
            "k_end": 8,
            "importance_weights": {"rms": 0.7, "voiced_fraction": 0.3},
            "pca_dims": {"clap": 64, "wav2vec": 32},
            "scaler_type": "RobustScaler",
            "storage_format": "tfrecord",
            "dtype": "float16"
        }
        
        config = SegmentConfig.from_dict(config_dict)
        
        self.assertEqual(config.segment_len, 5.0)
        self.assertEqual(config.hop, 2.0)
        self.assertEqual(config.max_seq_len, 64)
        self.assertEqual(config.k_start, 8)
        self.assertEqual(config.k_end, 8)
        self.assertEqual(config.importance_weights["rms"], 0.7)
        self.assertEqual(config.importance_weights["voiced_fraction"], 0.3)
        self.assertEqual(config.pca_dims["clap"], 64)
        self.assertEqual(config.pca_dims["wav2vec"], 32)
        self.assertEqual(config.scaler_type, "RobustScaler")
        self.assertEqual(config.storage_format, "tfrecord")
        self.assertEqual(config.dtype, "float16")


if __name__ == "__main__":
    unittest.main()
