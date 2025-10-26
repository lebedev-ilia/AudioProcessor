import asyncio
import time
import os
from typing import List, Dict, Any, Optional

from src.async_unified_celery_tasks import async_unified_batch_task
from src.segment_config import get_default_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Define the test video paths
TEST_VIDEO_DIR = "src/tests/test_videos"
TEST_VIDEO_FILES = [
    os.path.join(TEST_VIDEO_DIR, "-69HDT6DZEM.mp4"),
    os.path.join(TEST_VIDEO_DIR, "-JuF2ivdnAg.mp4"),
    os.path.join(TEST_VIDEO_DIR, "-niwQ0xGEGk.mp4"),
]

class CelerySingleWorkerTest:
    def __init__(self):
        self.available_videos = []
        for video_path in TEST_VIDEO_FILES:
            if os.path.exists(video_path):
                self.available_videos.append(video_path)
            else:
                logger.warning(f"Test video not found: {video_path}")
        
        logger.info(f"Found {len(self.available_videos)} test videos")
        
        # Use ALL extractors (no filtering)
        self.all_extractors = None  # None means use all available extractors
        
        logger.info("Using ALL available extractors (no filtering)")
        logger.info("Testing Celery single worker performance")

    def test_celery_single_worker(self) -> Dict[str, Any]:
        """Test Celery processing with single worker."""
        logger.info("=== Testing CELERY Single Worker (All Extractors) ===")
        
        # Prepare video data
        video_data = []
        for i, video_path in enumerate(self.available_videos):
            video_data.append({
                "video_id": f"celery_single_{i+1}",
                "input_uri": video_path,
            })

        start_time = time.time()
        
        # Run batch processing through Celery
        logger.info("Starting Celery batch processing...")
        result = async_unified_batch_task.delay(
            videos=video_data,
            output_dir="celery_single_output",
            extractor_names=self.all_extractors
        )
        
        # Wait for result
        logger.info("Waiting for Celery task to complete...")
        celery_result = result.get(timeout=600)  # 10 minute timeout
        
        end_time = time.time()
        
        # Add timing information
        celery_result["total_time"] = end_time - start_time
        
        logger.info(f"\nCelery single worker processing completed in {celery_result['total_time']:.2f} seconds.")
        return celery_result

    def analyze_results(self, celery_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Celery results."""
        total_time = celery_results["total_time"]
        
        celery_per_video = [r.get("processing_time", 0.0) for r in celery_results["results"] if r["success"]]
        
        analysis = {
            "celery_processing": {
                "total_time": total_time,
                "per_video_avg": sum(celery_per_video) / len(celery_per_video) if celery_per_video else 0,
                "successful": celery_results["successful"],
                "failed": celery_results["failed"]
            },
            "performance": {
                "total_videos": len(self.available_videos),
                "avg_time_per_video": total_time / len(self.available_videos) if self.available_videos else 0
            }
        }
        
        return analysis

    def run_celery_test(self):
        """Run the complete Celery test."""
        logger.info("Starting Celery single worker test with 3 videos")
        logger.info(f"Test videos: {self.available_videos}")
        logger.info("Note: No GPU available, using CPU-only extractors")
        
        # Run Celery test
        celery_results = self.test_celery_single_worker()
        
        # Analyze results
        analysis = self.analyze_results(celery_results)
        
        print("\n--- Celery Single Worker Test Results ---")
        print(f"Celery Processing Time: {celery_results['total_time']:.2f} seconds")
        for res in celery_results["results"]:
            print(f"  Video ID: {res['video_id']}, Success: {res['success']}, Processing Time: {res.get('processing_time', 'N/A')}")
        
        print(f"\nAnalysis:")
        print(f"  Total Time: {analysis['celery_processing']['total_time']:.2f}s")
        print(f"  Per Video Avg: {analysis['celery_processing']['per_video_avg']:.2f}s")
        print(f"  Successful: {analysis['celery_processing']['successful']}/{len(self.available_videos)}")
        print(f"  Failed: {analysis['celery_processing']['failed']}")
        
        assert celery_results["successful"] == len(self.available_videos), "Not all videos processed successfully."
        
        print("\n✅ All videos processed successfully with Celery single worker!")
        
        # Compare with previous async results (from full_async_test_results.json)
        # Expected: ~99.31 seconds for async processing
        expected_async_time = 99.31
        if celery_results['total_time'] <= expected_async_time * 1.1:  # 10% tolerance
            print(f"\n🎉 Celery single worker performance is comparable to direct async processing!")
            print(f"   Celery: {celery_results['total_time']:.2f}s")
            print(f"   Direct Async: {expected_async_time:.2f}s")
        else:
            print(f"\n⚠️ Celery single worker is slower than expected:")
            print(f"   Celery: {celery_results['total_time']:.2f}s")
            print(f"   Direct Async: {expected_async_time:.2f}s")
            print(f"   Difference: {celery_results['total_time'] - expected_async_time:.2f}s")

def main():
    tester = CelerySingleWorkerTest()
    tester.run_celery_test()

if __name__ == "__main__":
    os.makedirs("celery_single_output", exist_ok=True)
    main()
