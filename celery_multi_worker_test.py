import asyncio
import time
import os
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from src.async_unified_celery_tasks import async_unified_process_task
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

class CeleryMultiWorkerTest:
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
        logger.info("Testing Celery multi-worker performance with parallel tasks")

    def test_celery_parallel_tasks(self) -> Dict[str, Any]:
        """Test Celery with parallel individual tasks (one task per video)."""
        logger.info("=== Testing CELERY Parallel Tasks (One Task Per Video) ===")
        
        start_time = time.time()
        
        # Submit all tasks in parallel
        logger.info("Submitting parallel Celery tasks...")
        tasks = []
        for i, video_path in enumerate(self.available_videos):
            video_id = f"celery_parallel_{i+1}"
            task = async_unified_process_task.delay(
                video_id=video_id,
                input_uri=video_path,
                processing_mode="aggregates_only",
                extractor_names=self.all_extractors,
                output_dir="celery_parallel_output",
                max_cpu_workers=1000,  # High limits for maximum parallelism
                max_gpu_workers=1000,
                max_io_workers=1000
            )
            tasks.append((video_id, task))
        
        # Wait for all tasks to complete
        logger.info("Waiting for all parallel tasks to complete...")
        results = []
        for video_id, task in tasks:
            try:
                result = task.get(timeout=600)  # 10 minute timeout per task
                results.append(result)
                logger.info(f"Task {video_id} completed successfully")
            except Exception as e:
                logger.error(f"Task {video_id} failed: {e}")
                results.append({
                    "video_id": video_id,
                    "success": False,
                    "error": str(e),
                    "processing_time": 0.0
                })
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Aggregate results
        successful = sum(1 for r in results if r.get("success", False))
        failed = len(results) - successful
        
        celery_result = {
            "total_time": total_time,
            "successful": successful,
            "failed": failed,
            "results": results
        }
        
        logger.info(f"\nCelery parallel tasks processing completed in {total_time:.2f} seconds.")
        return celery_result

    def test_celery_batch_task(self) -> Dict[str, Any]:
        """Test Celery with single batch task (for comparison)."""
        logger.info("=== Testing CELERY Batch Task (Single Task) ===")
        
        # Prepare video data
        video_data = []
        for i, video_path in enumerate(self.available_videos):
            video_data.append({
                "video_id": f"celery_batch_{i+1}",
                "input_uri": video_path,
            })

        start_time = time.time()
        
        # Run batch processing through Celery
        logger.info("Starting Celery batch processing...")
        from src.async_unified_celery_tasks import async_unified_batch_task
        result = async_unified_batch_task.delay(
            videos=video_data,
            output_dir="celery_batch_output",
            extractor_names=self.all_extractors,
            max_cpu_workers=1000,  # High limits for maximum parallelism
            max_gpu_workers=1000,
            max_io_workers=1000
        )
        
        # Wait for result
        logger.info("Waiting for Celery batch task to complete...")
        celery_result = result.get(timeout=600)  # 10 minute timeout
        
        end_time = time.time()
        celery_result["total_time"] = end_time - start_time
        
        logger.info(f"\nCelery batch task processing completed in {celery_result['total_time']:.2f} seconds.")
        return celery_result

    def analyze_results(self, parallel_results: Dict[str, Any], batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and compare results."""
        analysis = {
            "parallel_processing": {
                "total_time": parallel_results["total_time"],
                "successful": parallel_results["successful"],
                "failed": parallel_results["failed"],
                "avg_time_per_video": parallel_results["total_time"] / len(self.available_videos)
            },
            "batch_processing": {
                "total_time": batch_results["total_time"],
                "successful": batch_results["successful"],
                "failed": batch_results["failed"],
                "avg_time_per_video": batch_results["total_time"] / len(self.available_videos)
            },
            "comparison": {
                "parallel_vs_batch_speedup": batch_results["total_time"] / parallel_results["total_time"] if parallel_results["total_time"] > 0 else 0,
                "parallel_vs_direct_async_speedup": 99.31 / parallel_results["total_time"] if parallel_results["total_time"] > 0 else 0,  # vs direct async
                "batch_vs_direct_async_speedup": 99.31 / batch_results["total_time"] if batch_results["total_time"] > 0 else 0
            }
        }
        
        return analysis

    def run_multi_worker_test(self):
        """Run the complete multi-worker test."""
        logger.info("Starting Celery multi-worker test with 3 videos")
        logger.info(f"Test videos: {self.available_videos}")
        logger.info("Note: No GPU available, using CPU-only extractors")
        logger.info("Testing both parallel tasks and batch task approaches")
        
        # Test 1: Parallel individual tasks
        parallel_results = self.test_celery_parallel_tasks()
        
        # Test 2: Single batch task
        batch_results = self.test_celery_batch_task()
        
        # Analyze results
        analysis = self.analyze_results(parallel_results, batch_results)
        
        print("\n--- Celery Multi-Worker Test Results ---")
        print(f"\n🔄 Parallel Tasks (One Task Per Video):")
        print(f"  Total Time: {parallel_results['total_time']:.2f}s")
        print(f"  Per Video Avg: {analysis['parallel_processing']['avg_time_per_video']:.2f}s")
        print(f"  Successful: {parallel_results['successful']}/{len(self.available_videos)}")
        print(f"  Failed: {parallel_results['failed']}")
        
        print(f"\n📦 Batch Task (Single Task):")
        print(f"  Total Time: {batch_results['total_time']:.2f}s")
        print(f"  Per Video Avg: {analysis['batch_processing']['avg_time_per_video']:.2f}s")
        print(f"  Successful: {batch_results['successful']}/{len(self.available_videos)}")
        print(f"  Failed: {batch_results['failed']}")
        
        print(f"\n📊 Performance Comparison:")
        print(f"  Parallel vs Batch Speedup: {analysis['comparison']['parallel_vs_batch_speedup']:.2f}x")
        print(f"  Parallel vs Direct Async: {analysis['comparison']['parallel_vs_direct_async_speedup']:.2f}x")
        print(f"  Batch vs Direct Async: {analysis['comparison']['batch_vs_direct_async_speedup']:.2f}x")
        
        # Assertions
        assert parallel_results["successful"] == len(self.available_videos), "Not all parallel tasks processed successfully."
        assert batch_results["successful"] == len(self.available_videos), "Not all batch tasks processed successfully."
        
        print("\n✅ All videos processed successfully with both approaches!")
        
        # Performance analysis
        if analysis['comparison']['parallel_vs_direct_async_speedup'] > 1.0:
            print(f"\n🎉 Parallel tasks are {analysis['comparison']['parallel_vs_direct_async_speedup']:.2f}x faster than direct async!")
        else:
            print(f"\n⚠️ Parallel tasks are {1/analysis['comparison']['parallel_vs_direct_async_speedup']:.2f}x slower than direct async")
        
        if analysis['comparison']['parallel_vs_batch_speedup'] > 1.0:
            print(f"🚀 Parallel tasks are {analysis['comparison']['parallel_vs_batch_speedup']:.2f}x faster than batch task!")
        else:
            print(f"📦 Batch task is {1/analysis['comparison']['parallel_vs_batch_speedup']:.2f}x faster than parallel tasks")

def main():
    tester = CeleryMultiWorkerTest()
    tester.run_multi_worker_test()

if __name__ == "__main__":
    os.makedirs("celery_parallel_output", exist_ok=True)
    os.makedirs("celery_batch_output", exist_ok=True)
    main()
