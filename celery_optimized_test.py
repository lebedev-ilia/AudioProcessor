import asyncio
import time
import os
from typing import List, Dict, Any, Optional

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

class CeleryOptimizedTest:
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
        logger.info("Testing OPTIMIZED Celery performance")

    def test_celery_parallel_tasks_optimized(self) -> Dict[str, Any]:
        """Test Celery with optimized parallel individual tasks."""
        logger.info("=== Testing CELERY Optimized Parallel Tasks ===")
        
        start_time = time.time()
        
        # Submit all tasks in parallel with optimized settings
        logger.info("Submitting optimized parallel Celery tasks...")
        tasks = []
        for i, video_path in enumerate(self.available_videos):
            video_id = f"celery_opt_{i+1}"
            task = async_unified_process_task.delay(
                video_id=video_id,
                input_uri=video_path,
                processing_mode="aggregates_only",
                extractor_names=self.all_extractors,
                output_dir="celery_optimized_output",
                max_cpu_workers=1000,  # Maximum parallelism
                max_gpu_workers=1000,
                max_io_workers=1000
            )
            tasks.append((video_id, task))
        
        # Wait for all tasks to complete with progress tracking
        logger.info("Waiting for all optimized parallel tasks to complete...")
        results = []
        completed_tasks = 0
        
        for video_id, task in tasks:
            try:
                logger.info(f"Waiting for task {video_id}...")
                result = task.get(timeout=600)  # 10 minute timeout per task
                results.append(result)
                completed_tasks += 1
                logger.info(f"✅ Task {video_id} completed successfully ({completed_tasks}/{len(tasks)})")
            except Exception as e:
                logger.error(f"❌ Task {video_id} failed: {e}")
                results.append({
                    "video_id": video_id,
                    "success": False,
                    "error": str(e),
                    "processing_time": 0.0
                })
                completed_tasks += 1
        
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
        
        logger.info(f"\n🎉 Celery optimized parallel tasks completed in {total_time:.2f} seconds.")
        return celery_result

    def analyze_optimized_results(self, celery_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimized Celery results."""
        total_time = celery_results["total_time"]
        
        celery_per_video = [r.get("processing_time", 0.0) for r in celery_results["results"] if r["success"]]
        
        analysis = {
            "celery_optimized": {
                "total_time": total_time,
                "per_video_avg": sum(celery_per_video) / len(celery_per_video) if celery_per_video else 0,
                "successful": celery_results["successful"],
                "failed": celery_results["failed"]
            },
            "performance": {
                "total_videos": len(self.available_videos),
                "avg_time_per_video": total_time / len(self.available_videos) if self.available_videos else 0
            },
            "comparison": {
                "vs_direct_async_speedup": 99.31 / total_time if total_time > 0 else 0,  # vs direct async (99.31s)
                "vs_single_worker_speedup": 218.17 / total_time if total_time > 0 else 0,  # vs single worker (218.17s)
                "vs_previous_parallel_speedup": 167.70 / total_time if total_time > 0 else 0  # vs previous parallel (167.70s)
            }
        }
        
        return analysis

    def run_optimized_test(self):
        """Run the optimized Celery test."""
        logger.info("🚀 Starting OPTIMIZED Celery test with 3 videos")
        logger.info(f"Test videos: {self.available_videos}")
        logger.info("Note: No GPU available, using CPU-only extractors")
        logger.info("Testing optimized parallel tasks with maximum concurrency")
        
        # Run optimized test
        celery_results = self.test_celery_parallel_tasks_optimized()
        
        # Analyze results
        analysis = self.analyze_optimized_results(celery_results)
        
        print("\n--- 🚀 Celery Optimized Test Results ---")
        print(f"\n⚡ Optimized Parallel Tasks:")
        print(f"  Total Time: {celery_results['total_time']:.2f}s")
        print(f"  Per Video Avg: {analysis['celery_optimized']['per_video_avg']:.2f}s")
        print(f"  Successful: {celery_results['successful']}/{len(self.available_videos)}")
        print(f"  Failed: {celery_results['failed']}")
        
        print(f"\n📊 Performance Comparison:")
        print(f"  vs Direct Async: {analysis['comparison']['vs_direct_async_speedup']:.2f}x")
        print(f"  vs Single Worker: {analysis['comparison']['vs_single_worker_speedup']:.2f}x")
        print(f"  vs Previous Parallel: {analysis['comparison']['vs_previous_parallel_speedup']:.2f}x")
        
        # Assertions
        assert celery_results["successful"] == len(self.available_videos), "Not all videos processed successfully."
        
        print("\n✅ All videos processed successfully with optimized Celery!")
        
        # Performance analysis
        if analysis['comparison']['vs_direct_async_speedup'] > 1.0:
            print(f"\n🎉 Optimized Celery is {analysis['comparison']['vs_direct_async_speedup']:.2f}x faster than direct async!")
        else:
            print(f"\n⚠️ Optimized Celery is {1/analysis['comparison']['vs_direct_async_speedup']:.2f}x slower than direct async")
        
        if analysis['comparison']['vs_single_worker_speedup'] > 1.0:
            print(f"🚀 Optimized Celery is {analysis['comparison']['vs_single_worker_speedup']:.2f}x faster than single worker!")
        
        if analysis['comparison']['vs_previous_parallel_speedup'] > 1.0:
            print(f"⚡ Optimized Celery is {analysis['comparison']['vs_previous_parallel_speedup']:.2f}x faster than previous parallel!")

def main():
    tester = CeleryOptimizedTest()
    tester.run_optimized_test()

if __name__ == "__main__":
    os.makedirs("celery_optimized_output", exist_ok=True)
    main()
