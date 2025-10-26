import time
import os
from typing import List, Dict, Any

from src.async_unified_celery_tasks import async_unified_process_task
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Define the test video paths
TEST_VIDEO_DIR = "src/tests/test_videos"
TEST_VIDEO_FILES = [
    os.path.join(TEST_VIDEO_DIR, "-69HDT6DZEM.mp4"),
    os.path.join(TEST_VIDEO_DIR, "-JuF2ivdnAg.mp4"),
    os.path.join(TEST_VIDEO_DIR, "-niwQ0xGEGk.mp4"),
]

def test_celery_simple():
    """Simple Celery test with progress tracking."""
    logger.info("=== Simple Celery Test ===")
    
    available_videos = []
    for video_path in TEST_VIDEO_FILES:
        if os.path.exists(video_path):
            available_videos.append(video_path)
        else:
            logger.warning(f"Test video not found: {video_path}")
    
    logger.info(f"Found {len(available_videos)} test videos")
    
    start_time = time.time()
    
    # Submit tasks one by one with progress tracking
    tasks = []
    for i, video_path in enumerate(available_videos):
        video_id = f"simple_{i+1}"
        logger.info(f"Submitting task {i+1}/{len(available_videos)}: {video_id}")
        
        task = async_unified_process_task.delay(
            video_id=video_id,
            input_uri=video_path,
            processing_mode="aggregates_only",
            extractor_names=None,  # Use all extractors
            output_dir="celery_simple_output",
            max_cpu_workers=1000,
            max_gpu_workers=1000,
            max_io_workers=1000
        )
        tasks.append((video_id, task))
    
    # Wait for tasks with timeout and progress tracking
    results = []
    for i, (video_id, task) in enumerate(tasks):
        logger.info(f"Waiting for task {i+1}/{len(tasks)}: {video_id}")
        
        try:
            # Check task status first
            if task.ready():
                logger.info(f"Task {video_id} already completed")
                result = task.result
            else:
                logger.info(f"Task {video_id} in progress, waiting...")
                result = task.get(timeout=300)  # 5 minute timeout per task
            
            results.append(result)
            logger.info(f"✅ Task {video_id} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Task {video_id} failed: {e}")
            results.append({
                "video_id": video_id,
                "success": False,
                "error": str(e),
                "processing_time": 0.0
            })
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Analyze results
    successful = sum(1 for r in results if r.get("success", False))
    failed = len(results) - successful
    
    print(f"\n--- Simple Celery Test Results ---")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Successful: {successful}/{len(available_videos)}")
    print(f"Failed: {failed}")
    
    for result in results:
        if result.get("success", False):
            print(f"  ✅ {result['video_id']}: {result.get('processing_time', 0):.2f}s")
        else:
            print(f"  ❌ {result['video_id']}: {result.get('error', 'Unknown error')}")
    
    # Compare with previous results
    print(f"\n📊 Performance Comparison:")
    print(f"  vs Direct Async (99.31s): {99.31 / total_time:.2f}x")
    print(f"  vs Single Worker (218.17s): {218.17 / total_time:.2f}x")
    print(f"  vs Previous Parallel (167.70s): {167.70 / total_time:.2f}x")
    
    if total_time < 99.31:
        print(f"\n🎉 Celery is {99.31 / total_time:.2f}x faster than direct async!")
    else:
        print(f"\n⚠️ Celery is {total_time / 99.31:.2f}x slower than direct async")
    
    return {
        "total_time": total_time,
        "successful": successful,
        "failed": failed,
        "results": results
    }

if __name__ == "__main__":
    os.makedirs("celery_simple_output", exist_ok=True)
    test_celery_simple()
