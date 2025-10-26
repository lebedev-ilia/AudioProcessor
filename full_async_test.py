import asyncio
import time
import os
from typing import List, Dict, Any, Optional

from src.unified_processor import UnifiedAudioProcessor
from src.async_unified_processor import AsyncUnifiedAudioProcessor
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

class FullAsyncTest:
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
        logger.info("Note: No GPU available, using CPU-only extractors")

    def test_sync_processing(self) -> Dict[str, Any]:
        """Test synchronous processing with all extractors."""
        logger.info("=== Testing SYNC Processing (All Extractors) ===")
        
        segment_config = get_default_config()
        processor = UnifiedAudioProcessor(config=segment_config)
        
        results = {
            "successful": 0,
            "failed": 0,
            "total_time": 0.0,
            "results": []
        }
        
        start_time = time.time()
        for i, video_path in enumerate(self.available_videos):
            video_id = f"sync_full_{i+1}"
            output_dir = os.path.join("sync_full_output", video_id)
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"Processing {video_id}: {video_path}")
            
            result = processor.process_audio(
                input_uri=video_path,
                video_id=video_id,
                output_dir=output_dir,
                extractor_names=self.all_extractors # Use all extractors
            )
            
            if result["success"]:
                results["successful"] += 1
            else:
                results["failed"] += 1
            results["results"].append(result)
        
        end_time = time.time()
        results["total_time"] = end_time - start_time
        
        logger.info(f"Synchronous batch processing completed in {results['total_time']:.2f} seconds.")
        return results

    async def test_async_processing(self) -> Dict[str, Any]:
        """Test asynchronous processing with all extractors and no semaphores."""
        logger.info("=== Testing ASYNC Processing (All Extractors, No Semaphores) ===")
        
        segment_config = get_default_config()
        
        # Create processor with maximum concurrency (no semaphore limits)
        processor = AsyncUnifiedAudioProcessor(
            config=segment_config,
            max_cpu_workers=1000,  # Very high limit - no semaphore blocking
            max_gpu_workers=1000,  # Very high limit - no semaphore blocking
            max_io_workers=1000    # Very high limit - no semaphore blocking
        )
        
        video_data = []
        for i, video_path in enumerate(self.available_videos):
            video_data.append({
                "video_id": f"async_full_{i+1}",
                "input_uri": video_path,
            })

        start_time = time.time()
        
        # Run batch processing asynchronously
        results = await processor.process_batch_async(
            video_data=video_data,
            output_dir="async_full_output",
            extractor_names=self.all_extractors # Use all extractors
        )
        
        end_time = time.time()
        results["total_time"] = end_time - start_time
        
        logger.info(f"\nAsynchronous batch processing completed in {results['total_time']:.2f} seconds.")
        return results

    def analyze_results(self, sync_results: Dict[str, Any], async_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and compare results."""
        sync_time = sync_results["total_time"]
        async_time = async_results["total_time"]
        
        sync_per_video = [r.get("processing_time", 0.0) for r in sync_results["results"] if r["success"]]
        async_per_video = [r.get("processing_time", 0.0) for r in async_results["results"] if r["success"]]
        
        analysis = {
            "sync_processing": {
                "total_time": sync_time,
                "per_video_avg": sum(sync_per_video) / len(sync_per_video) if sync_per_video else 0,
                "successful": sync_results["successful"],
                "failed": sync_results["failed"]
            },
            "async_processing": {
                "total_time": async_time,
                "per_video_avg": sum(async_per_video) / len(async_per_video) if async_per_video else 0,
                "successful": async_results["successful"],
                "failed": async_results["failed"]
            },
            "performance": {
                "speedup": sync_time / async_time if async_time > 0 else 0,
                "time_saved": sync_time - async_time,
                "best_method": "async" if async_time < sync_time else "sync"
            }
        }
        
        return analysis

    async def run_full_test(self):
        logger.info("Starting full async processing test with ALL extractors")
        logger.info(f"Test videos: {self.available_videos}")
        logger.info("Using ALL available extractors (no filtering)")
        logger.info("No semaphore limits (maximum concurrency)")
        logger.info("Note: No GPU available, using CPU-only extractors")

        # Run synchronous test
        sync_results = self.test_sync_processing()
        
        # Run asynchronous test
        async_results = await self.test_async_processing()
        
        # Analyze results
        analysis = self.analyze_results(sync_results, async_results)
        
        print("\n" + "="*70)
        print("FULL ASYNC PERFORMANCE ANALYSIS (ALL EXTRACTORS)")
        print("="*70)
        print()
        print(f"📊 SYNC Processing (All Extractors):")
        print(f"   Total Time: {analysis['sync_processing']['total_time']:.2f}s")
        print(f"   Per Video Avg: {analysis['sync_processing']['per_video_avg']:.2f}s")
        print(f"   Successful: {analysis['sync_processing']['successful']}/{len(self.available_videos)}")
        print()
        print(f"🚀 ASYNC Processing (All Extractors, No Semaphores):")
        print(f"   Total Time: {analysis['async_processing']['total_time']:.2f}s")
        print(f"   Per Video Avg: {analysis['async_processing']['per_video_avg']:.2f}s")
        print(f"   Speedup: {analysis['performance']['speedup']:.2f}x")
        print(f"   Successful: {analysis['async_processing']['successful']}/{len(self.available_videos)}")
        print()
        print(f"🏆 Best Method: {analysis['performance']['best_method']}")
        print(f"   Overall Speedup: {analysis['performance']['speedup']:.2f}x")
        if analysis['performance']['speedup'] > 1:
            print(f"   ✅ Async processing is {analysis['performance']['speedup']:.2f}x faster!")
        else:
            print(f"   ⚠️ Async processing was not faster than sync")
        
        # Save results
        import json
        results_data = {
            "test_type": "full_async_all_extractors",
            "sync_results": sync_results,
            "async_results": async_results,
            "analysis": analysis,
            "timestamp": time.time()
        }
        
        with open("full_async_test_results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        logger.info("Results saved to full_async_test_results.json")
        
        assert sync_results["successful"] == len(self.available_videos), "Not all sync videos processed successfully."
        assert async_results["successful"] == len(self.available_videos), "Not all async videos processed successfully."
        
        print(f"\n✅ Full test completed successfully!")
        return analysis

async def main():
    tester = FullAsyncTest()
    await tester.run_full_test()

if __name__ == "__main__":
    os.makedirs("sync_full_output", exist_ok=True)
    os.makedirs("async_full_output", exist_ok=True)
    asyncio.run(main())
