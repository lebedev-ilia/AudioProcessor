#!/usr/bin/env python3
"""
Clean test script for audio extractors with suppressed logging and result saving.
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime
from typing import Dict, Any, List

# Suppress verbose logging from various libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress tokenizer warnings

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('speechbrain').setLevel(logging.ERROR)
logging.getLogger('whisper').setLevel(logging.ERROR)
logging.getLogger('librosa').setLevel(logging.ERROR)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from extractors import discover_extractors


def suppress_stdout_stderr():
    """Suppress stdout and stderr for model loading."""
    import contextlib
    from io import StringIO
    
    @contextlib.contextmanager
    def suppress_output():
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = devnull
                sys.stderr = devnull
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
    
    return suppress_output()


def test_extractors(audio_file: str = 'test_audio.wav', output_dir: str = '/tmp') -> Dict[str, Any]:
    """
    Test all extractors and return results.
    
    Args:
        audio_file: Path to audio file to test
        output_dir: Directory to save results
        
    Returns:
        Dictionary with test results
    """
    print("🔍 Discovering extractors...")
    
    # Suppress output during extractor discovery and initialization
    with suppress_stdout_stderr():
        extractors = discover_extractors()
    
    print(f"📋 Found {len(extractors)} extractors")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'audio_file': audio_file,
        'total_extractors': len(extractors),
        'successful': 0,
        'failed': 0,
        'extractor_results': []
    }
    
    print("\n🧪 Testing extractors...")
    
    for i, extractor in enumerate(extractors, 1):
        print(f"[{i:2d}/{len(extractors)}] Testing {extractor.name}...", end=' ')
        
        try:
            # Suppress output during extraction
            with suppress_stdout_stderr():
                result = extractor.run(audio_file, output_dir)
            
            if result.success:
                print("✅ SUCCESS")
                results['successful'] += 1
                extractor_result = {
                    'name': extractor.name,
                    'status': 'success',
                    'error': None,
                    'features_count': len(result.payload) if result.payload else 0,
                    'features': list(result.payload.keys()) if result.payload else []
                }
            else:
                print(f"❌ ERROR - {result.error}")
                results['failed'] += 1
                extractor_result = {
                    'name': extractor.name,
                    'status': 'error',
                    'error': str(result.error),
                    'features_count': 0,
                    'features': []
                }
                
        except Exception as e:
            print(f"❌ EXCEPTION - {e}")
            results['failed'] += 1
            extractor_result = {
                'name': extractor.name,
                'status': 'exception',
                'error': str(e),
                'features_count': 0,
                'features': []
            }
        
        results['extractor_results'].append(extractor_result)
    
    # Calculate success rate
    results['success_rate'] = (results['successful'] / results['total_extractors']) * 100
    
    return results


def save_results(results: Dict[str, Any], output_file: str = 'extractor_test_results.json'):
    """
    Save test results to JSON file.
    
    Args:
        results: Test results dictionary
        output_file: Output file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {output_file}")


def print_summary(results: Dict[str, Any]):
    """
    Print a clean summary of test results.
    
    Args:
        results: Test results dictionary
    """
    print(f"\n📊 TEST SUMMARY")
    print(f"{'='*50}")
    print(f"📅 Timestamp: {results['timestamp']}")
    print(f"🎵 Audio file: {results['audio_file']}")
    print(f"🔢 Total extractors: {results['total_extractors']}")
    print(f"✅ Successful: {results['successful']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"📈 Success rate: {results['success_rate']:.1f}%")
    
    if results['failed'] > 0:
        print(f"\n❌ FAILED EXTRACTORS:")
        for result in results['extractor_results']:
            if result['status'] != 'success':
                print(f"  • {result['name']}: {result['error']}")
    
    print(f"\n✅ SUCCESSFUL EXTRACTORS:")
    for result in results['extractor_results']:
        if result['status'] == 'success':
            print(f"  • {result['name']}: {result['features_count']} features")


def main():
    """Main function to run the clean extractor test."""
    # Check if test audio file exists
    audio_file = 'test_audio.wav'
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        print("Please ensure test_audio.wav exists in the current directory.")
        return 1
    
    print("🚀 Starting clean extractor test...")
    print(f"🎵 Using audio file: {audio_file}")
    
    # Run tests
    results = test_extractors(audio_file)
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_file = f"extractor_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results(results, output_file)
    
    # Return exit code based on success
    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
