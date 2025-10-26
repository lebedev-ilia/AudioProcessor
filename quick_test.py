#!/usr/bin/env python3
"""
Quick test script for audio extractors with minimal output.
"""

import os
import sys
import json
import warnings
from datetime import datetime

# Suppress all warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

# Suppress stdout/stderr for model loading
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

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("🔍 Loading extractors...")
    
    with suppress_output():
        from extractors import discover_extractors
        extractors = discover_extractors()
    
    print(f"📋 Found {len(extractors)} extractors")
    print("🧪 Testing extractors...")
    
    successful = 0
    failed = 0
    results = []
    
    for i, extractor in enumerate(extractors, 1):
        print(f"[{i:2d}/{len(extractors)}] {extractor.name}...", end=' ')
        
        try:
            with suppress_output():
                result = extractor.run('test_audio.wav', '/tmp')
            
            if result.success:
                print("✅")
                successful += 1
                results.append({
                    'name': extractor.name,
                    'status': 'success',
                    'features': len(result.payload) if result.payload else 0
                })
            else:
                print(f"❌ {result.error}")
                failed += 1
                results.append({
                    'name': extractor.name,
                    'status': 'error',
                    'error': str(result.error)
                })
        except Exception as e:
            print(f"❌ {e}")
            failed += 1
            results.append({
                'name': extractor.name,
                'status': 'exception',
                'error': str(e)
            })
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'total': len(extractors),
        'successful': successful,
        'failed': failed,
        'success_rate': (successful / len(extractors)) * 100,
        'results': results
    }
    
    filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n📊 RESULTS:")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success rate: {successful/(successful+failed)*100:.1f}%")
    print(f"💾 Results saved to: {filename}")

if __name__ == "__main__":
    main()
