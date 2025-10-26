#!/usr/bin/env python3
"""
Test script that shows full extraction results, not just feature counts.
"""

import os
import sys
import json
import warnings
from datetime import datetime
from typing import Dict, Any, List

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

def format_value(value):
    """Format values for JSON serialization."""
    import numpy as np
    
    if isinstance(value, np.ndarray):
        if value.size > 100:  # Large arrays - show summary
            return {
                "type": "array",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": float(np.min(value)) if value.size > 0 else None,
                "max": float(np.max(value)) if value.size > 0 else None,
                "mean": float(np.mean(value)) if value.size > 0 else None,
                "sample": value.flatten()[:10].tolist() if value.size > 0 else []
            }
        else:
            return value.tolist()
    elif isinstance(value, (np.integer, np.floating)):
        return float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    else:
        return value

def main():
    print("🔍 Loading extractors...")
    
    with suppress_output():
        from extractors import discover_extractors
        extractors = discover_extractors()
    
    print(f"📋 Found {len(extractors)} extractors")
    print("🧪 Testing extractors with full results...")
    
    successful = 0
    failed = 0
    full_results = {
        'timestamp': datetime.now().isoformat(),
        'audio_file': 'test_audio.wav',
        'total_extractors': len(extractors),
        'successful': 0,
        'failed': 0,
        'extractors': {}
    }
    
    for i, extractor in enumerate(extractors, 1):
        print(f"[{i:2d}/{len(extractors)}] {extractor.name}...", end=' ')
        
        try:
            with suppress_output():
                result = extractor.run('test_audio.wav', '/tmp')
            
            if result.success:
                print("✅")
                successful += 1
                
                # Format the payload for JSON serialization
                formatted_payload = {}
                if result.payload:
                    for key, value in result.payload.items():
                        formatted_payload[key] = format_value(value)
                
                full_results['extractors'][extractor.name] = {
                    'status': 'success',
                    'error': None,
                    'features_count': len(result.payload) if result.payload else 0,
                    'features': formatted_payload
                }
            else:
                print(f"❌ {result.error}")
                failed += 1
                full_results['extractors'][extractor.name] = {
                    'status': 'error',
                    'error': str(result.error),
                    'features_count': 0,
                    'features': {}
                }
        except Exception as e:
            print(f"❌ {e}")
            failed += 1
            full_results['extractors'][extractor.name] = {
                'status': 'exception',
                'error': str(e),
                'features_count': 0,
                'features': {}
            }
    
    # Update summary
    full_results['successful'] = successful
    full_results['failed'] = failed
    full_results['success_rate'] = (successful / len(extractors)) * 100
    
    # Save full results
    filename = f"full_extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 SUMMARY:")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success rate: {successful/(successful+failed)*100:.1f}%")
    print(f"💾 Full results saved to: {filename}")
    
    # Show sample of results for first successful extractor
    for name, data in full_results['extractors'].items():
        if data['status'] == 'success' and data['features']:
            print(f"\n🔍 Sample from {name} (first 5 features):")
            features = data['features']
            for i, (key, value) in enumerate(features.items()):
                if i >= 5:
                    break
                if isinstance(value, dict) and 'type' in value:
                    print(f"  • {key}: {value['type']} {value.get('shape', '')} (min: {value.get('min', 'N/A')}, max: {value.get('max', 'N/A')})")
                else:
                    print(f"  • {key}: {value}")
            if len(features) > 5:
                print(f"  ... and {len(features) - 5} more features")
            break

if __name__ == "__main__":
    main()
