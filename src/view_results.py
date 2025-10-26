#!/usr/bin/env python3
"""
Script to view and analyze extraction results in a readable format.
"""

import json
import sys
import os
from typing import Dict, Any

def load_latest_results():
    """Load the most recent results file."""
    result_files = [f for f in os.listdir('.') if f.startswith('full_extraction_results_') and f.endswith('.json')]
    if not result_files:
        print("❌ No results files found. Run test_with_full_results.py first.")
        return None
    
    latest_file = sorted(result_files)[-1]
    print(f"📁 Loading results from: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_summary(results: Dict[str, Any]):
    """Print overall summary."""
    print(f"\n📊 EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"📅 Timestamp: {results['timestamp']}")
    print(f"🎵 Audio file: {results['audio_file']}")
    print(f"🔢 Total extractors: {results['total_extractors']}")
    print(f"✅ Successful: {results['successful']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"📈 Success rate: {results['success_rate']:.1f}%")

def print_extractor_results(results: Dict[str, Any], extractor_name: str = None):
    """Print detailed results for specific extractor or all extractors."""
    extractors = results['extractors']
    
    if extractor_name:
        if extractor_name not in extractors:
            print(f"❌ Extractor '{extractor_name}' not found.")
            return
        extractors = {extractor_name: extractors[extractor_name]}
    
    for name, data in extractors.items():
        print(f"\n🔍 {name.upper()}")
        print(f"{'='*60}")
        print(f"Status: {'✅ SUCCESS' if data['status'] == 'success' else '❌ FAILED'}")
        print(f"Features count: {data['features_count']}")
        
        if data['status'] != 'success':
            print(f"Error: {data['error']}")
            continue
        
        features = data['features']
        if not features:
            print("No features extracted.")
            continue
        
        # Group features by type
        scalar_features = {}
        array_features = {}
        complex_features = {}
        
        for key, value in features.items():
            if isinstance(value, dict) and 'type' in value:
                if value['type'] == 'array':
                    array_features[key] = value
                else:
                    complex_features[key] = value
            elif isinstance(value, (int, float, str, bool)):
                scalar_features[key] = value
            else:
                complex_features[key] = value
        
        # Print scalar features
        if scalar_features:
            print(f"\n📊 Scalar Features ({len(scalar_features)}):")
            for key, value in scalar_features.items():
                if isinstance(value, float):
                    print(f"  • {key}: {value:.6f}")
                else:
                    print(f"  • {key}: {value}")
        
        # Print array features summary
        if array_features:
            print(f"\n📈 Array Features ({len(array_features)}):")
            for key, value in array_features.items():
                print(f"  • {key}: {value['type']} {value['shape']} "
                      f"(min: {value.get('min', 'N/A'):.3f}, "
                      f"max: {value.get('max', 'N/A'):.3f}, "
                      f"mean: {value.get('mean', 'N/A'):.3f})")
        
        # Print complex features
        if complex_features:
            print(f"\n🔧 Complex Features ({len(complex_features)}):")
            for key, value in complex_features.items():
                if isinstance(value, dict):
                    if 'type' in value:
                        print(f"  • {key}: {value['type']} {value.get('shape', '')}")
                    else:
                        print(f"  • {key}: {type(value).__name__} with {len(value)} items")
                else:
                    print(f"  • {key}: {type(value).__name__}")

def list_extractors(results: Dict[str, Any]):
    """List all available extractors."""
    print(f"\n📋 AVAILABLE EXTRACTORS")
    print(f"{'='*60}")
    
    for name, data in results['extractors'].items():
        status = "✅" if data['status'] == 'success' else "❌"
        features = data['features_count']
        print(f"{status} {name:<25} ({features} features)")

def main():
    """Main function."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        command = "summary"
    
    results = load_latest_results()
    if not results:
        return 1
    
    if command == "summary":
        print_summary(results)
        list_extractors(results)
    elif command == "list":
        list_extractors(results)
    elif command == "show":
        if len(sys.argv) > 2:
            extractor_name = sys.argv[2]
            print_summary(results)
            print_extractor_results(results, extractor_name)
        else:
            print("Usage: python view_results.py show <extractor_name>")
            print("Available extractors:")
            for name in results['extractors'].keys():
                print(f"  • {name}")
    elif command == "all":
        print_summary(results)
        print_extractor_results(results)
    else:
        print("Usage:")
        print("  python view_results.py summary  - Show overall summary")
        print("  python view_results.py list     - List all extractors")
        print("  python view_results.py show <name> - Show specific extractor")
        print("  python view_results.py all      - Show all extractors")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
