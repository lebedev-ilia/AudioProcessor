#!/usr/bin/env python3
"""
Script to analyze manifest_test_video_local.json results in a readable format.
"""

import json
import sys
from typing import Dict, Any, List

def load_manifest():
    """Load the manifest file."""
    try:
        with open('manifest_test_video_local.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ manifest_test_video_local.json not found")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return None

def print_summary(data: Dict[str, Any]):
    """Print overall summary."""
    print(f"\n📊 MANIFEST SUMMARY")
    print(f"{'='*60}")
    print(f"🎬 Video ID: {data.get('video_id', 'N/A')}")
    print(f"📅 Timestamp: {data.get('timestamp', 'N/A')}")
    print(f"📊 Dataset: {data.get('dataset', 'N/A')}")
    print(f"🆔 Task ID: {data.get('task_id', 'N/A')}")
    
    extractors = data.get('extractors', [])
    print(f"🔢 Total extractors: {len(extractors)}")
    
    successful = sum(1 for ext in extractors if ext.get('success', False))
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {len(extractors) - successful}")
    print(f"📈 Success rate: {successful/len(extractors)*100:.1f}%" if extractors else "0%")

def analyze_extractor(extractor: Dict[str, Any]):
    """Analyze a single extractor."""
    name = extractor.get('name', 'Unknown')
    success = extractor.get('success', False)
    version = extractor.get('version', 'N/A')
    payload = extractor.get('payload', {})
    
    print(f"\n🔍 {name.upper()}")
    print(f"{'='*60}")
    print(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"Version: {version}")
    
    if not success:
        return
    
    if not payload:
        print("No payload data")
        return
    
    # Categorize features
    scalar_features = {}
    array_features = {}
    complex_features = {}
    
    for key, value in payload.items():
        if isinstance(value, list):
            if len(value) > 0:
                # Check if it's a numeric array
                if all(isinstance(x, (int, float)) or x is None for x in value):
                    array_features[key] = {
                        'length': len(value),
                        'non_null': sum(1 for x in value if x is not None),
                        'sample_values': [x for x in value[:5] if x is not None]
                    }
                else:
                    complex_features[key] = {'length': len(value), 'type': 'mixed_array'}
            else:
                array_features[key] = {'length': 0, 'non_null': 0, 'sample_values': []}
        elif isinstance(value, (int, float, str, bool)) or value is None:
            scalar_features[key] = value
        else:
            complex_features[key] = {'type': type(value).__name__}
    
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
        for key, info in array_features.items():
            if info['length'] > 0:
                non_null_pct = info['non_null'] / info['length'] * 100
                sample_str = ', '.join([f"{x:.3f}" for x in info['sample_values'][:3]])
                print(f"  • {key}: {info['length']} values ({non_null_pct:.1f}% non-null)")
                if sample_str:
                    print(f"    Sample: [{sample_str}{'...' if len(info['sample_values']) > 3 else ''}]")
            else:
                print(f"  • {key}: empty array")
    
    # Print complex features
    if complex_features:
        print(f"\n🔧 Complex Features ({len(complex_features)}):")
        for key, info in complex_features.items():
            if 'length' in info:
                print(f"  • {key}: {info['type']} with {info['length']} items")
            else:
                print(f"  • {key}: {info['type']}")

def list_extractors(data: Dict[str, Any]):
    """List all available extractors."""
    print(f"\n📋 AVAILABLE EXTRACTORS")
    print(f"{'='*60}")
    
    extractors = data.get('extractors', [])
    for ext in extractors:
        name = ext.get('name', 'Unknown')
        success = ext.get('success', False)
        payload = ext.get('payload', {})
        feature_count = len(payload) if payload else 0
        
        status = "✅" if success else "❌"
        print(f"{status} {name:<30} ({feature_count} features)")

def main():
    """Main function."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        command = "summary"
    
    data = load_manifest()
    if not data:
        return 1
    
    if command == "summary":
        print_summary(data)
        list_extractors(data)
    elif command == "list":
        list_extractors(data)
    elif command == "show":
        if len(sys.argv) > 2:
            extractor_name = sys.argv[2]
            extractors = data.get('extractors', [])
            found = False
            
            for ext in extractors:
                if ext.get('name') == extractor_name:
                    print_summary(data)
                    analyze_extractor(ext)
                    found = True
                    break
            
            if not found:
                print(f"❌ Extractor '{extractor_name}' not found.")
                print("Available extractors:")
                for ext in extractors:
                    print(f"  • {ext.get('name', 'Unknown')}")
        else:
            print("Usage: python analyze_manifest.py show <extractor_name>")
            print("Available extractors:")
            for ext in data.get('extractors', []):
                print(f"  • {ext.get('name', 'Unknown')}")
    elif command == "all":
        print_summary(data)
        for ext in data.get('extractors', []):
            analyze_extractor(ext)
    else:
        print("Usage:")
        print("  python analyze_manifest.py summary  - Show overall summary")
        print("  python analyze_manifest.py list     - List all extractors")
        print("  python analyze_manifest.py show <name> - Show specific extractor")
        print("  python analyze_manifest.py all      - Show all extractors")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
