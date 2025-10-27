"""
Script to fix all extractors with proper GPU fallback mechanisms.
This script ensures all extractors have proper imports, device detection, and fallback.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any

class FallbackFixer:
    """Utility class to fix extractors with proper fallback mechanisms."""
    
    def __init__(self, extractors_dir: str):
        """Initialize the fallback fixer."""
        self.extractors_dir = Path(extractors_dir)
        self.extractors_to_fix = [
            "pitch_extractor.py",
            "vad_extractor.py", 
            "tempo_extractor.py",
            "quality_extractor.py",
            "onset_extractor.py",
            "voice_quality_extractor.py",
            "phoneme_analysis_extractor.py",
            "rhythmic_analysis_extractor.py",
            "music_analysis_extractor.py",
            "video_audio_extractor.py",
            "sound_event_detection_extractor.py",
            "source_separation_extractor.py",
            "emotion_recognition_extractor.py"
        ]
    
    def fix_extractor(self, extractor_path: Path) -> bool:
        """Fix a single extractor with proper fallback mechanisms."""
        try:
            with open(extractor_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix imports
            content = self.fix_imports(content)
            
            # Fix __init__ method
            content = self.fix_init_method(content)
            
            # Add device info to features
            content = self.add_device_info_to_features(content)
            
            # Only write if content changed
            if content != original_content:
                with open(extractor_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ Fixed {extractor_path.name}")
                return True
            else:
                print(f"⏭️  No changes needed for {extractor_path.name}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to fix {extractor_path.name}: {e}")
            return False
    
    def fix_imports(self, content: str) -> str:
        """Fix imports to include torch."""
        # Add torch import if not present
        if "import torch" not in content:
            # Find the last import statement
            import_pattern = r'(from .* import .*|import .*)'
            imports = re.findall(import_pattern, content)
            if imports:
                last_import = imports[-1]
                last_import_pos = content.rfind(last_import)
                insert_pos = last_import_pos + len(last_import)
                content = content[:insert_pos] + '\nimport torch' + content[insert_pos:]
            else:
                # Add after docstring
                docstring_end = content.find('"""', content.find('"""') + 3) + 3
                content = content[:docstring_end] + '\n\nimport torch' + content[docstring_end:]
        
        return content
    
    def fix_init_method(self, content: str) -> str:
        """Fix __init__ method to include device parameter and proper detection."""
        # Pattern to find __init__ method
        init_pattern = r'def __init__\(self(?:, [^)]*)?\):'
        init_match = re.search(init_pattern, content)
        
        if not init_match:
            return content
        
        init_start = init_match.start()
        init_end = init_match.end()
        
        # Check if already has device parameter
        init_line = content[init_start:init_end]
        if "device" not in init_line:
            # Add device parameter
            if "self" in init_line and "(" in init_line:
                init_line = init_line.replace("self", "self, device: str = \"auto\"")
            else:
                init_line = init_line.replace("self)", "self, device: str = \"auto\")")
        
        # Replace the init line
        content = content[:init_start] + init_line + content[init_end:]
        
        # Find the super().__init__() call and add device detection after it
        super_pattern = r'super\(\)\.__init__\(\)'
        super_match = re.search(super_pattern, content)
        
        if super_match:
            super_end = super_match.end()
            # Check if device detection already exists
            if "torch.cuda.is_available()" not in content[super_end:super_end+200]:
                device_detection = '''
        
        # Device detection with fallback
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        self.logger.info(f"Initialized {self.name} v{self.version} on {self.device}")'''
                
                content = content[:super_end] + device_detection + content[super_end:]
        
        return content
    
    def add_device_info_to_features(self, content: str) -> str:
        """Add device information to feature extraction results."""
        # Look for feature dictionaries being created
        feature_dict_pattern = r'features\s*=\s*\{[^}]*\}'
        
        def add_device_info(match):
            feature_dict = match.group(0)
            if "device_used" not in feature_dict:
                # Add device info before closing brace
                feature_dict = feature_dict.rstrip('}') + ',\n                "device_used": self.device,\n                "gpu_accelerated": self.device == "cuda"\n            }'
            return feature_dict
        
        content = re.sub(feature_dict_pattern, add_device_info, content, flags=re.MULTILINE | re.DOTALL)
        return content
    
    def fix_all_extractors(self) -> Dict[str, Any]:
        """Fix all extractors that need fallback mechanisms."""
        results = {
            "fixed": [],
            "skipped": [],
            "failed": []
        }
        
        for extractor_name in self.extractors_to_fix:
            extractor_path = self.extractors_dir / extractor_name
            if not extractor_path.exists():
                results["skipped"].append(extractor_name)
                continue
            
            if self.fix_extractor(extractor_path):
                results["fixed"].append(extractor_name)
            else:
                results["failed"].append(extractor_name)
        
        return results

def main():
    """Main function to fix all extractors."""
    extractors_dir = "/Users/user/Desktop/MLService/DataProcessor/AudioProcessor/src/extractors"
    fixer = FallbackFixer(extractors_dir)
    
    print("🔧 Fixing extractors with proper fallback mechanisms...")
    results = fixer.fix_all_extractors()
    
    print(f"\n📊 Fix Results:")
    print(f"✅ Fixed: {len(results['fixed'])} extractors")
    print(f"⏭️  Skipped: {len(results['skipped'])} extractors")
    print(f"❌ Failed: {len(results['failed'])} extractors")
    
    if results["fixed"]:
        print(f"\n✅ Fixed extractors: {', '.join(results['fixed'])}")
    
    if results["failed"]:
        print(f"\n❌ Failed extractors: {', '.join(results['failed'])}")

if __name__ == "__main__":
    main()
