"""
Utility script to add GPU fallback mechanisms to all extractors.
This script automatically updates extractors to include GPU detection and CPU fallback.
"""

import os
import re
from typing import List, Dict, Any
from pathlib import Path

class FallbackUpdater:
    """Utility class to add fallback mechanisms to extractors."""
    
    def __init__(self, extractors_dir: str):
        """
        Initialize the fallback updater.
        
        Args:
            extractors_dir: Path to the extractors directory
        """
        self.extractors_dir = Path(extractors_dir)
        self.extractors_with_fallback = {
            "asr_extractor.py",
            "advanced_embeddings_extractor.py", 
            "speaker_diarization_extractor.py",
            "clap_extractor.py",
            "advanced_spectral_extractor.py",
            "mfcc_extractor.py",
            "chroma_extractor.py",
            "mel_extractor.py",
            "loudness_extractor.py"
        }
    
    def get_all_extractors(self) -> List[Path]:
        """Get all extractor files."""
        return list(self.extractors_dir.glob("*.py"))
    
    def needs_fallback_update(self, extractor_path: Path) -> bool:
        """Check if extractor needs fallback update."""
        if extractor_path.name in self.extractors_with_fallback:
            return False
        
        # Check if file already has GPU detection
        try:
            with open(extractor_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return "torch.cuda.is_available()" not in content
        except Exception:
            return True
    
    def add_gpu_detection_imports(self, content: str) -> str:
        """Add GPU detection imports to content."""
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
                # Add at the beginning after docstring
                docstring_end = content.find('"""', content.find('"""') + 3) + 3
                content = content[:docstring_end] + '\n\nimport torch' + content[docstring_end:]
        
        return content
    
    def add_device_detection(self, content: str, class_name: str) -> str:
        """Add device detection to __init__ method."""
        # Pattern to find __init__ method
        init_pattern = rf'def __init__\(self(?:, [^)]*)?\):'
        init_match = re.search(init_pattern, content)
        
        if not init_match:
            return content
        
        init_start = init_match.start()
        init_end = init_match.end()
        
        # Check if device detection already exists
        if "torch.cuda.is_available()" in content:
            return content
        
        # Add device parameter and detection
        init_line = content[init_start:init_end]
        if "device" not in init_line:
            # Add device parameter
            if "self" in init_line and "(" in init_line:
                init_line = init_line.replace("self", "self, device: str = \"auto\"")
            else:
                init_line = init_line.replace("self)", "self, device: str = \"auto\")")
        
        # Find the super().__init__() call
        super_pattern = r'super\(\)\.__init__\(\)'
        super_match = re.search(super_pattern, content[init_end:])
        
        if super_match:
            super_end = init_end + super_match.end()
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
    
    def update_extractor(self, extractor_path: Path) -> bool:
        """
        Update a single extractor with fallback mechanisms.
        
        Args:
            extractor_path: Path to the extractor file
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            with open(extractor_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Add GPU detection imports
            content = self.add_gpu_detection_imports(content)
            
            # Add device detection
            content = self.add_device_detection(content, extractor_path.stem)
            
            # Add device info to features
            content = self.add_device_info_to_features(content)
            
            # Only write if content changed
            if content != original_content:
                with open(extractor_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ Updated {extractor_path.name}")
                return True
            else:
                print(f"⏭️  No changes needed for {extractor_path.name}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to update {extractor_path.name}: {e}")
            return False
    
    def update_all_extractors(self) -> Dict[str, Any]:
        """
        Update all extractors that need fallback mechanisms.
        
        Returns:
            Dictionary with update results
        """
        results = {
            "updated": [],
            "skipped": [],
            "failed": []
        }
        
        extractors = self.get_all_extractors()
        
        for extractor_path in extractors:
            if extractor_path.name == "__init__.py":
                continue
                
            if not self.needs_fallback_update(extractor_path):
                results["skipped"].append(extractor_path.name)
                continue
            
            if self.update_extractor(extractor_path):
                results["updated"].append(extractor_path.name)
            else:
                results["failed"].append(extractor_path.name)
        
        return results

def main():
    """Main function to update all extractors."""
    extractors_dir = "/Users/user/Desktop/MLService/DataProcessor/AudioProcessor/src/extractors"
    updater = FallbackUpdater(extractors_dir)
    
    print("🔄 Updating extractors with fallback mechanisms...")
    results = updater.update_all_extractors()
    
    print(f"\n📊 Update Results:")
    print(f"✅ Updated: {len(results['updated'])} extractors")
    print(f"⏭️  Skipped: {len(results['skipped'])} extractors")
    print(f"❌ Failed: {len(results['failed'])} extractors")
    
    if results["updated"]:
        print(f"\n✅ Updated extractors: {', '.join(results['updated'])}")
    
    if results["failed"]:
        print(f"\n❌ Failed extractors: {', '.join(results['failed'])}")

if __name__ == "__main__":
    main()
