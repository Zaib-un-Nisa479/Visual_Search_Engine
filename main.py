#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main application entry point for Ring Visual Search Engine
"""

import sys
import os
import argparse
import cv2
import numpy as np

# Add the current directory to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Add modules to path
modules_path = os.path.join(BASE_DIR, 'modules')
if modules_path not in sys.path:
    sys.path.insert(0, modules_path)

# Now import your modules
try:
    from modules.classifier.ring_classifier import RingClassifier
    from modules.classifier.stone_classifier import StoneClassifier
    from modules.processor.ring_cropper import RingCropper
    from modules.processor.background_remover import BackgroundRemover
    from modules.matcher.stone_matcher import StoneRingMatcher
    from modules.matcher.plain_matcher import PlainRingMatcher
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nChecking module paths:")
    for path in sys.path:
        print(f"  {path}")
    sys.exit(1)

# Configuration
DATA_DIR = os.path.join(BASE_DIR, 'data')
CATALOG_DIR = os.path.join(DATA_DIR, 'catalog')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
UPLOAD_DIR = os.path.join(DATA_DIR, 'uploads')

# Create directories if they don't exist
for dir_path in [DATA_DIR, PROCESSED_DIR, FEATURES_DIR, UPLOAD_DIR]:
    os.makedirs(dir_path, exist_ok=True)

class RingSearchEngine:
    """Main class coordinating all modules."""
    
    def __init__(self):
        """Initialize all modules."""
        print("🔧 Initializing Ring Search Engine...")
        
        # Initialize classifiers with default thresholds
        self.ring_classifier = RingClassifier()
        self.stone_classifier = StoneClassifier()
        self.ring_cropper = RingCropper()
        self.bg_remover = BackgroundRemover()
        self.stone_matcher = StoneRingMatcher()
        self.plain_matcher = PlainRingMatcher()
        
        # Load databases
        self.stone_database = {}
        self.plain_database = {}
        self._load_databases()
        
        print("✅ All modules initialized successfully")
    
    def _load_databases(self):
        """Load pre-computed feature databases."""
        stone_db_path = os.path.join(FEATURES_DIR, 'stone_database.npy')
        plain_db_path = os.path.join(FEATURES_DIR, 'plain_database.npy')
        
        if os.path.exists(stone_db_path):
            try:
                self.stone_database = np.load(stone_db_path, allow_pickle=True).item()
                print(f"📚 Loaded {len(self.stone_database)} stone rings")
            except Exception as e:
                print(f"⚠ Could not load stone database: {e}")
        
        if os.path.exists(plain_db_path):
            try:
                self.plain_database = np.load(plain_db_path, allow_pickle=True).item()
                print(f"📚 Loaded {len(self.plain_database)} plain rings")
            except Exception as e:
                print(f"⚠ Could not load plain database: {e}")
    
    def build_database(self, catalog_path=CATALOG_DIR):
        """Build feature database from catalog images."""
        print(f"\n🏗️ Building database from: {catalog_path}")
        
        # Get all images
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif')
        image_paths = []
        
        if os.path.exists(catalog_path):
            image_paths = [
                os.path.join(catalog_path, f) for f in os.listdir(catalog_path)
                if f.lower().endswith(image_extensions)
            ]
        
        print(f"Found {len(image_paths)} images in catalog")
        
        if not image_paths:
            print("❌ No images found in catalog folder")
            return None, None
        
        stone_images = []
        plain_images = []
        
        for i, img_path in enumerate(image_paths, 1):
            print(f"\nProcessing {i}/{len(image_paths)}: {os.path.basename(img_path)}")
            
            # Check if image contains a ring
            result = self.ring_classifier.classify(img_path)
            
            if result['has_ring']:
                print(f"  → Ring detected with confidence: {result['confidence']:.2f}")
                
                # Crop and enhance
                cropped = self.ring_cropper.crop_and_enhance(img_path, result.get('ring_region'))
                if cropped['success']:
                    # Remove background
                    bg = self.bg_remover.remove_background(cropped['cropped_image'])
                    if bg['success']:
                        # Check for stone
                        stone = self.stone_classifier.classify(bg['image'])
                        
                        if stone['has_stone']:
                            stone_images.append(img_path)
                            print(f"    → Stone ring detected (type: {stone.get('stone_type', 'unknown')})")
                        else:
                            plain_images.append(img_path)
                            print(f"    → Plain ring detected")
                    else:
                        print(f"    → Background removal failed")
                else:
                    print(f"    → Cropping failed")
            else:
                print(f"  → No ring detected (confidence: {result['confidence']:.2f})")
                if result.get('debug_info'):
                    print(f"    Scores: shape={result['debug_info']['shape_score']:.2f}, "
                          f"edge={result['debug_info']['edge_score']:.2f}, "
                          f"contour={result['debug_info']['contour_score']:.2f}, "
                          f"texture={result['debug_info']['texture_score']:.2f}")
        
        # Build databases
        print(f"\n🏗️ Building stone ring database ({len(stone_images)} images)...")
        stone_db = {}
        for i, img_path in enumerate(stone_images, 1):
            print(f"  Processing stone ring {i}/{len(stone_images)}")
            # Extract features here in a real implementation
            stone_db[img_path] = {
                'path': img_path,
                'features': np.array([])  # Placeholder
            }
        
        print(f"🏗️ Building plain ring database ({len(plain_images)} images)...")
        plain_db = {}
        for i, img_path in enumerate(plain_images, 1):
            print(f"  Processing plain ring {i}/{len(plain_images)}")
            plain_db[img_path] = {
                'path': img_path,
                'features': np.array([])  # Placeholder
            }
        
        # Save databases
        os.makedirs(FEATURES_DIR, exist_ok=True)
        
        stone_db_path = os.path.join(FEATURES_DIR, 'stone_database.npy')
        plain_db_path = os.path.join(FEATURES_DIR, 'plain_database.npy')
        
        try:
            np.save(stone_db_path, stone_db)
            np.save(plain_db_path, plain_db)
            print(f"\n✅ Database built successfully:")
            print(f"   - Stone rings: {len(stone_db)}")
            print(f"   - Plain rings: {len(plain_db)}")
            print(f"   - Saved to: {FEATURES_DIR}")
        except Exception as e:
            print(f"❌ Error saving databases: {e}")
        
        return stone_db, plain_db
    
    def process_query(self, image_path):
        """Process a query image."""
        print(f"\n🔍 Processing query: {image_path}")
        
        if not os.path.exists(image_path):
            return {"success": False, "message": "Image not found"}
        
        # Classify ring
        result = self.ring_classifier.classify(image_path)
        
        if not result['has_ring']:
            return {
                "success": False, 
                "message": f"No ring detected in image (confidence: {result['confidence']:.2f})",
                "confidence": result['confidence'],
                "debug_info": result.get('debug_info', {})
            }
        
        # Crop and enhance
        cropped = self.ring_cropper.crop_and_enhance(image_path, result.get('ring_region'))
        
        if not cropped['success']:
            return {"success": False, "message": "Failed to crop ring"}
        
        # Remove background
        bg = self.bg_remover.remove_background(cropped['cropped_image'])
        
        if not bg['success']:
            return {"success": False, "message": "Failed to remove background"}
        
        # Check for stone
        stone = self.stone_classifier.classify(bg['image'])
        
        # Find matches (simplified for now)
        matches = []
        if stone['has_stone'] and self.stone_database:
            # Use stone matcher
            for path in list(self.stone_database.keys())[:5]:
                matches.append({
                    'path': path,
                    'filename': os.path.basename(path),
                    'similarity': 0.85  # Placeholder - replace with actual similarity
                })
        elif not stone['has_stone'] and self.plain_database:
            # Use plain matcher
            for path in list(self.plain_database.keys())[:5]:
                matches.append({
                    'path': path,
                    'filename': os.path.basename(path),
                    'similarity': 0.85  # Placeholder - replace with actual similarity
                })
        
        return {
            "success": True,
            "message": "Processing complete",
            "has_ring": True,
            "has_stone": stone['has_stone'],
            "stone_type": stone.get('stone_type', 'unknown'),
            "confidence": result['confidence'],
            "matches": matches[:5]
        }


def start_web_interface():
    """Start the web interface with multiple import attempts."""
    print("🌐 Starting web interface...")
    
    # Method 1: Try importing as package (with Web folder as package)
    try:
        # First, make sure Web is treated as a package
        web_init = os.path.join(BASE_DIR, 'Web', '__init__.py')
        if not os.path.exists(web_init):
            print("⚠ Web/__init__.py not found. Creating it...")
            with open(web_init, 'w') as f:
                f.write('"""Web package for Ring Visual Search Engine"""\n')
                f.write('from .app import app\n\n')
                f.write('__all__ = ["app"]\n')
        
        # Add parent directory to path
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)
        
        # Try package import
        from Web import app
        print("✅ Imported from Web package")
        return app
    except ImportError as e:
        print(f"⚠ Package import failed: {e}")
    
    # Method 2: Add Web to path and import directly
    try:
        web_path = os.path.join(BASE_DIR, 'Web')
        if web_path not in sys.path:
            sys.path.insert(0, web_path)
            print(f"✅ Added {web_path} to Python path")
        
        # Direct import (may show Pylance warning but works at runtime)
        from app import app  # type: ignore
        print("✅ Imported directly from app")
        return app
    except ImportError as e:
        print(f"⚠ Direct import failed: {e}")
    
    # Method 3: Use importlib (most reliable)
    try:
        import importlib.util
        web_path = os.path.join(BASE_DIR, 'Web', 'app.py')
        
        if os.path.exists(web_path):
            spec = importlib.util.spec_from_file_location("web_app", web_path)
            web_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(web_module)
            print("✅ Imported using importlib")
            return web_module.app
        else:
            print(f"❌ app.py not found at: {web_path}")
    except Exception as e:
        print(f"❌ Importlib failed: {e}")
    
    return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Ring Visual Search Engine')
    parser.add_argument('--query', type=str, help='Path to query image')
    parser.add_argument('--build-db', action='store_true', help='Build database from catalog')
    parser.add_argument('--catalog', type=str, default=CATALOG_DIR, help='Catalog directory')
    parser.add_argument('--web', action='store_true', help='Start web interface')
    parser.add_argument('--save', type=str, help='Save results to file')
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = RingSearchEngine()
    
    if args.web:
        # Check if Flask is installed
        try:
            import flask
            print(f"✅ Flask {flask.__version__} is installed")
        except ImportError:
            print("❌ Flask is not installed. Installing...")
            os.system('pip install flask')
            try:
                import flask
                print(f"✅ Flask {flask.__version__} installed successfully")
            except:
                print("❌ Failed to install Flask. Please install manually: pip install flask")
                sys.exit(1)
        
        # Start web interface
        app = start_web_interface()
        
        if app:
            print(f"🚀 Server running at: http://localhost:5000")
            print("Press Ctrl+C to stop")
            try:
                app.run(debug=True, host='0.0.0.0', port=5000)
            except KeyboardInterrupt:
                print("\n👋 Server stopped")
            except Exception as e:
                print(f"❌ Error running server: {e}")
        else:
            print("❌ Failed to start web interface")
            print("\nTroubleshooting steps:")
            print("1. Check that Web/app.py exists")
            print("2. Check that Web/__init__.py exists with proper content")
            print("3. Run: pip install flask")
            print("4. Try running: python -c \"from Web import app\"")
            sys.exit(1)
    
    elif args.build_db:
        # Build database
        stone_db, plain_db = engine.build_database(args.catalog)
        if stone_db or plain_db:
            print(f"\n✅ Database build complete!")
        else:
            print(f"\n❌ Database build failed - no rings detected")
            print("\nTips for better ring detection:")
            print("1. Make sure images clearly show the ring")
            print("2. Ensure good lighting and contrast")
            print("3. Try adjusting the confidence threshold in ring_classifier.py")
    
    elif args.query:
        # Process single query
        if not os.path.exists(args.query):
            print(f"❌ Query image not found: {args.query}")
            sys.exit(1)
        
        results = engine.process_query(args.query)
        
        print(f"\n{'='*50}")
        print(f"RESULTS")
        print(f"{'='*50}")
        print(f"Status: {results['message']}")
        
        if results.get('success'):
            print(f"Has Ring: {results.get('has_ring', False)}")
            print(f"Has Stone: {results.get('has_stone', False)}")
            print(f"Stone Type: {results.get('stone_type', 'N/A')}")
            print(f"Confidence: {results.get('confidence', 0):.2%}")
            
            if results.get('matches'):
                print(f"\nTop {len(results['matches'])} matches:")
                for i, match in enumerate(results['matches'], 1):
                    print(f"  {i}. {match['filename']} - Similarity: {match['similarity']:.2%}")
            else:
                print("\nNo matches found in database")
        else:
            if results.get('debug_info'):
                print(f"\nDebug Scores:")
                for key, value in results['debug_info'].items():
                    print(f"  {key}: {value:.3f}")
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python main.py --web                    # Start web interface")
        print("  python main.py --build-db               # Build database from catalog")
        print("  python main.py --query ring.jpg         # Search with query image")
        print("  python main.py --query ring.jpg --save  # Save results")


if __name__ == '__main__':
    # Print banner
    print("""
    ╔═══════════════════════════════════════════╗
    ║     Ring Visual Search Engine v1.0        ║
    ║         Find similar rings instantly       ║
    ╚═══════════════════════════════════════════╝
    """)
    
    main()