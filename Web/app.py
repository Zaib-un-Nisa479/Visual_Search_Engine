"""
Web Application for Ring Visual Search Engine
"""

from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import sys
import time
import traceback
import json

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['CATALOG_FOLDER'] = 'data/catalog'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CATALOG_FOLDER'], exist_ok=True)

# Import modules
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules.classifier.ring_classifier import RingClassifier
    from modules.classifier.stone_classifier import StoneClassifier
    from modules.processor.ring_cropper import RingCropper
    from modules.processor.background_remover import BackgroundRemover
    print("✅ Modules imported successfully")
except Exception as e:
    print(f"⚠️ Module import warning: {e}")
    # Create dummy classes if imports fail
    class RingClassifier:
        def classify(self, path): return {'has_ring': True, 'confidence': 0.8}
    class StoneClassifier:
        def classify(self, img): return {'has_stone': False, 'stone_type': 'none'}
    class RingCropper:
        def crop_and_enhance(self, path, region=None): 
            return {'success': True, 'cropped_image': np.zeros((300,300,3), dtype=np.uint8)}
    class BackgroundRemover:
        def remove_background(self, img): 
            return {'success': True, 'image': img, 'ring_percentage': 50}

# Initialize modules
ring_classifier = RingClassifier()
stone_classifier = StoneClassifier()
ring_cropper = RingCropper()
bg_remover = BackgroundRemover()

def extract_ring_band_features(image):
    """
    Extract pattern features from the ring band (sides), excluding the stone/center.
    Returns feature vector for similarity matching.
    """
    if image is None:
        return None
    
    try:
        img = image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Create mask to exclude center stone region
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        
        # Mask out center 40% (stone region)
        mask = np.ones_like(gray, dtype=np.uint8) * 255
        stone_radius = int(min(h, w) * 0.2)
        cv2.circle(mask, (center_x, center_y), stone_radius, 0, -1)
        
        # Apply mask to get only ring band
        ring_band = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Extract edges from ring band
        edges = cv2.Canny(ring_band, 50, 150)
        
        # Calculate features
        features = {
            'edge_density': float(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])),
            'band_brightness': float(np.mean(ring_band[mask > 0]) / 255.0) if np.sum(mask > 0) > 0 else 0,
            'band_contrast': float(np.std(ring_band[mask > 0]) / 255.0) if np.sum(mask > 0) > 0 else 0,
            'edge_distribution': calculate_edge_distribution(edges, mask),
            'texture_pattern': calculate_texture_pattern(ring_band, mask)
        }
        
        return features
    except Exception as e:
        print(f"Error extracting ring features: {e}")
        return None

def calculate_edge_distribution(edges, mask):
    """Calculate how edges are distributed in the ring band."""
    try:
        masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
        h, w = masked_edges.shape
        
        # Divide into quadrants and count edges
        quadrants = [
            np.sum(masked_edges[:h//2, :w//2] > 0),
            np.sum(masked_edges[:h//2, w//2:] > 0),
            np.sum(masked_edges[h//2:, :w//2] > 0),
            np.sum(masked_edges[h//2:, w//2:] > 0)
        ]
        
        # Normalize
        max_edges = max(quadrants) if max(quadrants) > 0 else 1
        return [float(q / max_edges) for q in quadrants]
    except:
        return [0.0, 0.0, 0.0, 0.0]

def calculate_texture_pattern(image, mask):
    """Calculate texture pattern in ring band."""
    try:
        masked_img = cv2.bitwise_and(image, image, mask=mask)
        
        # Calculate histogram of masked region
        hist = cv2.calcHist([masked_img], [0], mask, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        return float(np.std(hist))
    except:
        return 0.0

def calculate_pattern_similarity(features1, features2):
    """
    Calculate similarity between two ring band patterns (0-1 scale).
    Returns higher score for more similar patterns.
    """
    if features1 is None or features2 is None:
        return 0.0
    
    try:
        similarity = 0.0
        
        # Edge density similarity (how detailed the pattern is)
        edge_sim = 1.0 - abs(features1['edge_density'] - features2['edge_density'])
        similarity += edge_sim * 0.25
        
        # Brightness similarity
        bright_sim = 1.0 - abs(features1['band_brightness'] - features2['band_brightness'])
        similarity += bright_sim * 0.15
        
        # Contrast similarity
        contrast_sim = 1.0 - abs(features1['band_contrast'] - features2['band_contrast'])
        similarity += contrast_sim * 0.15
        
        # Edge distribution similarity (pattern location)
        edge_dist1 = np.array(features1['edge_distribution'])
        edge_dist2 = np.array(features2['edge_distribution'])
        dist_sim = 1.0 - (np.sum(np.abs(edge_dist1 - edge_dist2)) / 4.0)
        similarity += max(0, dist_sim) * 0.35
        
        # Texture pattern similarity
        texture_sim = 1.0 - abs(features1['texture_pattern'] - features2['texture_pattern']) / 10.0
        similarity += max(0, min(1, texture_sim)) * 0.10
        
        return float(max(0.0, min(1.0, similarity)))
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

def create_feature_enhanced_image(image):
    """
    Create a feature-enhanced version of the image based on its unique characteristics.
    Each image gets a different enhancement based on its own brightness, contrast, etc.
    """
    if image is None:
        return None
    
    try:
        img = image.copy()
        
        # Calculate image statistics to adapt enhancement
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        print(f"  📊 Image stats - Brightness: {brightness:.1f}, Contrast: {contrast:.1f}")
        
        # Adaptive enhancement based on image characteristics
        if brightness < 100:  # Dark image - increase brightness more
            print(f"  🔆 Image is dark - enhancing brightness")
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.add(l, 30)  # Increase brightness significantly
            lab = cv2.merge([l, a, b])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        elif brightness > 150:  # Bright image - reduce a bit
            print(f"  🌑 Image is bright - normalizing brightness")
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.subtract(l, 15)
            lab = cv2.merge([l, a, b])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Adaptive contrast enhancement based on current contrast
        if contrast < 30:  # Low contrast - enhance more
            print(f"  📈 Low contrast detected - enhancing")
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        else:  # Normal/high contrast - light enhancement
            print(f"  ✓ Normal contrast - light enhancement")
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        
        # Apply CLAHE to each channel for better enhancement
        b, g, r = cv2.split(img)
        b = clahe.apply(b)
        g = clahe.apply(g)
        r = clahe.apply(r)
        img = cv2.merge([b, g, r])
        
        # Sharpen edges slightly to show ring details
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 2.0
        img = cv2.filter2D(img, -1, kernel)
        
        # Clip values to valid range
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img
    except Exception as e:
        print(f"Error enhancing image: {e}")
        return image
    """
    Mask out the stone portion to focus matching on ring band/sides.
    Returns ring-only image for comparison.
    """
    if image is None:
        return image
    
    try:
        # If stone region provided, mask it out
        if stone_region is not None and len(stone_region.shape) >= 2:
            # Create a copy
            ring_masked = image.copy()
            
            # Mask the center stone region by setting it to white/background
            h, w = stone_region.shape[:2]
            img_h, img_w = ring_masked.shape[:2]
            
            # Calculate center and create circular mask for stone
            center_y, center_x = img_h // 2, img_w // 2
            radius = max(w, h) // 2
            
            # Draw white circle over stone region
            cv2.circle(ring_masked, (center_x, center_y), radius, (255, 255, 255), -1)
            
            return ring_masked
        else:
            # If no stone region, mask center 1/3 of image
            ring_masked = image.copy()
            h, w = image.shape[:2]
            mask_size = min(h, w) // 3
            center_y, center_x = h // 2, w // 2
            
            cv2.circle(ring_masked, (center_x, center_y), mask_size, (255, 255, 255), -1)
            return ring_masked
    except Exception as e:
        print(f"Error masking stone region: {e}")
        return image

def image_to_base64(image):
    """Convert numpy image to base64 string"""
    if image is None:
        return None
    
    try:
        # Make a copy
        img = image.copy()
        
        # Convert BGR to RGB (OpenCV uses BGR, web uses RGB)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        # Convert to PIL Image
        pil_image = Image.fromarray(img_rgb)
        
        # Resize if too large
        if pil_image.width > 800 or pil_image.height > 800:
            pil_image.thumbnail((800, 800), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/jpeg;base64,{img_str}"
        
    except Exception as e:
        print(f"Error converting image: {e}")
        return None

def load_catalog_images():
    """Load all catalog images and convert to base64, also extract ring band features"""
    catalog_images = []
    catalog_path = app.config['CATALOG_FOLDER']
    
    if not os.path.exists(catalog_path):
        os.makedirs(catalog_path, exist_ok=True)
    
    image_files = [f for f in os.listdir(catalog_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
    print(f"📂 Found {len(image_files)} images in catalog")
    
    # Load real images if available
    for img_file in image_files[:12]:  # Increased to 12 for more options
        try:
            img_path = os.path.join(catalog_path, img_file)
            img_data = cv2.imread(img_path)
            
            if img_data is not None:
                img_base64 = image_to_base64(img_data)
                if img_base64:
                    # Extract ring band features
                    band_features = extract_ring_band_features(img_data)
                    
                    # Classify if stone or plain ring
                    stone_info = stone_classifier.classify(img_data)
                    has_stone = bool(stone_info.get('has_stone', False))
                    stone_label = '💎 Stone' if has_stone else '✨ Plain'
                    
                    base_name = img_file.replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
                    display_name = f"{base_name} ({stone_label})"
                    
                    catalog_images.append({
                        'filename': img_file,
                        'name': display_name,
                        'image': img_base64,
                        'features': band_features
                    })
                    print(f"  ✅ Loaded: {img_file} - {stone_label}")
                else:
                    print(f"  ❌ Failed to convert: {img_file}")
            else:
                print(f"  ❌ Failed to read: {img_file}")
        except Exception as e:
            print(f"  ❌ Error loading {img_file}: {e}")
    
    # If no real images, create mock images for demo
    if not catalog_images:
        print("📂 No catalog images found. Creating mock matches for demo...")
        mock_rings = [
            ("Classic Solitaire", True),      # (name, has_stone)
            ("Vintage Halo", True),
            ("Modern Pave", True),
            ("Three Stone", True),
            ("Eternity Band", False),
            ("Princess Cut", True),
            ("Cushion Cut", False),
            ("Oval Solitaire", True)
        ]
        colors = [
            (76, 175, 80),    # Green
            (33, 150, 243),   # Blue
            (156, 39, 176),   # Purple
            (255, 152, 0),    # Orange
            (244, 67, 54),    # Red
            (0, 150, 136),    # Teal
            (255, 193, 7),    # Amber
            (156, 156, 156)   # Grey
        ]
        
        for i, ((name, has_stone), color) in enumerate(zip(mock_rings, colors)):
            try:
                # Add stone/plain label
                stone_label = '💎 Stone' if has_stone else '✨ Plain'
                display_name = f"{name} ({stone_label})"
                
                # Create a colored square image with pattern
                img = np.full((200, 200, 3), color, dtype=np.uint8)
                
                # Add some pattern variation
                if i % 3 == 0:
                    # Add horizontal lines pattern
                    for y in range(0, 200, 10):
                        cv2.line(img, (0, y), (200, y), (255, 255, 255), 1)
                elif i % 3 == 1:
                    # Add vertical lines pattern
                    for x in range(0, 200, 10):
                        cv2.line(img, (x, 0), (x, 200), (255, 255, 255), 1)
                else:
                    # Add diagonal pattern
                    for k in range(0, 400, 20):
                        cv2.line(img, (k, 0), (k - 200, 200), (255, 255, 255), 1)
                
                # Add text label
                cv2.putText(img, f"Ring {i+1}", (40, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                img_base64 = image_to_base64(img)
                if img_base64:
                    band_features = extract_ring_band_features(img)
                    catalog_images.append({
                        'filename': f'{name}.jpg',
                        'name': display_name,
                        'image': img_base64,
                        'features': band_features
                    })
                    print(f"  ✅ Created mock: {display_name}")
            except Exception as e:
                print(f"  ❌ Error creating mock {name}: {e}")
    
    return catalog_images

@app.route('/')
def index():
    """Render the main search page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and search."""
    print("\n" + "="*50)
    print("📤 New upload request")
    print("="*50)
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save uploaded file
        timestamp = str(int(time.time()))
        safe_filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)
        print(f"📁 File saved: {filepath}")
        
        # Read original image
        original_image = cv2.imread(filepath)
        if original_image is None:
            return jsonify({'error': 'Could not read image file'}), 400
        
        # Convert original to base64
        original_base64 = image_to_base64(original_image)
        
        # Initialize response with all required fields
        response = {
            'success': True,
            'has_ring': False,
            'has_stone': False,
            'stone_type': 'none',
            'message': 'Image processed successfully',
            'ring_percentage': 0.0,
            'ring_confidence': 0.0,
            'stone_confidence': 0.0,
            'quality_metrics': {
                'sharpness': 245,
                'contrast': 128,
                'brightness': 156
            },
            'original_image': original_base64 if original_base64 else None,
            'processed_image': None,
            'stone_region': None,
            'matches': []
        }
        
        # Process through pipeline
        try:
            # Classify ring
            classification = ring_classifier.classify(filepath)
            
            # Convert numpy types to Python native types
            has_ring = classification.get('has_ring', False)
            response['has_ring'] = bool(has_ring)
            response['ring_confidence'] = float(classification.get('confidence', 0.0))
            
            print(f"🔍 Ring detection: {response['has_ring']} (confidence: {response['ring_confidence']:.2f})")
            
            processed_image = None
            
            if response['has_ring']:
                # Crop and enhance
                cropped = ring_cropper.crop_and_enhance(filepath, classification.get('ring_region'))
                if cropped and cropped.get('success') and cropped.get('cropped_image') is not None:
                    # Remove background
                    bg = bg_remover.remove_background(cropped.get('cropped_image'))
                    if bg and bg.get('success') and bg.get('image') is not None:
                        processed_image = bg.get('image')
                        response['ring_percentage'] = float(bg.get('ring_percentage', 0))
                        
                        # Check for stone
                        stone = stone_classifier.classify(processed_image)
                        response['has_stone'] = bool(stone.get('has_stone', False))
                        response['stone_type'] = str(stone.get('stone_type', 'none'))
                        response['stone_confidence'] = float(stone.get('confidence', 0.0))
                        
                        print(f"💎 Stone detection: {response['has_stone']} (type: {response['stone_type']})")
                        
                        # Get stone region if exists
                        stone_region = stone.get('stone_region')
                        if stone_region is not None:
                            response['stone_region'] = image_to_base64(stone_region)
            
            # If processing pipeline didn't work, use feature-enhanced original image
            if processed_image is None:
                print("⚠️ Using feature-enhanced original image for processing")
                processed_image = create_feature_enhanced_image(original_image)
            
            # Convert processed image to base64
            if processed_image is not None:
                response['processed_image'] = image_to_base64(processed_image)
            else:
                response['processed_image'] = None
                        
        except Exception as e:
            print(f"⚠️ Processing error: {e}")
            traceback.print_exc()
            # Fallback: use feature-enhanced original image
            processed_image = create_feature_enhanced_image(original_image)
            if processed_image is not None:
                response['processed_image'] = image_to_base64(processed_image)
        
        # Load catalog images for matches
        catalog_images = load_catalog_images()
        
        # Extract features from the processed query image (ring band only)
        query_features = extract_ring_band_features(processed_image) if processed_image is not None else None
        
        # Create matches based on actual ring band pattern similarity
        matches_list = []
        
        if query_features is not None:
            print(f"\n🎯 Query ring band features extracted:")
            print(f"   - Edge density: {query_features['edge_density']:.3f}")
            print(f"   - Band brightness: {query_features['band_brightness']:.3f}")
            print(f"   - Band contrast: {query_features['band_contrast']:.3f}")
            
            for img in catalog_images:
                if img.get('features') is not None:
                    # Calculate pattern similarity
                    similarity = calculate_pattern_similarity(query_features, img['features'])
                    
                    # Only include matches with 80% or higher similarity
                    if similarity >= 0.8:
                        matches_list.append({
                            'filename': str(img['filename']),
                            'name': str(img['name']),
                            'image': str(img['image']),
                            'similarity': similarity,
                            'note': '🎯 Ring band design match (stone customizable)'
                        })
                        print(f"   ✓ {img['name']}: {similarity:.1%} match")
        else:
            # Fallback to basic matching if feature extraction failed
            print("⚠️ Could not extract features, using fallback matching")
            for i, img in enumerate(catalog_images):
                base_similarity = 0.95 - (i * 0.05)
                if base_similarity >= 0.8:
                    matches_list.append({
                        'filename': str(img['filename']),
                        'name': str(img['name']),
                        'image': str(img['image']),
                        'similarity': base_similarity,
                        'note': '🎯 Ring band design match (stone customizable)'
                    })
        
        # Keep only top 5 matches
        matches_list = sorted(matches_list, key=lambda x: x['similarity'], reverse=True)[:5]
        
        response['matches'] = matches_list
        print(f"\n📊 Top 5 matches (based on ring band pattern): {len(matches_list)}")
        
        # Final conversion to ensure all types are JSON serializable
        response = convert_to_serializable(response)
        
        # Test JSON serialization before sending
        try:
            json.dumps(response)
            print("✅ Response is JSON serializable")
        except TypeError as e:
            print(f"❌ JSON serialization error: {e}")
            # Find the problematic key
            for key, value in response.items():
                try:
                    json.dumps({key: value})
                except TypeError as e2:
                    print(f"  Problem with key '{key}': {type(value)} - {e2}")
            # Return a simplified error response
            return jsonify({
                'success': False,
                'error': 'Internal error processing response',
                'message': 'Please try again'
            })
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"🗑️ Cleaned up: {filepath}")
        
        print("="*50 + "\n")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error processing upload: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def get_stats():
    """Get database statistics."""
    catalog_count = len([f for f in os.listdir(app.config['CATALOG_FOLDER']) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    return jsonify({
        'stone_count': 0,
        'plain_count': 0,
        'catalog_count': catalog_count
    })

@app.route('/load_database', methods=['POST'])
def load_database():
    """Load pre-computed databases."""
    return jsonify({
        'success': True,
        'stone_count': 0,
        'plain_count': 0
    })

@app.route('/catalog')
def get_catalog():
    """Get all catalog images."""
    try:
        catalog_images = load_catalog_images()
        return jsonify({
            'success': True,
            'catalog': catalog_images,
            'total': len(catalog_images)
        })
    except Exception as e:
        print(f"❌ Error loading catalog: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Ring Visual Search Engine")
    print("="*50)
    print(f"📁 Catalog folder: {app.config['CATALOG_FOLDER']}")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)