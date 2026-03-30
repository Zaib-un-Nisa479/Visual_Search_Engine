"""
Module 6: Plain Ring Matcher
Matches plain rings (without stones) with orientation invariance.
Can be operated independently.
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlainRingMatcher:
    """
    Matches plain rings without stones.
    Uses shape, pattern, and texture analysis with orientation invariance.
    """
    
    def __init__(self, feature_extractor=None):
        """
        Initialize the plain ring matcher.
        
        Args:
            feature_extractor: Optional pre-trained feature extractor
        """
        self.feature_extractor = feature_extractor
        
    def match(self, query_image: np.ndarray, database: Dict, top_n: int = 5) -> List[Dict]:
        """
        Find similar plain rings with orientation invariance.
        
        Args:
            query_image: Query ring image (with white background)
            database: Dictionary of database images and their features
            top_n: Number of top matches to return
            
        Returns:
            List of top matches with scores
        """
        if query_image is None:
            return []
        
        # Step 1: Extract rotation-invariant features
        query_features = self._extract_comprehensive_features(query_image)
        
        # Step 2: Compare with database
        matches = []
        for db_path, db_data in database.items():
            if 'plain_features' in db_data:
                db_features = db_data['plain_features']
            else:
                # Compute features if not pre-computed
                db_image = cv2.imread(db_path)
                db_features = self._extract_comprehensive_features(db_image)
            
            # Calculate similarity with orientation invariance
            similarity = self._calculate_orientation_invariant_similarity(
                query_features, db_features
            )
            
            matches.append({
                'path': db_path,
                'similarity': similarity,
                'metadata': db_data.get('metadata', {})
            })
        
        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return matches[:top_n]
    
    def _extract_comprehensive_features(self, image: np.ndarray) -> Dict:
        """Extract comprehensive features from plain ring."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Shape features
        shape_features = self._extract_shape_features(gray)
        
        # 2. Pattern features (engravings, textures)
        pattern_features = self._extract_pattern_features(gray)
        
        # 3. Profile features (ring cross-section)
        profile_features = self._extract_profile_features(gray)
        
        # 4. Texture features
        texture_features = self._extract_texture_features(gray)
        
        return {
            'shape': shape_features,
            'pattern': pattern_features,
            'profile': profile_features,
            'texture': texture_features
        }
    
    def _extract_shape_features(self, gray: np.ndarray) -> np.ndarray:
        """Extract shape features from the ring."""
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = []
        if contours:
            # Get main contour
            main_contour = max(contours, key=cv2.contourArea)
            
            # Hu moments (rotation invariant)
            moments = cv2.moments(main_contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            # Log transform of Hu moments
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-6)
            features.extend(hu_moments)
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(main_contour)
            aspect_ratio = w / h
            features.append(aspect_ratio)
            
            # Circularity
            area = cv2.contourArea(main_contour)
            perimeter = cv2.arcLength(main_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                features.append(circularity)
        
        return np.array(features)
    
    def _extract_pattern_features(self, gray: np.ndarray) -> np.ndarray:
        """Extract pattern features (engravings, designs)."""
        # Apply Gabor filters at different orientations
        gabor_features = []
        
        # Define Gabor filter parameters
        ksize = 31
        sigma = 4.0
        theta = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Different orientations
        lambd = 10.0
        gamma = 0.5
        
        for angle in theta:
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, angle, lambd, gamma, 0)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            
            # Extract statistics
            gabor_features.extend([
                np.mean(filtered),
                np.std(filtered),
                np.max(filtered)
            ])
        
        # Local Binary Pattern for texture
        from skimage.feature import local_binary_pattern
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-6)
        
        features = np.concatenate([gabor_features, lbp_hist])
        return features
    
    def _extract_profile_features(self, gray: np.ndarray) -> np.ndarray:
        """Extract profile features (ring cross-section patterns)."""
        height, width = gray.shape
        center = (width // 2, height // 2)
        max_radius = min(center[0], center[1], width - center[0], height - center[1])
        
        # Polar transform
        polar = cv2.linearPolar(
            gray, 
            center, 
            max_radius, 
            cv2.WARP_FILL_OUTLIERS
        )
        
        # Extract radial profile (rotation invariant)
        radial_profile = np.mean(polar, axis=1)
        
        # Take first 100 points
        if len(radial_profile) > 100:
            radial_profile = radial_profile[:100]
        
        return radial_profile
    
    def _extract_texture_features(self, gray: np.ndarray) -> np.ndarray:
        """Extract texture features."""
        # GLCM (Gray-Level Co-occurrence Matrix) features
        from skimage.feature import graycomatrix, graycoprops
        
        # Quantize image
        quantized = (gray // 32).astype(np.uint8)
        
        # Compute GLCM
        glcm = graycomatrix(quantized, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=8)
        
        # Extract properties
        contrast = graycoprops(glcm, 'contrast').flatten()
        dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()
        energy = graycoprops(glcm, 'energy').flatten()
        correlation = graycoprops(glcm, 'correlation').flatten()
        
        features = np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation])
        return features
    
    def _calculate_orientation_invariant_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity with orientation invariance."""
        # Shape similarity (Hu moments are already rotation invariant)
        shape_sim = cosine_similarity([features1['shape']], [features2['shape']])[0][0]
        
        # Pattern similarity - test multiple orientations
        pattern_sim = self._calculate_rotational_similarity(
            features1['pattern'], features2['pattern']
        )
        
        # Profile similarity - shift invariance
        profile_sim = self._calculate_shift_invariant_similarity(
            features1['profile'], features2['profile']
        )
        
        # Texture similarity
        texture_sim = cosine_similarity([features1['texture']], [features2['texture']])[0][0]
        
        # Weighted combination
        total_sim = (
            shape_sim * 0.30 +
            pattern_sim * 0.30 +
            profile_sim * 0.25 +
            texture_sim * 0.15
        )
        
        return float(total_sim)
    
    def _calculate_rotational_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Calculate similarity considering rotational shifts."""
        # Reshape if 1D
        if len(feat1.shape) == 1:
            feat1 = feat1.reshape(1, -1)
        if len(feat2.shape) == 1:
            feat2 = feat2.reshape(1, -1)
        
        # Try different alignments
        best_sim = cosine_similarity(feat1, feat2)[0][0]
        
        # For longer feature vectors, try circular shifts
        if len(feat2) > 36:  # Only if we have enough elements
            for shift in range(1, min(36, len(feat2))):
                shifted = np.roll(feat2, shift)
                sim = cosine_similarity(feat1, [shifted])[0][0]
                best_sim = max(best_sim, sim)
        
        return best_sim
    
    def _calculate_shift_invariant_similarity(self, profile1: np.ndarray, profile2: np.ndarray) -> float:
        """Calculate similarity between profiles considering shifts."""
        # Ensure same length
        min_len = min(len(profile1), len(profile2))
        p1 = profile1[:min_len]
        p2 = profile2[:min_len]
        
        # Try different alignments
        best_corr = 0
        for shift in range(-min_len//4, min_len//4):
            shifted = np.roll(p2, shift)
            correlation = np.corrcoef(p1, shifted)[0, 1]
            best_corr = max(best_corr, correlation if not np.isnan(correlation) else 0)
        
        return best_corr
    
    def build_database(self, image_paths: List[str]) -> Dict:
        """Build feature database for plain rings."""
        database = {}
        
        for path in image_paths:
            image = cv2.imread(path)
            if image is None:
                continue
            
            # Extract features
            features = self._extract_comprehensive_features(image)
            
            # Store in database
            database[path] = {
                'plain_features': features,
                'path': path
            }
            
            logger.info(f"Added {os.path.basename(path)} to database")
        
        return database


# Independent execution example
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        query_path = sys.argv[1]
        db_paths = sys.argv[2:]
        
        # Load query
        query_image = cv2.imread(query_path)
        
        if query_image is not None:
            # Create matcher
            matcher = PlainRingMatcher()
            
            # Build database
            database = matcher.build_database(db_paths)
            
            # Find matches
            matches = matcher.match(query_image, database, top_n=5)
            
            print(f"\nTop {len(matches)} Matches:")
            for i, match in enumerate(matches):
                print(f"{i+1}. {os.path.basename(match['path'])} - Similarity: {match['similarity']:.4f}")
        else:
            print("Failed to load query image")
    else:
        print("Usage: python plain_matcher.py <query_image> <db_image1> <db_image2> ...")