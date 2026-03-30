"""
Module 5: Stone Ring Matcher
Matches rings with mid stones based on pattern and orientation.
Can be operated independently.
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StoneRingMatcher:
    """
    Matches rings with mid stones based on pattern analysis.
    Handles orientation invariance and focuses on side patterns.
    """
    
    def __init__(self, feature_extractor=None):
        """
        Initialize the stone ring matcher.
        
        Args:
            feature_extractor: Optional pre-trained feature extractor
        """
        self.feature_extractor = feature_extractor
        
    def match(self, query_image: np.ndarray, database: Dict, top_n: int = 5) -> List[Dict]:
        """
        Find similar stone rings ignoring stone color but matching side patterns.
        
        Args:
            query_image: Query ring image (with white background)
            database: Dictionary of database images and their features
            top_n: Number of top matches to return
            
        Returns:
            List of top matches with scores
        """
        if query_image is None:
            return []
        
        # Step 1: Mask out the stone
        query_without_stone = self._mask_out_stone(query_image)
        
        # Step 2: Extract rotation-invariant features
        query_features = self._extract_rotation_invariant_features(query_without_stone)
        
        # Step 3: Compare with database
        matches = []
        for db_path, db_data in database.items():
            # Get database features
            if 'stone_features' in db_data:
                db_features = db_data['stone_features']
            else:
                # Compute features if not pre-computed
                db_image = cv2.imread(db_path)
                db_without_stone = self._mask_out_stone(db_image)
                db_features = self._extract_rotation_invariant_features(db_without_stone)
            
            # Calculate similarity
            similarity = self._calculate_pattern_similarity(query_features, db_features)
            matches.append({
                'path': db_path,
                'similarity': similarity,
                'metadata': db_data.get('metadata', {})
            })
        
        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top N
        return matches[:top_n]
    
    def _mask_out_stone(self, image: np.ndarray) -> np.ndarray:
        """Mask out the stone region to focus on side patterns."""
        result = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Detect stone region (center area)
        center_x, center_y = width // 2, height // 2
        stone_radius = min(width, height) // 4
        
        # Create mask for stone
        stone_mask = np.zeros_like(gray)
        cv2.circle(stone_mask, (center_x, center_y), stone_radius, 255, -1)
        
        # Invert mask to keep everything except stone
        ring_mask = cv2.bitwise_not(stone_mask)
        
        # Apply mask
        mask_3channel = cv2.cvtColor(ring_mask, cv2.COLOR_GRAY2BGR) / 255.0
        result = (result * mask_3channel).astype(np.uint8)
        
        return result
    
    def _extract_rotation_invariant_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features that are invariant to rotation."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Circular Fourier Transform
        # Convert to polar coordinates for rotation invariance
        height, width = gray.shape
        center = (width // 2, height // 2)
        max_radius = min(center[0], center[1], width - center[0], height - center[1])
        
        # Polar transform
        polar_image = cv2.linearPolar(
            gray, 
            center, 
            max_radius, 
            cv2.WARP_FILL_OUTLIERS
        )
        
        # Extract features from polar image
        # 1. Radial profile (invariant to rotation)
        radial_profile = np.mean(polar_image, axis=1)
        
        # 2. Angular profile (will be shifted by rotation)
        angular_profile = np.mean(polar_image, axis=0)
        
        # 3. Use FFT of angular profile to achieve rotation invariance
        angular_fft = np.abs(np.fft.fft(angular_profile))
        
        # 4. Edge orientation histogram
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        orientation_hist = np.zeros(36)  # 36 bins for 10-degree intervals
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                bin_idx = int((theta * 180 / np.pi) / 10) % 36
                orientation_hist[bin_idx] += 1
        
        # Combine features
        features = np.concatenate([
            radial_profile[:100],  # Take first 100 radial values
            angular_fft[:50],      # Take first 50 FFT coefficients
            orientation_hist
        ])
        
        # Normalize
        features = features / (np.linalg.norm(features) + 1e-6)
        
        return features
    
    def _calculate_pattern_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between two feature vectors."""
        # Cosine similarity
        similarity = cosine_similarity([features1], [features2])[0][0]
        
        # Additional pattern matching
        # Check for complementary patterns (for orientation invariance)
        # This handles cases where ring is flipped/rotated
        
        # Try matching with slight shifts in angular features
        # (simulating rotation)
        best_similarity = similarity
        
        # Get angular part of features (last 36 elements)
        if len(features1) > 36 and len(features2) > 36:
            ang1 = features1[-36:]
            ang2 = features2[-36:]
            
            # Try circular shifts
            for shift in range(1, 36):
                shifted = np.roll(ang2, shift)
                shifted_features = np.concatenate([features2[:-36], shifted])
                shifted_sim = cosine_similarity([features1], [shifted_features])[0][0]
                best_similarity = max(best_similarity, shifted_sim)
        
        return float(best_similarity)
    
    def build_database(self, image_paths: List[str]) -> Dict:
        """Build feature database for stone rings."""
        database = {}
        
        for path in image_paths:
            image = cv2.imread(path)
            if image is None:
                continue
            
            # Mask out stone
            without_stone = self._mask_out_stone(image)
            
            # Extract features
            features = self._extract_rotation_invariant_features(without_stone)
            
            # Store in database
            database[path] = {
                'features': features,
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
            matcher = StoneRingMatcher()
            
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
        print("Usage: python stone_matcher.py <query_image> <db_image1> <db_image2> ...")