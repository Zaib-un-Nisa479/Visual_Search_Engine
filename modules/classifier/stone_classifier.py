"""
Module 4: Stone Classifier
Detects whether a ring has a mid stone.
Can be operated independently.
"""

import cv2
import numpy as np
import os
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StoneClassifier:
    """
    Classifies whether a ring has a mid stone.
    Uses color analysis, shape detection, and reflection patterns.
    """
    
    def __init__(self, confidence_threshold: float = 0.65):
        """
        Initialize the stone classifier.
        
        Args:
            confidence_threshold: Minimum confidence to classify as having stone
        """
        self.confidence_threshold = confidence_threshold
        
    def classify(self, image: np.ndarray) -> Dict:
        """
        Main method to classify if ring has a mid stone.
        
        Args:
            image: Input ring image (preferably with white background)
            
        Returns:
            Dictionary with classification results
        """
        if image is None:
            return {
                'has_stone': False,
                'confidence': 0.0,
                'message': 'Invalid image',
                'stone_region': None,
                'stone_type': None
            }
        
        # Run multiple detection methods
        color_score = self._detect_stone_color(image)
        shape_score = self._detect_stone_shape(image)
        reflection_score = self._detect_stone_reflection(image)
        texture_score = self._detect_stone_texture(image)
        
        # Calculate weighted confidence
        confidence = (
            color_score * 0.30 +
            shape_score * 0.35 +
            reflection_score * 0.20 +
            texture_score * 0.15
        )
        
        # Decision
        has_stone = confidence >= self.confidence_threshold
        
        # Extract stone region if present
        stone_region = None
        stone_type = None
        if has_stone:
            stone_region, stone_type = self._extract_stone_region(image)
        
        result = {
            'has_stone': has_stone,
            'confidence': float(confidence),
            'message': f"Stone {'detected' if has_stone else 'not detected'} (confidence: {confidence:.2f})",
            'stone_region': stone_region,
            'stone_type': stone_type,
            'debug_info': {
                'color_score': color_score,
                'shape_score': shape_score,
                'reflection_score': reflection_score,
                'texture_score': texture_score
            }
        }
        
        logger.info(f"Stone classification: {result['message']}")
        return result
    
    def _detect_stone_color(self, image: np.ndarray) -> float:
        """Detect stone by analyzing color patterns."""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for different stone types
        color_ranges = [
            # Diamonds/Clear stones (high value, low saturation)
            ((0, 0, 200), (180, 30, 255)),
            # Red stones (Ruby, Garnet)
            ((0, 50, 50), (10, 255, 255)),
            # Blue stones (Sapphire)
            ((100, 50, 50), (130, 255, 255)),
            # Green stones (Emerald)
            ((40, 50, 50), (80, 255, 255)),
            # Purple stones (Amethyst)
            ((130, 50, 50), (160, 255, 255)),
            # Yellow stones (Topaz, Citrine)
            ((20, 50, 50), (35, 255, 255))
        ]
        
        max_score = 0
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            stone_pixels = np.sum(mask > 0)
            total_pixels = mask.shape[0] * mask.shape[1]
            
            if total_pixels > 0:
                score = stone_pixels / total_pixels
                max_score = max(max_score, score)
        
        return min(1.0, max_score * 3)  # Scale up since stone is small
    
    def _detect_stone_shape(self, image: np.ndarray) -> float:
        """Detect stone by analyzing shape patterns."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 30, 100)
        
        # Look for circular/oval shapes in the center
        height, width = gray.shape
        center_x, center_y = width // 2, height // 2
        
        # Create a mask focusing on center region
        center_mask = np.zeros_like(gray)
        cv2.circle(center_mask, (center_x, center_y), min(width, height) // 4, 255, -1)
        
        # Apply mask to edges
        center_edges = cv2.bitwise_and(edges, center_mask)
        
        # Find contours in center region
        contours, _ = cv2.findContours(center_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stone_contours = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # Too small
                continue
            
            # Check if contour is roughly circular
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.6:  # Circular/oval shape
                stone_contours += 1
        
        score = min(1.0, stone_contours * 0.5)
        return score
    
    def _detect_stone_reflection(self, image: np.ndarray) -> float:
        """Detect stone by analyzing reflection patterns."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Look for bright spots (reflections)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Find local maxima (bright spots)
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(blurred, kernel)
        local_max = (blurred == dilated) & (blurred > 200)
        
        # Count significant bright spots in center region
        height, width = gray.shape
        center_x, center_y = width // 2, height // 2
        
        # Create center mask
        center_mask = np.zeros_like(gray)
        cv2.circle(center_mask, (center_x, center_y), min(width, height) // 3, 255, -1)
        
        # Apply mask to local maxima
        center_max = cv2.bitwise_and(local_max.astype(np.uint8) * 255, center_mask)
        
        # Count bright spots
        num_spots = np.sum(center_max > 0) / 255
        
        # Normalize score
        score = min(1.0, num_spots / 10)
        
        return score
    
    def _detect_stone_texture(self, image: np.ndarray) -> float:
        """Detect stone by analyzing texture patterns."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Focus on center region
        height, width = gray.shape
        center_x, center_y = width // 2, height // 2
        
        # Extract center region
        radius = min(width, height) // 4
        center_region = gray[center_y-radius:center_y+radius, center_x-radius:center_x+radius]
        
        if center_region.size == 0:
            return 0.0
        
        # Calculate texture features
        # 1. Local variance (high for faceted stones)
        local_var = cv2.Laplacian(center_region, cv2.CV_64F).var()
        
        # 2. Gradient magnitude
        sobelx = cv2.Sobel(center_region, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(center_region, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2).mean()
        
        # Combine features
        texture_score = (local_var / 1000 + gradient_mag / 100) / 2
        texture_score = min(1.0, texture_score)
        
        return texture_score
    
    def _extract_stone_region(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Extract the stone region and determine stone type."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        center_x, center_y = width // 2, height // 2
        
        # Look for circular/oval region in center
        circles = cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=30,
            param1=50, 
            param2=25, 
            minRadius=10, 
            maxRadius=min(width, height) // 3
        )
        
        stone_region = None
        stone_type = None
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Find circle closest to center
            best_circle = None
            min_distance = float('inf')
            
            for (x, y, r) in circles:
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if distance < min_distance:
                    min_distance = distance
                    best_circle = (x, y, r)
            
            if best_circle:
                x, y, r = best_circle
                # Extract stone region
                x1 = max(0, x - r - 5)
                y1 = max(0, y - r - 5)
                x2 = min(width, x + r + 5)
                y2 = min(height, y + r + 5)
                
                stone_region = image[y1:y2, x1:x2]
                
                # Determine stone type based on color
                stone_type = self._determine_stone_type(stone_region)
        
        return stone_region, stone_type
    
    def _determine_stone_type(self, stone_region: np.ndarray) -> str:
        """Determine the type of stone based on color analysis."""
        if stone_region is None or stone_region.size == 0:
            return "unknown"
        
        # Convert to HSV
        hsv = cv2.cvtColor(stone_region, cv2.COLOR_BGR2HSV)
        
        # Calculate average color
        avg_hue = np.mean(hsv[:, :, 0])
        avg_saturation = np.mean(hsv[:, :, 1])
        avg_value = np.mean(hsv[:, :, 2])
        
        # Classify based on color
        if avg_saturation < 30 and avg_value > 200:
            return "diamond/clear"
        elif avg_hue < 10 or avg_hue > 160:
            return "red/ruby"
        elif 100 <= avg_hue <= 130:
            return "blue/sapphire"
        elif 40 <= avg_hue <= 80:
            return "green/emerald"
        elif 130 <= avg_hue <= 160:
            return "purple/amethyst"
        elif 20 <= avg_hue <= 35:
            return "yellow/citrine"
        else:
            return "other"
    
    def batch_classify(self, images: list) -> list:
        """Classify multiple images."""
        results = []
        for image in images:
            results.append(self.classify(image))
        return results


# Independent execution example
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Load image
        image = cv2.imread(sys.argv[1])
        if image is not None:
            classifier = StoneClassifier()
            result = classifier.classify(image)
            
            print(f"\nStone Classification Result:")
            print(f"Has Stone: {result['has_stone']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Stone Type: {result['stone_type']}")
            print(f"Message: {result['message']}")
            print(f"Debug Info: {result['debug_info']}")
            
            # Save stone region if detected
            if result['stone_region'] is not None:
                output_path = f"stone_{os.path.basename(sys.argv[1])}"
                cv2.imwrite(output_path, result['stone_region'])
                print(f"Saved stone region to: {output_path}")
        else:
            print("Failed to load image")
    else:
        print("Usage: python stone_classifier.py <image_path>")