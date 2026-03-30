"""
Module 1: Ring Classifier
Detects whether an image contains a ring using multiple computer vision techniques.
Can be operated independently.
"""

import cv2
import numpy as np
import os
from typing import Dict, Optional
import logging

from skimage.feature import local_binary_pattern  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RingClassifier:
    """
    Classifies images to determine if they contain a ring.
    Uses shape detection, circular patterns, and edge analysis.
    """
    
    def __init__(self, confidence_threshold: float = 0.15):
        """
        Initialize the ring classifier.
        
        Args:
            confidence_threshold: Minimum confidence score to classify as ring
        """
        self.confidence_threshold = confidence_threshold
        
    def classify(self, image_path: str) -> Dict:
        """
        Main method to classify if an image contains a ring.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with classification results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {
                'has_ring': False,
                'confidence': 0.0,
                'message': 'Failed to load image',
                'debug_info': {}
            }
        
        # Run multiple detection methods
        shape_score = self._detect_circular_shape(image)
        edge_score = self._detect_ring_edges(image)
        contour_score = self._detect_ring_contours(image)
        texture_score = self._detect_metal_texture(image)
        
        # Calculate weighted confidence
        confidence = (
            shape_score * 0.35 +
            edge_score * 0.30 +
            contour_score * 0.25 +
            texture_score * 0.10
        )
        
        # Decision
        has_ring = confidence >= self.confidence_threshold
        
        # Extract ring region if present
        ring_region = None
        if has_ring:
            ring_region = self._extract_ring_region(image)
        
        result = {
            'has_ring': has_ring,
            'confidence': float(confidence),
            'message': f"Ring {'detected' if has_ring else 'not detected'} (confidence: {confidence:.2f})",
            'debug_info': {
                'shape_score': shape_score,
                'edge_score': edge_score,
                'contour_score': contour_score,
                'texture_score': texture_score
            },
            'ring_region': ring_region
        }
        
        logger.info(f"Classification result for {os.path.basename(image_path)}: {result['message']}")
        return result
    
    def _detect_circular_shape(self, image: np.ndarray) -> float:
        """Detect circular/elliptical shapes that might indicate a ring."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=50,
            param1=50, 
            param2=30, 
            minRadius=30, 
            maxRadius=300
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return min(1.0, len(circles) * 0.3)
        
        return 0.0
    
    def _detect_ring_edges(self, image: np.ndarray) -> float:
        """Detect characteristic ring edges (inner and outer circles)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for concentric circles pattern
        circles = cv2.HoughCircles(
            edges, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=30,
            param1=50, 
            param2=25, 
            minRadius=20, 
            maxRadius=200
        )
        
        if circles is not None:
            circles = circles[0]
            # Check for concentric pairs
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    center_dist = np.sqrt(
                        (circles[i][0] - circles[j][0])**2 + 
                        (circles[i][1] - circles[j][1])**2
                    )
                    if center_dist < 20:  # Roughly concentric
                        return 0.9
        
        # Alternative: Check for strong circular gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        if np.mean(gradient_magnitude) > 50:
            return 0.5
        
        return 0.0
    
    def _detect_ring_contours(self, image: np.ndarray) -> float:
        """Detect ring-like contours in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to find potential ring regions
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ring_like_contours = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Too small
                continue
            
            # Check if contour is roughly circular
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Check for hole (inner circle)
            if circularity > 0.7:  # Circular shape
                # Look for inner contour
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                inner_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(inner_contours) > 1:  # Has inner contour (hole)
                    ring_like_contours += 1
        
        score = min(1.0, ring_like_contours * 0.5)
        return score
    
    def _detect_metal_texture(self, image: np.ndarray) -> float:
        """Detect metallic texture patterns common in rings."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture features using Local Binary Pattern
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Calculate texture contrast
        texture_contrast = np.std(lbp)
        
        # Normalize score
        score = min(1.0, texture_contrast / 50.0)
        
        return score
    
    def _extract_ring_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract the region containing the ring for further processing."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find the largest circular contour
        circles = cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=50,
            param1=50, 
            param2=30, 
            minRadius=30, 
            maxRadius=300
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Get the largest circle
            largest = max(circles, key=lambda c: c[2])
            x, y, r = largest
            
            # Extract region with padding
            padding = 20
            x1 = max(0, x - r - padding)
            y1 = max(0, y - r - padding)
            x2 = min(image.shape[1], x + r + padding)
            y2 = min(image.shape[0], y + r + padding)
            
            ring_region = image[y1:y2, x1:x2]
            return ring_region
        
        return None
    
    def batch_classify(self, image_paths: list) -> list:
        """Classify multiple images."""
        results = []
        for path in image_paths:
            results.append(self.classify(path))
        return results


# Independent execution example
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        classifier = RingClassifier()
        result = classifier.classify(sys.argv[1])
        print(f"\nClassification Result:")
        print(f"Has Ring: {result['has_ring']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Message: {result['message']}")
        print(f"Debug Info: {result['debug_info']}")
    else:
        print("Usage: python ring_classifier.py <image_path>")