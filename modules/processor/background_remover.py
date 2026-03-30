"""
Module 3: Background Remover
Removes background from ring images and sets it to white.
Can be operated independently.
"""

import cv2
import numpy as np
import os
from typing import Optional, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackgroundRemover:
    """
    Removes background from ring images and sets it to white.
    """
    
    def __init__(self, bg_color: Tuple[int, int, int] = (255, 255, 255)):
        """
        Initialize the background remover.
        
        Args:
            bg_color: Background color to use (default: white)
        """
        self.bg_color = bg_color
        
    def remove_background(self, image: np.ndarray) -> Dict:
        """
        Main method to remove background from ring image.
        
        Args:
            image: Input ring image
            
        Returns:
            Dictionary with background-removed image and mask
        """
        if image is None:
            return {
                'success': False,
                'message': 'Invalid image',
                'image': None,
                'mask': None
            }
        
        # Create mask for the ring
        mask = self._create_ring_mask(image)
        
        # Remove background and set to white
        result = self._apply_background_removal(image, mask)
        
        # Clean up any remaining noise
        result = self._remove_noise(result, mask)
        
        # Calculate statistics
        ring_area = np.sum(mask > 0) / 255.0
        total_area = mask.shape[0] * mask.shape[1]
        ring_percentage = (ring_area / total_area) * 100
        
        result_dict = {
            'success': True,
            'message': 'Background successfully removed',
            'image': result,
            'mask': mask,
            'ring_percentage': ring_percentage,
            'original_shape': image.shape[:2]
        }
        
        logger.info(f"Background removed. Ring occupies {ring_percentage:.1f}% of image")
        return result_dict
    
    def _create_ring_mask(self, image: np.ndarray) -> np.ndarray:
        """Create an accurate mask for the ring."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple segmentation techniques
        
        # 1. Adaptive thresholding
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # 2. Edge-based segmentation
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3,3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # 3. Combine masks
        combined = cv2.bitwise_or(thresh1, edges_dilated)
        
        # 4. Morphological operations to clean up
        kernel = np.ones((5,5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 5. Find and keep largest connected components (the ring)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined, 8, cv2.CV_32S)
        
        # Create mask with only significant components
        mask = np.zeros_like(combined)
        min_area = 500  # Minimum area to keep
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > min_area:
                mask[labels == i] = 255
        
        # 6. Fill holes in the ring
        mask = self._fill_ring_holes(mask)
        
        return mask
    
    def _fill_ring_holes(self, mask: np.ndarray) -> np.ndarray:
        """Fill holes in the ring mask (the inner empty space of the ring)."""
        # Invert mask to find holes
        inverted = cv2.bitwise_not(mask)
        
        # Find contours in inverted mask
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a mask of holes to fill
        holes_mask = np.zeros_like(mask)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5000:  # Likely a hole in the ring
                cv2.drawContours(holes_mask, [contour], -1, 255, -1)
        
        # Add holes back to original mask
        mask = cv2.bitwise_or(mask, holes_mask)
        
        return mask
    
    def _apply_background_removal(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply background removal and set to white."""
        # Create white background
        result = np.full_like(image, self.bg_color)
        
        # Apply mask
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        result = (result * (1 - mask_3channel) + image * mask_3channel).astype(np.uint8)
        
        return result
    
    def _remove_noise(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Remove any remaining noise around the ring."""
        # Create a clean mask with morphological operations
        kernel = np.ones((3,3), np.uint8)
        clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Apply clean mask
        clean_mask_3channel = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR) / 255.0
        result = np.full_like(image, self.bg_color)
        result = (result * (1 - clean_mask_3channel) + image * clean_mask_3channel).astype(np.uint8)
        
        return result
    
    def batch_process(self, images: list) -> list:
        """Process multiple images."""
        results = []
        for image in images:
            results.append(self.remove_background(image))
        return results


# Independent execution example
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Load image
        image = cv2.imread(sys.argv[1])
        if image is not None:
            remover = BackgroundRemover()
            result = remover.remove_background(image)
            
            if result['success']:
                print(f"\nSuccessfully removed background:")
                print(f"Ring occupies: {result['ring_percentage']:.1f}% of image")
                print(f"Original shape: {result['original_shape']}")
                
                # Save results
                base_name = os.path.basename(sys.argv[1])
                output_path = f"nobg_{base_name}"
                mask_path = f"mask_{base_name}"
                
                cv2.imwrite(output_path, result['image'])
                cv2.imwrite(mask_path, result['mask'])
                
                print(f"Saved result to: {output_path}")
                print(f"Saved mask to: {mask_path}")
            else:
                print(f"Failed: {result['message']}")
        else:
            print("Failed to load image")
    else:
        print("Usage: python background_remover.py <image_path>")