"""
Module 2: Ring Cropper and Enhancer
Crops the ring from the image and enhances its quality.
Can be operated independently.
"""

import cv2
import numpy as np
import os
from typing import Optional, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RingCropper:
    """
    Crops rings from images and enhances their quality.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        """
        Initialize the ring cropper.
        
        Args:
            target_size: Target size for the output image
        """
        self.target_size = target_size
        
    def crop_and_enhance(self, image_path: str, ring_region: Optional[np.ndarray] = None) -> Dict:
        """
        Main method to crop and enhance the ring.
        
        Args:
            image_path: Path to the image file
            ring_region: Optional pre-detected ring region
            
        Returns:
            Dictionary with cropped and enhanced image and metadata
        """
        # Load image
        if ring_region is None:
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'success': False,
                    'message': 'Failed to load image',
                    'cropped_image': None
                }
        else:
            image = ring_region
        
        # Step 1: Find and crop ring
        cropped = self._find_and_crop_ring(image)
        if cropped is None:
            return {
                'success': False,
                'message': 'Could not locate ring in image',
                'cropped_image': None
            }
        
        # Step 2: Enhance image quality
        enhanced = self._enhance_quality(cropped)
        
        # Step 3: Resize to target size
        resized = cv2.resize(enhanced, self.target_size)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(resized)
        
        result = {
            'success': True,
            'message': 'Ring successfully cropped and enhanced',
            'cropped_image': resized,
            'original_size': cropped.shape[:2],
            'quality_metrics': quality_metrics
        }
        
        logger.info(f"Successfully processed {os.path.basename(image_path)}")
        return result
    
    def _find_and_crop_ring(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Find the ring in the image and crop tightly around it."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try multiple methods to find the ring
        
        # Method 1: Circle detection
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
            
            # Crop with padding
            padding = int(r * 0.2)  # 20% padding
            x1 = max(0, x - r - padding)
            y1 = max(0, y - r - padding)
            x2 = min(image.shape[1], x + r + padding)
            y2 = min(image.shape[0], y + r + padding)
            
            return image[y1:y2, x1:x2]
        
        # Method 2: Contour detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            return image[y1:y2, x1:x2]
        
        return None
    
    def _enhance_quality(self, image: np.ndarray) -> np.ndarray:
        """Enhance the quality of the cropped ring image."""
        enhanced = image.copy()
        
        # 1. Denoising
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        # 2. Contrast enhancement
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # 3. Sharpening
        kernel_sharpen = np.array([[-1,-1,-1],
                                   [-1, 9,-1],
                                   [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel_sharpen)
        
        # 4. Auto white balance
        enhanced = self._auto_white_balance(enhanced)
        
        # 5. Gamma correction if needed
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 80:  # Too dark
            gamma = 1.5
            lookUpTable = np.empty((1,256), np.uint8)
            for i in range(256):
                lookUpTable[0,i] = np.clip(pow(i/255.0, 1.0/gamma) * 255.0, 0, 255)
            enhanced = cv2.LUT(enhanced, lookUpTable)
        elif mean_brightness > 200:  # Too bright
            enhanced = cv2.convertScaleAbs(enhanced, alpha=0.8, beta=-20)
        
        return enhanced
    
    def _auto_white_balance(self, image: np.ndarray) -> np.ndarray:
        """Apply auto white balance to the image."""
        result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result
    
    def _calculate_quality_metrics(self, image: np.ndarray) -> Dict:
        """Calculate quality metrics for the enhanced image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Contrast
        contrast = gray.std()
        
        # Brightness
        brightness = gray.mean()
        
        # Noise level (estimated)
        noise = np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0))
        
        return {
            'sharpness': float(sharpness),
            'contrast': float(contrast),
            'brightness': float(brightness),
            'noise_level': float(noise)
        }


# Independent execution example
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        cropper = RingCropper()
        result = cropper.crop_and_enhance(sys.argv[1])
        
        if result['success']:
            print(f"\nSuccessfully processed image:")
            print(f"Original size: {result['original_size']}")
            print(f"Quality metrics: {result['quality_metrics']}")
            
            # Save the result
            output_path = f"cropped_{os.path.basename(sys.argv[1])}"
            cv2.imwrite(output_path, result['cropped_image'])
            print(f"Saved cropped image to: {output_path}")
        else:
            print(f"Failed: {result['message']}")
    else:
        print("Usage: python ring_cropper.py <image_path>")