"""
Image processing utilities for dermatological analysis.
Simplified preprocessing for proof-of-concept.
"""
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Optional

def process_uploaded_image(image_path: str) -> Optional[np.ndarray]:
    """
    Process uploaded image for dermatological analysis.
    
    Args:
        image_path: Path to the uploaded image file
        
    Returns:
        Processed image array or None if processing fails
    """
    try:
        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Could not load image from {image_path}")
            return None
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to standard size for processing
        target_size = (224, 224)  # Standard CNN input size
        image_resized = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_AREA)
        
        # Apply basic preprocessing
        processed_image = preprocess_dermatology_image(image_resized)
        
        return processed_image
        
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return None

def preprocess_dermatology_image(image: np.ndarray) -> np.ndarray:
    """
    Apply dermatology-specific preprocessing to the image.
    
    Args:
        image: Input image array (RGB)
        
    Returns:
        Preprocessed image array
    """
    try:
        # Convert to grayscale for certain analyses
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Apply edge detection to highlight boundaries
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Combine enhanced image with edge information
        # This is a simplified approach - in production, more sophisticated methods would be used
        combined = cv2.addWeighted(enhanced, 0.8, edges, 0.2, 0)
        
        # Normalize pixel values to 0-1 range
        normalized = combined.astype(np.float32) / 255.0
        
        return normalized
        
    except Exception as e:
        logging.error(f"Error in dermatology preprocessing: {str(e)}")
        # Return original image if preprocessing fails
        return image.astype(np.float32) / 255.0

def extract_roi(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Extract Region of Interest (ROI) from the image.
    Simplified version of what would be a U-Net segmentation in production.
    
    Args:
        image: Input image array
        threshold: Threshold for ROI detection
        
    Returns:
        ROI mask or full image if ROI detection fails
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply Otsu's thresholding to separate lesion from skin
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (assumed to be the main lesion)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create mask for the largest contour
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [largest_contour], 255)
            
            # Apply mask to original image
            roi = cv2.bitwise_and(image, image, mask=mask)
            
            return roi
        else:
            # If no contours found, return original image
            return image
            
    except Exception as e:
        logging.error(f"Error extracting ROI: {str(e)}")
        return image

def validate_medical_image(image_path: str) -> dict:
    """
    Validate that the uploaded image is suitable for medical analysis.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Validation result dictionary
    """
    try:
        # Load image to check properties
        with Image.open(image_path) as img:
            width, height = img.size
            format_type = img.format
            mode = img.mode
        
        validation_result = {
            'valid': True,
            'messages': [],
            'warnings': []
        }
        
        # Check minimum resolution
        min_resolution = 100  # pixels
        if width < min_resolution or height < min_resolution:
            validation_result['valid'] = False
            validation_result['messages'].append(
                f"Image resolution too low. Minimum {min_resolution}x{min_resolution} pixels required."
            )
        
        # Check maximum resolution (to prevent processing issues)
        max_resolution = 4000  # pixels
        if width > max_resolution or height > max_resolution:
            validation_result['warnings'].append(
                "High resolution image detected. Processing time may be longer."
            )
        
        # Check image format
        supported_formats = ['JPEG', 'PNG', 'GIF']
        if format_type not in supported_formats:
            validation_result['valid'] = False
            validation_result['messages'].append(
                f"Unsupported image format: {format_type}. Supported formats: {', '.join(supported_formats)}"
            )
        
        # Check color mode
        if mode not in ['RGB', 'RGBA', 'L']:
            validation_result['warnings'].append(
                f"Unusual color mode detected: {mode}. Results may vary."
            )
        
        return validation_result
        
    except Exception as e:
        return {
            'valid': False,
            'messages': [f"Error validating image: {str(e)}"],
            'warnings': []
        }

def create_visualization_overlay(original_image: np.ndarray, processed_image: np.ndarray) -> np.ndarray:
    """
    Create a visualization overlay showing the processing results.
    This simulates the Grad-CAM functionality mentioned in requirements.
    
    Args:
        original_image: Original input image
        processed_image: Processed image with features highlighted
        
    Returns:
        Overlay visualization
    """
    try:
        # Ensure images are the same size
        if original_image.shape != processed_image.shape:
            processed_image = cv2.resize(processed_image, 
                                       (original_image.shape[1], original_image.shape[0]))
        
        # Create a simple heatmap overlay
        # In production, this would be actual Grad-CAM output
        heatmap = cv2.applyColorMap((processed_image * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend original image with heatmap
        overlay = cv2.addWeighted(original_image, 0.7, heatmap, 0.3, 0)
        
        return overlay
        
    except Exception as e:
        logging.error(f"Error creating visualization overlay: {str(e)}")
        return original_image
