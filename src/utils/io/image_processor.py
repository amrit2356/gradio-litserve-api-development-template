# utils/image_processor.py
import base64
import io
from PIL import Image
from typing import Dict, Any, Union

from src.utils.config.settings import get_settings
from src.utils.log.logger import get_module_logger

# Initialize settings and logger
settings = get_settings()
logger = get_module_logger(__name__)

class ImageProcessor:
    """Handles image decoding, encoding, and preprocessing"""
    
    @staticmethod
    def decode_base64_image(request: Dict[str, Any]) -> Image.Image:
        """
        Decode base64 image from request
        
        Args:
            request: Request dictionary containing image data
            
        Returns:
            PIL Image object
        """
        try:
            # Handle different input formats
            image_data = ImageProcessor._extract_image_data(request)
            
            # Decode base64 image
            if isinstance(image_data, str):
                image_bytes = ImageProcessor._decode_base64_string(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                # Assume it's already image bytes
                image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Error decoding image: {str(e)}")
            raise ValueError(f"Invalid image format: {str(e)}")
    
    @staticmethod
    def _extract_image_data(request: Dict[str, Any]) -> Union[str, bytes]:
        """Extract image data from request"""
        if "image" in request:
            return request["image"]
        elif "file" in request:
            return request["file"]
        elif "data" in request:
            return request["data"]
        else:
            raise ValueError("No image data found in request. Expected 'image', 'file', or 'data' field.")
    
    @staticmethod
    def _decode_base64_string(image_data: str) -> bytes:
        """Decode base64 string to bytes"""
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        return base64.b64decode(image_data)
    
    @staticmethod
    def encode_image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
        """
        Encode PIL Image to base64 string
        
        Args:
            image: PIL Image object
            format: Image format (JPEG, PNG, etc.)
            
        Returns:
            Base64 encoded string
        """
        try:
            buffer = io.BytesIO()
            image.save(buffer, format=format)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return img_str
            
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            raise ValueError(f"Image encoding failed: {str(e)}")
    
    @staticmethod
    def validate_image(image: Image.Image) -> bool:
        """
        Validate image properties
        
        Args:
            image: PIL Image object
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Get validation settings
            img_settings = settings.image_processing
            
            # Check if image has valid dimensions
            if image.width <= 0 or image.height <= 0:
                return False
            
            # Check dimension limits
            if img_settings.strict_validation:
                if (image.width > img_settings.max_width or 
                    image.height > img_settings.max_height or
                    image.width < img_settings.min_width or 
                    image.height < img_settings.min_height):
                    return False
            
            # Check if image has valid mode
            if image.mode not in ['RGB', 'RGBA', 'L', 'P']:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Image validation failed: {str(e)}")
            return False
    
    @staticmethod
    def preprocess_image(image: Image.Image, target_size: tuple = None) -> Image.Image:
        """
        Preprocess image for inference
        
        Args:
            image: PIL Image object
            target_size: Target size tuple (width, height)
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            # Get preprocessing settings
            img_settings = settings.image_processing
            
            # Ensure RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Auto-resize if enabled
            if img_settings.auto_resize or target_size:
                resize_target = target_size or (settings.model.image_size, settings.model.image_size)
                
                if img_settings.maintain_aspect_ratio:
                    # Maintain aspect ratio
                    image.thumbnail(resize_target, Image.Resampling.LANCZOS)
                else:
                    # Direct resize
                    image = image.resize(resize_target, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")
    
    @staticmethod
    def get_image_info(image: Image.Image) -> Dict[str, Any]:
        """
        Get image information
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with image information
        """
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "size_bytes": len(image.tobytes())
        }