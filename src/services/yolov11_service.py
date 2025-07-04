# services/yolo_service.py
import time
from typing import Dict, Any, List, Optional
from PIL import Image

from src.models.yolo_pipeline import YOLOModel
from src.utils.io.image_processor import ImageProcessor
from src.utils.io.response_formatter import ResponseFormatter
from src.utils.config.settings import get_settings
from src.utils.log.logger import get_module_logger, log_execution_time

# Initialize settings and logger
settings = get_settings()
logger = get_module_logger(__name__)

class YOLOService:
    """
    Main service class that orchestrates YOLO inference
    Handles the business logic and coordinates between different components
    """
    
    def __init__(self, model_size: str = None, device: str = None):
        """
        Initialize YOLO service
        
        Args:
            model_size: YOLO model size (n, s, m, l, x)
            device: Device to run inference on
        """
        # Use settings if parameters not provided
        model_size = model_size or settings.model.default_size
        device = device or settings.model.device_preferred
        
        self.model = YOLOModel(model_size=model_size, device=device)
        self.image_processor = ImageProcessor()
        self.response_formatter = ResponseFormatter()
        
        # Load model on initialization
        self.model.load_model()
        
        logger.info(f"YOLOService initialized with model size: {model_size}, device: {device}")
    
    def process_image(
        self, 
        request: Dict[str, Any],
        include_image_info: bool = None,
        include_performance: bool = None,
        response_format: str = None
    ) -> Dict[str, Any]:
        """
        Process image and return detection results
        
        Args:
            request: Request containing image data
            include_image_info: Whether to include image info in response
            include_performance: Whether to include performance metrics
            response_format: Response format ("detailed", "minimal")
            
        Returns:
            Detection results dictionary
        """
        # Use settings defaults if not provided
        include_image_info = include_image_info if include_image_info is not None else settings.response.include_metadata
        include_performance = include_performance if include_performance is not None else settings.response.include_timing
        response_format = response_format or settings.response.default_format
        
        with log_execution_time(__name__, "process_image"):
            start_time = time.time()
            
            try:
                # Decode image
                image = self.image_processor.decode_base64_image(request)
                
                # Validate image
                if not self.image_processor.validate_image(image):
                    return self.response_formatter.format_error_response(
                        "Invalid image format or dimensions",
                        error_type="ImageValidationError"
                    )
                
                # Preprocess image if needed
                if settings.image_processing.auto_resize:
                    image = self.image_processor.preprocess_image(image)
                
                # Run inference
                detections = self.model.predict(image)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Format response based on requested format
                if response_format == "minimal":
                    return self.response_formatter.format_minimal_response(detections)
                else:
                    return self._format_detailed_response(
                        detections, 
                        image, 
                        processing_time,
                        include_image_info,
                        include_performance
                    )
            
            except ValueError as e:
                logger.error(f"Image processing error: {str(e)}")
                return self.response_formatter.format_error_response(
                    str(e),
                    error_type="ImageProcessingError"
                )
            
            except RuntimeError as e:
                logger.error(f"Inference error: {str(e)}")
                return self.response_formatter.format_error_response(
                    str(e),
                    error_type="InferenceError"
                )
            
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return self.response_formatter.format_error_response(
                    "An unexpected error occurred during processing",
                    error_type="UnknownError",
                    details={"original_error": str(e)}
                )
    
    def _format_detailed_response(
        self,
        detections: List[Dict[str, Any]],
        image: Image.Image,
        processing_time: float,
        include_image_info: bool,
        include_performance: bool
    ) -> Dict[str, Any]:
        """Format detailed response with optional components"""
        
        # Format detections for API
        formatted_detections = [
            self.response_formatter.format_detection_for_api(detection)
            for detection in detections
        ]
        
        # Prepare optional components
        model_info = self.model.get_model_info() if include_performance else None
        image_info = self.image_processor.get_image_info(image) if include_image_info else None
        proc_time = processing_time if include_performance else None
        
        return self.response_formatter.format_success_response(
            formatted_detections,
            model_info=model_info,
            processing_time=proc_time,
            image_info=image_info
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get service health status
        
        Returns:
            Health status dictionary
        """
        model_info = self.model.get_model_info()
        
        return self.response_formatter.format_health_response(
            model_loaded=self.model.is_loaded(),
            model_info=model_info
        )
    
    def batch_process(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch
        
        Args:
            requests: List of image requests
            
        Returns:
            List of detection results
        """
        results = []
        
        for i, request in enumerate(requests):
            try:
                # Add batch identifier
                request_with_id = {**request, "batch_id": i}
                result = self.process_image(request_with_id, response_format="minimal")
                result["batch_id"] = i
                results.append(result)
                
            except Exception as e:
                logger.error(f"Batch processing error for item {i}: {str(e)}")
                results.append(
                    self.response_formatter.format_error_response(
                        f"Batch item {i} processing failed: {str(e)}",
                        error_type="BatchProcessingError"
                    )
                )
        
        return results
    
    def update_model_config(self, conf_threshold: float = None, iou_threshold: float = None) -> Dict[str, Any]:
        """
        Update model configuration
        
        Args:
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
            
        Returns:
            Updated configuration
        """
        config = {}
        
        if conf_threshold is not None:
            config["conf_threshold"] = conf_threshold
        
        if iou_threshold is not None:
            config["iou_threshold"] = iou_threshold
        
        return {
            "success": True,
            "message": "Configuration updated",
            "config": config
        }
    
    def get_supported_classes(self) -> Dict[str, Any]:
        """
        Get list of supported object classes
        
        Returns:
            Dictionary with supported classes
        """
        if not self.model.is_loaded():
            return self.response_formatter.format_error_response(
                "Model not loaded",
                error_type="ModelNotLoadedError"
            )
        
        try:
            class_names = self.model.model.names
            return {
                "success": True,
                "classes": {
                    "total": len(class_names),
                    "names": class_names
                }
            }
        except Exception as e:
            return self.response_formatter.format_error_response(
                f"Failed to get classes: {str(e)}",
                error_type="ClassRetrievalError"
            )