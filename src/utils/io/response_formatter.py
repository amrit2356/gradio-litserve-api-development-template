# utils/response_formatter.py
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.utils.config.settings import get_settings
from src.utils.log.logger import get_module_logger

# Initialize settings and logger
settings = get_settings()
logger = get_module_logger(__name__)

class ResponseFormatter:
    """Handles response formatting and standardization"""
    
    @staticmethod
    def format_success_response(
        detections: List[Dict[str, Any]], 
        model_info: Dict[str, Any] = None,
        processing_time: float = None,
        image_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Format successful detection response
        
        Args:
            detections: List of detection dictionaries
            model_info: Model information
            processing_time: Processing time in seconds
            image_info: Image information
            
        Returns:
            Formatted response dictionary
        """
        # Get response settings
        response_config = settings.response
        
        response = {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "results": {
                "detections": detections,
                "count": len(detections),
                "summary": ResponseFormatter._generate_detection_summary(detections)
            }
        }
        
        # Add model info if provided and metadata is enabled
        if model_info and response_config.include_metadata:
            response["model"] = {
                "name": "YOLOv11",
                "version": model_info.get("model_size", "unknown"),
                "device": model_info.get("device", "unknown")
            }
        
        # Add performance metrics if provided and timing is enabled
        if processing_time is not None and response_config.include_timing:
            response["performance"] = {
                "processing_time_seconds": round(processing_time, response_config.decimal_precision),
                "fps": round(1.0 / processing_time, 2) if processing_time > 0 else 0
            }
        
        # Add image info if provided and metadata is enabled
        if image_info and response_config.include_metadata:
            response["image_info"] = image_info
        
        return response
    
    @staticmethod
    def format_error_response(
        error_message: str,
        error_type: str = "InferenceError",
        details: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Format error response
        
        Args:
            error_message: Error message
            error_type: Type of error
            details: Additional error details
            
        Returns:
            Formatted error response dictionary
        """
        response = {
            "success": False,
            "timestamp": datetime.utcnow().isoformat(),
            "error": {
                "type": error_type,
                "message": error_message
            }
        }
        
        if details:
            response["error"]["details"] = details
        
        return response
    
    @staticmethod
    def _generate_detection_summary(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics for detections
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Summary statistics dictionary
        """
        if not detections:
            return {
                "total_objects": 0,
                "classes_detected": [],
                "class_counts": {},
                "confidence_stats": {
                    "min": 0,
                    "max": 0,
                    "avg": 0
                }
            }
        
        # Count classes
        class_counts = {}
        confidences = []
        
        for detection in detections:
            class_name = detection.get("class_name", "unknown")
            confidence = detection.get("confidence", 0)
            
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(confidence)
        
        # Calculate confidence statistics
        conf_stats = {
            "min": round(min(confidences), settings.response.decimal_precision),
            "max": round(max(confidences), settings.response.decimal_precision),
            "avg": round(sum(confidences) / len(confidences), settings.response.decimal_precision)
        }
        
        return {
            "total_objects": len(detections),
            "classes_detected": list(class_counts.keys()),
            "class_counts": class_counts,
            "confidence_stats": conf_stats
        }
    
    @staticmethod
    def format_health_response(
        model_loaded: bool,
        model_info: Dict[str, Any] = None,
        system_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Format health check response
        
        Args:
            model_loaded: Whether model is loaded
            model_info: Model information
            system_info: System information
            
        Returns:
            Health check response dictionary
        """
        response = {
            "status": "healthy" if model_loaded else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "model_status": "loaded" if model_loaded else "not_loaded"
        }
        
        if model_info:
            response["model_info"] = model_info
        
        if system_info:
            response["system_info"] = system_info
        
        return response
    
    @staticmethod
    def format_detection_for_api(detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format individual detection for API response
        
        Args:
            detection: Raw detection dictionary
            
        Returns:
            Formatted detection dictionary
        """
        # Get response settings
        response_config = settings.response
        precision = response_config.decimal_precision
        
        formatted = {
            "object_id": detection.get("object_id"),
            "class": {
                "id": detection.get("class_id"),
                "name": detection.get("class_name"),
                "confidence": round(detection.get("confidence", 0), precision)
            }
        }
        
        # Format bounding box according to settings
        bbox = detection.get("bbox", {})
        if response_config.bbox_format == "xyxy":
            formatted["bounding_box"] = {
                "x1": round(bbox.get("x1", 0), 1),
                "y1": round(bbox.get("y1", 0), 1),
                "x2": round(bbox.get("x2", 0), 1),
                "y2": round(bbox.get("y2", 0), 1),
                "width": round(bbox.get("x2", 0) - bbox.get("x1", 0), 1),
                "height": round(bbox.get("y2", 0) - bbox.get("y1", 0), 1)
            }
        # Add other bbox formats as needed
        
        return formatted
    
    @staticmethod
    def format_minimal_response(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format minimal response for lightweight applications
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Minimal response dictionary
        """
        return {
            "success": True,
            "detections": detections,
            "count": len(detections),
            "model": "YOLOv11",
            "message": f"Detected {len(detections)} objects"
        }