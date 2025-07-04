# apis/runpod_api.py
import runpod
from typing import Dict, Any

from src.services.yolov11_service import YOLOService
from src.utils.log.logger import get_module_logger

logger = get_module_logger(__name__)


class YOLOv11RunPodAPI:
    """
    RunPod serverless API implementation for YOLO inference
    Handles synchronous operations and integrates with YOLOService
    """
    
    def __init__(self, model_size: str = "n", device: str = "auto"):
        """
        Initialize RunPod API
        
        Args:
            model_size: YOLO model size
            device: Device to run inference on
        """
        self.model_size = model_size
        self.device = device
        self.service = None
        
        # Initialize service
        self._setup_service()
    
    def _setup_service(self) -> None:
        """Initialize the YOLO service"""
        try:
            logger.info(f"Setting up RunPod API with model size: {self.model_size}, device: {self.device}")
            
            # Initialize service
            self.service = YOLOService(model_size=self.model_size, device=self.device)
            
            logger.info("RunPod API setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup RunPod API: {str(e)}")
            raise RuntimeError(f"API setup failed: {str(e)}")
    
    def handler(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main RunPod job handler
        
        Args:
            job: RunPod job dictionary
            
        Returns:
            Job result dictionary
        """
        try:
            # Extract job input
            job_input = job.get("input", {})
            
            # Add RunPod-specific metadata
            job_input["runpod_job_id"] = job.get("id", "unknown")
            
            # Process request
            result = self.service.process_image(
                job_input,
                include_image_info=job_input.get("include_image_info", False),
                include_performance=job_input.get("include_performance", True),  # Default to True for RunPod
                response_format=job_input.get("response_format", "detailed")
            )
            
            # Add RunPod-specific metadata to response
            result["server_type"] = "runpod"
            result["job_id"] = job.get("id", "unknown")
            
            return result
            
        except Exception as e:
            logger.error(f"RunPod handler error: {str(e)}")
            return self.service.response_formatter.format_error_response(
                f"RunPod handler failed: {str(e)}",
                error_type="RunPodHandlerError",
                details={"job_id": job.get("id", "unknown")}
            )
    
    def health_check(self) -> Dict[str, Any]:
        """
        Health check endpoint
        
        Returns:
            Health status dictionary
        """
        try:
            health_status = self.service.get_health_status()
            health_status["server_type"] = "runpod"
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return self.service.response_formatter.format_error_response(
                f"Health check failed: {str(e)}",
                error_type="HealthCheckError"
            )
    
    def get_supported_classes(self) -> Dict[str, Any]:
        """
        Get supported object classes
        
        Returns:
            Supported classes dictionary
        """
        try:
            return self.service.get_supported_classes()
            
        except Exception as e:
            logger.error(f"Failed to get supported classes: {str(e)}")
            return self.service.response_formatter.format_error_response(
                f"Failed to get supported classes: {str(e)}",
                error_type="ClassRetrievalError"
            )

class YOLOv11RunPodManager:
    """
    Manager class for RunPod serverless operations
    """
    
    def __init__(self, model_size: str = "n", device: str = "auto"):
        """
        Initialize RunPod manager
        
        Args:
            model_size: YOLO model size
            device: Device to run inference on
        """
        self.model_size = model_size
        self.device = device
        self.api = None
    
    def create_api(self) -> YOLOv11RunPodAPI:
        """
        Create RunPod API instance
        
        Returns:
            YOLOv11RunPodAPI instance
        """
        try:
            self.api = YOLOv11RunPodAPI(model_size=self.model_size, device=self.device)
            logger.info("RunPod API created successfully")
            return self.api
            
        except Exception as e:
            logger.error(f"Failed to create RunPod API: {str(e)}")
            raise RuntimeError(f"API creation failed: {str(e)}")
    
    def start_serverless(self) -> None:
        """
        Start RunPod serverless function
        """
        try:
            if not self.api:
                self.create_api()
            
            logger.info("Starting RunPod serverless function...")
            
            # Start RunPod serverless with the handler
            runpod.serverless.start({
                "handler": self.api.handler,
                "return_aggregate_stream": True
            })
            
        except Exception as e:
            logger.error(f"Failed to start RunPod serverless: {str(e)}")
            raise RuntimeError(f"Serverless start failed: {str(e)}")
    
    def get_api_info(self) -> Dict[str, Any]:
        """
        Get API information
        
        Returns:
            API information dictionary
        """
        return {
            "server_type": "runpod",
            "model_size": self.model_size,
            "device": self.device,
            "api_initialized": self.api is not None
        }

# Helper functions for RunPod-specific operations
def create_test_input(output_file: str = "test_input.json") -> str:
    """
    Create test input file for RunPod local testing
    
    Args:
        output_file: Output file path
        
    Returns:
        Path to created test file
    """
    import json
    import base64
    import io
    from PIL import Image
    
    try:
        # Create a simple test image
        img = Image.new('RGB', (640, 480), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Create test input
        test_input = {
            "input": {
                "image": img_str,
                "include_image_info": True,
                "include_performance": True,
                "response_format": "detailed"
            }
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(test_input, f, indent=2)
        
        logger.info(f"Created test input file: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Failed to create test input: {str(e)}")
        raise RuntimeError(f"Test input creation failed: {str(e)}")

def validate_runpod_environment() -> Dict[str, Any]:
    """
    Validate RunPod environment
    
    Returns:
        Environment validation results
    """
    import os
    
    validation_results = {
        "is_runpod_environment": False,
        "environment_variables": {},
        "missing_variables": []
    }
    
    # Check for RunPod-specific environment variables
    runpod_vars = [
        "RUNPOD_POD_ID",
        "RUNPOD_POD_HOSTNAME",
        "RUNPOD_API_KEY"
    ]
    
    for var in runpod_vars:
        value = os.getenv(var)
        if value:
            validation_results["environment_variables"][var] = value
            validation_results["is_runpod_environment"] = True
        else:
            validation_results["missing_variables"].append(var)
    
    return validation_results