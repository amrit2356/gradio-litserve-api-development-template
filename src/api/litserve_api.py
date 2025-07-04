# apis/litserve_api.py
import litserve as ls
from typing import Dict, Any

from src.services.yolov11_service import YOLOService
from src.utils.log.logger import get_module_logger

logger = get_module_logger(__name__)

class YOLOv11LitServeAPI(ls.LitAPI):
    """
    LitServe API implementation for YOLO inference
    Handles async operations and integrates with YOLOService
    """
    
    def __init__(self, model_size: str = "n", **kwargs):
        """
        Initialize LitServe API
        
        Args:
            model_size: YOLO model size
            **kwargs: Additional LitAPI arguments
        """
        super().__init__(**kwargs)
        self.model_size = model_size
        self.service = None
        
    def setup(self, device):
        """
        Initialize the YOLO service on startup
        
        Args:
            device: Device to run inference on
        """
        logger.info(f"Setting up LitServe API with device: {device}")
        
        try:
            # Initialize service with specified device
            self.service = YOLOService(model_size=self.model_size, device=device)
            logger.info("LitServe API setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup LitServe API: {str(e)}")
            raise RuntimeError(f"API setup failed: {str(e)}")

    async def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode and validate incoming request
        
        Args:
            request: Raw request from client
            
        Returns:
            Processed request dictionary
        """
        try:
            # Add default processing options if not specified
            processed_request = {
                "image_data": request,
                "options": {
                    "include_image_info": request.get("include_image_info", False),
                    "include_performance": request.get("include_performance", False),
                    "response_format": request.get("response_format", "detailed")
                }
            }
            
            return processed_request
            
        except Exception as e:
            logger.error(f"Request decoding failed: {str(e)}")
            raise ValueError(f"Invalid request format: {str(e)}")

    async def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference using YOLOService
        
        Args:
            request: Processed request dictionary
            
        Returns:
            Inference results
        """
        try:
            # Extract request components
            image_data = request["image_data"]
            options = request["options"]
            
            # Process image using service
            result = self.service.process_image(
                image_data,
                include_image_info=options["include_image_info"],
                include_performance=options["include_performance"],
                response_format=options["response_format"]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            # Return error response in consistent format
            return self.service.response_formatter.format_error_response(
                f"Prediction failed: {str(e)}",
                error_type="PredictionError"
            )

    async def encode_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode response for client
        
        Args:
            result: Inference result from predict method
            
        Returns:
            Final response dictionary
        """
        try:
            # Add API-specific metadata
            result["api_version"] = "1.0.0"
            result["server_type"] = "litserve"
            
            return result
            
        except Exception as e:
            logger.error(f"Response encoding failed: {str(e)}")
            return self.service.response_formatter.format_error_response(
                f"Response encoding failed: {str(e)}",
                error_type="ResponseEncodingError"
            )

class YOLOv11LitServeManager:
    """
    Manager class for LitServe server operations
    """
    
    def __init__(self, model_size: str = "n"):
        """
        Initialize LitServe manager
        
        Args:
            model_size: YOLO model size
        """
        self.model_size = model_size
        self.api = None
        self.server = None
    
    def create_server(
        self,
        workers_per_device: int = 1,
        timeout: int = 30,
        accelerator: str = "auto"
    ) -> ls.LitServer:
        """
        Create LitServe server instance
        
        Args:
            workers_per_device: Number of workers per device
            timeout: Request timeout in seconds
            accelerator: Accelerator type
            
        Returns:
            LitServer instance
        """
        try:
            # Initialize API
            self.api = YOLOv11LitServeAPI(
                model_size=self.model_size,
                enable_async=True
            )
            
            # Create server
            self.server = ls.LitServer(
                lit_api=self.api,
                accelerator=accelerator,
                workers_per_device=workers_per_device,
                timeout=timeout,
            )
            
            logger.info(f"LitServe server created successfully")
            return self.server
            
        except Exception as e:
            logger.error(f"Failed to create LitServe server: {str(e)}")
            raise RuntimeError(f"Server creation failed: {str(e)}")
    
    def run_server(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers_per_device: int = 1,
        timeout: int = 30
    ) -> None:
        """
        Run LitServe server
        
        Args:
            host: Host address
            port: Port number
            workers_per_device: Number of workers per device
            timeout: Request timeout
        """
        try:
            if not self.server:
                self.create_server(workers_per_device=workers_per_device, timeout=timeout)
            
            logger.info(f"Starting LitServe server on {host}:{port}")
            self.server.run(host=host, port=port, generate_client_file=False)
            
        except Exception as e:
            logger.error(f"Failed to run LitServe server: {str(e)}")
            raise RuntimeError(f"Server run failed: {str(e)}")
    
    def get_server_info(self) -> Dict[str, Any]:
        """
        Get server information
        
        Returns:
            Server information dictionary
        """
        return {
            "server_type": "litserve",
            "model_size": self.model_size,
            "api_initialized": self.api is not None,
            "server_initialized": self.server is not None
        }