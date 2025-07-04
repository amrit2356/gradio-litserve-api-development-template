
import torch
from ultralytics import YOLO
from typing import List, Dict, Any
from PIL import Image

from src.utils.config.settings import get_settings
from src.utils.log.logger import get_module_logger

# Initialize settings and logger
settings = get_settings()
logger = get_module_logger(__name__)

class YOLOModel:
    """Handles YOLO model loading, inference, and management"""
    
    def __init__(self, model_size: str = None, device: str = None):
        """
        Initialize YOLO model
        
        Args:
            model_size: Model size (n=nano, s=small, m=medium, l=large, x=extra large)
            device: Device to run on ('auto', 'cpu', 'cuda')
        """
        # Get model configuration from settings
        model_config = settings.model
        
        self.model_size = model_size or model_config.default_size
        self.device = self._get_device(device or model_config.device_preferred)
        self.model = None
        self._is_loaded = False
        
        # Load configuration values
        self.confidence_threshold = model_config.confidence_threshold
        self.iou_threshold = model_config.iou_threshold
        self.max_detections = model_config.max_detections
        self.image_size = model_config.image_size
        self.warmup_enabled = model_config.warmup_enabled
    
    def _get_device(self, device: str) -> str:
        """Determine the appropriate device"""
        if device == "auto":
            if settings.model.device_auto_detect:
                return "cuda" if torch.cuda.is_available() else settings.model.device_fallback
            else:
                return settings.model.device_fallback
        return device
    
    def load_model(self) -> None:
        """Load the YOLO model"""
        if self._is_loaded:
            return
        
        logger.info(f"Loading YOLOv11-{self.model_size} model on device: {self.device}")
        
        try:
            # Get model path from settings
            model_path = settings.model.get_model_path(self.model_size)
            
            self.model = YOLO(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            if self.warmup_enabled:
                self._warmup()
            
            self._is_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def _warmup(self) -> None:
        """Warm up the model with a dummy inference"""
        logger.info("Warming up model...")
        
        # Get warmup configuration from settings
        warmup_size = settings.model.get('warmup.dummy_image_size', [640, 640])
        channels = settings.model.get('warmup.channels', 3)
        
        dummy_image = torch.zeros((1, channels, *warmup_size)).to(self.device)
        
        with torch.no_grad():
            _ = self.model.predict(dummy_image, verbose=False)
        
        logger.info("Model warmed up successfully")
    
    def predict(self, image: Image.Image, conf: float = None, iou: float = None) -> List[Dict[str, Any]]:
        """
        Run inference on an image
        
        Args:
            image: PIL Image to run inference on
            conf: Confidence threshold (uses config default if None)
            iou: IoU threshold for NMS (uses config default if None)
            
        Returns:
            List of detection dictionaries
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Use configuration defaults if not provided
        conf = conf or self.confidence_threshold
        iou = iou or self.iou_threshold
        
        try:
            results = self.model.predict(
                source=image,
                conf=conf,
                iou=iou,
                verbose=False,
                save=False,
                imgsz=self.image_size
            )
            
            return self._extract_detections(results)
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise RuntimeError(f"Inference failed: {str(e)}")
    
    def _extract_detections(self, results) -> List[Dict[str, Any]]:
        """Extract detection data from YOLO results"""
        detections = []
        max_detections = self.max_detections
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                # Limit number of detections
                num_detections = min(len(boxes), max_detections)
                
                for i in range(num_detections):
                    # Get bounding box coordinates
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    # Get class name
                    class_name = self.model.names[cls]
                    
                    detection = {
                        "bbox": {
                            "x1": float(xyxy[0]),
                            "y1": float(xyxy[1]),
                            "x2": float(xyxy[2]),
                            "y2": float(xyxy[3])
                        },
                        "confidence": conf,
                        "class_id": cls,
                        "class_name": class_name
                    }
                    detections.append(detection)
        
        return detections
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._is_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "loaded": self._is_loaded,
            "model_path": f'yolo11{self.model_size}.pt'
        }