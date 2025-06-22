# litserve_yolo.py
import litserve as ls
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOv11API(ls.LitAPI):
    def setup(self, device):
        """Initialize the YOLO model on startup"""
        self.device = device
        logger.info(f"Loading YOLOv11 model on device: {device}")
        
        # Load YOLOv11 model - you can change model size here
        self.model = YOLO('yolo11n.pt')  # n=nano, s=small, m=medium, l=large, x=extra large
        
        # Warm up the model with a dummy inference
        dummy_image = torch.zeros((1, 3, 640, 640)).to(device)
        logger.info("Warming up model...")
        with torch.no_grad():
            _ = self.model.predict(dummy_image, verbose=False)
        logger.info("Model ready for inference")

    def decode_request(self, request: Dict[str, Any]) -> Image.Image:
        """Decode base64 image from request"""
        try:
            # Handle different input formats
            if "image" in request:
                image_data = request["image"]
            elif "file" in request:
                image_data = request["file"]
            else:
                raise ValueError("No image data found in request")
            
            # Decode base64 image
            if isinstance(image_data, str):
                # Remove data URL prefix if present
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
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

    def predict(self, image: Image.Image) -> List[Dict]:
        """Run YOLO inference on the image"""
        try:
            # Run inference
            results = self.model.predict(
                source=image,
                conf=0.25,  # Confidence threshold
                iou=0.45,   # IoU threshold for NMS
                verbose=False,
                save=False
            )
            
            # Extract results
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
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
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise RuntimeError(f"Inference failed: {str(e)}")

    def encode_response(self, detections: List[Dict]) -> Dict[str, Any]:
        """Format the response"""
        return {
            "success": True,
            "detections": detections,
            "count": len(detections),
            "model": "YOLOv11",
            "message": f"Detected {len(detections)} objects"
        }

def create_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """Create and configure the LitServe server"""
    
    # Initialize the API
    api = YOLOv11API()
    
    # Create server with configuration
    server = ls.LitServer(
        api,
        accelerator="auto",  # Will use GPU if available, else CPU
        workers_per_device=workers,
        timeout=30,  # 30 second timeout
        max_batch_size=1,  # Process up to 4 images in a batch
        batch_timeout=0.1,  # 100ms batch timeout for low latency
    )
    
    return server

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv11 LitServe API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers per device")
    
    args = parser.parse_args()
    
    # Create and run server
    server = create_server(host=args.host, port=args.port, workers=args.workers)
    server.run(host=args.host, port=args.port)