# gradio_yolo_frontend.py
import gradio as gr
import requests
import base64
import io
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOGradioApp:
    def __init__(self, litserve_url: str = "http://localhost:8000"):
        self.litserve_url = litserve_url.rstrip('/')
        self.colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
            "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
        ]
    
    def encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def draw_detections(self, image: Image.Image, detections: list) -> Image.Image:
        """Draw bounding boxes and labels on image"""
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        for i, detection in enumerate(detections):
            bbox = detection["bbox"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            
            # Get color for this class
            color = self.colors[i % len(self.colors)]
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Prepare label
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate text size and position
            bbox_coords = draw.textbbox((x1, y1), label, font=font)
            text_width = bbox_coords[2] - bbox_coords[0]
            text_height = bbox_coords[3] - bbox_coords[1]
            
            # Draw label background
            draw.rectangle([x1, y1-text_height-5, x1+text_width+10, y1], fill=color)
            
            # Draw label text
            draw.text((x1+5, y1-text_height-2), label, fill="white", font=font)
        
        return image
    
    def predict(self, image: Image.Image, confidence_threshold: float = 0.25) -> Tuple[Optional[Image.Image], str]:
        """Send image to LitServe backend and return annotated image with results"""
        if image is None:
            return None, "Please upload an image first."
        
        try:
            # Encode image
            image_b64 = self.encode_image(image)
            
            # Prepare request
            payload = {
                "image": image_b64
            }
            
            # Send request to LitServe
            response = requests.post(
                f"{self.litserve_url}/predict",
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                return None, f"Error: Server returned status {response.status_code}"
            
            result = response.json()
            
            if not result.get("success", False):
                return None, f"Prediction failed: {result.get('message', 'Unknown error')}"
            
            detections = result.get("detections", [])
            
            # Filter detections by confidence threshold
            filtered_detections = [
                det for det in detections 
                if det["confidence"] >= confidence_threshold
            ]
            
            # Draw detections on image
            annotated_image = image.copy()
            if filtered_detections:
                annotated_image = self.draw_detections(annotated_image, filtered_detections)
            
            # Prepare results text
            results_text = self.format_results(filtered_detections)
            
            return annotated_image, results_text
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Connection error: {str(e)}\nMake sure LitServe is running at {self.litserve_url}"
            return None, error_msg
        except Exception as e:
            return None, f"Unexpected error: {str(e)}"
    
    def format_results(self, detections: list) -> str:
        """Format detection results as readable text"""
        if not detections:
            return "No objects detected."
        
        results = [f"Detected {len(detections)} object(s):\n"]
        
        for i, detection in enumerate(detections, 1):
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            bbox = detection["bbox"]
            
            results.append(
                f"{i}. {class_name} (confidence: {confidence:.3f})\n"
                f"   Location: ({bbox['x1']:.0f}, {bbox['y1']:.0f}) to "
                f"({bbox['x2']:.0f}, {bbox['y2']:.0f})"
            )
        
        return "\n".join(results)
    
    def test_connection(self) -> str:
        """Test connection to LitServe backend"""
        try:
            response = requests.get(f"{self.litserve_url}/health", timeout=5)
            if response.status_code == 200:
                return f"✅ Connected to LitServe at {self.litserve_url}"
            else:
                return f"❌ LitServe responded with status {response.status_code}"
        except requests.exceptions.RequestException:
            return f"❌ Cannot connect to LitServe at {self.litserve_url}"
    
    def create_interface(self) -> gr.Interface:
        """Create the Gradio interface"""
        
        with gr.Blocks(title="YOLOv11 Object Detection", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🔍 YOLOv11 Object Detection")
            gr.Markdown("Upload an image to detect objects using YOLOv11 model")
            
            # Connection status
            with gr.Row():
                status_btn = gr.Button("Test Connection", variant="secondary")
                status_output = gr.Textbox(label="Connection Status", interactive=False)
            
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        label="Upload Image",
                        type="pil",
                        sources=["upload", "webcam"],
                        height=400
                    )
                    
                    confidence_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.25,
                        step=0.05,
                        label="Confidence Threshold"
                    )
                    
                    predict_btn = gr.Button("🚀 Detect Objects", variant="primary", size="lg")
                
                with gr.Column():
                    image_output = gr.Image(
                        label="Detection Results",
                        type="pil",
                        height=400
                    )
                    
                    results_output = gr.Textbox(
                        label="Detection Details",
                        lines=10,
                        max_lines=15
                    )
            
            # Event handlers
            predict_btn.click(
                fn=self.predict,
                inputs=[image_input, confidence_slider],
                outputs=[image_output, results_output]
            )
            
            status_btn.click(
                fn=self.test_connection,
                outputs=status_output
            )
            
            # Auto-test connection on load
            interface.load(
                fn=self.test_connection,
                outputs=status_output
            )
            
            # Examples section
            gr.Markdown("## 📸 Example Images")
            gr.Examples(
                examples=[
                    ["https://ultralytics.com/images/bus.jpg"],
                    ["https://ultralytics.com/images/zidane.jpg"],
                ],
                inputs=image_input
            )
        
        return interface

def create_app(litserve_url: str = "http://localhost:8000", share: bool = False, port: int = 7860):
    """Create and launch the Gradio app"""
    app = YOLOGradioApp(litserve_url)
    interface = app.create_interface()
    
    return interface

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv11 Gradio Frontend")
    parser.add_argument("--litserve-url", default="http://localhost:8000", 
                       help="URL of the LitServe backend")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio app")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    
    args = parser.parse_args()
    
    # Create and launch app
    app = YOLOGradioApp(args.litserve_url)
    interface = app.create_interface()
    
    print(f"Starting Gradio app on port {args.port}")
    print(f"LitServe backend: {args.litserve_url}")
    
    interface.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0"
    )
    