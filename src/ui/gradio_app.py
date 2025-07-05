# gradio_app.py - FIXED VERSION
import os
import sys
import time
import json
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import pandas as pd
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import plotly.graph_objects as go

# Import configuration and logging
from src.utils.config.settings import get_settings
from src.utils.log.logger import get_module_logger

# Import clients and utilities
from src.ui.backend_manager import backend_manager, BackendType, JobStatus
from src.ui.gradio_utils import GradioUtils

# Initialize settings and logger
settings = get_settings()
logger = get_module_logger(__name__)

class GradioApp:
    """Main Gradio application class - FIXED VERSION"""
    
    def __init__(self):
        self.backend_manager = backend_manager
        self.utils = GradioUtils()
        
        # Get gradio configuration
        self.gradio_config = settings.get_section('gradio')
        if not self.gradio_config:
            logger.error("No Gradio configuration found")
            # Use default configuration
            self.gradio_config = {
                'app': {'title': 'YOLOv11 Object Detection', 'theme': 'soft'},
                'server': {'host': '0.0.0.0', 'port': 7860},
                'ui': {'components': {}},
                'detection': {
                    'confidence_threshold': {'min': 0.01, 'max': 0.99, 'default': 0.25, 'step': 0.01},
                    'iou_threshold': {'min': 0.01, 'max': 0.99, 'default': 0.45, 'step': 0.01},
                    'max_detections': {'min': 1, 'max': 1000, 'default': 100, 'step': 1}
                },
                'response': {'formats': ['detailed', 'minimal'], 'default_format': 'detailed'}
            }
        
        self.app_config = self.gradio_config.get('app', {})
        self.server_config = self.gradio_config.get('server', {})
        self.ui_config = self.gradio_config.get('ui', {})
        self.detection_config = self.gradio_config.get('detection', {})
        
        # Initialize state
        self.current_job_id = None
        
        logger.info("Gradio app initialized")
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        # Get theme and styling
        theme = self.app_config.get('theme', 'soft')
        title = self.app_config.get('title', 'YOLOv11 Object Detection')
        
        # Custom CSS
        custom_css = """
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-healthy { background-color: #10b981; }
        .status-unhealthy { background-color: #ef4444; }
        .status-unknown { background-color: #6b7280; }
        
        .notification {
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
            font-weight: 500;
        }
        .notification-success { background-color: #d1fae5; color: #065f46; }
        .notification-error { background-color: #fee2e2; color: #991b1b; }
        .notification-warning { background-color: #fef3c7; color: #92400e; }
        .notification-info { background-color: #dbeafe; color: #1e40af; }
        """
        
        with gr.Blocks(theme=theme, css=custom_css, title=title) as app:
            # Header
            with gr.Row():
                gr.Markdown(f"# {title}")
                
            # Backend status
            with gr.Row():
                backend_status = gr.HTML(self._get_backend_status_html())
                refresh_status_btn = gr.Button("🔄 Refresh Status", size="sm")
            
            # Main interface
            with gr.Tab("Object Detection"):
                with gr.Row():
                    # Left column - Input
                    with gr.Column(scale=1):
                        # Image upload
                        image_input = gr.Image(
                            label="Upload Image",
                            type="pil",
                            sources=['upload', 'webcam'],
                            height=400
                        )
                        
                        # Backend selection
                        backend_choice = gr.Radio(
                            choices=self._get_backend_choices(),
                            value=self._get_default_backend(),
                            label="Backend",
                            info="Choose between live server or serverless inference"
                        )
                        
                        # Model settings
                        with gr.Accordion("Model Settings", open=False):
                            confidence_slider = gr.Slider(
                                minimum=self.detection_config.get('confidence_threshold', {}).get('min', 0.01),
                                maximum=self.detection_config.get('confidence_threshold', {}).get('max', 0.99),
                                value=self.detection_config.get('confidence_threshold', {}).get('default', 0.25),
                                step=self.detection_config.get('confidence_threshold', {}).get('step', 0.01),
                                label="Confidence Threshold"
                            )
                            
                            iou_slider = gr.Slider(
                                minimum=self.detection_config.get('iou_threshold', {}).get('min', 0.01),
                                maximum=self.detection_config.get('iou_threshold', {}).get('max', 0.99),
                                value=self.detection_config.get('iou_threshold', {}).get('default', 0.45),
                                step=self.detection_config.get('iou_threshold', {}).get('step', 0.01),
                                label="IoU Threshold"
                            )
                            
                            max_detections_slider = gr.Slider(
                                minimum=self.detection_config.get('max_detections', {}).get('min', 1),
                                maximum=self.detection_config.get('max_detections', {}).get('max', 1000),
                                value=self.detection_config.get('max_detections', {}).get('default', 100),
                                step=self.detection_config.get('max_detections', {}).get('step', 1),
                                label="Max Detections"
                            )
                        
                        # Action buttons
                        with gr.Row():
                            detect_btn = gr.Button("🔍 Detect Objects", variant="primary")
                            clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    
                    # Right column - Output
                    with gr.Column(scale=2):
                        # Status and notifications
                        status_output = gr.HTML()
                        
                        # Results tabs
                        with gr.Tabs():
                            with gr.Tab("Annotated Image"):
                                output_image = gr.Image(label="Detection Results", type="pil")
                            
                            with gr.Tab("Detection Table"):
                                detection_table = gr.Dataframe(
                                    headers=["ID", "Class", "Confidence", "X1", "Y1", "X2", "Y2", "Width", "Height"],
                                    label="Detections"
                                )
                            
                            with gr.Tab("Raw Response"):
                                raw_response = gr.JSON(label="API Response")
            
            # Debug Tab
            with gr.Tab("Debug"):
                with gr.Row():
                    debug_info = gr.JSON(label="Debug Information")
                    refresh_debug_btn = gr.Button("🔄 Refresh Debug Info")
            
            # Event handlers
            self._setup_event_handlers(
                # Input components
                image_input, backend_choice, confidence_slider, iou_slider, max_detections_slider,
                # Buttons
                detect_btn, clear_btn, refresh_status_btn, refresh_debug_btn,
                # Output components
                backend_status, status_output, output_image, detection_table, raw_response, debug_info
            )
        
        return app
    
    def _get_backend_choices(self) -> List[Tuple[str, str]]:
        """Get backend choices for radio button"""
        try:
            choices = []
            available_backends = self.backend_manager.get_available_backends()
            
            for backend in available_backends:
                if backend['enabled']:
                    choices.append((backend['name'], backend['type']))
            
            if not choices:
                choices = [("No backends available", "none")]
                
            logger.info(f"Available backend choices: {choices}")
            return choices
            
        except Exception as e:
            logger.error(f"Error getting backend choices: {e}")
            return [("Error loading backends", "none")]
    
    def _get_default_backend(self) -> str:
        """Get default backend"""
        try:
            choices = self._get_backend_choices()
            if choices and choices[0][1] != "none":
                return choices[0][1]
            return "none"
        except Exception as e:
            logger.error(f"Error getting default backend: {e}")
            return "none"
    
    def _get_backend_status_html(self) -> str:
        """Get backend status HTML"""
        try:
            health_status = self.backend_manager.get_backend_health()
            html_parts = ["<div style='display: flex; gap: 20px; align-items: center;'>"]
            
            for backend_type, status in health_status.items():
                is_healthy = status.get('is_healthy', False)
                error_message = status.get('error_message', '')
                
                if is_healthy:
                    icon = "✅"
                    status_text = "Healthy"
                    color = "#10b981"
                else:
                    icon = "❌"
                    status_text = f"Unhealthy: {error_message}"
                    color = "#ef4444"
                
                backend_name = backend_type.replace('_', ' ').title()
                
                html_parts.append(f"""
                <div style='display: flex; align-items: center; gap: 8px;'>
                    <span style='font-size: 16px;'>{icon}</span>
                    <span><strong>{backend_name}:</strong> <span style='color: {color};'>{status_text}</span></span>
                </div>
                """)
            
            html_parts.append("</div>")
            return "".join(html_parts)
            
        except Exception as e:
            logger.error(f"Error getting backend status: {e}")
            return f"<div style='color: red;'>Error loading backend status: {e}</div>"
    
    def _setup_event_handlers(self, *components):
        """Setup event handlers for all components"""
        try:
            (image_input, backend_choice, confidence_slider, iou_slider, max_detections_slider,
             detect_btn, clear_btn, refresh_status_btn, refresh_debug_btn,
             backend_status, status_output, output_image, detection_table, raw_response, debug_info) = components
            
            # Detection button
            detect_btn.click(
                fn=self.process_detection,
                inputs=[
                    image_input, backend_choice, confidence_slider, iou_slider, max_detections_slider
                ],
                outputs=[
                    status_output, output_image, detection_table, raw_response
                ]
            )
            
            # Clear button
            clear_btn.click(
                fn=self.clear_results,
                inputs=[],
                outputs=[
                    image_input, output_image, detection_table, raw_response, status_output
                ]
            )
            
            # Refresh backend status
            refresh_status_btn.click(
                fn=self.refresh_backend_status,
                inputs=[],
                outputs=[backend_status, status_output]
            )
            
            # Refresh debug info
            refresh_debug_btn.click(
                fn=self.get_debug_info,
                inputs=[],
                outputs=[debug_info]
            )
            
            logger.info("Event handlers setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up event handlers: {e}")
    
    def process_detection(self, image, backend_type, confidence, iou, max_detections):
        """Process object detection request - FIXED VERSION"""
        try:
            logger.info(f"Processing detection request with backend: {backend_type}")
            
            # Validate inputs
            if image is None:
                return self._create_error_outputs("Please upload an image")
            
            if backend_type == "none":
                return self._create_error_outputs("No backend selected")
            
            # Validate image
            if not isinstance(image, Image.Image):
                return self._create_error_outputs("Invalid image format")
            
            # Prepare request data
            request_data = {
                "image": self.utils.encode_image_to_base64(image),
                "confidence_threshold": confidence,
                "iou_threshold": iou,
                "max_detections": int(max_detections)
            }
            
            logger.info(f"Request data prepared, image encoded: {len(request_data['image'])} chars")
            
            # Submit job synchronously
            backend_enum = BackendType(backend_type)
            
            logger.info(f"Submitting job to {backend_enum.value}")
            job_id = self.backend_manager.sync_submit_job(backend_enum, request_data)
            self.current_job_id = job_id
            
            logger.info(f"Job {job_id} submitted successfully")
            
            # Get job result
            job = self.backend_manager.get_job(job_id)
            
            if job and job.status == JobStatus.COMPLETED:
                logger.info(f"Job {job_id} completed successfully")
                return self._create_success_outputs(job.response_data, image)
            elif job and job.status == JobStatus.FAILED:
                logger.error(f"Job {job_id} failed: {job.error_message}")
                return self._create_error_outputs(f"Job failed: {job.error_message}")
            else:
                logger.error(f"Job {job_id} in unexpected state: {job.status if job else 'Not found'}")
                return self._create_error_outputs("Job failed or timed out")
                
        except Exception as e:
            logger.error(f"Detection processing error: {e}")
            return self._create_error_outputs(f"Detection failed: {str(e)}")
    
    def _create_success_outputs(self, result: Dict[str, Any], original_image: Image.Image):
        """Create success outputs for detection"""
        try:
            logger.info("Creating success outputs")
            
            # Extract detections - handle different response formats
            detections = []
            if isinstance(result, dict):
                # Try different possible keys
                if 'detections' in result:
                    detections = result['detections']
                elif 'results' in result and isinstance(result['results'], dict):
                    detections = result['results'].get('detections', [])
                elif 'results' in result and isinstance(result['results'], list):
                    detections = result['results']
            
            logger.info(f"Extracted {len(detections)} detections")
            
            # Create annotated image
            annotated_image = self.utils.draw_bounding_boxes(original_image, detections)
            
            # Create detection table
            detection_df = self.utils.create_detection_table(detections)
            
            # Status message
            status_msg = f"✅ Detection completed: {len(detections)} objects found"
            
            return (
                f"<div class='notification notification-success'>{status_msg}</div>",
                annotated_image,
                detection_df,
                result
            )
            
        except Exception as e:
            logger.error(f"Error creating success outputs: {e}")
            return self._create_error_outputs(f"Error processing results: {str(e)}")
    
    def _create_error_outputs(self, error_message: str):
        """Create error outputs"""
        logger.error(f"Creating error outputs: {error_message}")
        
        return (
            f"<div class='notification notification-error'>❌ {error_message}</div>",
            None,  # output_image
            pd.DataFrame(),  # detection_table
            {"error": error_message}  # raw_response
        )
    
    def clear_results(self):
        """Clear all results"""
        return (
            None,  # image_input
            None,  # output_image
            pd.DataFrame(),  # detection_table
            {},  # raw_response
            ""   # status_output
        )
    
    def refresh_backend_status(self):
        """Refresh backend status"""
        try:
            logger.info("Refreshing backend status")
            status_html = self._get_backend_status_html()
            
            return (
                status_html,
                "<div class='notification notification-info'>✅ Backend status refreshed</div>"
            )
            
        except Exception as e:
            logger.error(f"Error refreshing backend status: {e}")
            return (
                f"<div style='color: red;'>Error: {e}</div>",
                f"<div class='notification notification-error'>❌ Refresh failed: {e}</div>"
            )
    
    def get_debug_info(self):
        """Get debug information"""
        try:
            debug_data = {
                "backend_manager": {
                    "available_backends": self.backend_manager.get_available_backends(),
                    "health_status": self.backend_manager.get_backend_health(),
                    "total_jobs": len(self.backend_manager.jobs)
                },
                "configuration": {
                    "gradio_config": self.gradio_config,
                    "backends_config": self.backend_manager.backends_config
                },
                "current_job_id": self.current_job_id,
                "timestamp": datetime.now().isoformat()
            }
            
            return debug_data
            
        except Exception as e:
            logger.error(f"Error getting debug info: {e}")
            return {"error": str(e)}
    
    def run(self):
        """Run the Gradio app"""
        try:
            app = self.create_interface()
            
            # Get server configuration
            host = self.server_config.get('host', '0.0.0.0')
            port = self.server_config.get('port', 7860)
            share = self.server_config.get('share', False)
            debug = self.server_config.get('debug', False)
            
            logger.info(f"Starting Gradio app on {host}:{port}")
            
            # Start the app
            app.launch(
                server_name=host,
                server_port=port,
                share=share,
                debug=debug,
                show_api=True,
                show_error=True
            )
            
        except Exception as e:
            logger.error(f"Failed to start Gradio app: {e}")
            raise

def main():
    """Main entry point"""
    try:
        app = GradioApp()
        app.run()
    except KeyboardInterrupt:
        logger.info("Gradio app interrupted by user")
    except Exception as e:
        logger.error(f"Gradio app failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()