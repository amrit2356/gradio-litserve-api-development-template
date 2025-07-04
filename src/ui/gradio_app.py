# gradio_app.py
import os
import sys
import asyncio
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
    """Main Gradio application class"""
    
    def __init__(self):
        self.backend_manager = backend_manager
        self.utils = GradioUtils()
        
        # Get gradio configuration
        self.gradio_config = settings.get_section('gradio')
        if not self.gradio_config:
            logger.error("No Gradio configuration found")
            raise ValueError("Gradio configuration is missing")
        
        self.app_config = self.gradio_config.get('app', {})
        self.server_config = self.gradio_config.get('server', {})
        self.ui_config = self.gradio_config.get('ui', {})
        self.detection_config = self.gradio_config.get('detection', {})
        
        # Initialize state
        self.current_job_id = None
        self.polling_active = False
        
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
        
        .job-status-pending { color: #f59e0b; }
        .job-status-running { color: #3b82f6; }
        .job-status-completed { color: #10b981; }
        .job-status-failed { color: #ef4444; }
        .job-status-cancelled { color: #6b7280; }
        
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
                            sources=self.ui_config.get('components', {}).get('image_upload', {}).get('sources', ['upload', 'webcam']),
                            height=self.ui_config.get('components', {}).get('image_upload', {}).get('height', 400)
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
                        
                        # Output settings
                        with gr.Accordion("Output Settings", open=False):
                            response_format = gr.Radio(
                                choices=self.gradio_config.get('response', {}).get('formats', ['detailed', 'minimal']),
                                value=self.gradio_config.get('response', {}).get('default_format', 'detailed'),
                                label="Response Format"
                            )
                            
                            include_metadata = gr.Checkbox(
                                value=self.gradio_config.get('response', {}).get('include_metadata', True),
                                label="Include Metadata"
                            )
                            
                            include_performance = gr.Checkbox(
                                value=self.gradio_config.get('response', {}).get('include_performance', True),
                                label="Include Performance Metrics"
                            )
                        
                        # Action buttons
                        with gr.Row():
                            detect_btn = gr.Button("🔍 Detect Objects", variant="primary")
                            clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    
                    # Right column - Output
                    with gr.Column(scale=2):
                        # Status and notifications
                        status_output = gr.HTML()
                        notification_output = gr.HTML()
                        
                        # Results tabs
                        with gr.Tabs():
                            with gr.Tab("Annotated Image"):
                                output_image = gr.Image(label="Detection Results", type="pil")
                            
                            with gr.Tab("Detection Table"):
                                detection_table = gr.Dataframe(
                                    headers=["ID", "Class", "Confidence", "X1", "Y1", "X2", "Y2", "Width", "Height"],
                                    label="Detections"
                                )
                            
                            with gr.Tab("Statistics"):
                                with gr.Row():
                                    confidence_chart = gr.Plot(label="Confidence Distribution")
                                    class_chart = gr.Plot(label="Class Distribution")
                            
                            with gr.Tab("Raw Response"):
                                raw_response = gr.JSON(label="API Response")
                        
                        # Export options
                        with gr.Accordion("Export Results", open=False):
                            with gr.Row():
                                export_json_btn = gr.Button("📄 Export JSON")
                                export_csv_btn = gr.Button("📊 Export CSV")
                            
                            export_output = gr.File(label="Download")
            
            # Job Management Tab
            with gr.Tab("Job Management"):
                with gr.Row():
                    job_refresh_btn = gr.Button("🔄 Refresh Jobs")
                    job_clear_btn = gr.Button("🗑️ Clear Completed")
                
                # Job queue
                with gr.Row():
                    with gr.Column(scale=2):
                        job_queue = gr.Dataframe(
                            headers=["ID", "Backend", "Status", "Created", "Duration", "Error"],
                            label="Job Queue"
                        )
                    
                    with gr.Column(scale=1):
                        job_details = gr.JSON(label="Job Details")
                        cancel_job_btn = gr.Button("❌ Cancel Selected Job", variant="stop")
                
                # Job polling
                job_polling_checkbox = gr.Checkbox(
                    value=False,
                    label="Auto-refresh job status",
                    info="Automatically refresh job status every 2 seconds"
                )
            
            # Settings Tab
            with gr.Tab("Settings"):
                with gr.Accordion("Backend Configuration", open=True):
                    backend_config_display = gr.JSON(
                        value=self._get_backend_config_display(),
                        label="Backend Configuration"
                    )
                
                with gr.Accordion("App Configuration", open=False):
                    app_config_display = gr.JSON(
                        value=self.app_config,
                        label="App Configuration"
                    )
                
                with gr.Accordion("Detection Settings", open=False):
                    detection_config_display = gr.JSON(
                        value=self.detection_config,
                        label="Detection Configuration"
                    )
            
            # Event handlers
            self._setup_event_handlers(
                # Input components
                image_input, backend_choice, confidence_slider, iou_slider, max_detections_slider,
                response_format, include_metadata, include_performance,
                # Buttons
                detect_btn, clear_btn, refresh_status_btn, export_json_btn, export_csv_btn,
                job_refresh_btn, job_clear_btn, cancel_job_btn,
                # Output components
                backend_status, status_output, notification_output, output_image,
                detection_table, confidence_chart, class_chart, raw_response,
                job_queue, job_details, job_polling_checkbox, export_output
            )
        
        return app
    
    def _get_backend_choices(self) -> List[Tuple[str, str]]:
        """Get backend choices for radio button"""
        choices = []
        available_backends = self.backend_manager.get_available_backends()
        
        for backend in available_backends:
            if backend['enabled']:
                choices.append((backend['name'], backend['type']))
        
        return choices if choices else [("No backends available", "none")]
    
    def _get_default_backend(self) -> str:
        """Get default backend"""
        choices = self._get_backend_choices()
        if choices and choices[0][1] != "none":
            return choices[0][1]
        return "none"
    
    def _get_backend_status_html(self) -> str:
        """Get backend status HTML"""
        health_status = self.backend_manager.get_backend_health()
        html_parts = ["<div style='display: flex; gap: 20px; align-items: center;'>"]
        
        for backend_type, status in health_status.items():
            icon, status_text = self.utils.get_backend_status_indicator(status)
            backend_name = backend_type.replace('_', ' ').title()
            
            html_parts.append(f"""
            <div style='display: flex; align-items: center; gap: 8px;'>
                <span style='font-size: 16px;'>{icon}</span>
                <span><strong>{backend_name}:</strong> {status_text}</span>
            </div>
            """)
        
        html_parts.append("</div>")
        return "".join(html_parts)
    
    def _get_backend_config_display(self) -> Dict[str, Any]:
        """Get backend configuration for display"""
        return {
            "available_backends": self.backend_manager.get_available_backends(),
            "health_status": self.backend_manager.get_backend_health()
        }
    
    def _setup_event_handlers(self, *components):
        """Setup event handlers for all components"""
        (image_input, backend_choice, confidence_slider, iou_slider, max_detections_slider,
         response_format, include_metadata, include_performance,
         detect_btn, clear_btn, refresh_status_btn, export_json_btn, export_csv_btn,
         job_refresh_btn, job_clear_btn, cancel_job_btn,
         backend_status, status_output, notification_output, output_image,
         detection_table, confidence_chart, class_chart, raw_response,
         job_queue, job_details, job_polling_checkbox, export_output) = components
        
        # Detection button
        detect_btn.click(
            fn=self.process_detection,
            inputs=[
                image_input, backend_choice, confidence_slider, iou_slider, max_detections_slider,
                response_format, include_metadata, include_performance
            ],
            outputs=[
                status_output, notification_output, output_image, detection_table,
                confidence_chart, class_chart, raw_response
            ]
        )
        
        # Clear button
        clear_btn.click(
            fn=self.clear_results,
            inputs=[],
            outputs=[
                image_input, output_image, detection_table, confidence_chart,
                class_chart, raw_response, status_output, notification_output
            ]
        )
        
        # Refresh backend status
        refresh_status_btn.click(
            fn=self.refresh_backend_status,
            inputs=[],
            outputs=[backend_status, notification_output]
        )
        
        # Export buttons
        export_json_btn.click(
            fn=self.export_json,
            inputs=[raw_response],
            outputs=[export_output, notification_output]
        )
        
        export_csv_btn.click(
            fn=self.export_csv,
            inputs=[detection_table],
            outputs=[export_output, notification_output]
        )
        
        # Job management
        job_refresh_btn.click(
            fn=self.refresh_jobs,
            inputs=[],
            outputs=[job_queue, notification_output]
        )
        
        job_clear_btn.click(
            fn=self.clear_completed_jobs,
            inputs=[],
            outputs=[job_queue, notification_output]
        )
        
        # Job selection
        job_queue.select(
            fn=self.select_job,
            inputs=[job_queue],
            outputs=[job_details]
        )
        
        cancel_job_btn.click(
            fn=self.cancel_selected_job,
            inputs=[job_details],
            outputs=[job_queue, notification_output]
        )
        
        # Job polling
        job_polling_checkbox.change(
            fn=self.toggle_job_polling,
            inputs=[job_polling_checkbox],
            outputs=[notification_output]
        )
    
    def process_detection(self, image, backend_type, confidence, iou, max_detections, 
                         response_format, include_metadata, include_performance):
        """Process object detection request"""
        try:
            # Validate inputs
            if image is None:
                return self._create_error_outputs("Please upload an image")
            
            if backend_type == "none":
                return self._create_error_outputs("No backend selected")
            
            # Validate image
            is_valid, validation_message = self.utils.validate_image(image)
            if not is_valid:
                return self._create_error_outputs(validation_message)
            
            # Prepare request data
            request_data = {
                "image": self.utils.encode_image_to_base64(image),
                "confidence_threshold": confidence,
                "iou_threshold": iou,
                "max_detections": int(max_detections),
                "response_format": response_format,
                "include_metadata": include_metadata,
                "include_performance": include_performance
            }
            
            # Submit job synchronously
            backend_enum = BackendType(backend_type)
            job_id = self.backend_manager.sync_submit_job(backend_enum, request_data)
            self.current_job_id = job_id
            
            # Wait for job completion (with timeout)
            result = self._wait_for_job_completion(job_id, timeout=60)
            
            if result:
                return self._create_success_outputs(result, image)
            else:
                return self._create_error_outputs("Job failed or timed out")
                
        except Exception as e:
            logger.error(f"Detection processing error: {e}")
            return self._create_error_outputs(f"Detection failed: {str(e)}")
    
    def _wait_for_job_completion(self, job_id: str, timeout: int = 60) -> Optional[Dict[str, Any]]:
        """Wait for job completion with polling"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            job = self.backend_manager.get_job(job_id)
            if not job:
                return None
            
            if job.status == JobStatus.COMPLETED:
                return job.response_data
            elif job.status == JobStatus.FAILED:
                return None
            
            time.sleep(1)  # Poll every second
        
        return None
    
    def _create_success_outputs(self, result: Dict[str, Any], original_image: Image.Image):
        """Create success outputs for detection"""
        try:
            # Extract detections
            detections = result.get('results', {}).get('detections', [])
            if not detections:
                detections = result.get('detections', [])
            
            # Create annotated image
            annotated_image = self.utils.draw_bounding_boxes(original_image, detections)
            
            # Create detection table
            detection_df = self.utils.create_detection_table(detections)
            
            # Create charts
            confidence_chart = self.utils.create_confidence_chart(detections)
            class_chart = self.utils.create_class_distribution_chart(detections)
            
            # Status message
            status_msg = f"✅ Detection completed: {len(detections)} objects found"
            notification_msg = self.utils.create_notification_message(
                "success", "Detection Complete", f"Found {len(detections)} objects"
            )
            
            return (
                f"<div class='notification notification-success'>{status_msg}</div>",
                f"<div class='notification notification-success'>{notification_msg}</div>",
                annotated_image,
                detection_df,
                confidence_chart,
                class_chart,
                result
            )
            
        except Exception as e:
            logger.error(f"Error creating success outputs: {e}")
            return self._create_error_outputs(f"Error processing results: {str(e)}")
    
    def _create_error_outputs(self, error_message: str):
        """Create error outputs"""
        error_notification = self.utils.create_notification_message(
            "error", "Error", error_message
        )
        
        return (
            f"<div class='notification notification-error'>❌ {error_message}</div>",
            f"<div class='notification notification-error'>{error_notification}</div>",
            None,  # output_image
            pd.DataFrame(),  # detection_table
            go.Figure(),  # confidence_chart
            go.Figure(),  # class_chart
            {"error": error_message}  # raw_response
        )
    
    def clear_results(self):
        """Clear all results"""
        return (
            None,  # image_input
            None,  # output_image
            pd.DataFrame(),  # detection_table
            go.Figure(),  # confidence_chart
            go.Figure(),  # class_chart
            {},  # raw_response
            "",  # status_output
            ""   # notification_output
        )
    
    def refresh_backend_status(self):
        """Refresh backend status"""
        try:
            status_html = self._get_backend_status_html()
            notification = self.utils.create_notification_message(
                "info", "Status Refreshed", "Backend status updated"
            )
            
            return (
                status_html,
                f"<div class='notification notification-info'>{notification}</div>"
            )
            
        except Exception as e:
            logger.error(f"Error refreshing backend status: {e}")
            error_notification = self.utils.create_notification_message(
                "error", "Refresh Failed", str(e)
            )
            return (
                self._get_backend_status_html(),
                f"<div class='notification notification-error'>{error_notification}</div>"
            )
    
    async def _refresh_all_backends(self):
        """Refresh all backend health checks"""
        tasks = []
        
        if self.backend_manager.litserve_client:
            tasks.append(self.backend_manager.litserve_client.health_check())
        
        if self.backend_manager.runpod_client:
            tasks.append(self.backend_manager.runpod_client.health_check())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def export_json(self, raw_response):
        """Export results to JSON"""
        try:
            if not raw_response or raw_response.get('error'):
                return None, self._create_export_error_notification("No results to export")
            
            # Create JSON export
            json_data = self.utils.export_results_to_json(
                raw_response.get('results', {}).get('detections', []),
                raw_response.get('metadata', {})
            )
            
            # Create temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_results_{timestamp}.json"
            
            with open(filename, 'w') as f:
                f.write(json_data)
            
            notification = self.utils.create_notification_message(
                "success", "Export Complete", f"Results exported to {filename}"
            )
            
            return (
                filename,
                f"<div class='notification notification-success'>{notification}</div>"
            )
            
        except Exception as e:
            logger.error(f"JSON export error: {e}")
            return None, self._create_export_error_notification(str(e))
    
    def export_csv(self, detection_table):
        """Export results to CSV"""
        try:
            if detection_table.empty:
                return None, self._create_export_error_notification("No results to export")
            
            # Create CSV export
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_results_{timestamp}.csv"
            
            detection_table.to_csv(filename, index=False)
            
            notification = self.utils.create_notification_message(
                "success", "Export Complete", f"Results exported to {filename}"
            )
            
            return (
                filename,
                f"<div class='notification notification-success'>{notification}</div>"
            )
            
        except Exception as e:
            logger.error(f"CSV export error: {e}")
            return None, self._create_export_error_notification(str(e))
    
    def _create_export_error_notification(self, error_message: str):
        """Create export error notification"""
        notification = self.utils.create_notification_message(
            "error", "Export Failed", error_message
        )
        return f"<div class='notification notification-error'>{notification}</div>"
    
    def refresh_jobs(self):
        """Refresh job queue"""
        try:
            jobs = self.backend_manager.get_jobs()
            job_data = [job.to_dict() for job in jobs]
            job_df = self.utils.create_job_history_table(job_data)
            
            notification = self.utils.create_notification_message(
                "info", "Jobs Refreshed", f"Found {len(jobs)} jobs"
            )
            
            return (
                job_df,
                f"<div class='notification notification-info'>{notification}</div>"
            )
            
        except Exception as e:
            logger.error(f"Error refreshing jobs: {e}")
            error_notification = self.utils.create_notification_message(
                "error", "Refresh Failed", str(e)
            )
            return (
                pd.DataFrame(),
                f"<div class='notification notification-error'>{error_notification}</div>"
            )
    
    def clear_completed_jobs(self):
        """Clear completed jobs"""
        try:
            # Clear completed jobs from local storage
            jobs_to_remove = []
            for job_id, job in self.backend_manager.jobs.items():
                if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self.backend_manager.jobs[job_id]
            
            # Refresh job list
            jobs = self.backend_manager.get_jobs()
            job_data = [job.to_dict() for job in jobs]
            job_df = self.utils.create_job_history_table(job_data)
            
            notification = self.utils.create_notification_message(
                "success", "Jobs Cleared", f"Cleared {len(jobs_to_remove)} completed jobs"
            )
            
            return (
                job_df,
                f"<div class='notification notification-success'>{notification}</div>"
            )
            
        except Exception as e:
            logger.error(f"Error clearing jobs: {e}")
            error_notification = self.utils.create_notification_message(
                "error", "Clear Failed", str(e)
            )
            return (
                pd.DataFrame(),
                f"<div class='notification notification-error'>{error_notification}</div>"
            )
    
    def select_job(self, job_queue_data):
        """Select a job from the queue"""
        try:
            # This would extract the selected job from the dataframe
            # For now, return empty details
            return {}
            
        except Exception as e:
            logger.error(f"Error selecting job: {e}")
            return {}
    
    def cancel_selected_job(self, job_details):
        """Cancel the selected job"""
        try:
            if not job_details or 'id' not in job_details:
                return pd.DataFrame(), self._create_job_error_notification("No job selected")
            
            job_id = job_details['id']
            
            # Cancel job locally
            job = self.backend_manager.jobs.get(job_id)
            if job and job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
                job.status = JobStatus.CANCELLED
                job.updated_at = datetime.now()
                
                # Refresh job list
                jobs = self.backend_manager.get_jobs()
                job_data = [job.to_dict() for job in jobs]
                job_df = self.utils.create_job_history_table(job_data)
                
                notification = self.utils.create_notification_message(
                    "success", "Job Cancelled", f"Job {job_id[:8]} has been cancelled"
                )
                
                return (
                    job_df,
                    f"<div class='notification notification-success'>{notification}</div>"
                )
            else:
                return pd.DataFrame(), self._create_job_error_notification("Job cannot be cancelled")
                
        except Exception as e:
            logger.error(f"Error cancelling job: {e}")
            return pd.DataFrame(), self._create_job_error_notification(str(e))
    
    def toggle_job_polling(self, enable_polling):
        """Toggle job polling"""
        try:
            self.polling_active = enable_polling
            
            if enable_polling:
                # Start polling (this would typically use a separate thread)
                notification = self.utils.create_notification_message(
                    "info", "Polling Enabled", "Job status will auto-refresh every 2 seconds"
                )
            else:
                notification = self.utils.create_notification_message(
                    "info", "Polling Disabled", "Job status auto-refresh disabled"
                )
            
            return f"<div class='notification notification-info'>{notification}</div>"
            
        except Exception as e:
            logger.error(f"Error toggling polling: {e}")
            return self._create_job_error_notification(str(e))
    
    def _create_job_error_notification(self, error_message: str):
        """Create job error notification"""
        notification = self.utils.create_notification_message(
            "error", "Job Error", error_message
        )
        return f"<div class='notification notification-error'>{notification}</div>"
    
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
            
            # Start the app - this will create an event loop
            app.launch(
                server_name=host,
                server_port=port,
                share=share,
                debug=debug,
                show_api=self.server_config.get('show_api', True),
                show_error=self.server_config.get('show_error', True),
                max_threads=self.server_config.get('max_threads', 40)
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