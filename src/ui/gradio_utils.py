# src/ui/gradio_utils.py
import base64
import io
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import colorsys

from src.utils.config.settings import get_settings
from src.utils.log.logger import get_module_logger

# Initialize settings and logger
settings = get_settings()
logger = get_module_logger(__name__)

class GradioUtils:
    """Utility functions for Gradio interface"""
    
    @staticmethod
    def encode_image_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        try:
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return img_str
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise
    
    @staticmethod
    def decode_base64_to_image(base64_str: str) -> Image.Image:
        """Convert base64 string to PIL Image"""
        try:
            if base64_str.startswith('data:image'):
                base64_str = base64_str.split(',')[1]
            
            image_bytes = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_bytes))
            return image
        except Exception as e:
            logger.error(f"Error decoding base64 image: {e}")
            raise
    
    @staticmethod
    def validate_image(image: Image.Image) -> Tuple[bool, str]:
        """Validate uploaded image"""
        try:
            if image is None:
                return False, "No image provided"
            
            # Get gradio config
            gradio_config = settings.get_section('gradio')
            if not gradio_config:
                # Use default values if no config
                max_size = 10485760  # 10MB
                min_size = 1024  # 1KB
            else:
                detection_config = gradio_config.get('detection', {})
                max_size = detection_config.get('max_image_size', 10485760)
                min_size = detection_config.get('min_image_size', 1024)
            
            # Check image size
            image_size = len(image.tobytes())
            
            if image_size > max_size:
                return False, f"Image too large. Maximum size: {max_size / 1024 / 1024:.1f}MB"
            
            if image_size < min_size:
                return False, f"Image too small. Minimum size: {min_size}bytes"
            
            # Check dimensions
            if image.width <= 0 or image.height <= 0:
                return False, "Invalid image dimensions"
            
            # Check format
            if image.mode not in ['RGB', 'RGBA', 'L']:
                return False, f"Unsupported image mode: {image.mode}"
            
            return True, "Valid image"
            
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            return False, f"Image validation failed: {str(e)}"
    
    @staticmethod
    def generate_colors(num_colors: int) -> List[str]:
        """Generate distinct colors for bounding boxes"""
        try:
            # Get predefined colors from config
            gradio_config = settings.get_section('gradio')
            predefined_colors = [
                "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#ffeaa7",
                "#dda0dd", "#98d8c8", "#f7dc6f", "#bb8fce", "#85c1e9"
            ]
            
            if gradio_config:
                viz_config = gradio_config.get('visualization', {})
                predefined_colors = viz_config.get('bbox_colors', predefined_colors)
            
            if num_colors <= len(predefined_colors):
                return predefined_colors[:num_colors]
            
            # Generate additional colors using HSV
            colors = predefined_colors.copy()
            
            for i in range(len(predefined_colors), num_colors):
                hue = (i * 137.508) % 360  # Golden angle approximation
                saturation = 0.7 + (i % 3) * 0.1  # Vary saturation
                value = 0.8 + (i % 2) * 0.2  # Vary value
                
                rgb = colorsys.hsv_to_rgb(hue/360, saturation, value)
                hex_color = "#{:02x}{:02x}{:02x}".format(
                    int(rgb[0] * 255),
                    int(rgb[1] * 255),
                    int(rgb[2] * 255)
                )
                colors.append(hex_color)
            
            return colors
            
        except Exception as e:
            logger.error(f"Color generation error: {e}")
            # Fallback to simple colors
            return [f"#{i*50:02x}{i*30:02x}{i*70:02x}" for i in range(num_colors)]
    
    @staticmethod
    def draw_bounding_boxes(image: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
        """Draw bounding boxes on image"""
        try:
            # Get visualization config
            gradio_config = settings.get_section('gradio')
            viz_config = gradio_config.get('visualization', {}) if gradio_config else {}
            
            # Create a copy of the image
            img_with_boxes = image.copy()
            draw = ImageDraw.Draw(img_with_boxes)
            
            # Get visualization settings with defaults
            bbox_thickness = viz_config.get('bbox_thickness', 2)
            font_size = viz_config.get('font_size', 12)
            font_color = viz_config.get('font_color', '#ffffff')
            show_confidence = viz_config.get('show_confidence_in_label', True)
            
            # Get colors
            unique_classes = list(set(det.get('class_name', 'unknown') for det in detections))
            colors = GradioUtils.generate_colors(len(unique_classes))
            class_colors = {cls: colors[i] for i, cls in enumerate(unique_classes)}
            
            # Try to load font
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Draw each detection
            for detection in detections:
                bbox = detection.get('bbox', {})
                class_name = detection.get('class_name', 'unknown')
                confidence = detection.get('confidence', 0.0)
                
                # Get bounding box coordinates
                x1 = int(bbox.get('x1', 0))
                y1 = int(bbox.get('y1', 0))
                x2 = int(bbox.get('x2', 0))
                y2 = int(bbox.get('y2', 0))
                
                # Get color for this class
                color = class_colors.get(class_name, '#ff0000')
                
                # Draw bounding box
                for i in range(bbox_thickness):
                    draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color, width=1)
                
                # Prepare label
                if show_confidence:
                    label = f"{class_name} ({confidence:.2f})"
                else:
                    label = class_name
                
                # Get text size
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
                
                # Draw label background
                label_bg = [x1, y1 - text_height - 4, x1 + text_width + 4, y1]
                draw.rectangle(label_bg, fill=color)
                
                # Draw label text
                draw.text((x1 + 2, y1 - text_height - 2), label, fill=font_color, font=font)
            
            return img_with_boxes
            
        except Exception as e:
            logger.error(f"Error drawing bounding boxes: {e}")
            return image
    
    @staticmethod
    def create_detection_table(detections: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create a pandas DataFrame from detections"""
        try:
            if not detections:
                return pd.DataFrame()
            
            # Extract data for table
            table_data = []
            for i, detection in enumerate(detections):
                bbox = detection.get('bbox', {})
                row = {
                    'ID': i + 1,
                    'Class': detection.get('class_name', 'unknown'),
                    'Confidence': f"{detection.get('confidence', 0.0):.3f}",
                    'X1': int(bbox.get('x1', 0)),
                    'Y1': int(bbox.get('y1', 0)),
                    'X2': int(bbox.get('x2', 0)),
                    'Y2': int(bbox.get('y2', 0)),
                    'Width': int(bbox.get('x2', 0) - bbox.get('x1', 0)),
                    'Height': int(bbox.get('y2', 0) - bbox.get('y1', 0))
                }
                table_data.append(row)
            
            return pd.DataFrame(table_data)
            
        except Exception as e:
            logger.error(f"Error creating detection table: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def create_confidence_chart(detections: List[Dict[str, Any]]) -> go.Figure:
        """Create confidence distribution chart"""
        try:
            if not detections:
                return go.Figure()
            
            # Extract confidence values
            confidences = [det.get('confidence', 0.0) for det in detections]
            class_names = [det.get('class_name', 'unknown') for det in detections]
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=class_names,
                    y=confidences,
                    marker_color='lightblue',
                    text=[f"{c:.3f}" for c in confidences],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Detection Confidence Scores",
                xaxis_title="Class",
                yaxis_title="Confidence",
                yaxis=dict(range=[0, 1]),
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating confidence chart: {e}")
            return go.Figure()
    
    @staticmethod
    def create_class_distribution_chart(detections: List[Dict[str, Any]]) -> go.Figure:
        """Create class distribution pie chart"""
        try:
            if not detections:
                return go.Figure()
            
            # Count classes
            class_counts = {}
            for detection in detections:
                class_name = detection.get('class_name', 'unknown')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Create pie chart
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(class_counts.keys()),
                    values=list(class_counts.values()),
                    hole=0.3
                )
            ])
            
            fig.update_layout(
                title="Class Distribution",
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating class distribution chart: {e}")
            return go.Figure()
    
    @staticmethod
    def format_job_for_display(job: Dict[str, Any]) -> Dict[str, Any]:
        """Format job data for display in Gradio"""
        try:
            return {
                "ID": job.get('id', 'unknown')[:8],
                "Backend": job.get('backend_type', 'unknown').title(),
                "Status": job.get('status', 'unknown').title(),
                "Created": job.get('created_at', '')[:19] if job.get('created_at') else '',
                "Duration": f"{job.get('execution_time', 0):.2f}s" if job.get('execution_time') else 'N/A',
                "Error": job.get('error_message', '')[:50] + '...' if job.get('error_message') and len(job.get('error_message', '')) > 50 else job.get('error_message', '')
            }
        except Exception as e:
            logger.error(f"Error formatting job for display: {e}")
            return {}
    
    @staticmethod
    def create_job_history_table(jobs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create job history table"""
        try:
            if not jobs:
                return pd.DataFrame()
            
            formatted_jobs = [GradioUtils.format_job_for_display(job) for job in jobs]
            return pd.DataFrame(formatted_jobs)
            
        except Exception as e:
            logger.error(f"Error creating job history table: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_backend_status_indicator(health_status: Dict[str, Any]) -> Tuple[str, str]:
        """Get backend status indicator"""
        try:
            is_healthy = health_status.get('is_healthy', False)
            response_time = health_status.get('response_time')
            error_message = health_status.get('error_message')
            
            if is_healthy:
                if response_time is not None:
                    return "🟢", f"Healthy ({response_time:.2f}s)"
                else:
                    return "🟢", "Healthy"
            else:
                if error_message:
                    return "🔴", f"Unhealthy: {error_message[:30]}..."
                else:
                    return "🔴", "Unhealthy"
                    
        except Exception as e:
            logger.error(f"Error getting backend status: {e}")
            return "⚫", "Unknown"
    
    @staticmethod
    def create_notification_message(message_type: str, title: str, message: str) -> str:
        """Create formatted notification message"""
        try:
            # Get notification config
            gradio_config = settings.get_section('gradio') or {}
            notification_config = gradio_config.get('notifications', {})
            
            if not notification_config.get('enabled', True):
                return message
            
            # Get icon for message type
            icons = notification_config.get('types', {})
            icon = icons.get(message_type, {}).get('icon', '')
            
            return f"{icon} {title}: {message}"
            
        except Exception as e:
            logger.error(f"Error creating notification: {e}")
            return message
    
    @staticmethod
    def export_results_to_json(detections: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> str:
        """Export detection results to JSON"""
        try:
            export_data = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "detections": detections,
                "metadata": metadata or {},
                "total_detections": len(detections)
            }
            
            return json.dumps(export_data, indent=2)
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return "{}"
    
    @staticmethod
    def export_results_to_csv(detections: List[Dict[str, Any]]) -> str:
        """Export detection results to CSV"""
        try:
            df = GradioUtils.create_detection_table(detections)
            return df.to_csv(index=False)
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return ""
    
    @staticmethod
    def get_example_images() -> List[str]:
        """Get list of example images"""
        try:
            # Get examples directory from config
            gradio_config = settings.get_section('gradio') or {}
            examples_dir = gradio_config.get('paths', {}).get('examples_dir', './examples')
            
            # This would normally scan the directory for images
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Error getting example images: {e}")
            return []