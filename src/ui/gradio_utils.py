# src/ui/gradio_utils.py - FIXED VERSION for API Response Format
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
import os
from pathlib import Path

from src.utils.config.settings import get_settings
from src.utils.log.logger import get_module_logger

# Initialize settings and logger
settings = get_settings()
logger = get_module_logger(__name__)

class GradioUtils:
    """Utility functions for Gradio interface - FIXED VERSION"""
    
    @staticmethod
    def encode_image_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string with size optimization using configuration"""
        try:
            # Get settings from configuration
            settings = get_settings()
            
            # Make a copy to avoid modifying the original
            img_copy = image.copy()
            
            # Convert to RGB if it's not already
            if img_copy.mode != 'RGB':
                img_copy = img_copy.convert('RGB')
            
            # Get auto resize setting from image processing config
            auto_resize = settings.image_processing.auto_resize
            maintain_aspect_ratio = settings.image_processing.maintain_aspect_ratio
            
            # Resize if image is too large and auto_resize is enabled
            max_dimension = 1920  # Reasonable max for web usage
            if auto_resize and (img_copy.width > max_dimension or img_copy.height > max_dimension):
                if maintain_aspect_ratio:
                    # Calculate resize ratio
                    ratio = min(max_dimension / img_copy.width, max_dimension / img_copy.height)
                    new_width = int(img_copy.width * ratio)
                    new_height = int(img_copy.height * ratio)
                else:
                    # Just cap the dimensions
                    new_width = min(img_copy.width, max_dimension)
                    new_height = min(img_copy.height, max_dimension)
                
                # Resize with high quality
                img_copy = img_copy.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized image from {image.width}x{image.height} to {new_width}x{new_height}")
            
            buffer = io.BytesIO()
            # Use JPEG with good quality to balance size and quality
            img_copy.save(buffer, format='JPEG', quality=85, optimize=True)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            logger.info(f"Encoded image to base64: {len(img_str)} characters")
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
        """Validate uploaded image using configuration settings"""
        try:
            if image is None:
                return False, "No image provided"
            
            # Get settings from configuration
            settings = get_settings()
            
            # Get validation settings
            strict_mode = settings.image_processing.get('validation.strict_mode', True)
            check_dimensions = settings.image_processing.get('validation.check_dimensions', True)
            check_file_size = settings.image_processing.get('validation.check_file_size', True)
            
            # If not in strict mode, only do basic validation
            if not strict_mode:
                # Basic validation - just check if image exists and has valid mode
                if image.mode not in ['RGB', 'RGBA', 'L', 'P']:
                    try:
                        image = image.convert('RGB')
                        logger.info(f"Converted image from {image.mode} to RGB (non-strict mode)")
                    except:
                        return False, f"Cannot convert image mode: {image.mode}"
                
                logger.info(f"Image validation passed (non-strict mode): {image.width}x{image.height}, {image.mode}")
                return True, "Valid image (non-strict validation)"
            
            # Strict mode validation
            validation_errors = []
            
            # Check dimensions if enabled
            if check_dimensions:
                max_width = settings.image_processing.max_width
                max_height = settings.image_processing.max_height
                min_width = settings.image_processing.min_width
                min_height = settings.image_processing.min_height
                
                if image.width > max_width or image.height > max_height:
                    validation_errors.append(f"Image too large. Maximum size: {max_width}x{max_height} pixels")
                
                if image.width < min_width or image.height < min_height:
                    validation_errors.append(f"Image too small. Minimum size: {min_width}x{min_height} pixels")
                
                # Check for reasonable pixel count (to prevent extremely large images)
                total_pixels = image.width * image.height
                max_pixels = max_width * max_height
                
                if total_pixels > max_pixels:
                    validation_errors.append(f"Image has too many pixels: {total_pixels:,}. Maximum: {max_pixels:,}")
            else:
                logger.info("Dimension checking disabled in configuration")
            
            # Check file size if enabled
            if check_file_size:
                max_image_size = settings.image_processing.max_image_size
                min_image_size = settings.image_processing.min_image_size
                
                # Estimate file size from image data
                try:
                    # Create a temporary buffer to estimate compressed size
                    buffer = io.BytesIO()
                    temp_img = image.copy()
                    if temp_img.mode != 'RGB':
                        temp_img = temp_img.convert('RGB')
                    temp_img.save(buffer, format='JPEG', quality=85)
                    estimated_size = len(buffer.getvalue())
                    
                    if estimated_size > max_image_size:
                        validation_errors.append(f"Image file too large. Estimated size: {estimated_size / 1024 / 1024:.1f}MB, Maximum: {max_image_size / 1024 / 1024:.1f}MB")
                    
                    if estimated_size < min_image_size:
                        validation_errors.append(f"Image file too small. Estimated size: {estimated_size}bytes, Minimum: {min_image_size}bytes")
                        
                except Exception as e:
                    logger.warning(f"Could not estimate file size: {e}")
            else:
                logger.info("File size checking disabled in configuration")
            
            # Check format and convert if necessary
            supported_formats = settings.image_processing.supported_formats
            if image.format and image.format not in supported_formats:
                logger.warning(f"Image format {image.format} not in supported formats, converting to RGB")
            
            if image.mode not in ['RGB', 'RGBA', 'L']:
                try:
                    image = image.convert('RGB')
                    logger.info(f"Converted image from {image.mode} to RGB")
                except:
                    validation_errors.append(f"Cannot convert image mode: {image.mode}")
            
            # Return validation results
            if validation_errors:
                error_message = "; ".join(validation_errors)
                logger.error(f"Image validation failed: {error_message}")
                return False, error_message
            
            logger.info(f"Image validation passed (strict mode): {image.width}x{image.height}, {image.mode}")
            return True, "Valid image"
            
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            return False, f"Image validation failed: {str(e)}"
    
    @staticmethod
    def parse_api_response(api_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse API response and convert to standardized detection format"""
        try:
            detections = []
            
            # Handle the specific API response format you provided
            if 'results' in api_response and 'detections' in api_response['results']:
                raw_detections = api_response['results']['detections']
                
                for detection in raw_detections:
                    # Extract class information
                    class_info = detection.get('class', {})
                    class_name = class_info.get('name', 'unknown')
                    class_id = class_info.get('id', 0)
                    confidence = class_info.get('confidence', 0.0)
                    
                    # Extract bounding box
                    bbox = detection.get('bounding_box', {})
                    
                    # Convert to standardized format
                    standardized_detection = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': {
                            'x1': float(bbox.get('x1', 0)),
                            'y1': float(bbox.get('y1', 0)),
                            'x2': float(bbox.get('x2', 0)),
                            'y2': float(bbox.get('y2', 0)),
                            'width': float(bbox.get('width', 0)),
                            'height': float(bbox.get('height', 0))
                        },
                        'object_id': detection.get('object_id')
                    }
                    
                    detections.append(standardized_detection)
            
            # Handle other possible formats
            elif 'detections' in api_response:
                detections = api_response['detections']
            elif 'results' in api_response and isinstance(api_response['results'], list):
                detections = api_response['results']
            else:
                logger.warning(f"Unknown API response format: {list(api_response.keys())}")
            
            logger.info(f"Parsed {len(detections)} detections from API response")
            return detections
            
        except Exception as e:
            logger.error(f"Error parsing API response: {e}")
            return []
    
    @staticmethod
    def generate_colors(num_colors: int) -> List[str]:
        """Generate distinct colors for bounding boxes using configuration"""
        try:
            # Get colors from configuration
            settings = get_settings()
            predefined_colors = settings.gradio.get('visualization.bbox_colors', [
                "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#ffeaa7",
                "#dda0dd", "#98d8c8", "#f7dc6f", "#bb8fce", "#85c1e9",
                "#f1948a", "#82e0aa", "#85c1e9", "#f8c471", "#d2b4de",
                "#aed6f1", "#a3e4d7", "#f9e79f", "#fadbd8", "#d5dbdb"
            ])
            
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
        """Draw bounding boxes on image using configuration settings"""
        try:
            if not detections:
                return image
            
            # Get settings from configuration
            settings = get_settings()
            
            # Create a copy of the image
            img_with_boxes = image.copy()
            draw = ImageDraw.Draw(img_with_boxes)
            
            # Get visualization settings from config
            bbox_thickness = settings.gradio.get('visualization.bbox_thickness', 2)
            font_size = settings.gradio.get('visualization.font_size', 12)
            font_color = settings.gradio.get('visualization.font_color', '#ffffff')
            show_confidence = settings.gradio.get('visualization.show_confidence_in_label', True)
            
            # Get unique classes and colors
            unique_classes = list(set(det.get('class_name', 'unknown') for det in detections))
            colors = GradioUtils.generate_colors(len(unique_classes))
            class_colors = {cls: colors[i] for i, cls in enumerate(unique_classes)}
            
            # Try to load font
            try:
                # Try different font paths
                font_paths = [
                    "/System/Library/Fonts/Arial.ttf",  # macOS
                    "/Windows/Fonts/arial.ttf",         # Windows
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                    "/usr/share/fonts/TTF/arial.ttf"    # Linux alternative
                ]
                
                font = None
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, font_size)
                        break
                
                if font is None:
                    font = ImageFont.load_default()
                    
            except Exception as e:
                logger.warning(f"Could not load custom font: {e}")
                font = ImageFont.load_default()
            
            # Draw each detection
            for i, detection in enumerate(detections):
                bbox = detection.get('bbox', {})
                class_name = detection.get('class_name', 'unknown')
                confidence = detection.get('confidence', 0.0)
                
                # Get bounding box coordinates
                x1 = int(bbox.get('x1', 0))
                y1 = int(bbox.get('y1', 0))
                x2 = int(bbox.get('x2', 0))
                y2 = int(bbox.get('y2', 0))
                
                # Ensure coordinates are valid
                if x1 >= x2 or y1 >= y2:
                    logger.warning(f"Invalid bounding box coordinates: {bbox}")
                    continue
                
                # Get color for this class
                color = class_colors.get(class_name, '#ff0000')
                
                # Draw bounding box with thickness
                for thickness in range(bbox_thickness):
                    draw.rectangle([x1-thickness, y1-thickness, x2+thickness, y2+thickness], 
                                 outline=color, width=1)
                
                # Prepare label
                if show_confidence:
                    label = f"{class_name} {confidence:.2f}"
                else:
                    label = class_name
                
                # Get text size using textbbox
                try:
                    bbox_text = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox_text[2] - bbox_text[0]
                    text_height = bbox_text[3] - bbox_text[1]
                except:
                    # Fallback for older PIL versions
                    text_width, text_height = draw.textsize(label, font=font)
                
                # Draw label background
                label_bg = [x1, y1 - text_height - 6, x1 + text_width + 8, y1]
                
                # Ensure label doesn't go outside image bounds
                if label_bg[1] < 0:
                    label_bg = [x1, y2, x1 + text_width + 8, y2 + text_height + 6]
                
                draw.rectangle(label_bg, fill=color)
                
                # Draw label text
                text_x = x1 + 4
                text_y = y1 - text_height - 3 if y1 - text_height - 3 >= 0 else y2 + 3
                draw.text((text_x, text_y), label, fill=font_color, font=font)
            
            logger.info(f"Drew {len(detections)} bounding boxes on image")
            return img_with_boxes
            
        except Exception as e:
            logger.error(f"Error drawing bounding boxes: {e}")
            return image
    
    @staticmethod
    def create_detection_table(detections: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create a pandas DataFrame from detections - FIXED for API response format"""
        try:
            if not detections:
                return pd.DataFrame(columns=["ID", "Class", "Confidence", "X1", "Y1", "X2", "Y2", "Width", "Height"])
            
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
                    'Width': int(bbox.get('width', bbox.get('x2', 0) - bbox.get('x1', 0))),
                    'Height': int(bbox.get('height', bbox.get('y2', 0) - bbox.get('y1', 0)))
                }
                table_data.append(row)
            
            df = pd.DataFrame(table_data)
            logger.info(f"Created detection table with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error creating detection table: {e}")
            return pd.DataFrame(columns=["ID", "Class", "Confidence", "X1", "Y1", "X2", "Y2", "Width", "Height"])
    
    @staticmethod
    def create_confidence_chart(detections: List[Dict[str, Any]]) -> go.Figure:
        """Create confidence distribution chart - FIXED for API response format"""
        try:
            if not detections:
                fig = go.Figure()
                fig.add_annotation(text="No detections to display", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                fig.update_layout(title="Detection Confidence Scores", height=400)
                return fig
            
            # Extract confidence values and class names
            confidences = [det.get('confidence', 0.0) for det in detections]
            class_names = [f"{det.get('class_name', 'unknown')} {i+1}" for i, det in enumerate(detections)]
            
            # Create color scale
            colors = GradioUtils.generate_colors(len(detections))
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=class_names,
                    y=confidences,
                    marker_color=colors,
                    text=[f"{c:.3f}" for c in confidences],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Confidence: %{y:.3f}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title="Detection Confidence Scores",
                xaxis_title="Detections",
                yaxis_title="Confidence",
                yaxis=dict(range=[0, 1]),
                height=400,
                showlegend=False,
                xaxis_tickangle=-45
            )
            
            logger.info(f"Created confidence chart with {len(detections)} detections")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating confidence chart: {e}")
            return go.Figure()
    
    @staticmethod
    def create_class_distribution_chart(detections: List[Dict[str, Any]]) -> go.Figure:
        """Create class distribution pie chart - FIXED for API response format"""
        try:
            if not detections:
                fig = go.Figure()
                fig.add_annotation(text="No detections to display", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                fig.update_layout(title="Class Distribution", height=400)
                return fig
            
            # Count classes
            class_counts = {}
            for detection in detections:
                class_name = detection.get('class_name', 'unknown')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Get colors
            colors = GradioUtils.generate_colors(len(class_counts))
            
            # Create pie chart
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(class_counts.keys()),
                    values=list(class_counts.values()),
                    hole=0.3,
                    marker_colors=colors,
                    textinfo='label+percent+value',
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title="Class Distribution",
                height=400,
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
            )
            
            logger.info(f"Created class distribution chart with {len(class_counts)} classes")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating class distribution chart: {e}")
            return go.Figure()
    
    @staticmethod
    def create_summary_stats(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary statistics from detections"""
        try:
            if not detections:
                return {
                    "total_detections": 0,
                    "unique_classes": 0,
                    "avg_confidence": 0.0,
                    "max_confidence": 0.0,
                    "min_confidence": 0.0
                }
            
            confidences = [det.get('confidence', 0.0) for det in detections]
            unique_classes = set(det.get('class_name', 'unknown') for det in detections)
            
            return {
                "total_detections": len(detections),
                "unique_classes": len(unique_classes),
                "avg_confidence": sum(confidences) / len(confidences),
                "max_confidence": max(confidences),
                "min_confidence": min(confidences),
                "class_names": list(unique_classes)
            }
            
        except Exception as e:
            logger.error(f"Error creating summary stats: {e}")
            return {}
    
    @staticmethod
    def get_example_images() -> List[Tuple[str, str]]:
        """Get list of example images with paths and descriptions using configuration"""
        try:
            # Get settings from configuration
            settings = get_settings()
            
            # Get examples directory from gradio config
            examples_dir = Path(settings.gradio.get('paths.examples_dir', './examples'))
            examples_dir.mkdir(exist_ok=True)
            
            # Define example image URLs and descriptions
            example_urls = [
                ("https://images.unsplash.com/photo-1583337130417-3346a1be7dee?w=800", "Living room scene"),
                ("https://images.unsplash.com/photo-1555685812-4b943f1cb0eb?w=800", "Kitchen scene"),
                ("https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=800", "Office workspace"),
                ("https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=800", "Street scene"),
                ("https://images.unsplash.com/photo-1549398632-fcd296ee2653?w=800", "Market scene")
            ]
            
            # Create placeholder images if real ones aren't available
            example_images = []
            for i, (url, description) in enumerate(example_urls):
                filename = f"example_{i+1}.jpg"
                filepath = examples_dir / filename
                
                if not filepath.exists():
                    # Create a simple placeholder image
                    img = Image.new('RGB', (800, 600), color=(70, 130, 180))
                    draw = ImageDraw.Draw(img)
                    
                    # Add text to the image
                    try:
                        font = ImageFont.load_default()
                        text = f"Example {i+1}\n{description}"
                        
                        # Get text size
                        bbox = draw.textbbox((0, 0), text, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        
                        # Center text
                        x = (800 - text_width) // 2
                        y = (600 - text_height) // 2
                        
                        draw.text((x, y), text, fill=(255, 255, 255), font=font)
                    except:
                        pass
                    
                    img.save(filepath, "JPEG")
                
                example_images.append((str(filepath), description))
            
            logger.info(f"Found {len(example_images)} example images")
            return example_images
            
        except Exception as e:
            logger.error(f"Error getting example images: {e}")
            return []
    
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
        """Create formatted notification message using configuration"""
        try:
            # Get settings from configuration
            settings = get_settings()
            
            # Check if notifications are enabled
            if not settings.gradio.get('notifications.enabled', True):
                return message
            
            # Get icon for message type from config
            notification_types = settings.gradio.get('notifications.types', {})
            type_config = notification_types.get(message_type, {})
            icon = type_config.get('icon', '')
            
            # Fallback icons if not in config
            if not icon:
                icons = {
                    'success': '✅',
                    'error': '❌', 
                    'warning': '⚠️',
                    'info': 'ℹ️'
                }
                icon = icons.get(message_type, '')
            
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
                "total_detections": len(detections),
                "summary": GradioUtils.create_summary_stats(detections)
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