# utils/settings.py
"""
Settings module that provides easy access to configuration values
using the ConfigManager singleton.
"""

from typing import Dict, Any, Optional, Union, List
from src.utils.config.config_manager import get_config
from src.utils.log.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class Settings:
    """
    Settings class that provides structured access to configuration values.
    This class acts as a facade over the ConfigManager singleton.
    """
    
    def __init__(self):
        self._config = get_config()
        logger.info("Settings initialized")
    
    @property
    def app(self) -> 'AppSettings':
        """Get application settings"""
        return AppSettings(self._config)
    
    @property
    def model(self) -> 'ModelSettings':
        """Get model settings"""
        return ModelSettings(self._config)
    
    @property
    def image_processing(self) -> 'ImageProcessingSettings':
        """Get image processing settings"""
        return ImageProcessingSettings(self._config)
    
    @property
    def response(self) -> 'ResponseSettings':
        """Get response settings"""
        return ResponseSettings(self._config)
    
    @property
    def litserve(self) -> 'LitServeSettings':
        """Get LitServe settings"""
        return LitServeSettings(self._config)
    
    @property
    def runpod(self) -> 'RunPodSettings':
        """Get RunPod settings"""
        return RunPodSettings(self._config)
    
    @property
    def performance(self) -> 'PerformanceSettings':
        """Get performance settings"""
        return PerformanceSettings(self._config)
    
    @property
    def security(self) -> 'SecuritySettings':
        """Get security settings"""
        return SecuritySettings(self._config)
    
    @property
    def monitoring(self) -> 'MonitoringSettings':
        """Get monitoring settings"""
        return MonitoringSettings(self._config)
    
    @property
    def development(self) -> 'DevelopmentSettings':
        """Get development settings"""
        return DevelopmentSettings(self._config)
    
    @property
    def paths(self) -> 'PathSettings':
        """Get path settings"""
        return PathSettings(self._config)
    
    @property
    def features(self) -> 'FeatureSettings':
        """Get feature settings"""
        return FeatureSettings(self._config)
    
    @property
    def gradio(self) -> 'GradioSettings':
        """Get Gradio settings"""
        return GradioSettings(self._config)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section
        
        Args:
            section: Section name (e.g., 'model', 'litserve', 'gradio')
            
        Returns:
            Dictionary containing section configuration
        """
        return self._config.get_section(section)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot notation key
        
        Args:
            key: Dot notation key (e.g., 'model.device.preferred')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot notation key
        
        Args:
            key: Dot notation key (e.g., 'model.device.preferred')
            value: Value to set
        """
        self._config.set(key, value)
    
    def reload(self) -> bool:
        """
        Reload configuration from files
        
        Returns:
            True if configuration was reloaded, False otherwise
        """
        return self._config.reload_if_needed()
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate configuration
        
        Returns:
            Validation results
        """
        return self._config.validate_config()
    
    def get_environment(self) -> str:
        """Get current environment"""
        return self._config.get_environment()
    
    def set_environment(self, environment: str) -> None:
        """Set current environment"""
        self._config.set_environment(environment)
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled"""
        return self._config.is_feature_enabled(feature_name)

class BaseSettings:
    """Base class for settings sections"""
    
    def __init__(self, config_manager, section_name: str):
        self._config = config_manager
        self._section = section_name
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from this section"""
        return self._config.get(f"{self._section}.{key}", default)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in this section"""
        self._config.set(f"{self._section}.{key}", value)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all values from this section"""
        return self._config.get_section(self._section)

class AppSettings(BaseSettings):
    """Application settings"""
    
    def __init__(self, config_manager):
        super().__init__(config_manager, 'app')
    
    @property
    def name(self) -> str:
        return self.get('name', 'YOLOv11 API')
    
    @property
    def version(self) -> str:
        return self.get('version', '1.0.0')
    
    @property
    def description(self) -> str:
        return self.get('description', 'YOLO object detection API')
    
    @property
    def author(self) -> str:
        return self.get('author', 'Unknown')
    
    @property
    def contact(self) -> str:
        return self.get('contact', '')

class ModelSettings(BaseSettings):
    """Model settings"""
    
    def __init__(self, config_manager):
        super().__init__(config_manager, 'model')
    
    @property
    def default_size(self) -> str:
        return self.get('default_size', 'n')
    
    @property
    def available_sizes(self) -> List[str]:
        return self.get('available_sizes', ['n', 's', 'm', 'l', 'x'])
    
    @property
    def model_paths(self) -> Dict[str, str]:
        return self.get('model_paths', {})
    
    @property
    def device_auto_detect(self) -> bool:
        return self.get('device.auto_detect', True)
    
    @property
    def device_preferred(self) -> str:
        return self.get('device.preferred', 'auto')
    
    @property
    def device_fallback(self) -> str:
        return self.get('device.fallback', 'cpu')
    
    @property
    def half_precision_enabled(self) -> str:
        return self.get('device.half_precision_enabled', 'false')
    
    @property
    def confidence_threshold(self) -> float:
        return self.get('inference.confidence_threshold', 0.25)
    
    @property
    def iou_threshold(self) -> float:
        return self.get('inference.iou_threshold', 0.45)
    
    @property
    def max_detections(self) -> int:
        return self.get('inference.max_detections', 100)
    
    @property
    def image_size(self) -> int:
        return self.get('inference.image_size', 640)
    
    @property
    def warmup_enabled(self) -> bool:
        return self.get('warmup.enabled', True)
    
    def get_model_path(self, size: str) -> str:
        """Get model path for specific size"""
        return self.model_paths.get(size, f'yolo11{size}.pt')

class ImageProcessingSettings(BaseSettings):
    """Image processing settings"""
    
    def __init__(self, config_manager):
        super().__init__(config_manager, 'image_processing')
    
    @property
    def supported_formats(self) -> List[str]:
        return self.get('supported_formats', ['JPEG', 'PNG', 'WEBP', 'BMP'])
    
    @property
    def max_image_size(self) -> int:
        return self.get('max_image_size', 10485760)  # 10MB
    
    @property
    def min_image_size(self) -> int:
        return self.get('min_image_size', 1024)  # 1KB
    
    @property
    def max_width(self) -> int:
        return self.get('max_dimensions.width', 4096)
    
    @property
    def max_height(self) -> int:
        return self.get('max_dimensions.height', 4096)
    
    @property
    def min_width(self) -> int:
        return self.get('min_dimensions.width', 32)
    
    @property
    def min_height(self) -> int:
        return self.get('min_dimensions.height', 32)
    
    @property
    def auto_resize(self) -> bool:
        return self.get('preprocessing.auto_resize', False)
    
    @property
    def maintain_aspect_ratio(self) -> bool:
        return self.get('preprocessing.maintain_aspect_ratio', True)
    
    @property
    def strict_validation(self) -> bool:
        return self.get('validation.strict_mode', True)

class ResponseSettings(BaseSettings):
    """Response settings"""
    
    def __init__(self, config_manager):
        super().__init__(config_manager, 'response')
    
    @property
    def available_formats(self) -> List[str]:
        return self.get('formats', ['detailed', 'minimal', 'compact'])
    
    @property
    def default_format(self) -> str:
        return self.get('default_format', 'detailed')
    
    @property
    def include_metadata(self) -> bool:
        return self.get('include_metadata', True)
    
    @property
    def include_timing(self) -> bool:
        return self.get('include_timing', True)
    
    @property
    def decimal_precision(self) -> int:
        return self.get('decimal_precision', 3)
    
    @property
    def bbox_format(self) -> str:
        return self.get('bbox_format', 'xyxy')
    
    @property
    def confidence_format(self) -> str:
        return self.get('confidence_format', 'decimal')

class LitServeSettings(BaseSettings):
    """LitServe settings"""
    
    def __init__(self, config_manager):
        super().__init__(config_manager, 'litserve')
    
    @property
    def host(self) -> str:
        return self.get('server.host', '0.0.0.0')
    
    @property
    def port(self) -> int:
        return self.get('server.port', 8000)
    
    @property
    def workers_per_device(self) -> int:
        return self.get('server.workers_per_device', 1)
    
    @property
    def timeout(self) -> int:
        return self.get('server.timeout', 30)
    
    @property
    def accelerator(self) -> str:
        return self.get('server.accelerator', 'auto')
    
    @property
    def max_batch_size(self) -> int:
        return self.get('server.max_batch_size', 8)
    
    @property
    def batch_timeout(self) -> float:
        return self.get('server.batch_timeout', 0.1)
    
    @property
    def enable_async(self) -> bool:
        return self.get('api.enable_async', True)
    
    @property
    def enable_cors(self) -> bool:
        return self.get('api.enable_cors', True)
    
    @property
    def cors_origins(self) -> List[str]:
        return self.get('api.cors_origins', ['*'])
    
    @property
    def max_request_size(self) -> int:
        return self.get('api.max_request_size', 10485760)

class RunPodSettings(BaseSettings):
    """RunPod settings"""
    
    def __init__(self, config_manager):
        super().__init__(config_manager, 'runpod')
    
    @property
    def return_aggregate_stream(self) -> bool:
        return self.get('serverless.return_aggregate_stream', True)
    
    @property
    def rp_serve_api(self) -> bool:
        return self.get('serverless.rp_serve_api', True)
    
    @property
    def rp_log_level(self) -> str:
        return self.get('serverless.rp_log_level', 'INFO')
    
    @property
    def handler_timeout(self) -> int:
        return self.get('handler.timeout', 60)
    
    @property
    def memory_limit(self) -> int:
        return self.get('handler.memory_limit', 4096)
    
    @property
    def concurrent_requests(self) -> int:
        return self.get('handler.concurrent_requests', 10)
    
    @property
    def required_env_vars(self) -> List[str]:
        return self.get('environment.required_vars', [])
    
    @property
    def optional_env_vars(self) -> List[str]:
        return self.get('environment.optional_vars', [])
    
    @property
    def test_input_filename(self) -> str:
        return self.get('test_input.filename', 'test_input.json')
    
    @property
    def auto_create_test_input(self) -> bool:
        return self.get('test_input.auto_create', True)

class PerformanceSettings(BaseSettings):
    """Performance settings"""
    
    def __init__(self, config_manager):
        super().__init__(config_manager, 'performance')
    
    @property
    def enable_profiling(self) -> bool:
        return self.get('enable_profiling', False)
    
    @property
    def enable_metrics(self) -> bool:
        return self.get('enable_metrics', True)
    
    @property
    def metrics_interval(self) -> int:
        return self.get('metrics_interval', 60)
    
    @property
    def clear_cache_after_requests(self) -> int:
        return self.get('memory_management.clear_cache_after_requests', 100)
    
    @property
    def max_memory_usage(self) -> float:
        return self.get('memory_management.max_memory_usage', 0.8)
    
    @property
    def enable_mixed_precision(self) -> bool:
        return self.get('gpu_optimization.enable_mixed_precision', True)
    
    @property
    def enable_cudnn_benchmark(self) -> bool:
        return self.get('gpu_optimization.enable_cudnn_benchmark', True)
    
    @property
    def max_workers(self) -> int:
        return self.get('threading.max_workers', 4)

class SecuritySettings(BaseSettings):
    """Security settings"""
    
    def __init__(self, config_manager):
        super().__init__(config_manager, 'security')
    
    @property
    def enable_input_validation(self) -> bool:
        return self.get('enable_input_validation', True)
    
    @property
    def max_upload_size(self) -> int:
        return self.get('max_upload_size', 10485760)
    
    @property
    def allowed_mime_types(self) -> List[str]:
        return self.get('allowed_mime_types', [])
    
    @property
    def rate_limiting_enabled(self) -> bool:
        return self.get('rate_limiting.enabled', False)
    
    @property
    def requests_per_minute(self) -> int:
        return self.get('rate_limiting.requests_per_minute', 60)
    
    @property
    def api_keys_enabled(self) -> bool:
        return self.get('api_keys.enabled', False)
    
    @property
    def api_key_header(self) -> str:
        return self.get('api_keys.header_name', 'X-API-Key')

class MonitoringSettings(BaseSettings):
    """Monitoring settings"""
    
    def __init__(self, config_manager):
        super().__init__(config_manager, 'monitoring')
    
    @property
    def enable_health_checks(self) -> bool:
        return self.get('enable_health_checks', True)
    
    @property
    def health_check_interval(self) -> int:
        return self.get('health_check_interval', 30)
    
    @property
    def enable_performance_monitoring(self) -> bool:
        return self.get('enable_performance_monitoring', True)
    
    @property
    def performance_check_interval(self) -> int:
        return self.get('performance_check_interval', 60)
    
    @property
    def enable_prometheus(self) -> bool:
        return self.get('metrics.enable_prometheus', False)
    
    @property
    def prometheus_port(self) -> int:
        return self.get('metrics.prometheus_port', 9090)

class DevelopmentSettings(BaseSettings):
    """Development settings"""
    
    def __init__(self, config_manager):
        super().__init__(config_manager, 'development')
    
    @property
    def debug_mode(self) -> bool:
        return self.get('debug_mode', False)
    
    @property
    def hot_reload(self) -> bool:
        return self.get('hot_reload', False)
    
    @property
    def auto_reload_config(self) -> bool:
        return self.get('auto_reload_config', True)
    
    @property
    def enable_debug_endpoints(self) -> bool:
        return self.get('enable_debug_endpoints', False)
    
    @property
    def mock_model(self) -> bool:
        return self.get('mock_model', False)
    
    @property
    def save_debug_images(self) -> bool:
        return self.get('save_debug_images', False)
    
    @property
    def debug_image_path(self) -> str:
        return self.get('debug_image_path', './debug_images')

class PathSettings(BaseSettings):
    """Path settings"""
    
    def __init__(self, config_manager):
        super().__init__(config_manager, 'paths')
    
    @property
    def models_dir(self) -> str:
        return self.get('models_dir', './models')
    
    @property
    def cache_dir(self) -> str:
        return self.get('cache_dir', './cache')
    
    @property
    def logs_dir(self) -> str:
        return self.get('logs_dir', './logs')
    
    @property
    def temp_dir(self) -> str:
        return self.get('temp_dir', './temp')
    
    @property
    def data_dir(self) -> str:
        return self.get('data_dir', './data')
    
    @property
    def output_dir(self) -> str:
        return self.get('output_dir', './output')
    
    @property
    def config_dir(self) -> str:
        return self.get('config_dir', './config')

class FeatureSettings(BaseSettings):
    """Feature settings"""
    
    def __init__(self, config_manager):
        super().__init__(config_manager, 'features')
    
    @property
    def enable_batch_processing(self) -> bool:
        return self.get('enable_batch_processing', True)
    
    @property
    def enable_async_processing(self) -> bool:
        return self.get('enable_async_processing', True)
    
    @property
    def enable_model_caching(self) -> bool:
        return self.get('enable_model_caching', True)
    
    @property
    def enable_response_caching(self) -> bool:
        return self.get('enable_response_caching', False)
    
    @property
    def enable_health_checks(self) -> bool:
        return self.get('enable_health_checks', True)
    
    @property
    def enable_metrics_collection(self) -> bool:
        return self.get('enable_metrics_collection', True)
    
    @property
    def enable_auto_scaling(self) -> bool:
        return self.get('enable_auto_scaling', False)
    
    @property
    def enable_model_switching(self) -> bool:
        return self.get('enable_model_switching', False)
    
    @property
    def enable_custom_models(self) -> bool:
        return self.get('enable_custom_models', False)

class GradioSettings(BaseSettings):
    """Gradio settings"""
    
    def __init__(self, config_manager):
        super().__init__(config_manager, 'gradio')
    
    @property
    def app_name(self) -> str:
        return self.get('app.name', 'YOLOv11 Object Detection')
    
    @property
    def app_title(self) -> str:
        return self.get('app.title', 'YOLOv11 Object Detection Interface')
    
    @property
    def app_theme(self) -> str:
        return self.get('app.theme', 'soft')
    
    @property
    def server_host(self) -> str:
        return self.get('server.host', '0.0.0.0')
    
    @property
    def server_port(self) -> int:
        return self.get('server.port', 7860)
    
    @property
    def server_share(self) -> bool:
        return self.get('server.share', False)
    
    @property
    def server_debug(self) -> bool:
        return self.get('server.debug', False)
    
    @property
    def litserve_enabled(self) -> bool:
        return self.get('backends.litserve.enabled', True)
    
    @property
    def litserve_base_url(self) -> str:
        return self.get('backends.litserve.base_url', 'http://localhost:8000')
    
    @property
    def runpod_enabled(self) -> bool:
        return self.get('backends.runpod.enabled', True)
    
    @property
    def runpod_base_url(self) -> str:
        return self.get('backends.runpod.base_url', 'https://api.runpod.ai/v2')
    
    @property
    def detection_confidence_default(self) -> float:
        return self.get('detection.confidence_threshold.default', 0.25)
    
    @property
    def detection_iou_default(self) -> float:
        return self.get('detection.iou_threshold.default', 0.45)
    
    @property
    def detection_max_detections_default(self) -> int:
        return self.get('detection.max_detections.default', 100)
    
    @property
    def detection_max_image_size(self) -> int:
        return self.get('detection.max_image_size', 10485760)
    
    @property
    def notifications_enabled(self) -> bool:
        return self.get('notifications.enabled', True)
    
    @property
    def export_enabled(self) -> bool:
        return self.get('features.enable_export_results', True)
    
    @property
    def job_management_enabled(self) -> bool:
        return self.get('backends.runpod.job_management.enabled', True)

# Global settings instance
settings = Settings()

# Convenience functions
def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings

def reload_settings() -> bool:
    """Reload settings from configuration files"""
    return settings.reload()

def validate_settings() -> Dict[str, Any]:
    """Validate current settings"""
    return settings.validate()

def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled"""
    return settings.is_feature_enabled(feature_name)