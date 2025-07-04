# utils/config_manager.py
import os
import yaml
import threading
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ConfigPaths:
    """Configuration file paths"""
    config_yaml: str = "utils/config.yaml"
    logging_yaml: str = "utils/logging.yaml"
    config_dir: str = "utils"

class ConfigManager:
    """
    Configuration Manager
    
    Manages application configuration loaded from YAML files.
    Provides thread-safe access to configuration values.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._config_lock = threading.RLock()
            self._config_data: Dict[str, Any] = {}
            self._logging_config: Dict[str, Any] = {}
            self._paths = ConfigPaths()
            self._environment = os.getenv('ENVIRONMENT', 'development')
            self._auto_reload = True
            self._file_timestamps = {}
            
            # Load configurations
            self._load_configurations()
    
    def _load_configurations(self) -> None:
        """Load all configuration files"""
        with self._config_lock:
            self._load_config_file()
            self._load_logging_config()
            self._update_file_timestamps()
    
    def _load_config_file(self) -> None:
        """Load main configuration file"""
        try:
            config_path = Path(self._paths.config_yaml)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as file:
                self._config_data = yaml.safe_load(file) or {}
            
            # Apply environment-specific overrides
            self._apply_environment_overrides()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {str(e)}")
    
    def _load_logging_config(self) -> None:
        """Load logging configuration file"""
        try:
            logging_path = Path(self._paths.logging_yaml)
            if not logging_path.exists():
                raise FileNotFoundError(f"Logging configuration file not found: {logging_path}")
            
            with open(logging_path, 'r', encoding='utf-8') as file:
                self._logging_config = yaml.safe_load(file) or {}
            
            # Apply environment-specific logging config
            self._apply_logging_environment_overrides()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load logging configuration: {str(e)}")
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment-specific configuration overrides"""
        # Override with environment variables
        env_overrides = {
            'model.device.preferred': os.getenv('MODEL_DEVICE', self.get('model.device.preferred')),
            'litserve.server.host': os.getenv('LITSERVE_HOST', self.get('litserve.server.host')),
            'litserve.server.port': int(os.getenv('LITSERVE_PORT', self.get('litserve.server.port'))),
            'development.debug_mode': os.getenv('DEBUG_MODE', '').lower() == 'true',
            'performance.enable_profiling': os.getenv('ENABLE_PROFILING', '').lower() == 'true',
        }
        
        for key, value in env_overrides.items():
            if value is not None:
                self.set(key, value)
    
    def _apply_logging_environment_overrides(self) -> None:
        """Apply environment-specific logging configuration overrides"""
        env = self._environment
        
        if 'environments' in self._logging_config and env in self._logging_config['environments']:
            env_config = self._logging_config['environments'][env]
            
            # Apply environment-specific logging configuration
            for key, value in env_config.items():
                if key in self._logging_config:
                    if isinstance(value, dict) and isinstance(self._logging_config[key], dict):
                        self._logging_config[key].update(value)
                    else:
                        self._logging_config[key] = value
    
    def _update_file_timestamps(self) -> None:
        """Update file modification timestamps for reload detection"""
        config_files = [self._paths.config_yaml, self._paths.logging_yaml]
        
        for file_path in config_files:
            if Path(file_path).exists():
                self._file_timestamps[file_path] = os.path.getmtime(file_path)
    
    def _should_reload(self) -> bool:
        """Check if configuration files have been modified"""
        if not self._auto_reload:
            return False
        
        for file_path, old_timestamp in self._file_timestamps.items():
            if Path(file_path).exists():
                current_timestamp = os.path.getmtime(file_path)
                if current_timestamp > old_timestamp:
                    return True
        
        return False
    
    def reload_if_needed(self) -> bool:
        """Reload configuration if files have been modified"""
        if self._should_reload():
            self._load_configurations()
            return True
        return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot notation key
        
        Args:
            key: Dot notation key (e.g., 'model.device.preferred')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        with self._config_lock:
            # Auto-reload if needed
            self.reload_if_needed()
            
            keys = key.split('.')
            value = self._config_data
            
            try:
                for k in keys:
                    if isinstance(value, dict) and k in value:
                        value = value[k]
                    else:
                        return default
                return value
            except (KeyError, TypeError):
                return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot notation key
        
        Args:
            key: Dot notation key (e.g., 'model.device.preferred')
            value: Value to set
        """
        with self._config_lock:
            keys = key.split('.')
            target = self._config_data
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            
            # Set the final value
            target[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section
        
        Args:
            section: Section name (e.g., 'model', 'litserve')
            
        Returns:
            Dictionary containing section configuration
        """
        return self.get(section, {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration
        
        Returns:
            Logging configuration dictionary
        """
        with self._config_lock:
            return self._logging_config.copy()
    
    def get_environment(self) -> str:
        """
        Get current environment
        
        Returns:
            Current environment name
        """
        return self._environment
    
    def set_environment(self, environment: str) -> None:
        """
        Set current environment
        
        Args:
            environment: Environment name
        """
        self._environment = environment
        self._apply_environment_overrides()
        self._apply_logging_environment_overrides()
    
    def enable_auto_reload(self, enable: bool = True) -> None:
        """
        Enable or disable automatic configuration reloading
        
        Args:
            enable: Whether to enable auto-reload
        """
        self._auto_reload = enable
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate configuration
        
        Returns:
            Validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate required sections
        required_sections = ['app', 'model', 'image_processing', 'response']
        for section in required_sections:
            if not self.get(section):
                validation_results['valid'] = False
                validation_results['errors'].append(f"Missing required section: {section}")
        
        # Validate model configuration
        model_size = self.get('model.default_size')
        available_sizes = self.get('model.available_sizes', [])
        if model_size not in available_sizes:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Invalid model size: {model_size}")
        
        # Validate paths
        paths_to_check = [
            'paths.models_dir',
            'paths.logs_dir',
            'paths.cache_dir'
        ]
        
        for path_key in paths_to_check:
            path_value = self.get(path_key)
            if path_value:
                path_obj = Path(path_value)
                if not path_obj.exists():
                    validation_results['warnings'].append(f"Path does not exist: {path_value}")
        
        return validation_results
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return self.get_section('model')
    
    def get_api_config(self, api_type: str) -> Dict[str, Any]:
        """
        Get API-specific configuration
        
        Args:
            api_type: API type ('litserve' or 'runpod')
            
        Returns:
            API configuration dictionary
        """
        return self.get_section(api_type)
    
    def get_image_processing_config(self) -> Dict[str, Any]:
        """Get image processing configuration"""
        return self.get_section('image_processing')
    
    def get_response_config(self) -> Dict[str, Any]:
        """Get response formatting configuration"""
        return self.get_section('response')
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.get_section('performance')
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return self.get_section('security')
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self.get_section('monitoring')
    
    def get_feature_flags(self) -> Dict[str, Any]:
        """Get feature flags"""
        return self.get_section('features')
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        Check if a feature is enabled
        
        Args:
            feature_name: Feature name
            
        Returns:
            True if feature is enabled, False otherwise
        """
        return self.get(f'features.{feature_name}', False)
    
    def dump_config(self) -> Dict[str, Any]:
        """
        Dump entire configuration (for debugging)
        
        Returns:
            Complete configuration dictionary
        """
        with self._config_lock:
            return {
                'config': self._config_data.copy(),
                'logging': self._logging_config.copy(),
                'environment': self._environment,
                'auto_reload': self._auto_reload,
                'file_timestamps': self._file_timestamps.copy()
            }
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults by reloading from files"""
        with self._config_lock:
            self._load_configurations()

# Convenience function to get the singleton instance
def get_config() -> ConfigManager:
    """
    Get the ConfigManager singleton instance
    
    Returns:
        ConfigManager instance
    """
    return ConfigManager()