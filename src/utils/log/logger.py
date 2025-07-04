# utils/logger_manager.py
import logging
import logging.config
import threading
import os
import sys
from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import json
from contextlib import contextmanager

from src.utils.config.config_manager import get_config

class LoggerManager:
    """
    Singleton Logger Manager
    
    Manages application logging configuration and provides
    centralized logging functionality with thread-safe access.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._loggers: Dict[str, logging.Logger] = {}
            self._config_manager = get_config()
            self._logging_config = {}
            self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        try:
            # Get logging configuration from config manager
            self._logging_config = self._config_manager.get_logging_config()
            
            # Create logs directory if it doesn't exist
            self._create_log_directories()
            
            # Check if JSON formatter is available
            self._check_json_formatter_availability()
            
            # Configure logging
            self._configure_logging()
            
        except Exception as e:
            # Fallback to basic logging if configuration fails
            self._setup_fallback_logging()
            print(f"Warning: Failed to setup logging configuration: {e}")
    
    def _check_json_formatter_availability(self) -> None:
        """Check if JSON formatter is available and update config accordingly"""
        try:
            import pythonjsonlogger.jsonlogger
            
            # JSON formatter is available, add it to formatters
            if 'formatters' in self._logging_config:
                self._logging_config['formatters']['json'] = {
                    'format': '%(asctime)s %(name)s %(levelname)s %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S',
                    'class': 'pythonjsonlogger.jsonlogger.JsonFormatter'
                }
                
        except ImportError:
            # JSON formatter not available, ensure we don't reference it
            if 'formatters' in self._logging_config:
                # Remove any JSON formatter references
                self._logging_config['formatters'].pop('json', None)
                
                # Replace json formatter references with structured formatter
                self._replace_json_formatter_references()
    
    def _create_log_directories(self) -> None:
        """Create log directories if they don't exist"""
        # Get log directory from config
        log_dir = self._config_manager.get('paths.logs_dir', './logs')
        
        # Create directory
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Update file paths in logging config to use absolute paths
        self._update_log_file_paths(log_dir)
    
    def _update_log_file_paths(self, log_dir: str) -> None:
        """Update file paths in logging configuration"""
        if 'handlers' in self._logging_config:
            for handler_name, handler_config in self._logging_config['handlers'].items():
                if 'filename' in handler_config:
                    filename = handler_config['filename']
                    if filename.startswith('./logs/'):
                        # Replace relative path with absolute path
                        filename = filename.replace('./logs/', '')
                        handler_config['filename'] = os.path.join(log_dir, filename)
    
    def _configure_logging(self) -> None:
        """Configure logging using dictConfig"""
        if not self._logging_config:
            raise ValueError("No logging configuration available")
        
        try:
            # Configure logging
            logging.config.dictConfig(self._logging_config)
            
            # Set module-specific logging levels
            self._set_module_logging_levels()
            
        except Exception as e:
            # If JSON formatter fails, fall back to basic configuration
            if "json" in str(e).lower() or "pythonjsonlogger" in str(e).lower():
                logger = logging.getLogger(__name__)
                logger.warning(f"JSON formatter not available, using fallback configuration: {e}")
                self._setup_fallback_logging()
            else:
                raise e
    
    def _replace_json_formatter_references(self) -> None:
        """Replace JSON formatter references with structured formatter"""
        def replace_in_dict(d):
            if isinstance(d, dict):
                for key, value in d.items():
                    if key == 'formatter' and value == 'json':
                        d[key] = 'structured'
                    elif isinstance(value, dict):
                        replace_in_dict(value)
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            if isinstance(item, dict):
                                replace_in_dict(item)
        
        replace_in_dict(self._logging_config)
    
    def _set_module_logging_levels(self) -> None:
        """Set logging levels for external modules"""
        module_levels = self._logging_config.get('module_levels', {})
        
        for module_name, level in module_levels.items():
            logger = logging.getLogger(module_name)
            logger.setLevel(getattr(logging, level.upper()))
    
    def _setup_fallback_logging(self) -> None:
        """Setup basic logging as fallback"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(name)s:%(lineno)d %(asctime)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
        
        return self._loggers[name]
    
    def get_module_logger(self, module_name: str) -> logging.Logger:
        """
        Get a module-specific logger
        
        Args:
            module_name: Module name (e.g., 'models.yolo_model')
            
        Returns:
            Logger instance
        """
        return self.get_logger(module_name)
    
    def set_log_level(self, logger_name: str, level: Union[str, int]) -> None:
        """
        Set log level for a specific logger
        
        Args:
            logger_name: Logger name
            level: Log level (string or integer)
        """
        logger = self.get_logger(logger_name)
        
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        logger.setLevel(level)
    
    def log_performance(self, 
                       operation: str, 
                       duration: float, 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log performance metrics
        
        Args:
            operation: Operation name
            duration: Operation duration in seconds
            metadata: Additional metadata
        """
        perf_logger = self.get_logger('performance')
        
        perf_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'operation': operation,
            'duration_seconds': duration,
            'metadata': metadata or {}
        }
        
        perf_logger.info(json.dumps(perf_data))
    
    def log_error(self, 
                  logger_name: str, 
                  error: Exception, 
                  context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log error with context
        
        Args:
            logger_name: Logger name
            error: Exception instance
            context: Error context
        """
        logger = self.get_logger(logger_name)
        
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }
        
        logger.error(f"Error occurred: {json.dumps(error_data)}", exc_info=True)
    
    def log_request(self, 
                   logger_name: str, 
                   request_id: str, 
                   method: str, 
                   endpoint: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log request information
        
        Args:
            logger_name: Logger name
            request_id: Request identifier
            method: HTTP method
            endpoint: API endpoint
            metadata: Additional metadata
        """
        if not self._config_manager.get('custom.log_requests', True):
            return
        
        logger = self.get_logger(logger_name)
        
        request_data = {
            'request_id': request_id,
            'method': method,
            'endpoint': endpoint,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }
        
        logger.info(f"Request: {json.dumps(request_data)}")
    
    def log_response(self, 
                    logger_name: str, 
                    request_id: str, 
                    status_code: int, 
                    duration: float, 
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log response information
        
        Args:
            logger_name: Logger name
            request_id: Request identifier
            status_code: HTTP status code
            duration: Response time in seconds
            metadata: Additional metadata
        """
        if not self._config_manager.get('custom.log_responses', True):
            return
        
        logger = self.get_logger(logger_name)
        
        response_data = {
            'request_id': request_id,
            'status_code': status_code,
            'duration_seconds': duration,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }
        
        logger.info(f"Response: {json.dumps(response_data)}")
    
    @contextmanager
    def log_execution_time(self, logger_name: str, operation: str):
        """
        Context manager to log execution time
        
        Args:
            logger_name: Logger name
            operation: Operation name
        """
        start_time = datetime.utcnow()
        logger = self.get_logger(logger_name)
        
        try:
            logger.debug(f"Starting operation: {operation}")
            yield
        finally:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            if self._config_manager.get('custom.log_processing_time', True):
                logger.info(f"Operation '{operation}' completed in {duration:.4f} seconds")
            
            # Log to performance logger
            self.log_performance(operation, duration)
    
    def configure_debug_logging(self, enable: bool = True) -> None:
        """
        Configure debug logging
        
        Args:
            enable: Whether to enable debug logging
        """
        if enable:
            # Set all loggers to DEBUG level
            for logger_name in self._loggers:
                self.set_log_level(logger_name, 'DEBUG')
            
            # Set root logger to DEBUG
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            # Restore original levels
            self._set_module_logging_levels()
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """
        Get logging statistics
        
        Returns:
            Dictionary containing logging statistics
        """
        stats = {
            'total_loggers': len(self._loggers),
            'logger_names': list(self._loggers.keys()),
            'handlers': list(self._logging_config.get('handlers', {}).keys()),
            'formatters': list(self._logging_config.get('formatters', {}).keys()),
            'root_level': logging.getLogger().level,
            'effective_level': logging.getLogger().getEffectiveLevel()
        }
        
        return stats
    
    def reload_configuration(self) -> None:
        """Reload logging configuration"""
        try:
            # Reload config from config manager
            self._config_manager.reload_if_needed()
            self._logging_config = self._config_manager.get_logging_config()
            
            # Reconfigure logging
            self._configure_logging()
            
        except Exception as e:
            logger = self.get_logger('logger_manager')
            logger.error(f"Failed to reload logging configuration: {e}")
    
    def shutdown(self) -> None:
        """Shutdown logging system"""
        logging.shutdown()
    
    def add_custom_handler(self, 
                          handler_name: str, 
                          handler: logging.Handler) -> None:
        """
        Add custom handler to root logger
        
        Args:
            handler_name: Handler name
            handler: Handler instance
        """
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        
        # Store reference
        if not hasattr(self, '_custom_handlers'):
            self._custom_handlers = {}
        self._custom_handlers[handler_name] = handler
    
    def remove_custom_handler(self, handler_name: str) -> None:
        """
        Remove custom handler from root logger
        
        Args:
            handler_name: Handler name
        """
        if hasattr(self, '_custom_handlers') and handler_name in self._custom_handlers:
            handler = self._custom_handlers[handler_name]
            root_logger = logging.getLogger()
            root_logger.removeHandler(handler)
            del self._custom_handlers[handler_name]
    
    def get_logger_config(self) -> Dict[str, Any]:
        """
        Get current logging configuration
        
        Returns:
            Logging configuration dictionary
        """
        return self._logging_config.copy()

# Convenience functions
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return LoggerManager().get_logger(name)

def get_module_logger(module_name: str) -> logging.Logger:
    """
    Get a module-specific logger
    
    Args:
        module_name: Module name
        
    Returns:
        Logger instance
    """
    return LoggerManager().get_module_logger(module_name)

def log_performance(operation: str, 
                   duration: float, 
                   metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Log performance metrics
    
    Args:
        operation: Operation name
        duration: Operation duration in seconds
        metadata: Additional metadata
    """
    LoggerManager().log_performance(operation, duration, metadata)

def log_execution_time(logger_name: str, operation: str):
    """
    Context manager to log execution time
    
    Args:
        logger_name: Logger name
        operation: Operation name
    """
    return LoggerManager().log_execution_time(logger_name, operation)