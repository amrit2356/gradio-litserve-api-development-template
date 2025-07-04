# main.py
import os
import sys
import argparse
from typing import Dict, Any
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize configuration and logging first
from src.utils.config.settings import get_settings, validate_settings
from src.utils.log.logger import get_logger, get_module_logger
from src.utils.config.config_manager import get_config

# Initialize settings and logger
settings = get_settings()
logger = get_module_logger(__name__)

# Import API managers after configuration is initialized
from src.api.litserve_api import YOLOv11LitServeManager
from src.api.runpod_api import YOLOv11RunPodManager, create_test_input, validate_runpod_environment

class YOLOv11Application:
    """
    Main application class that handles different deployment modes
    """
    
    def __init__(self):
        self.mode = None
        self.config = get_config()
        
        # Validate configuration on startup
        validation_results = validate_settings()
        if not validation_results['valid']:
            logger.error(f"Configuration validation failed: {validation_results['errors']}")
            for error in validation_results['errors']:
                logger.error(f"  - {error}")
        
        if validation_results['warnings']:
            for warning in validation_results['warnings']:
                logger.warning(f"  - {warning}")
    
    def parse_arguments(self) -> argparse.Namespace:
        """
        Parse command line arguments (utility commands only)
        
        Returns:
            Parsed arguments
        """
        parser = argparse.ArgumentParser(
            description="YOLOv11 API with RunPod and LitServe support",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
                    Examples:
                    # Run with default configuration
                    python main.py
                    
                    # Create test input for RunPod
                    python main.py --create-test
                    
                    # Check environment
                    python main.py --check-env
                    
                    # Validate configuration
                    python main.py --validate-config
                    
                    # Show current configuration
                    python main.py --show-config
                    
                    Note: All runtime configuration is managed through config.yaml files.
                    Edit src/utils/config/config.yaml to modify application settings.
            """
        )
        
        # Utility commands only
        parser.add_argument(
            "--create-test",
            action="store_true",
            help="Create test_input.json file for RunPod testing"
        )
        
        parser.add_argument(
            "--check-env",
            action="store_true",
            help="Check environment and system information"
        )
        
        parser.add_argument(
            "--validate-runpod",
            action="store_true",
            help="Validate RunPod environment"
        )
        
        parser.add_argument(
            "--validate-config",
            action="store_true",
            help="Validate configuration"
        )
        
        parser.add_argument(
            "--show-config",
            action="store_true",
            help="Show current configuration"
        )
        
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode (overrides config)"
        )
        
        return parser.parse_args()
    
    def setup_debug_mode(self, enable_debug: bool) -> None:
        """
        Setup debug mode if requested
        
        Args:
            enable_debug: Whether to enable debug mode
        """
        if enable_debug:
            logger.info("Debug mode enabled via command line")
            settings.development.set('debug_mode', True)
            from src.utils.log.logger import LoggerManager
            LoggerManager().configure_debug_logging(True)
    
    def check_environment(self) -> Dict[str, Any]:
        """
        Check system environment and requirements
        
        Returns:
            Environment information dictionary
        """
        import torch
        import platform
        
        env_info = {
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "architecture": platform.architecture()[0]
            },
            "torch": {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            },
            "environment_variables": {
                "RUNPOD_POD_ID": os.getenv("RUNPOD_POD_ID"),
                "RUNPOD_POD_HOSTNAME": os.getenv("RUNPOD_POD_HOSTNAME"),
                "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
                "ENVIRONMENT": os.getenv("ENVIRONMENT")
            },
            "configuration": {
                "current_environment": settings.get_environment(),
                "model_size": settings.model.default_size,
                "device": settings.model.device_preferred,
                "debug_mode": settings.development.debug_mode
            }
        }
        
        return env_info
    
    def run_litserve_mode(self) -> None:
        """
        Run application in LitServe mode
        """
        try:
            model_size = settings.model.default_size
            logger.info(f"Starting LitServe mode with model size: {model_size}")
            
            # Create LitServe manager
            manager = YOLOv11LitServeManager(model_size=model_size)
            
            # Get configuration from settings
            litserve_config = settings.litserve
            
            logger.info(f"LitServe configuration:")
            logger.info(f"  Host: {litserve_config.host}")
            logger.info(f"  Port: {litserve_config.port}")
            logger.info(f"  Workers per device: {litserve_config.workers_per_device}")
            logger.info(f"  Timeout: {litserve_config.timeout}")
            
            # Run server
            manager.run_server(
                host=litserve_config.host,
                port=litserve_config.port,
                workers_per_device=litserve_config.workers_per_device,
                timeout=litserve_config.timeout
            )
            
        except Exception as e:
            logger.error(f"LitServe mode failed: {str(e)}")
            raise
    
    def run_runpod_mode(self) -> None:
        """
        Run application in RunPod mode
        """
        try:
            model_size = settings.model.default_size
            device = settings.model.device_preferred
            
            logger.info(f"Starting RunPod mode with model size: {model_size}, device: {device}")
            
            # Log RunPod configuration
            runpod_config = settings.runpod
            logger.info(f"RunPod configuration:")
            logger.info(f"  Handler timeout: {runpod_config.handler_timeout}")
            logger.info(f"  Memory limit: {runpod_config.memory_limit}")
            logger.info(f"  Concurrent requests: {runpod_config.concurrent_requests}")
            
            # Create RunPod manager
            manager = YOLOv11RunPodManager(
                model_size=model_size,
                device=device
            )
            
            # Start serverless function
            manager.start_serverless()
            
        except Exception as e:
            logger.error(f"RunPod mode failed: {str(e)}")
            raise
    
    def create_test_input(self) -> None:
        """Create test input file for RunPod"""
        try:
            filename = settings.runpod.test_input_filename
            output_file = create_test_input(filename)
            print(f"✅ Created test input file: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to create test input: {str(e)}")
            raise
    
    def validate_runpod_env(self) -> None:
        """Validate RunPod environment"""
        try:
            results = validate_runpod_environment()
            
            print("🔍 RunPod Environment Validation:")
            print(f"Is RunPod Environment: {'✅' if results['is_runpod_environment'] else '❌'}")
            
            if results['environment_variables']:
                print("\nEnvironment Variables:")
                for var, value in results['environment_variables'].items():
                    print(f"  {var}: {value}")
            
            if results['missing_variables']:
                print("\nMissing Variables:")
                for var in results['missing_variables']:
                    print(f"  ❌ {var}")
            
        except Exception as e:
            logger.error(f"Environment validation failed: {str(e)}")
            raise
    
    def validate_configuration(self) -> None:
        """Validate configuration"""
        try:
            results = validate_settings()
            
            print("⚙️  Configuration Validation:")
            print(f"Valid: {'✅' if results['valid'] else '❌'}")
            
            if results['errors']:
                print("\nErrors:")
                for error in results['errors']:
                    print(f"  ❌ {error}")
            
            if results['warnings']:
                print("\nWarnings:")
                for warning in results['warnings']:
                    print(f"  ⚠️  {warning}")
            
            if results['valid']:
                print("\n✅ Configuration is valid!")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise
    
    def show_configuration(self) -> None:
        """Show current configuration"""
        try:
            print("📋 Current Configuration:")
            print(f"Environment: {settings.get_environment()}")
            print(f"App Name: {settings.app.name}")
            print(f"App Version: {settings.app.version}")
            print("")
            
            print("🤖 Model Configuration:")
            print(f"Model Size: {settings.model.default_size}")
            print(f"Device: {settings.model.device_preferred}")
            print(f"Confidence Threshold: {settings.model.confidence_threshold}")
            print(f"IoU Threshold: {settings.model.iou_threshold}")
            print("")
            
            print("🖥️  LitServe Configuration:")
            print(f"Host: {settings.litserve.host}")
            print(f"Port: {settings.litserve.port}")
            print(f"Workers per Device: {settings.litserve.workers_per_device}")
            print(f"Timeout: {settings.litserve.timeout}")
            print("")
            
            print("☁️  RunPod Configuration:")
            print(f"Handler Timeout: {settings.runpod.handler_timeout}")
            print(f"Memory Limit: {settings.runpod.memory_limit}")
            print(f"Concurrent Requests: {settings.runpod.concurrent_requests}")
            print("")
            
            print("🔧 Development Configuration:")
            print(f"Debug Mode: {settings.development.debug_mode}")
            print(f"Mock Model: {settings.development.mock_model}")
            print(f"Auto Reload Config: {settings.development.auto_reload_config}")
            print("")
            
            print("🚀 Feature Flags:")
            print(f"Batch Processing: {settings.features.enable_batch_processing}")
            print(f"Async Processing: {settings.features.enable_async_processing}")
            print(f"Model Caching: {settings.features.enable_model_caching}")
            print(f"Health Checks: {settings.features.enable_health_checks}")
            print(f"Metrics Collection: {settings.features.enable_metrics_collection}")
            print("")
            
            print("🛡️  Security Configuration:")
            print(f"Input Validation: {settings.security.enable_input_validation}")
            print(f"Rate Limiting: {settings.security.rate_limiting_enabled}")
            print(f"Max Upload Size: {settings.security.max_upload_size} bytes")
            print("")
            
            print("📊 Performance Configuration:")
            print(f"Enable Profiling: {settings.performance.enable_profiling}")
            print(f"Enable Metrics: {settings.performance.enable_metrics}")
            print(f"Mixed Precision: {settings.performance.enable_mixed_precision}")
            print(f"Max Workers: {settings.performance.max_workers}")
            
        except Exception as e:
            logger.error(f"Failed to show configuration: {str(e)}")
            raise
    
    def check_system_info(self) -> None:
        """Check and display system information"""
        try:
            env_info = self.check_environment()
            
            print("🖥️  System Information:")
            print(f"Platform: {env_info['system']['platform']}")
            print(f"Python: {env_info['system']['python_version']}")
            print(f"Architecture: {env_info['system']['architecture']}")
            
            print("\n🔥 PyTorch Information:")
            print(f"Version: {env_info['torch']['version']}")
            print(f"CUDA Available: {'✅' if env_info['torch']['cuda_available'] else '❌'}")
            if env_info['torch']['cuda_available']:
                print(f"CUDA Version: {env_info['torch']['cuda_version']}")
                print(f"GPU Count: {env_info['torch']['device_count']}")
            
            print("\n🌍 Environment Variables:")
            for var, value in env_info['environment_variables'].items():
                if value:
                    print(f"{var}: {value}")
                else:
                    print(f"{var}: Not set")
            
            print("\n⚙️  Current Configuration:")
            print(f"Environment: {env_info['configuration']['current_environment']}")
            print(f"Model Size: {env_info['configuration']['model_size']}")
            print(f"Device: {env_info['configuration']['device']}")
            print(f"Debug Mode: {env_info['configuration']['debug_mode']}")
            
        except Exception as e:
            logger.error(f"System check failed: {str(e)}")
            raise
    
    def determine_deployment_mode(self) -> str:
        """
        Determine deployment mode based on environment and configuration
        
        Returns:
            Deployment mode ('runpod' or 'litserve')
        """
        # Check environment variables first
        if os.getenv("RUNPOD_POD_ID"):
            logger.info("Detected RunPod environment via RUNPOD_POD_ID")
            return "runpod"
        
        # Check configuration setting
        # You can add a deployment mode setting to config.yaml if needed
        default_mode = self.config.get('app.default_deployment_mode', 'litserve')
        logger.info(f"Using default deployment mode: {default_mode}")
        
        return default_mode
    
    def run(self) -> None:
        """Main application entry point"""
        try:
            # Parse arguments (utility commands only)
            args = self.parse_arguments()
            
            # Setup debug mode if requested
            self.setup_debug_mode(args.debug)
            
            # Handle utility commands
            if args.create_test:
                self.create_test_input()
                return
            
            if args.check_env:
                self.check_system_info()
                return
            
            if args.validate_runpod:
                self.validate_runpod_env()
                return
            
            if args.validate_config:
                self.validate_configuration()
                return
            
            if args.show_config:
                self.show_configuration()
                return
            
            # Log startup information
            logger.info("="*50)
            logger.info(f"🚀 Starting {settings.app.name} v{settings.app.version}")
            logger.info(f"Environment: {settings.get_environment()}")
            logger.info(f"Model Size: {settings.model.default_size}")
            logger.info(f"Device: {settings.model.device_preferred}")
            logger.info("="*50)
            
            # Determine deployment mode
            mode = self.determine_deployment_mode()
            
            # Run application
            if mode == "runpod":
                self.run_runpod_mode()
            else:
                self.run_litserve_mode()
                
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application failed: {str(e)}")
            sys.exit(1)
        finally:
            # Cleanup
            logger.info("Shutting down application...")
            from src.utils.log.logger import LoggerManager
            LoggerManager().shutdown()

def main():
    """Main entry point"""
    app = YOLOv11Application()
    app.run()

if __name__ == "__main__":
    main()