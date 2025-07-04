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
        Parse command line arguments
        
        Returns:
            Parsed arguments
        """
        parser = argparse.ArgumentParser(
            description="YOLOv11 API with RunPod and LitServe support",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run with LitServe
  python main.py --mode litserve --host 0.0.0.0 --port 8000
  
  # Run with RunPod
  python main.py --mode runpod --model-size s
  
  # Create test input for RunPod
  python main.py --create-test
  
  # Check environment
  python main.py --check-env
  
  # Validate configuration
  python main.py --validate-config
            """
        )
        
        # Mode selection
        parser.add_argument(
            "--mode",
            choices=["runpod", "litserve"],
            default=None,
            help="Deployment mode: runpod or litserve"
        )
        
        # Model configuration
        parser.add_argument(
            "--model-size",
            choices=settings.model.available_sizes,
            default=None,
            help="YOLO model size"
        )
        
        parser.add_argument(
            "--device",
            default=None,
            help="Device to run inference on (auto, cpu, cuda)"
        )
        
        # LitServe specific arguments
        parser.add_argument(
            "--host",
            default=None,
            help="Host to bind to (LitServe only)"
        )
        
        parser.add_argument(
            "--port",
            type=int,
            default=None,
            help="Port to bind to (LitServe only)"
        )
        
        parser.add_argument(
            "--workers",
            type=int,
            default=None,
            help="Number of workers per device (LitServe only)"
        )
        
        parser.add_argument(
            "--timeout",
            type=int,
            default=None,
            help="Request timeout in seconds (LitServe only)"
        )
        
        # Configuration options
        parser.add_argument(
            "--config-file",
            default=None,
            help="Path to configuration file"
        )
        
        parser.add_argument(
            "--environment",
            choices=["development", "production", "testing"],
            default=None,
            help="Environment mode"
        )
        
        # Utility commands
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
            help="Enable debug mode"
        )
        
        return parser.parse_args()
    
    def setup_from_args(self, args: argparse.Namespace) -> None:
        """
        Setup configuration from command line arguments
        
        Args:
            args: Parsed command line arguments
        """
        # Set environment if provided
        if args.environment:
            settings.set_environment(args.environment)
        
        # Override configuration with command line arguments
        if args.model_size:
            settings.model.set('default_size', args.model_size)
        
        if args.device:
            settings.model.set('device.preferred', args.device)
        
        if args.host:
            settings.litserve.set('server.host', args.host)
        
        if args.port:
            settings.litserve.set('server.port', args.port)
        
        if args.workers:
            settings.litserve.set('server.workers_per_device', args.workers)
        
        if args.timeout:
            settings.litserve.set('server.timeout', args.timeout)
        
        # Enable debug mode if requested
        if args.debug:
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
    
    def run_litserve_mode(self, args: argparse.Namespace) -> None:
        """
        Run application in LitServe mode
        
        Args:
            args: Parsed command line arguments
        """
        try:
            model_size = settings.model.default_size
            logger.info(f"Starting LitServe mode with model size: {model_size}")
            
            # Create LitServe manager
            manager = YOLOv11LitServeManager(model_size=model_size)
            
            # Get configuration from settings
            litserve_config = settings.litserve
            
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
    
    def run_runpod_mode(self, args: argparse.Namespace) -> None:
        """
        Run application in RunPod mode
        
        Args:
            args: Parsed command line arguments
        """
        try:
            model_size = settings.model.default_size
            device = settings.model.device_preferred
            
            logger.info(f"Starting RunPod mode with model size: {model_size}, device: {device}")
            
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
            print(f"Model Size: {settings.model.default_size}")
            print(f"Device: {settings.model.device_preferred}")
            print(f"LitServe Host: {settings.litserve.host}")
            print(f"LitServe Port: {settings.litserve.port}")
            print(f"Debug Mode: {settings.development.debug_mode}")
            
            # Show feature flags
            print(f"\n🚀 Feature Flags:")
            print(f"Batch Processing: {settings.features.enable_batch_processing}")
            print(f"Async Processing: {settings.features.enable_async_processing}")
            print(f"Model Caching: {settings.features.enable_model_caching}")
            print(f"Health Checks: {settings.features.enable_health_checks}")
            
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
            
            print("\n⚙️  Configuration:")
            print(f"Environment: {env_info['configuration']['current_environment']}")
            print(f"Model Size: {env_info['configuration']['model_size']}")
            print(f"Device: {env_info['configuration']['device']}")
            print(f"Debug Mode: {env_info['configuration']['debug_mode']}")
            
        except Exception as e:
            logger.error(f"System check failed: {str(e)}")
            raise
    
    def determine_deployment_mode(self, args: argparse.Namespace) -> str:
        """
        Determine deployment mode based on arguments and environment
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Deployment mode ('runpod' or 'litserve')
        """
        # Check explicit mode argument
        if args.mode:
            return args.mode
        
        # Check environment variables
        if os.getenv("RUNPOD_POD_ID"):
            return "runpod"
        
        # Default to litserve
        return "litserve"
    
    def run(self) -> None:
        """Main application entry point"""
        try:
            # Parse arguments
            args = self.parse_arguments()
            
            # Setup configuration from arguments
            self.setup_from_args(args)
            
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
            
            # Determine deployment mode
            mode = self.determine_deployment_mode(args)
            
            # Run application
            if mode == "runpod":
                self.run_runpod_mode(args)
            else:
                self.run_litserve_mode(args)
                
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application failed: {str(e)}")
            sys.exit(1)
        finally:
            # Cleanup
            from src.utils.log.logger import LoggerManager
            LoggerManager().shutdown()

def main():
    """Main entry point"""
    app = YOLOv11Application()
    app.run()

if __name__ == "__main__":
    main()