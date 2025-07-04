# start_gradio.py
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize configuration and logging
from src.utils.config.settings import get_settings, validate_settings
from src.utils.log.logger import get_module_logger
from src.utils.config.config_manager import get_config

# Initialize settings and logger
settings = get_settings()
logger = get_module_logger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="YOLOv11 Gradio Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default configuration
  python start_gradio.py
  
  # Start with custom host and port
  python start_gradio.py --host 0.0.0.0 --port 7860
  
  # Start with public sharing enabled
  python start_gradio.py --share
  
  # Start in debug mode
  python start_gradio.py --debug
  
  # Show configuration and exit
  python start_gradio.py --show-config
        """
    )
    
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind to (overrides config)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (overrides config)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public link (overrides config)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (overrides config)"
    )
    
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show Gradio configuration and exit"
    )
    
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit"
    )
    
    parser.add_argument(
        "--check-backends",
        action="store_true",
        help="Check backend connectivity and exit"
    )
    
    return parser.parse_args()

def show_gradio_config():
    """Show Gradio configuration"""
    try:
        gradio_config = settings.get_section('gradio')
        
        print("📋 Gradio Configuration:")
        print("=" * 50)
        
        # App settings
        print(f"App Name: {settings.gradio.app_name}")
        print(f"Title: {settings.gradio.app_title}")
        print(f"Theme: {settings.gradio.app_theme}")
        print()
        
        # Server settings
        print(f"Server Host: {settings.gradio.server_host}")
        print(f"Server Port: {settings.gradio.server_port}")
        print(f"Share: {settings.gradio.server_share}")
        print(f"Debug: {settings.gradio.server_debug}")
        print()
        
        # Backend settings
        print("Available Backends:")
        if settings.gradio.litserve_enabled:
            print(f"  ✅ LitServe: {settings.gradio.litserve_base_url}")
        else:
            print(f"  ❌ LitServe: Disabled")
        
        if settings.gradio.runpod_enabled:
            print(f"  ✅ RunPod: {settings.gradio.runpod_base_url}")
        else:
            print(f"  ❌ RunPod: Disabled")
        print()
        
        # Detection settings
        print("Detection Settings:")
        print(f"  Confidence Threshold: {settings.gradio.detection_confidence_default}")
        print(f"  IoU Threshold: {settings.gradio.detection_iou_default}")
        print(f"  Max Detections: {settings.gradio.detection_max_detections_default}")
        print(f"  Max Image Size: {settings.gradio.detection_max_image_size} bytes")
        print()
        
        # Feature flags
        print("Feature Flags:")
        print(f"  ✅ Notifications: {settings.gradio.notifications_enabled}")
        print(f"  ✅ Export Results: {settings.gradio.export_enabled}")
        print(f"  ✅ Job Management: {settings.gradio.job_management_enabled}")
        
    except Exception as e:
        logger.error(f"Error showing config: {e}")
        print(f"❌ Error showing configuration: {e}")

def validate_gradio_config():
    """Validate Gradio configuration"""
    try:
        print("🔍 Validating Gradio Configuration...")
        
        # Validate main configuration
        validation_results = validate_settings()
        
        if validation_results['valid']:
            print("✅ Main configuration is valid")
        else:
            print("❌ Main configuration has errors:")
            for error in validation_results['errors']:
                print(f"  - {error}")
        
        if validation_results['warnings']:
            print("⚠️  Configuration warnings:")
            for warning in validation_results['warnings']:
                print(f"  - {warning}")
        
        # Validate Gradio-specific configuration
        gradio_config = settings.get_section('gradio')
        gradio_errors = []
        gradio_warnings = []
        
        if not gradio_config:
            gradio_errors.append("No Gradio configuration found")
            print("\n🎯 Gradio-specific validation:")
            print("❌ Gradio configuration errors:")
            for error in gradio_errors:
                print(f"  - {error}")
            return False
        
        # Check required sections
        required_sections = ['app', 'server', 'backends', 'detection']
        for section in required_sections:
            if section not in gradio_config:
                gradio_errors.append(f"Missing required section: {section}")
        
        # Check backends
        if not (settings.gradio.litserve_enabled or settings.gradio.runpod_enabled):
            gradio_errors.append("No backends enabled")
        
        # Check server configuration
        port = settings.gradio.server_port
        if not isinstance(port, int) or port < 1 or port > 65535:
            gradio_errors.append(f"Invalid server port: {port}")
        
        # Report Gradio-specific validation
        print("\n🎯 Gradio-specific validation:")
        if gradio_errors:
            print("❌ Gradio configuration errors:")
            for error in gradio_errors:
                print(f"  - {error}")
        else:
            print("✅ Gradio configuration is valid")
        
        if gradio_warnings:
            print("⚠️  Gradio configuration warnings:")
            for warning in gradio_warnings:
                print(f"  - {warning}")
        
        return len(gradio_errors) == 0
        
    except Exception as e:
        logger.error(f"Error validating config: {e}")
        print(f"❌ Error validating configuration: {e}")
        return False

def check_backend_connectivity():
    """Check backend connectivity"""
    try:
        print("🔍 Checking Backend Connectivity...")
        
        # Import backend manager
        from src.ui.backend_manager import backend_manager
        
        # Get available backends
        available_backends = backend_manager.get_available_backends()
        
        if not available_backends:
            print("❌ No backends available")
            return False
        
        print(f"Found {len(available_backends)} backend(s)")
        
        # Check each backend
        import asyncio
        
        async def check_backends():
            results = []
            
            for backend in available_backends:
                backend_type = backend['type']
                backend_name = backend['name']
                
                print(f"\n🔍 Checking {backend_name}...")
                
                try:
                    if backend_type == 'litserve':
                        if backend_manager.litserve_client:
                            healthy = await backend_manager.litserve_client.health_check()
                            if healthy:
                                print(f"  ✅ {backend_name} is healthy")
                                results.append(True)
                            else:
                                print(f"  ❌ {backend_name} is unhealthy")
                                results.append(False)
                        else:
                            print(f"  ❌ {backend_name} client not initialized")
                            results.append(False)
                    
                    elif backend_type == 'runpod':
                        if backend_manager.runpod_client:
                            healthy = await backend_manager.runpod_client.health_check()
                            if healthy:
                                print(f"  ✅ {backend_name} is accessible")
                                results.append(True)
                            else:
                                print(f"  ❌ {backend_name} is not accessible")
                                results.append(False)
                        else:
                            print(f"  ❌ {backend_name} client not initialized")
                            results.append(False)
                    
                except Exception as e:
                    print(f"  ❌ {backend_name} check failed: {e}")
                    results.append(False)
            
            return results
        
        # Run async checks
        results = asyncio.run(check_backends())
        
        # Summary
        healthy_count = sum(results)
        total_count = len(results)
        
        print(f"\n📊 Backend Health Summary:")
        print(f"  Healthy: {healthy_count}/{total_count}")
        
        if healthy_count == total_count:
            print("✅ All backends are healthy")
            return True
        elif healthy_count > 0:
            print("⚠️  Some backends are unhealthy")
            return True
        else:
            print("❌ No backends are healthy")
            return False
        
    except Exception as e:
        logger.error(f"Error checking backends: {e}")
        print(f"❌ Error checking backend connectivity: {e}")
        return False

def apply_command_line_overrides(args):
    """Apply command line argument overrides"""
    try:
        overrides_applied = []
        
        if args.host is not None:
            settings.set('gradio.server.host', args.host)
            overrides_applied.append(f"host={args.host}")
        
        if args.port is not None:
            settings.set('gradio.server.port', args.port)
            overrides_applied.append(f"port={args.port}")
        
        if args.share:
            settings.set('gradio.server.share', True)
            overrides_applied.append("share=True")
        
        if args.debug:
            settings.set('gradio.server.debug', True)
            settings.set('gradio.development.debug_mode', True)
            overrides_applied.append("debug=True")
        
        if overrides_applied:
            logger.info(f"Applied command line overrides: {', '.join(overrides_applied)}")
            print(f"🔧 Applied overrides: {', '.join(overrides_applied)}")
        
    except Exception as e:
        logger.error(f"Error applying overrides: {e}")
        print(f"❌ Error applying command line overrides: {e}")

def main():
    """Main entry point"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Handle utility commands
        if args.show_config:
            show_gradio_config()
            return
        
        if args.validate_config:
            is_valid = validate_gradio_config()
            sys.exit(0 if is_valid else 1)
        
        if args.check_backends:
            is_healthy = check_backend_connectivity()
            sys.exit(0 if is_healthy else 1)
        
        # Apply command line overrides
        apply_command_line_overrides(args)
        
        # Validate configuration before starting
        print("🔍 Validating configuration...")
        is_valid = validate_gradio_config()
        
        if not is_valid:
            print("❌ Configuration validation failed. Please fix errors before starting.")
            sys.exit(1)
        
        # Check backend connectivity
        print("\n🔍 Checking backend connectivity...")
        backends_healthy = check_backend_connectivity()
        
        if not backends_healthy:
            print("⚠️  Warning: No backends are healthy. The app will start but may not function properly.")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
        
        # Start Gradio app
        print("\n🚀 Starting Gradio application...")
        logger.info("Starting Gradio application")
        
        # Import and run the app
        from src.ui.gradio_app import GradioApp
        
        app = GradioApp()
        app.run()
        
    except KeyboardInterrupt:
        logger.info("Gradio startup interrupted by user")
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Gradio startup failed: {e}")
        print(f"❌ Failed to start Gradio app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()