# src/utils/sync_backend_checker.py
import os
import requests
from typing import Dict, Any, List
from src.utils.config.settings import get_settings
from src.utils.log.logger import get_module_logger

# Initialize settings and logger
settings = get_settings()
logger = get_module_logger(__name__)

class SyncBackendChecker:
    """Synchronous backend health checker for startup validation"""
    
    def __init__(self):
        self.gradio_config = settings.get_section('gradio')
        self.backends_config = self.gradio_config.get('backends', {}) if self.gradio_config else {}
    
    def get_available_backends(self) -> List[Dict[str, Any]]:
        """Get list of available backends"""
        backends = []
        
        # Check LitServe
        litserve_config = self.backends_config.get('litserve', {})
        if litserve_config.get('enabled', False):
            backends.append({
                "type": "litserve",
                "name": litserve_config.get('name', 'LitServe'),
                "base_url": litserve_config.get('base_url', 'http://localhost:8000'),
                "config": litserve_config
            })
        
        # Check RunPod
        runpod_config = self.backends_config.get('runpod', {})
        if runpod_config.get('enabled', False):
            backends.append({
                "type": "runpod",
                "name": runpod_config.get('name', 'RunPod'),
                "base_url": runpod_config.get('base_url', 'https://api.runpod.ai/v2'),
                "config": runpod_config
            })
        
        return backends
    
    def check_litserve_health(self, backend_config: Dict[str, Any]) -> bool:
        """Check LitServe backend health"""
        try:
            base_url = backend_config.get('base_url', 'http://localhost:8000')
            health_endpoint = backend_config.get('endpoints', {}).get('health', '/health')
            health_url = f"{base_url}{health_endpoint}"
            
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
            
        except Exception as e:
            logger.debug(f"LitServe health check failed: {e}")
            return False
    
    def check_runpod_health(self, backend_config: Dict[str, Any]) -> bool:
        """Check RunPod backend configuration"""
        try:
            # For RunPod, we mainly check if credentials are configured
            api_key = backend_config.get('api_key') or os.getenv('RUNPOD_API_KEY')
            endpoint_id = backend_config.get('endpoint_id') or os.getenv('RUNPOD_ENDPOINT_ID')
            
            if not api_key:
                logger.debug("RunPod API key not configured")
                return False
            
            if not endpoint_id:
                logger.debug("RunPod endpoint ID not configured")
                return False
            
            # Basic credential validation
            if len(api_key) < 10:
                logger.debug("RunPod API key appears invalid")
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"RunPod health check failed: {e}")
            return False
    
    def check_all_backends(self) -> Dict[str, Any]:
        """Check all available backends"""
        results = {
            "total": 0,
            "healthy": 0,
            "backends": []
        }
        
        available_backends = self.get_available_backends()
        results["total"] = len(available_backends)
        
        for backend in available_backends:
            backend_type = backend["type"]
            backend_name = backend["name"]
            backend_config = backend["config"]
            
            if backend_type == "litserve":
                is_healthy = self.check_litserve_health(backend_config)
            elif backend_type == "runpod":
                is_healthy = self.check_runpod_health(backend_config)
            else:
                is_healthy = False
            
            if is_healthy:
                results["healthy"] += 1
            
            results["backends"].append({
                "type": backend_type,
                "name": backend_name,
                "healthy": is_healthy,
                "url": backend.get("base_url", "")
            })
        
        return results

# Global instance
sync_backend_checker = SyncBackendChecker()