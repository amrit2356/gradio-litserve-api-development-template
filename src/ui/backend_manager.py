# src/clients/backend_client.py - FIXED VERSION
import asyncio
import json
import os
import time
import uuid
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import aiohttp
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading

from src.utils.config.settings import get_settings
from src.utils.log.logger import get_module_logger

# Initialize settings and logger
settings = get_settings()
logger = get_module_logger(__name__)

class BackendType(Enum):
    LITSERVE = "litserve"
    RUNPOD = "runpod"

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Job:
    """Represents a job for processing"""
    id: str
    backend_type: BackendType
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    request_data: Dict[str, Any]
    response_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "backend_type": self.backend_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "request_data": self.request_data,
            "response_data": self.response_data,
            "error_message": self.error_message,
            "execution_time": self.execution_time
        }

class BackendHealthStatus:
    """Represents backend health status"""
    def __init__(self, backend_type: BackendType):
        self.backend_type = backend_type
        self.is_healthy = False
        self.last_check = None
        self.response_time = None
        self.error_message = None
        self.consecutive_failures = 0
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend_type": self.backend_type.value,
            "is_healthy": self.is_healthy,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "response_time": self.response_time,
            "error_message": self.error_message,
            "consecutive_failures": self.consecutive_failures
        }

class LitServeClient:
    """Client for LitServe backend"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get("base_url", "http://localhost:8000")
        self.endpoints = config.get("endpoints", {})
        self.timeout = config.get("request_timeout", 30)
        self.health_status = BackendHealthStatus(BackendType.LITSERVE)
        
        # Add default endpoints if not provided
        if not self.endpoints:
            self.endpoints = {
                "health": "/health",
                "predict": "/predict",
                "info": "/info"
            }
        
        logger.info(f"LitServe client initialized with base_url: {self.base_url}")
    
    def sync_health_check(self) -> bool:
        """Synchronous health check for LitServe backend"""
        try:
            start_time = time.time()
            health_url = f"{self.base_url}{self.endpoints.get('health', '/health')}"
            
            logger.debug(f"Checking LitServe health at: {health_url}")
            response = requests.get(health_url, timeout=5)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                self.health_status.is_healthy = True
                self.health_status.response_time = response_time
                self.health_status.error_message = None
                self.health_status.consecutive_failures = 0
                logger.info(f"LitServe health check passed ({response_time:.2f}s)")
                return True
            else:
                raise Exception(f"Health check failed with status {response.status_code}")
                
        except Exception as e:
            self.health_status.is_healthy = False
            self.health_status.error_message = str(e)
            self.health_status.consecutive_failures += 1
            logger.error(f"LitServe health check failed: {e}")
            return False
        finally:
            self.health_status.last_check = datetime.now()
    
    async def health_check(self) -> bool:
        """Asynchronous health check for LitServe backend"""
        try:
            start_time = time.time()
            health_url = f"{self.base_url}{self.endpoints.get('health', '/health')}"
            
            logger.debug(f"Checking LitServe health at: {health_url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=5) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        self.health_status.is_healthy = True
                        self.health_status.response_time = response_time
                        self.health_status.error_message = None
                        self.health_status.consecutive_failures = 0
                        logger.info(f"LitServe health check passed ({response_time:.2f}s)")
                        return True
                    else:
                        raise Exception(f"Health check failed with status {response.status}")
                        
        except Exception as e:
            self.health_status.is_healthy = False
            self.health_status.error_message = str(e)
            self.health_status.consecutive_failures += 1
            logger.error(f"LitServe health check failed: {e}")
            return False
        finally:
            self.health_status.last_check = datetime.now()
    
    def sync_predict(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous prediction request to LitServe"""
        try:
            predict_url = f"{self.base_url}{self.endpoints.get('predict', '/predict')}"
            
            logger.debug(f"Sending prediction request to: {predict_url}")
            logger.debug(f"Request data keys: {list(request_data.keys())}")
            
            response = requests.post(
                predict_url,
                json=request_data,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"LitServe prediction successful")
                return result
            else:
                error_text = response.text
                logger.error(f"LitServe prediction failed with status {response.status_code}: {error_text}")
                raise Exception(f"Prediction failed with status {response.status_code}: {error_text}")
                
        except Exception as e:
            logger.error(f"LitServe prediction failed: {e}")
            raise
    
    async def predict(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronous prediction request to LitServe"""
        try:
            predict_url = f"{self.base_url}{self.endpoints.get('predict', '/predict')}"
            
            logger.debug(f"Sending async prediction request to: {predict_url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    predict_url,
                    json=request_data,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"LitServe async prediction successful")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"LitServe async prediction failed with status {response.status}: {error_text}")
                        raise Exception(f"Prediction failed with status {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"LitServe async prediction failed: {e}")
            raise
    
    def get_info(self) -> Dict[str, Any]:
        """Get backend information"""
        try:
            info_url = f"{self.base_url}{self.endpoints.get('info', '/info')}"
            response = requests.get(info_url, timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get info: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Failed to get LitServe info: {e}")
            return {"error": str(e)}

class RunPodClient:
    """Client for RunPod backend"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get("base_url", "https://api.runpod.ai/v2")
        self.endpoints = config.get("endpoints", {})
        self.api_key = config.get("api_key") or os.getenv("RUNPOD_API_KEY")
        self.endpoint_id = config.get("endpoint_id") or os.getenv("RUNPOD_ENDPOINT_ID")
        self.timeout = config.get("request_timeout", 300)
        self.polling_interval = config.get("polling_interval", 2)
        self.health_status = BackendHealthStatus(BackendType.RUNPOD)
        
        if not self.api_key:
            logger.warning("RunPod API key not found. Set RUNPOD_API_KEY environment variable.")
        if not self.endpoint_id:
            logger.warning("RunPod endpoint ID not found. Set RUNPOD_ENDPOINT_ID environment variable.")
        
        logger.info(f"RunPod client initialized with endpoint: {self.endpoint_id}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for RunPod API requests"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def sync_health_check(self) -> bool:
        """Synchronous health check for RunPod backend"""
        try:
            if not self.api_key or not self.endpoint_id:
                self.health_status.is_healthy = False
                self.health_status.error_message = "Missing API key or endpoint ID"
                return False
            
            start_time = time.time()
            # Test with a simple status check
            url = f"{self.base_url}/{self.endpoint_id}/health"
            
            logger.debug(f"Checking RunPod health at: {url}")
            
            response = requests.get(
                url,
                headers=self._get_headers(),
                timeout=10
            )
            
            response_time = time.time() - start_time
            
            if response.status_code in [200, 404]:  # 404 is acceptable for health check
                self.health_status.is_healthy = True
                self.health_status.response_time = response_time
                self.health_status.error_message = None
                self.health_status.consecutive_failures = 0
                logger.info(f"RunPod health check passed ({response_time:.2f}s)")
                return True
            else:
                raise Exception(f"Health check failed with status {response.status_code}")
                
        except Exception as e:
            self.health_status.is_healthy = False
            self.health_status.error_message = str(e)
            self.health_status.consecutive_failures += 1
            logger.error(f"RunPod health check failed: {e}")
            return False
        finally:
            self.health_status.last_check = datetime.now()
    
    async def health_check(self) -> bool:
        """Asynchronous health check for RunPod backend"""
        try:
            if not self.api_key or not self.endpoint_id:
                self.health_status.is_healthy = False
                self.health_status.error_message = "Missing API key or endpoint ID"
                return False
            
            start_time = time.time()
            # Test with a simple status check
            url = f"{self.base_url}/{self.endpoint_id}/health"
            
            logger.debug(f"Checking RunPod health at: {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self._get_headers(),
                    timeout=10
                ) as response:
                    response_time = time.time() - start_time
                    
                    if response.status in [200, 404]:  # 404 is acceptable for health check
                        self.health_status.is_healthy = True
                        self.health_status.response_time = response_time
                        self.health_status.error_message = None
                        self.health_status.consecutive_failures = 0
                        logger.info(f"RunPod health check passed ({response_time:.2f}s)")
                        return True
                    else:
                        raise Exception(f"Health check failed with status {response.status}")
                        
        except Exception as e:
            self.health_status.is_healthy = False
            self.health_status.error_message = str(e)
            self.health_status.consecutive_failures += 1
            logger.error(f"RunPod health check failed: {e}")
            return False
        finally:
            self.health_status.last_check = datetime.now()
    
    def sync_run_sync(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous request to RunPod"""
        try:
            url = f"{self.base_url}/{self.endpoint_id}/runsync"
            
            payload = {
                "input": request_data,
                "webhook": None
            }
            
            logger.debug(f"Sending sync request to RunPod: {url}")
            
            response = requests.post(
                url,
                json=payload,
                headers=self._get_headers(),
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"RunPod sync request successful")
                return result
            else:
                error_text = response.text
                logger.error(f"RunPod sync request failed with status {response.status_code}: {error_text}")
                raise Exception(f"RunPod request failed with status {response.status_code}: {error_text}")
                
        except Exception as e:
            logger.error(f"RunPod sync request failed: {e}")
            raise
    
    async def run_sync(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronous request to RunPod"""
        try:
            url = f"{self.base_url}/{self.endpoint_id}/runsync"
            
            payload = {
                "input": request_data,
                "webhook": None
            }
            
            logger.debug(f"Sending async request to RunPod: {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=self._get_headers(),
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"RunPod async request successful")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"RunPod async request failed with status {response.status}: {error_text}")
                        raise Exception(f"RunPod request failed with status {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"RunPod async request failed: {e}")
            raise

class BackendManager:
    """Manages multiple backend clients with proper sync/async handling"""
    
    def __init__(self):
        # Get gradio configuration
        self.gradio_config = settings.get_section('gradio')
        self.backends_config = self.gradio_config.get('backends', {}) if self.gradio_config else {}
        
        # Initialize clients
        self.litserve_client = None
        self.runpod_client = None
        
        # Initialize clients based on configuration
        if self.backends_config.get('litserve', {}).get('enabled', False):
            self.litserve_client = LitServeClient(self.backends_config['litserve'])
            logger.info("LitServe client initialized")
        
        if self.backends_config.get('runpod', {}).get('enabled', False):
            self.runpod_client = RunPodClient(self.backends_config['runpod'])
            logger.info("RunPod client initialized")
        
        # Job management
        self.jobs: Dict[str, Job] = {}
        self.job_cleanup_interval = self.backends_config.get('job_management', {}).get('cleanup_interval', 3600)
        
        # Thread pool for sync operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Background task management
        self.background_tasks_started = False
        self.background_task_lock = threading.Lock()
        
        logger.info("Backend manager initialized")
    
    def sync_submit_job(self, backend_type: BackendType, request_data: Dict[str, Any]) -> str:
        """Submit a job synchronously - FIXED VERSION"""
        job_id = str(uuid.uuid4())
        
        # Create job
        job = Job(
            id=job_id,
            backend_type=backend_type,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            request_data=request_data
        )
        
        self.jobs[job_id] = job
        
        logger.info(f"Job {job_id} created for backend {backend_type.value}")
        
        # Process job synchronously
        try:
            job.status = JobStatus.RUNNING
            job.updated_at = datetime.now()
            
            start_time = time.time()
            
            if backend_type == BackendType.LITSERVE:
                if not self.litserve_client:
                    raise Exception("LitServe client not initialized")
                
                logger.info(f"Processing job {job_id} with LitServe")
                result = self.litserve_client.sync_predict(request_data)
                
            elif backend_type == BackendType.RUNPOD:
                if not self.runpod_client:
                    raise Exception("RunPod client not initialized")
                
                logger.info(f"Processing job {job_id} with RunPod")
                result = self.runpod_client.sync_run_sync(request_data)
                
            else:
                raise ValueError(f"Unknown backend type: {backend_type}")
            
            execution_time = time.time() - start_time
            
            job.status = JobStatus.COMPLETED
            job.response_data = result
            job.execution_time = execution_time
            job.updated_at = datetime.now()
            
            logger.info(f"Job {job_id} completed successfully in {execution_time:.2f}s")
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.updated_at = datetime.now()
            
            logger.error(f"Job {job_id} failed: {e}")
            raise
        
        return job_id
    
    def get_available_backends(self) -> List[Dict[str, Any]]:
        """Get list of available backends"""
        backends = []
        
        if self.litserve_client:
            backends.append({
                "type": BackendType.LITSERVE.value,
                "name": self.backends_config.get('litserve', {}).get('name', 'LitServe'),
                "description": self.backends_config.get('litserve', {}).get('description', 'Live server'),
                "enabled": True,
                "health_status": self.litserve_client.health_status.to_dict()
            })
        
        if self.runpod_client:
            backends.append({
                "type": BackendType.RUNPOD.value,
                "name": self.backends_config.get('runpod', {}).get('name', 'RunPod'),
                "description": self.backends_config.get('runpod', {}).get('description', 'Serverless'),
                "enabled": True,
                "health_status": self.runpod_client.health_status.to_dict()
            })
        
        return backends
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    def get_jobs(self, status: Optional[JobStatus] = None) -> List[Job]:
        """Get jobs, optionally filtered by status"""
        jobs = list(self.jobs.values())
        
        if status:
            jobs = [job for job in jobs if job.status == status]
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        return jobs
    
    def get_backend_health(self) -> Dict[str, Any]:
        """Get health status of all backends"""
        health_status = {}
        
        if self.litserve_client:
            # Perform sync health check
            self.litserve_client.sync_health_check()
            health_status["litserve"] = self.litserve_client.health_status.to_dict()
        
        if self.runpod_client:
            # Perform sync health check
            self.runpod_client.sync_health_check()
            health_status["runpod"] = self.runpod_client.health_status.to_dict()
        
        return health_status
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False
        
        job.status = JobStatus.CANCELLED
        job.updated_at = datetime.now()
        
        logger.info(f"Job {job_id} cancelled")
        return True

# Global backend manager instance
backend_manager = BackendManager()