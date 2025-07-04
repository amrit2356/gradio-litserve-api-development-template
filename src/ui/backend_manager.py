# src/clients/backend_client.py
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
        
    async def health_check(self) -> bool:
        """Check if LitServe backend is healthy"""
        try:
            start_time = time.time()
            health_url = f"{self.base_url}{self.endpoints.get('health', '/health')}"
            
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
    
    async def predict(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send prediction request to LitServe"""
        try:
            predict_url = f"{self.base_url}{self.endpoints.get('predict', '/predict')}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    predict_url,
                    json=request_data,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        raise Exception(f"Prediction failed with status {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"LitServe prediction failed: {e}")
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
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for RunPod API requests"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def health_check(self) -> bool:
        """Check if RunPod backend is accessible"""
        try:
            if not self.api_key or not self.endpoint_id:
                self.health_status.is_healthy = False
                self.health_status.error_message = "Missing API key or endpoint ID"
                return False
            
            start_time = time.time()
            # Test with a simple status check
            url = f"{self.base_url}/{self.endpoint_id}/health"
            
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
    
    async def run_sync(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run synchronous request on RunPod"""
        try:
            url = f"{self.base_url}/{self.endpoint_id}/runsync"
            
            payload = {
                "input": request_data,
                "webhook": None
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=self._get_headers(),
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        raise Exception(f"RunPod request failed with status {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"RunPod sync request failed: {e}")
            raise
    
    async def run_async(self, request_data: Dict[str, Any]) -> str:
        """Run asynchronous request on RunPod"""
        try:
            url = f"{self.base_url}/{self.endpoint_id}/run"
            
            payload = {
                "input": request_data,
                "webhook": None
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=self._get_headers(),
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("id")
                    else:
                        error_text = await response.text()
                        raise Exception(f"RunPod async request failed with status {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"RunPod async request failed: {e}")
            raise
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a RunPod job"""
        try:
            url = f"{self.base_url}/{self.endpoint_id}/status/{job_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self._get_headers(),
                    timeout=10
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"Status check failed with status {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"RunPod status check failed: {e}")
            raise
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a RunPod job"""
        try:
            url = f"{self.base_url}/{self.endpoint_id}/cancel/{job_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=self._get_headers(),
                    timeout=10
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"RunPod job cancellation failed: {e}")
            return False

class BackendManager:
    """Manages multiple backend clients"""
    
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
        
        if self.backends_config.get('runpod', {}).get('enabled', False):
            self.runpod_client = RunPodClient(self.backends_config['runpod'])
        
        # Job management
        self.jobs: Dict[str, Job] = {}
        self.job_cleanup_interval = self.backends_config.get('runpod', {}).get('job_management', {}).get('cleanup_interval', 3600)
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background tasks for health checks and job cleanup"""
        # Only start background tasks if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, create tasks
            
            # Health check intervals
            if self.litserve_client:
                litserve_interval = self.backends_config.get('litserve', {}).get('health_check', {}).get('interval', 30)
                loop.create_task(self._periodic_health_check(BackendType.LITSERVE, litserve_interval))
            
            # Job cleanup task
            loop.create_task(self._periodic_job_cleanup())
            
        except RuntimeError:
            # No event loop running, skip background tasks for now
            logger.debug("No event loop running, skipping background tasks initialization")
            pass
    
    async def _periodic_health_check(self, backend_type: BackendType, interval: int):
        """Periodic health check for backends"""
        while True:
            try:
                if backend_type == BackendType.LITSERVE and self.litserve_client:
                    await self.litserve_client.health_check()
                elif backend_type == BackendType.RUNPOD and self.runpod_client:
                    await self.runpod_client.health_check()
                
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in periodic health check for {backend_type}: {e}")
                await asyncio.sleep(interval)
    
    async def _periodic_job_cleanup(self):
        """Periodic cleanup of old jobs"""
        while True:
            try:
                await asyncio.sleep(self.job_cleanup_interval)
                await self.cleanup_old_jobs()
            except Exception as e:
                logger.error(f"Error in job cleanup: {e}")
    
    def start_background_tasks_if_needed(self):
        """Start background tasks if not already started"""
        try:
            loop = asyncio.get_running_loop()
            
            # Health check intervals
            if self.litserve_client:
                litserve_interval = self.backends_config.get('litserve', {}).get('health_check', {}).get('interval', 30)
                loop.create_task(self._periodic_health_check(BackendType.LITSERVE, litserve_interval))
            
            # Job cleanup task
            loop.create_task(self._periodic_job_cleanup())
            
            logger.info("Background tasks started")
            
        except RuntimeError:
            # No event loop running
            logger.debug("No event loop running, background tasks not started")
            pass
    
    def sync_submit_job(self, backend_type: BackendType, request_data: Dict[str, Any]) -> str:
        """Submit a job synchronously (for use in sync contexts)"""
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
        
        # Try to start processing if event loop is available
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._process_job(job))
        except RuntimeError:
            # No event loop, job will be processed later
            logger.debug(f"Job {job_id} queued, no event loop for immediate processing")
        
        return job_id
    
    async def cleanup_old_jobs(self):
        """Clean up old completed jobs"""
        cutoff_time = datetime.now() - timedelta(seconds=self.job_cleanup_interval)
        jobs_to_remove = []
        
        for job_id, job in self.jobs.items():
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                if job.updated_at < cutoff_time:
                    jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.jobs[job_id]
            
        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
    
    def get_available_backends(self) -> List[Dict[str, Any]]:
        """Get list of available backends"""
        backends = []
        
        if self.litserve_client:
            backends.append({
                "type": BackendType.LITSERVE.value,
                "name": self.backends_config['litserve'].get('name', 'LitServe'),
                "description": self.backends_config['litserve'].get('description', 'Live server'),
                "enabled": True,
                "health_status": self.litserve_client.health_status.to_dict()
            })
        
        if self.runpod_client:
            backends.append({
                "type": BackendType.RUNPOD.value,
                "name": self.backends_config['runpod'].get('name', 'RunPod'),
                "description": self.backends_config['runpod'].get('description', 'Serverless'),
                "enabled": True,
                "health_status": self.runpod_client.health_status.to_dict()
            })
        
        return backends
    
    async def submit_job(self, backend_type: BackendType, request_data: Dict[str, Any]) -> str:
        """Submit a job to the specified backend"""
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
        
        # Start processing
        asyncio.create_task(self._process_job(job))
        
        return job_id
    
    async def _process_job(self, job: Job):
        """Process a job"""
        try:
            job.status = JobStatus.RUNNING
            job.updated_at = datetime.now()
            
            start_time = time.time()
            
            if job.backend_type == BackendType.LITSERVE:
                result = await self.litserve_client.predict(job.request_data)
            elif job.backend_type == BackendType.RUNPOD:
                result = await self.runpod_client.run_sync(job.request_data)
            else:
                raise ValueError(f"Unknown backend type: {job.backend_type}")
            
            execution_time = time.time() - start_time
            
            job.status = JobStatus.COMPLETED
            job.response_data = result
            job.execution_time = execution_time
            job.updated_at = datetime.now()
            
            logger.info(f"Job {job.id} completed in {execution_time:.2f}s")
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.updated_at = datetime.now()
            
            logger.error(f"Job {job.id} failed: {e}")
    
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
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False
        
        job.status = JobStatus.CANCELLED
        job.updated_at = datetime.now()
        
        # If it's a RunPod job, try to cancel it on the server
        if job.backend_type == BackendType.RUNPOD and self.runpod_client:
            # This would require storing the RunPod job ID
            pass
        
        logger.info(f"Job {job_id} cancelled")
        return True
    
    def get_backend_health(self) -> Dict[str, Any]:
        """Get health status of all backends"""
        health_status = {}
        
        if self.litserve_client:
            health_status["litserve"] = self.litserve_client.health_status.to_dict()
        
        if self.runpod_client:
            health_status["runpod"] = self.runpod_client.health_status.to_dict()
        
        return health_status

# Global backend manager instance
backend_manager = BackendManager()