"""
Health check implementations for AudioProcessor.
"""
import asyncio
import time
from typing import Dict, Any, Optional
from enum import Enum
import redis
import httpx
from ..config import get_settings
from ..utils.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class HealthCheck:
    """Base class for health checks."""
    
    def __init__(self, name: str, timeout: float = 5.0):
        self.name = name
        self.timeout = timeout
    
    async def check(self) -> Dict[str, Any]:
        """Perform health check."""
        start_time = time.time()
        try:
            result = await asyncio.wait_for(self._check(), timeout=self.timeout)
            duration = time.time() - start_time
            
            return {
                "status": HealthStatus.HEALTHY,
                "duration": duration,
                "details": result
            }
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            logger.warning(f"Health check {self.name} timed out after {duration:.2f}s")
            return {
                "status": HealthStatus.UNHEALTHY,
                "duration": duration,
                "error": "Timeout"
            }
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Health check {self.name} failed: {e}")
            return {
                "status": HealthStatus.UNHEALTHY,
                "duration": duration,
                "error": str(e)
            }
    
    async def _check(self) -> Dict[str, Any]:
        """Implement specific health check logic."""
        raise NotImplementedError


class RedisHealthCheck(HealthCheck):
    """Health check for Redis connection."""
    
    def __init__(self):
        super().__init__("redis", timeout=3.0)
    
    async def _check(self) -> Dict[str, Any]:
        """Check Redis connection."""
        try:
            # Parse Redis URL
            redis_url = str(settings.celery_broker_url)
            if redis_url.startswith("redis://"):
                redis_url = redis_url[8:]  # Remove redis:// prefix
            
            # Connect to Redis
            client = redis.from_url(f"redis://{redis_url}")
            
            # Test connection
            await asyncio.get_event_loop().run_in_executor(
                None, client.ping
            )
            
            # Get Redis info
            info = await asyncio.get_event_loop().run_in_executor(
                None, client.info
            )
            
            client.close()
            
            return {
                "connected": True,
                "version": info.get("redis_version"),
                "uptime": info.get("uptime_in_seconds"),
                "memory_used": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients")
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e)
            }


class S3HealthCheck(HealthCheck):
    """Health check for S3 connection."""
    
    def __init__(self):
        super().__init__("s3", timeout=10.0)
    
    async def _check(self) -> Dict[str, Any]:
        """Check S3 connection."""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            # Create S3 client
            s3_client = boto3.client(
                's3',
                endpoint_url=str(settings.s3_endpoint),
                aws_access_key_id=settings.s3_access_key,
                aws_secret_access_key=settings.s3_secret_key,
                region_name=settings.s3_region
            )
            
            # Test connection by listing buckets
            response = await asyncio.get_event_loop().run_in_executor(
                None, s3_client.list_buckets
            )
            
            # Check if our bucket exists
            bucket_exists = any(
                bucket['Name'] == settings.s3_bucket 
                for bucket in response.get('Buckets', [])
            )
            
            return {
                "connected": True,
                "bucket_exists": bucket_exists,
                "bucket_name": settings.s3_bucket,
                "total_buckets": len(response.get('Buckets', []))
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e)
            }


class MasterMLHealthCheck(HealthCheck):
    """Health check for MasterML connection."""
    
    def __init__(self):
        super().__init__("masterml", timeout=5.0)
    
    async def _check(self) -> Dict[str, Any]:
        """Check MasterML connection."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{settings.masterml_url}/health")
                response.raise_for_status()
                
                data = response.json()
                return {
                    "connected": True,
                    "status": data.get("status"),
                    "version": data.get("version"),
                    "uptime": data.get("uptime")
                }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e)
            }


class CeleryHealthCheck(HealthCheck):
    """Health check for Celery workers."""
    
    def __init__(self):
        super().__init__("celery", timeout=5.0)
    
    async def _check(self) -> Dict[str, Any]:
        """Check Celery workers."""
        try:
            from .celery_app import celery_app
            
            # Get active workers
            inspect = celery_app.control.inspect()
            active_workers = await asyncio.get_event_loop().run_in_executor(
                None, inspect.active
            )
            
            # Get worker stats
            stats = await asyncio.get_event_loop().run_in_executor(
                None, inspect.stats
            )
            
            # Get registered tasks
            registered = await asyncio.get_event_loop().run_in_executor(
                None, inspect.registered
            )
            
            worker_count = len(active_workers) if active_workers else 0
            
            return {
                "workers_active": worker_count,
                "active_workers": list(active_workers.keys()) if active_workers else [],
                "stats": stats,
                "registered_tasks": registered
            }
        except Exception as e:
            return {
                "workers_active": 0,
                "error": str(e)
            }


class SystemHealthCheck(HealthCheck):
    """Health check for system resources."""
    
    def __init__(self):
        super().__init__("system", timeout=2.0)
    
    async def _check(self) -> Dict[str, Any]:
        """Check system resources."""
        try:
            import psutil
            
            # Get system info
            cpu_percent = await asyncio.get_event_loop().run_in_executor(
                None, psutil.cpu_percent, 1
            )
            
            memory = await asyncio.get_event_loop().run_in_executor(
                None, psutil.virtual_memory
            )
            
            disk = await asyncio.get_event_loop().run_in_executor(
                None, psutil.disk_usage, "/"
            )
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            }
        except Exception as e:
            return {
                "error": str(e)
            }


class HealthChecker:
    """Main health checker that runs all checks."""
    
    def __init__(self):
        self.checks = {
            "redis": RedisHealthCheck(),
            "s3": S3HealthCheck(),
            "masterml": MasterMLHealthCheck(),
            "celery": CeleryHealthCheck(),
            "system": SystemHealthCheck()
        }
    
    async def check_all(self) -> Dict[str, Any]:
        """Run all health checks."""
        start_time = time.time()
        
        # Run all checks concurrently
        check_tasks = {
            name: check.check() 
            for name, check in self.checks.items()
        }
        
        results = await asyncio.gather(
            *check_tasks.values(),
            return_exceptions=True
        )
        
        # Combine results
        health_results = {}
        overall_status = HealthStatus.HEALTHY
        
        for i, (name, result) in enumerate(zip(self.checks.keys(), results)):
            if isinstance(result, Exception):
                health_results[name] = {
                    "status": HealthStatus.UNHEALTHY,
                    "error": str(result)
                }
                overall_status = HealthStatus.UNHEALTHY
            else:
                health_results[name] = result
                if result["status"] == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif result["status"] == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
        
        total_duration = time.time() - start_time
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "duration": total_duration,
            "checks": health_results
        }
    
    async def check_specific(self, check_name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if check_name not in self.checks:
            raise ValueError(f"Unknown health check: {check_name}")
        
        return await self.checks[check_name].check()


# Global health checker instance
health_checker = HealthChecker()
