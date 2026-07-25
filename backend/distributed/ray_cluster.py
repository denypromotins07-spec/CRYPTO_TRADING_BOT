"""
Ray Cluster Initialization: Local Ray cluster capped at 4GB RAM.
Implements graceful shutdown and memory management for the 4-hour trading window.
Uses Actor Model pattern for parallel asset processing.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import logging
import os
import sys
import time
import psutil
import ray
from ray import actor
from ray.actor import ActorHandle

logger = logging.getLogger(__name__)


class RayClusterConfig:
    """Configuration for Ray cluster initialization."""
    
    def __init__(
        self,
        max_memory_gb: float = 4.0,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
        object_store_memory_gb: Optional[float] = None,
        temp_dir: Optional[str] = None,
        include_dashboard: bool = False,
        dashboard_port: int = 8265,
    ):
        self.max_memory_gb = max_memory_gb
        self.num_cpus = num_cpus or min(4, os.cpu_count() or 4)
        self.num_gpus = num_gpus or 0
        self.object_store_memory_gb = object_store_memory_gb or (max_memory_gb * 0.3)
        self.temp_dir = temp_dir
        self.include_dashboard = include_dashboard
        self.dashboard_port = dashboard_port
        
        # Enforce 8GB total system limit
        self._validate_constraints()
    
    def _validate_constraints(self) -> None:
        """Validate configuration against system constraints."""
        if self.max_memory_gb > 4.0:
            logger.warning(f"Reducing max_memory_gb from {self.max_memory_gb} to 4.0 (limit)")
            self.max_memory_gb = 4.0
        
        if self.object_store_memory_gb > self.max_memory_gb * 0.5:
            logger.warning("Object store memory limited to 50% of max memory")
            self.object_store_memory_gb = self.max_memory_gb * 0.5
    
    def to_ray_init_args(self) -> Dict[str, Any]:
        """Convert to Ray init arguments."""
        args = {
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
            "_memory": int(self.max_memory_gb * 1024**3),  # Convert to bytes
            "object_store_memory": int(self.object_store_memory_gb * 1024**3),
            "include_dashboard": self.include_dashboard,
            "log_to_driver": True,
            "logging_level": logging.INFO,
        }
        
        if self.temp_dir:
            args["temp_dir"] = self.temp_dir
        
        if not self.include_dashboard:
            args["dashboard_host"] = "127.0.0.1"
        
        return args


class RayClusterManager:
    """
    Manages the lifecycle of the Ray cluster.
    Implements Singleton pattern for global access.
    Ensures graceful shutdown when 4-hour trading window closes.
    """
    
    _instance: Optional["RayClusterManager"] = None
    _initialized: bool = False
    
    def __new__(cls) -> "RayClusterManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.config: Optional[RayClusterConfig] = None
        self._is_running: bool = False
        self._start_time: Optional[float] = None
        self._asset_actors: Dict[str, ActorHandle] = {}
        self._monitor_thread: Optional[Any] = None
        self._shutdown_callbacks: List[Any] = []
        
        logger.info("RayClusterManager initialized")
    
    @classmethod
    def get_instance(cls) -> "RayClusterManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def initialize(self, config: Optional[RayClusterConfig] = None) -> bool:
        """
        Initialize the Ray cluster with given configuration.
        
        Args:
            config: RayClusterConfig instance or None for defaults
            
        Returns:
            True if initialization successful, False otherwise
        """
        if self._is_running:
            logger.warning("Ray cluster already running")
            return True
        
        try:
            self.config = config or RayClusterConfig()
            
            # Check if Ray is already initialized
            if not ray.is_initialized():
                init_args = self.config.to_ray_init_args()
                
                logger.info(f"Initializing Ray cluster with {self.config.max_memory_gb}GB memory limit")
                logger.info(f"CPUs: {self.config.num_cpus}, GPUs: {self.config.num_gpus}")
                
                ray.init(**init_args)
                
                self._is_running = True
                self._start_time = time.time()
                
                logger.info(f"Ray cluster initialized successfully")
                logger.info(f"Dashboard: http://localhost:{self.config.dashboard_port}" 
                           if self.config.include_dashboard else "Dashboard disabled")
                
                # Start memory monitoring
                self._start_memory_monitoring()
                
                return True
            else:
                logger.info("Ray already initialized by external process")
                self._is_running = True
                self._start_time = time.time()
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize Ray cluster: {e}")
            return False
    
    def _start_memory_monitoring(self) -> None:
        """Start background thread for memory monitoring."""
        import threading
        
        def monitor_loop():
            while self._is_running:
                self._check_memory_usage()
                time.sleep(5.0)  # Check every 5 seconds
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.debug("Memory monitoring started")
    
    def _check_memory_usage(self) -> None:
        """Check current memory usage and log warnings."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            max_memory_mb = self.config.max_memory_gb * 1024 if self.config else 4096
            
            usage_pct = (memory_mb / max_memory_mb) * 100
            
            if usage_pct > 90:
                logger.critical(f"CRITICAL: Memory usage at {usage_pct:.1f}% ({memory_mb:.0f}/{max_memory_mb:.0f}MB)")
            elif usage_pct > 75:
                logger.warning(f"WARNING: Memory usage at {usage_pct:.1f}% ({memory_mb:.0f}/{max_memory_mb:.0f}MB)")
            elif usage_pct > 50:
                logger.info(f"Memory usage at {usage_pct:.1f}% ({memory_mb:.0f}/{max_memory_mb:.0f}MB)")
                
        except Exception as e:
            logger.debug(f"Memory check failed: {e}")
    
    def register_asset_actor(self, asset: str, actor_handle: ActorHandle) -> None:
        """Register an asset worker actor."""
        self._asset_actors[asset] = actor_handle
        logger.info(f"Registered actor for {asset}")
    
    def unregister_asset_actor(self, asset: str) -> None:
        """Unregister an asset worker actor."""
        if asset in self._asset_actors:
            del self._asset_actors[asset]
            logger.info(f"Unregistered actor for {asset}")
    
    def get_asset_actor(self, asset: str) -> Optional[ActorHandle]:
        """Get actor handle for an asset."""
        return self._asset_actors.get(asset)
    
    def get_all_actors(self) -> Dict[str, ActorHandle]:
        """Get all registered actors."""
        return self._asset_actors.copy()
    
    def register_shutdown_callback(self, callback: Any) -> None:
        """Register a callback to be called on shutdown."""
        self._shutdown_callbacks.append(callback)
    
    def get_uptime(self) -> float:
        """Get cluster uptime in seconds."""
        if self._start_time:
            return time.time() - self._start_time
        return 0.0
    
    def is_within_trading_window(self, max_window_hours: float = 4.0) -> bool:
        """Check if still within the 4-hour trading window."""
        if not self._start_time:
            return False
        
        elapsed_hours = self.get_uptime() / 3600.0
        return elapsed_hours < max_window_hours
    
    def shutdown(self, force: bool = False) -> None:
        """
        Gracefully shutdown the Ray cluster.
        
        Args:
            force: If True, force immediate shutdown without waiting
        """
        if not self._is_running:
            logger.warning("Ray cluster not running")
            return
        
        logger.info("Initiating Ray cluster shutdown...")
        
        # Call shutdown callbacks
        for callback in self._shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Shutdown callback error: {e}")
        
        # Stop asset actors gracefully
        if not force:
            logger.info("Stopping asset actors...")
            for asset, actor in list(self._asset_actors.items()):
                try:
                    ray.kill(actor)
                    logger.info(f"Stopped actor for {asset}")
                except Exception as e:
                    logger.error(f"Error stopping actor for {asset}: {e}")
        
        self._asset_actors.clear()
        
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray cluster shut down successfully")
        
        self._is_running = False
        self._start_time = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("Ray cluster shutdown complete")
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get current cluster statistics."""
        if not self._is_running or not ray.is_initialized():
            return {"status": "not_running"}
        
        try:
            resources = ray.available_resources()
            
            stats = {
                "status": "running",
                "uptime_seconds": self.get_uptime(),
                "uptime_hours": self.get_uptime() / 3600.0,
                "within_trading_window": self.is_within_trading_window(),
                "available_cpus": resources.get("CPU", 0),
                "available_memory_gb": resources.get("memory", 0) / (1024**3),
                "active_actors": len(self._asset_actors),
                "assets": list(self._asset_actors.keys()),
            }
            
            # Add process memory info
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            stats["process_memory_mb"] = memory_info.rss / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cluster stats: {e}")
            return {"status": "error", "message": str(e)}


# Convenience functions
def initialize_ray_cluster(max_memory_gb: float = 4.0) -> bool:
    """Initialize Ray cluster with specified memory limit."""
    config = RayClusterConfig(max_memory_gb=max_memory_gb)
    manager = RayClusterManager.get_instance()
    return manager.initialize(config)


def get_ray_cluster_manager() -> RayClusterManager:
    """Get the Ray cluster manager singleton."""
    return RayClusterManager.get_instance()


def shutdown_ray_cluster(force: bool = False) -> None:
    """Shutdown the Ray cluster."""
    manager = RayClusterManager.get_instance()
    manager.shutdown(force=force)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Ray Cluster Manager...")
    
    # Initialize
    success = initialize_ray_cluster(max_memory_gb=2.0)  # Use 2GB for testing
    print(f"Initialization successful: {success}")
    
    if success:
        manager = get_ray_cluster_manager()
        
        # Get stats
        stats = manager.get_cluster_stats()
        print(f"Cluster stats: {stats}")
        
        # Check trading window
        in_window = manager.is_within_trading_window()
        print(f"Within trading window: {in_window}")
        
        # Shutdown
        shutdown_ray_cluster()
        print("Shutdown complete")
    
    print("Ray Cluster module test complete.")
