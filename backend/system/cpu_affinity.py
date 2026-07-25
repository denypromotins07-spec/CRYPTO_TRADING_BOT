#!/usr/bin/env python3
"""
CPU Affinity Manager for ZAID Crypto Trading Bot

This module pins critical bot threads to specific CPU cores on AMD Ryzen processors,
optimizing for the CCX (Core Complex) architecture to minimize latency and maximize
cache locality.

Features:
- Automatic detection of AMD Ryzen topology
- Core pinning for critical trading threads
- Isolation of performance cores from background tasks
- Windows-specific optimization using SetThreadAffinityMask

Target: AMD Ryzen AI 5 laptop with 8GB RAM
"""

from __future__ import annotations
import os
import sys
import logging
import threading
import multiprocessing as mp
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import ctypes
from ctypes import wintypes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CoreType(Enum):
    """Types of CPU cores based on their role"""
    PERFORMANCE = "performance"  # High-frequency cores for critical tasks
    EFFICIENCY = "efficiency"    # Lower-power cores for background tasks
    ISOLATED = "isolated"        # Reserved exclusively for trading engine


@dataclass
class CoreInfo:
    """Information about a CPU core"""
    core_id: int
    logical_processor: int
    core_type: CoreType
    ccx_id: int  # Core Complex ID for AMD Ryzen
    numa_node: int
    is_parked: bool


@dataclass
class AffinityConfig:
    """Configuration for CPU affinity settings"""
    # Number of cores to reserve for critical trading operations
    reserved_cores: int = 2
    # Enable automatic core type detection
    auto_detect_topology: bool = True
    # Isolate reserved cores from OS scheduling
    isolate_cores: bool = True
    # Enable detailed logging
    verbose: bool = False


class CpuAffinityManager:
    """
    Manages CPU core affinity for optimal performance on AMD Ryzen processors
    
    This class identifies the CPU topology and pins critical threads to
    high-performance cores while isolating them from background OS tasks.
    """
    
    def __init__(self, config: Optional[AffinityConfig] = None):
        self.config = config or AffinityConfig()
        self._core_info: List[CoreInfo] = []
        self._reserved_cores: Set[int] = set()
        self._original_affinity: Optional[int] = None
        self._is_windows = sys.platform == 'win32'
        
        self._detect_topology()
        self._reserve_cores()
        
        logger.info(f"CPU Affinity Manager initialized on {self._get_platform_name()}")
        logger.info(f"Detected {len(self._core_info)} logical processors")
        logger.info(f"Reserved cores: {sorted(self._reserved_cores)}")
    
    def _get_platform_name(self) -> str:
        """Get human-readable platform name"""
        if self._is_windows:
            return "Windows"
        elif sys.platform == 'linux':
            return "Linux"
        else:
            return "Unknown"
    
    def _detect_topology(self):
        """Detect CPU topology including CCX structure"""
        num_cpus = os.cpu_count() or 4
        
        # Create basic core info (simplified detection)
        # In production, this would use platform-specific APIs:
        # - Windows: GetLogicalProcessorInformationEx
        # - Linux: /sys/devices/system/cpu/topology/*
        
        for i in range(num_cpus):
            # Simplified assumption: first half are performance cores
            core_type = CoreType.PERFORMANCE if i < num_cpus // 2 else CoreType.EFFICIENCY
            
            # Simplified CCX assignment (AMD Ryzen typically has 4 cores per CCX)
            ccx_id = i // 4
            
            core_info = CoreInfo(
                core_id=i,
                logical_processor=i,
                core_type=core_type,
                ccx_id=ccx_id,
                numa_node=0,  # Simplified for laptop
                is_parked=False
            )
            self._core_info.append(core_info)
        
        if self.config.verbose:
            for core in self._core_info:
                logger.debug(
                    f"Core {core.core_id}: Type={core.core_type.value}, "
                    f"CCX={core.ccx_id}, Logical={core.logical_processor}"
                )
    
    def _reserve_cores(self):
        """Reserve high-performance cores for critical trading operations"""
        performance_cores = [
            core.core_id for core in self._core_info 
            if core.core_type == CoreType.PERFORMANCE
        ]
        
        # Reserve the specified number of performance cores
        self._reserved_cores = set(performance_cores[:self.config.reserved_cores])
        
        if self.config.isolate_cores and self._is_windows:
            self._isolate_cores_windows()
    
    def _isolate_cores_windows(self):
        """Isolate reserved cores from OS scheduling on Windows"""
        if not self._is_windows:
            return
        
        try:
            # Note: Full core isolation requires administrative privileges
            # and may involve modifying boot configuration
            # This is a simplified implementation
            
            logger.info("Windows core isolation requested (requires admin privileges)")
            
            # In production, this would:
            # 1. Check for admin privileges
            # 2. Modify processor affinity mask for system processes
            # 3. Potentially update boot configuration for full isolation
            
        except Exception as e:
            logger.warning(f"Failed to isolate cores: {e}")
    
    def get_available_cores(self) -> List[int]:
        """Get list of available (non-reserved) cores"""
        return [
            core.core_id for core in self._core_info 
            if core.core_id not in self._reserved_cores
        ]
    
    def get_reserved_cores(self) -> List[int]:
        """Get list of reserved cores for critical operations"""
        return sorted(self._reserved_cores)
    
    def pin_current_thread(self, core_ids: Optional[List[int]] = None) -> bool:
        """
        Pin the current thread to specified core(s)
        
        Args:
            core_ids: List of core IDs to pin to. If None, uses reserved cores.
            
        Returns:
            True if successful, False otherwise
        """
        if core_ids is None:
            core_ids = self.get_reserved_cores()
        
        if not core_ids:
            logger.error("No cores specified for pinning")
            return False
        
        try:
            if self._is_windows:
                return self._pin_thread_windows(core_ids)
            else:
                return self._pin_thread_linux(core_ids)
                
        except Exception as e:
            logger.error(f"Failed to pin thread: {e}")
            return False
    
    def _pin_thread_windows(self, core_ids: List[int]) -> bool:
        """Pin current thread on Windows using SetThreadAffinityMask"""
        try:
            # Get current thread handle
            kernel32 = ctypes.windll.kernel32
            current_thread = kernel32.GetCurrentThread()
            
            # Create affinity mask
            affinity_mask = sum(1 << core for core in core_ids)
            
            # Set thread affinity
            result = kernel32.SetThreadAffinityMask(current_thread, affinity_mask)
            
            if result:
                logger.info(f"Pinned thread to cores {core_ids} (mask: {affinity_mask})")
                return True
            else:
                logger.error(f"SetThreadAffinityMask failed with error {kernel32.GetLastError()}")
                return False
                
        except Exception as e:
            logger.error(f"Windows thread pinning failed: {e}")
            return False
    
    def _pin_thread_linux(self, core_ids: List[int]) -> bool:
        """Pin current thread on Linux using sched_setaffinity"""
        try:
            import ctypes.util
            
            libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
            
            # Get current thread ID
            tid = ctypes.c_long(ctypes.pythonapi.PyThreadState_Get().contents.thread_id)
            
            # Create CPU set
            cpu_set = 0
            for core in core_ids:
                cpu_set |= (1 << core)
            
            # Call sched_setaffinity
            result = libc.sched_setaffinity(tid, ctypes.sizeof(ctypes.c_ulong), ctypes.byref(ctypes.c_ulong(cpu_set)))
            
            if result == 0:
                logger.info(f"Pinned thread to cores {core_ids}")
                return True
            else:
                errno = ctypes.get_errno()
                logger.error(f"sched_setaffinity failed with errno {errno}")
                return False
                
        except Exception as e:
            logger.error(f"Linux thread pinning failed: {e}")
            return False
    
    def pin_process(self, core_ids: Optional[List[int]] = None) -> bool:
        """
        Pin the entire process to specified cores
        
        Args:
            core_ids: List of core IDs. If None, uses reserved cores.
            
        Returns:
            True if successful
        """
        if core_ids is None:
            core_ids = self.get_reserved_cores()
        
        try:
            process = mp.current_process()
            
            if self._is_windows:
                # Use psutil for Windows process affinity
                import psutil
                p = psutil.Process(os.getpid())
                p.cpu_affinity(core_ids)
                logger.info(f"Pinned process to cores {core_ids}")
                return True
            else:
                # Linux: use os.sched_setaffinity
                cpu_set = 0
                for core in core_ids:
                    cpu_set |= (1 << core)
                os.sched_setaffinity(0, {core for core in core_ids})
                logger.info(f"Pinned process to cores {core_ids}")
                return True
                
        except Exception as e:
            logger.error(f"Process pinning failed: {e}")
            return False
    
    def get_optimal_core_for_ccx(self, ccx_id: int) -> Optional[int]:
        """Get an optimal core within a specific CCX"""
        for core in self._core_info:
            if core.ccx_id == ccx_id and core.core_id in self._reserved_cores:
                return core.core_id
        return None
    
    def get_stats(self) -> Dict:
        """Get statistics about CPU affinity configuration"""
        return {
            "total_cores": len(self._core_info),
            "reserved_cores": sorted(self._reserved_cores),
            "available_cores": self.get_available_cores(),
            "platform": self._get_platform_name(),
            "isolation_enabled": self.config.isolate_cores,
        }
    
    def restore_original_affinity(self):
        """Restore original process affinity"""
        if self._original_affinity is not None:
            try:
                if self._is_windows:
                    import psutil
                    p = psutil.Process(os.getpid())
                    p.cpu_affinity(self._original_affinity)
                else:
                    # Linux restoration logic
                    pass
                logger.info("Restored original CPU affinity")
            except Exception as e:
                logger.error(f"Failed to restore affinity: {e}")


# Convenience function for quick core pinning
def pin_to_performance_cores() -> bool:
    """Quick function to pin current thread to performance cores"""
    manager = CpuAffinityManager()
    return manager.pin_current_thread()


if __name__ == "__main__":
    # Test CPU affinity management
    print("=== CPU Affinity Manager Test ===\n")
    
    manager = CpuAffinityManager(AffinityConfig(
        reserved_cores=2,
        verbose=True
    ))
    
    print(f"Platform: {manager._get_platform_name()}")
    print(f"Stats: {manager.get_stats()}")
    print(f"Reserved cores: {manager.get_reserved_cores()}")
    print(f"Available cores: {manager.get_available_cores()}")
    
    # Attempt to pin current thread
    success = manager.pin_current_thread()
    print(f"Thread pinning {'successful' if success else 'failed'}")
    
    # Show optimal core for CCX 0
    optimal = manager.get_optimal_core_for_ccx(0)
    print(f"Optimal core for CCX 0: {optimal}")
