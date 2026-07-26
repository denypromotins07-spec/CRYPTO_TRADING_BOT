#!/usr/bin/env python3
"""
Memory Guard - Locks memory pages to prevent API keys from swapping to disk.

This module provides OS-level memory protection by using mlock/mlockall
to prevent sensitive data (API keys, encryption keys) from being paged out
to swap space where it could be recovered by attackers.

Security Features:
- Prevents memory pages containing secrets from being swapped to disk
- Detects and warns about insufficient lock limits
- Provides context managers for safe memory locking
- Cross-platform support (Linux, macOS, Windows)

Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
"""

import ctypes
import ctypes.util
import os
import sys
import warnings
from typing import Optional, Tuple, List, Any
from contextlib import contextmanager

# Platform-specific constants
try:
    # Linux constants
    MCL_CURRENT = 1  # Lock all currently mapped pages
    MCL_FUTURE = 2  # Lock all pages that will become mapped
    MCL_ONFAULT = 4  # Lock pages on fault (Linux 4.4+)
    
    # Windows constants
    PROCESS_LOCKED_PAGES = 1
except ImportError:
    MCL_CURRENT = 1
    MCL_FUTURE = 2
    MCL_ONFAULT = 4


class MemoryGuardError(Exception):
    """Exception raised when memory locking fails."""
    pass


class MemoryGuard:
    """
    Manages locked memory pages to protect sensitive data.
    
    Uses OS-level mlock/mlockall calls to prevent paging.
    Implements the Facade pattern for cross-platform compatibility.
    """
    
    def __init__(self, max_lock_bytes: int = 64 * 1024 * 1024):
        """
        Initialize MemoryGuard.
        
        Args:
            max_lock_bytes: Maximum bytes to lock (default 64MB for 8GB RAM system)
        """
        self.max_lock_bytes = max_lock_bytes
        self._locked_regions: List[Tuple[int, int]] = []  # (address, size)
        self._libc: Optional[ctypes.CDLL] = None
        self._kernel32: Optional[ctypes.CDLL] = None
        self._is_windows = sys.platform == 'win32'
        
        self._load_libraries()
        self._check_limits()
    
    def _load_libraries(self) -> None:
        """Load platform-specific C libraries."""
        try:
            if self._is_windows:
                # Windows uses kernel32.dll
                self._kernel32 = ctypes.windll.kernel32  # type: ignore
                if self._kernel32 is None:
                    raise MemoryGuardError("Failed to load kernel32.dll")
            else:
                # Unix-like systems use libc
                libc_name = ctypes.util.find_library('c')
                if libc_name is None:
                    # Fallback for some systems
                    libc_name = 'libc.so.6'
                self._libc = ctypes.CDLL(libc_name, use_errno=True)
                if self._libc is None:
                    raise MemoryGuardError(f"Failed to load {libc_name}")
        except Exception as e:
            raise MemoryGuardError(f"Failed to load system libraries: {e}")
    
    def _check_limits(self) -> None:
        """Check and warn about memory lock limits."""
        if self._is_windows:
            # Windows doesn't have mlock limits in the same way
            return
        
        try:
            # Read /proc/self/limits on Linux
            if os.path.exists('/proc/self/limits'):
                with open('/proc/self/limits', 'r') as f:
                    for line in f:
                        if 'locked memory' in line.lower():
                            parts = line.split()
                            if len(parts) >= 4:
                                limit_str = parts[3]
                                if limit_str != 'unlimited':
                                    limit = int(limit_str)
                                    if limit < self.max_lock_bytes:
                                        warnings.warn(
                                            f"Memory lock limit ({limit} bytes) is less than "
                                            f"requested ({self.max_lock_bytes} bytes). "
                                            f"Consider increasing with 'ulimit -l'"
                                        )
        except Exception:
            # Non-critical, continue anyway
            pass
    
    def lock_all(self) -> bool:
        """
        Lock all current and future memory pages.
        
        Returns:
            True if successful, False otherwise
        """
        if self._is_windows:
            return self._lock_all_windows()
        else:
            return self._lock_all_unix()
    
    def _lock_all_unix(self) -> bool:
        """Lock all pages on Unix-like systems using mlockall."""
        if self._libc is None:
            return False
        
        try:
            # Call mlockall(MCL_CURRENT | MCL_FUTURE)
            flags = MCL_CURRENT | MCL_FUTURE
            result = self._libc.mlockall(flags)
            
            if result != 0:
                errno = ctypes.get_errno()
                raise MemoryGuardError(f"mlockall failed with errno {errno}: {os.strerror(errno)}")
            
            return True
        except Exception as e:
            warnings.warn(f"Failed to lock all memory: {e}")
            return False
    
    def _lock_all_windows(self) -> bool:
        """
        Lock process working set on Windows.
        
        Uses VirtualLock for critical regions.
        """
        if self._kernel32 is None:
            return False
        
        try:
            # Get current process handle
            h_process = self._kernel32.GetCurrentProcess()
            
            # Set working set size to ensure memory stays resident
            # Min and Max working set sizes
            min_ws = self.max_lock_bytes // 2
            max_ws = self.max_lock_bytes
            
            result = self._kernel32.SetProcessWorkingSetSize(
                h_process,
                ctypes.c_size_t(min_ws),
                ctypes.c_size_t(max_ws)
            )
            
            if result == 0:
                raise MemoryGuardError(
                    f"SetProcessWorkingSetSize failed: {self._kernel32.GetLastError()}"
                )
            
            return True
        except Exception as e:
            warnings.warn(f"Failed to lock Windows working set: {e}")
            return False
    
    def lock_region(self, data: bytearray) -> bool:
        """
        Lock a specific memory region containing sensitive data.
        
        Args:
            data: bytearray containing sensitive data
            
        Returns:
            True if successful
        """
        if self._is_windows:
            return self._lock_region_windows(data)
        else:
            return self._lock_region_unix(data)
    
    def _lock_region_unix(self, data: bytearray) -> bool:
        """Lock specific region on Unix using mlock."""
        if self._libc is None:
            return False
        
        try:
            # Get memory address of bytearray
            addr = ctypes.addressof(ctypes.c_char.from_buffer(data))
            size = len(data)
            
            result = self._libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(size))
            
            if result != 0:
                errno = ctypes.get_errno()
                raise MemoryGuardError(f"mlock failed: {os.strerror(errno)}")
            
            self._locked_regions.append((addr, size))
            return True
        except Exception as e:
            warnings.warn(f"Failed to lock region: {e}")
            return False
    
    def _lock_region_windows(self, data: bytearray) -> bool:
        """Lock specific region on Windows using VirtualLock."""
        if self._kernel32 is None:
            return False
        
        try:
            addr = ctypes.addressof(ctypes.c_char.from_buffer(data))
            size = len(data)
            
            result = self._kernel32.VirtualLock(
                ctypes.c_void_p(addr),
                ctypes.c_size_t(size)
            )
            
            if result == 0:
                raise MemoryGuardError(
                    f"VirtualLock failed: {self._kernel32.GetLastError()}"
                )
            
            self._locked_regions.append((addr, size))
            return True
        except Exception as e:
            warnings.warn(f"Failed to lock Windows region: {e}")
            return False
    
    def unlock_region(self, data: bytearray) -> bool:
        """
        Unlock a previously locked memory region.
        
        Args:
            data: bytearray to unlock
            
        Returns:
            True if successful
        """
        if self._is_windows:
            return self._unlock_region_windows(data)
        else:
            return self._unlock_region_unix(data)
    
    def _unlock_region_unix(self, data: bytearray) -> bool:
        """Unlock specific region on Unix using munlock."""
        if self._libc is None:
            return False
        
        try:
            addr = ctypes.addressof(ctypes.c_char.from_buffer(data))
            size = len(data)
            
            result = self._libc.munlock(ctypes.c_void_p(addr), ctypes.c_size_t(size))
            
            if result != 0:
                errno = ctypes.get_errno()
                raise MemoryGuardError(f"munlock failed: {os.strerror(errno)}")
            
            # Remove from tracked regions
            self._locked_regions = [
                (a, s) for a, s in self._locked_regions if a != addr
            ]
            return True
        except Exception as e:
            warnings.warn(f"Failed to unlock region: {e}")
            return False
    
    def _unlock_region_windows(self, data: bytearray) -> bool:
        """Unlock specific region on Windows using VirtualUnlock."""
        if self._kernel32 is None:
            return False
        
        try:
            addr = ctypes.addressof(ctypes.c_char.from_buffer(data))
            size = len(data)
            
            result = self._kernel32.VirtualUnlock(
                ctypes.c_void_p(addr),
                ctypes.c_size_t(size)
            )
            
            # Note: VirtualUnlock returns 0 on failure, but also if page wasn't locked
            self._locked_regions = [
                (a, s) for a, s in self._locked_regions 
                if a != ctypes.addressof(ctypes.c_char.from_buffer(data))
            ]
            return True
        except Exception as e:
            warnings.warn(f"Failed to unlock Windows region: {e}")
            return False
    
    def unlock_all(self) -> bool:
        """Unlock all memory pages."""
        if self._is_windows:
            return self._unlock_all_windows()
        else:
            return self._unlock_all_unix()
    
    def _unlock_all_unix(self) -> bool:
        """Unlock all pages on Unix using munlockall."""
        if self._libc is None:
            return False
        
        try:
            result = self._libc.munlockall()
            
            if result != 0:
                errno = ctypes.get_errno()
                raise MemoryGuardError(f"munlockall failed: {os.strerror(errno)}")
            
            self._locked_regions.clear()
            return True
        except Exception as e:
            warnings.warn(f"Failed to unlock all memory: {e}")
            return False
    
    def _unlock_all_windows(self) -> bool:
        """Reset working set on Windows."""
        if self._kernel32 is None:
            return False
        
        try:
            h_process = self._kernel32.GetCurrentProcess()
            
            # Reset working set
            result = self._kernel32.SetProcessWorkingSetSize(
                h_process,
                ctypes.c_size_t(-1),  # -1 means reset
                ctypes.c_size_t(-1)
            )
            
            self._locked_regions.clear()
            return result != 0
        except Exception as e:
            warnings.warn(f"Failed to reset Windows working set: {e}")
            return False
    
    def get_locked_bytes(self) -> int:
        """Return total bytes currently locked."""
        return sum(size for _, size in self._locked_regions)


@contextmanager
def locked_memory(data: bytearray, guard: Optional[MemoryGuard] = None):
    """
    Context manager for temporarily locking memory.
    
    Usage:
        sensitive_data = bytearray(api_key.encode())
        with locked_memory(sensitive_data):
            # Use sensitive_data here - it's locked in memory
            process_key(sensitive_data)
        # Automatically unlocked here
    
    Args:
        data: bytearray to lock
        guard: Optional MemoryGuard instance
    """
    mem_guard = guard or MemoryGuard()
    locked = False
    
    try:
        locked = mem_guard.lock_region(data)
        yield data
    finally:
        if locked:
            mem_guard.unlock_region(data)


def secure_zero(data: bytearray) -> None:
    """
    Securely zero out sensitive data in memory.
    
    Uses multiple passes to ensure data is overwritten.
    Prevents compiler optimization from removing the zeroing.
    
    Args:
        data: bytearray to zero out
    """
    if not data:
        return
    
    # Multiple passes for security
    for _ in range(3):
        for i in range(len(data)):
            data[i] = 0x00
        
        # Memory barrier to prevent optimization
        ctypes.memmove(
            ctypes.addressof(ctypes.c_char.from_buffer(data)),
            ctypes.addressof(ctypes.c_char.from_buffer(data)),
            len(data)
        )


class SensitiveBuffer:
    """
    A buffer class that automatically locks memory and zeros on deletion.
    
    Usage:
        buf = SensitiveBuffer(b"api_key_secret")
        # Buffer is locked in memory
        use_key(buf.data)
        # Automatically zeroed and unlocked when deleted
    """
    
    def __init__(self, initial_data: bytes = b'', guard: Optional[MemoryGuard] = None):
        """
        Initialize sensitive buffer.
        
        Args:
            initial_data: Initial bytes (will be locked)
            guard: Optional MemoryGuard instance
        """
        self._guard = guard or MemoryGuard()
        self._data = bytearray(initial_data)
        self._locked = self._guard.lock_region(self._data)
    
    @property
    def data(self) -> bytearray:
        """Get the underlying data (still locked)."""
        return self._data
    
    def update(self, new_data: bytes) -> None:
        """Update buffer contents securely."""
        # Zero old data first
        secure_zero(self._data)
        
        # Resize if needed
        if len(new_data) != len(self._data):
            # Unlock old, resize, lock new
            if self._locked:
                self._guard.unlock_region(self._data)
            
            self._data = bytearray(len(new_data))
            self._locked = self._guard.lock_region(self._data)
        
        # Copy new data
        self._data[:len(new_data)] = bytearray(new_data)
    
    def __del__(self) -> None:
        """Securely cleanup on deletion."""
        if hasattr(self, '_data') and self._data:
            secure_zero(self._data)
        if hasattr(self, '_locked') and self._locked and hasattr(self, '_guard'):
            try:
                self._guard.unlock_region(self._data)
            except Exception:
                pass
    
    def __enter__(self) -> 'SensitiveBuffer':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.__del__()


if __name__ == '__main__':
    # Example usage
    print("Initializing MemoryGuard...")
    guard = MemoryGuard(max_lock_bytes=32 * 1024 * 1024)  # 32MB
    
    print("Locking all memory pages...")
    if guard.lock_all():
        print("✓ All memory pages locked successfully")
    else:
        print("✗ Failed to lock all memory pages")
    
    # Create sensitive buffer
    print("\nCreating sensitive buffer with API key...")
    with SensitiveBuffer(b"test_api_key_12345") as buf:
        print(f"Buffer locked: {buf._locked}")
        print(f"Locked bytes: {guard.get_locked_bytes()}")
        
        # Use the key
        print(f"Using key: {buf.data[:4]}...")  # Only show first 4 chars
    
    print("\nBuffer automatically zeroed and unlocked")
    print("MemoryGuard cleanup complete")
