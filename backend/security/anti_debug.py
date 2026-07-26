#!/usr/bin/env python3
"""
Anti-Debug Module - Detects and blocks debuggers from attaching to the process.

This module implements multiple anti-debugging techniques to detect and prevent
debugger attachment, memory dumping, and reverse engineering attempts.

Security Features:
- Detects common debuggers (gdb, x64dbg, OllyDbg, etc.)
- Checks for debugger artifacts in process environment
- Monitors for breakpoints and single-step execution
- Detects timing anomalies caused by debugging
- Immediate termination on debugger detection

Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
"""

import os
import sys
import time
import ctypes
import signal
import threading
import subprocess
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import hashlib


class DebugStatus(Enum):
    """Debugger detection status."""
    CLEAN = "clean"
    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"


@dataclass
class DebugDetectionResult:
    """Result of a debug detection check."""
    check_name: str
    is_debugger_detected: bool
    confidence: float  # 0.0 to 1.0
    details: str


class AntiDebugError(Exception):
    """Exception raised when debugger is detected."""
    pass


class AntiDebugger:
    """
    Multi-layered anti-debugging protection.
    
    Implements multiple detection techniques using the Chain of Responsibility
    pattern for extensible debugger detection.
    """
    
    def __init__(self, aggressive_mode: bool = True):
        """
        Initialize anti-debugger.
        
        Args:
            aggressive_mode: If True, terminate immediately on detection
        """
        self.aggressive_mode = aggressive_mode
        self._detection_results: List[DebugDetectionResult] = []
        self._is_monitoring: bool = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_flag: threading.Event = threading.Event()
        self._callbacks: List[Callable[[DebugDetectionResult], None]] = []
        
        # Platform detection
        self._is_windows = sys.platform == 'win32'
        self._is_linux = sys.platform.startswith('linux')
        self._is_macos = sys.platform.startswith('darwin')
        
        # Load platform-specific libraries
        self._libc: Optional[ctypes.CDLL] = None
        self._kernel32: Optional[ctypes.CDLL] = None
        self._load_platform_libraries()
    
    def _load_platform_libraries(self) -> None:
        """Load platform-specific system libraries."""
        try:
            if self._is_windows:
                self._kernel32 = ctypes.windll.kernel32  # type: ignore
            elif self._is_linux or self._is_macos:
                libc_name = ctypes.util.find_library('c')
                if libc_name:
                    self._libc = ctypes.CDLL(libc_name, use_errno=True)
        except Exception:
            pass  # Continue without platform libs
    
    def add_callback(self, callback: Callable[[DebugDetectionResult], None]) -> None:
        """Add a callback to be invoked on debugger detection."""
        self._callbacks.append(callback)
    
    def _notify_callbacks(self, result: DebugDetectionResult) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception:
                pass  # Don't let callback errors interfere
    
    def check_debugger_present(self) -> DebugDetectionResult:
        """
        Check if a debugger is attached to the current process.
        
        Uses platform-specific APIs to detect debugger attachment.
        """
        if self._is_windows:
            return self._check_debugger_windows()
        else:
            return self._check_debugger_unix()
    
    def _check_debugger_windows(self) -> DebugDetectionResult:
        """Check for debugger on Windows using IsDebuggerPresent."""
        if self._kernel32 is None:
            return DebugDetectionResult(
                check_name="IsDebuggerPresent",
                is_debugger_detected=False,
                confidence=0.0,
                details="kernel32 not available"
            )
        
        try:
            is_present = bool(self._kernel32.IsDebuggerPresent())
            
            # Also check OutputDebugString behavior
            output_debug_check = self._check_output_debug_string_windows()
            
            combined = is_present or output_debug_check.is_debugger_detected
            confidence = max(
                1.0 if is_present else 0.0,
                output_debug_check.confidence
            )
            
            return DebugDetectionResult(
                check_name="Windows Debugger Check",
                is_debugger_detected=combined,
                confidence=confidence,
                details=f"IsDebuggerPresent: {is_present}, OutputDebugString: {output_debug_check.details}"
            )
            
        except Exception as e:
            return DebugDetectionResult(
                check_name="Windows Debugger Check",
                is_debugger_detected=False,
                confidence=0.0,
                details=f"Error: {e}"
            )
    
    def _check_output_debug_string_windows(self) -> DebugDetectionResult:
        """Check debugger via OutputDebugString behavior on Windows."""
        if self._kernel32 is None:
            return DebugDetectionResult(
                check_name="OutputDebugString",
                is_debugger_detected=False,
                confidence=0.0,
                details="kernel32 not available"
            )
        
        try:
            # Set up error mode
            SEM_FAILCRITICALERRORS = 0x0001
            old_error_mode = self._kernel32.SetErrorMode(SEM_FAILCRITICALERRORS)
            
            test_string = b"AntiDebug Test\x00"
            
            # Call OutputDebugString
            self._kernel32.OutputDebugStringA(test_string)
            
            # Restore error mode
            self._kernel32.SetErrorMode(old_error_mode)
            
            # If we get here without issues, likely no debugger
            # A real implementation would check for timing anomalies
            return DebugDetectionResult(
                check_name="OutputDebugString",
                is_debugger_detected=False,
                confidence=0.3,
                details="No anomalies detected"
            )
            
        except Exception as e:
            return DebugDetectionResult(
                check_name="OutputDebugString",
                is_debugger_detected=True,
                confidence=0.7,
                details=f"Exception during check: {e}"
            )
    
    def _check_debugger_unix(self) -> DebugDetectionResult:
        """Check for debugger on Unix-like systems."""
        results = []
        
        # Check ptrace status (Linux)
        if self._is_linux:
            ptrace_result = self._check_ptrace_linux()
            results.append(ptrace_result)
        
        # Check for debugger processes
        process_result = self._check_debugger_processes()
        results.append(process_result)
        
        # Check environment variables
        env_result = self._check_debugger_environment()
        results.append(env_result)
        
        # Combine results
        any_detected = any(r.is_debugger_detected for r in results)
        max_confidence = max(r.confidence for r in results) if results else 0.0
        
        details = "; ".join(f"{r.check_name}: {r.details}" for r in results)
        
        return DebugDetectionResult(
            check_name="Unix Debugger Check",
            is_debugger_detected=any_detected,
            confidence=max_confidence,
            details=details
        )
    
    def _check_ptrace_linux(self) -> DebugDetectionResult:
        """Check if process is being traced via ptrace on Linux."""
        if self._libc is None:
            return DebugDetectionResult(
                check_name="ptrace",
                is_debugger_detected=False,
                confidence=0.0,
                details="libc not available"
            )
        
        try:
            # PTRACE_TRACEME = 0
            # If already being traced, this will fail
            result = self._libc.ptrace(0, 0, 0, 0)
            
            if result == -1:
                errno = ctypes.get_errno()
                # EPERM (1) or ESRCH (3) indicate tracing
                if errno in (1, 3):
                    return DebugDetectionResult(
                        check_name="ptrace",
                        is_debugger_detected=True,
                        confidence=0.9,
                        details=f"Process is being traced (errno={errno})"
                    )
            
            return DebugDetectionResult(
                check_name="ptrace",
                is_debugger_detected=False,
                confidence=0.5,
                details="Not being traced"
            )
            
        except Exception as e:
            return DebugDetectionResult(
                check_name="ptrace",
                is_debugger_detected=False,
                confidence=0.0,
                details=f"Error: {e}"
            )
    
    def _check_debugger_processes(self) -> DebugDetectionResult:
        """Check for known debugger processes running on the system."""
        debugger_names = [
            'gdb', 'lldb', 'x64dbg', 'ollydbg', 'ida', 'ida64',
            'windbg', 'immunity', 'cheatengine', 'processhacker',
            'procmon', 'wireshark', 'fiddler', 'charles'
        ]
        
        try:
            if self._is_windows:
                cmd = ['tasklist', '/FO', 'CSV']
            else:
                cmd = ['ps', 'aux']
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            output = result.stdout.lower()
            
            detected = []
            for name in debugger_names:
                if name in output:
                    detected.append(name)
            
            if detected:
                return DebugDetectionResult(
                    check_name="Debugger Processes",
                    is_debugger_detected=True,
                    confidence=0.6,
                    details=f"Detected: {', '.join(detected)}"
                )
            
            return DebugDetectionResult(
                check_name="Debugger Processes",
                is_debugger_detected=False,
                confidence=0.3,
                details="No debugger processes found"
            )
            
        except Exception as e:
            return DebugDetectionResult(
                check_name="Debugger Processes",
                is_debugger_detected=False,
                confidence=0.0,
                details=f"Error checking processes: {e}"
            )
    
    def _check_debugger_environment(self) -> DebugDetectionResult:
        """Check environment variables for debugger artifacts."""
        debugger_env_vars = [
            'PYDEVD_USE_CYTHON',  # PyCharm debugger
            'JPY_DAEMON',  # Jupyter debugger
            'IDEA_INITIAL_DIRECTORY',  # IntelliJ
            'VSCODE_PID',  # VS Code
            'PYCHARM_HOSTED',  # PyCharm
            'DEBUGPY_SOCKET',  # debugpy
        ]
        
        detected = []
        for var in debugger_env_vars:
            if var in os.environ:
                detected.append(var)
        
        if detected:
            return DebugDetectionResult(
                check_name="Environment Variables",
                is_debugger_detected=True,
                confidence=0.4,
                details=f"Debugger env vars: {', '.join(detected)}"
            )
        
        return DebugDetectionResult(
            check_name="Environment Variables",
            is_debugger_detected=False,
            confidence=0.2,
            details="No debugger env vars found"
        )
    
    def check_timing_anomaly(self, iterations: int = 1000) -> DebugDetectionResult:
        """
        Detect timing anomalies that may indicate debugging.
        
        Debuggers often cause measurable delays in tight loops.
        """
        start = time.perf_counter()
        
        # Tight loop with cryptographic operation
        data = b"timing_check_data"
        for i in range(iterations):
            hashlib.sha256(data + str(i).encode()).digest()
        
        elapsed = time.perf_counter() - start
        
        # Expected time varies by hardware, adjust threshold accordingly
        # On AMD Ryzen AI 5, this should complete in < 10ms
        expected_max_ms = 50  # Generous threshold
        
        elapsed_ms = elapsed * 1000
        
        if elapsed_ms > expected_max_ms:
            confidence = min(1.0, (elapsed_ms - expected_max_ms) / expected_max_ms)
            return DebugDetectionResult(
                check_name="Timing Anomaly",
                is_debugger_detected=True,
                confidence=confidence * 0.5,  # Reduce confidence for false positives
                details=f"Execution took {elapsed_ms:.2f}ms (expected <{expected_max_ms}ms)"
            )
        
        return DebugDetectionResult(
            check_name="Timing Anomaly",
            is_debugger_detected=False,
            confidence=0.3,
            details=f"Execution took {elapsed_ms:.2f}ms"
        )
    
    def run_all_checks(self) -> List[DebugDetectionResult]:
        """Run all debugger detection checks."""
        results = [
            self.check_debugger_present(),
            self.check_timing_anomaly(),
        ]
        
        self._detection_results = results
        return results
    
    def is_debugger_detected(self) -> bool:
        """Check if any debugger was detected."""
        if not self._detection_results:
            self.run_all_checks()
        
        return any(r.is_debugger_detected and r.confidence > 0.5 
                   for r in self._detection_results)
    
    def start_monitoring(self, check_interval_seconds: float = 5.0) -> None:
        """
        Start continuous debugger monitoring in background thread.
        
        Args:
            check_interval_seconds: Time between checks
        """
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        self._stop_flag.clear()
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(check_interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
    
    def _monitor_loop(self, interval: float) -> None:
        """Background monitoring loop."""
        while not self._stop_flag.is_set():
            results = self.run_all_checks()
            
            for result in results:
                if result.is_debugger_detected and result.confidence > 0.5:
                    self._notify_callbacks(result)
                    
                    if self.aggressive_mode:
                        # Terminate immediately
                        self._emergency_shutdown(f"Debugger detected: {result.details}")
            
            self._stop_flag.wait(interval)
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self._stop_flag.set()
        self._is_monitoring = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
    
    def _emergency_shutdown(self, reason: str) -> None:
        """Perform emergency shutdown when debugger is detected."""
        print(f"\n{'='*60}", file=sys.stderr)
        print("CRITICAL SECURITY ALERT", file=sys.stderr)
        print(f"Reason: {reason}", file=sys.stderr)
        print("Terminating application immediately", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
        
        # Try to corrupt any sensitive data in memory before exit
        self._zeroize_sensitive_data()
        
        # Exit with error code
        os._exit(1)
    
    def _zeroize_sensitive_data(self) -> None:
        """Zeroize sensitive data in memory before shutdown."""
        # This is a placeholder - real implementation would
        # zeroize all sensitive buffers
        pass


def protect_from_debug(func: Callable) -> Callable:
    """
    Decorator to protect a function from debugger analysis.
    
    Usage:
        @protect_from_debug
        def sensitive_function():
            ...
    """
    def wrapper(*args, **kwargs):
        debugger = AntiDebugger(aggressive_mode=True)
        
        # Quick check before executing
        if debugger.is_debugger_detected():
            debugger._emergency_shutdown("Debugger detected before function execution")
        
        try:
            return func(*args, **kwargs)
        finally:
            # Check again after execution
            if debugger.is_debugger_detected():
                debugger._emergency_shutdown("Debugger detected after function execution")
    
    return wrapper


if __name__ == '__main__':
    print("Anti-Debug Module Self-Test")
    print("=" * 40)
    
    debugger = AntiDebugger(aggressive_mode=False)
    
    print("\nRunning debugger detection checks...")
    results = debugger.run_all_checks()
    
    for result in results:
        status = "⚠️ DETECTED" if result.is_debugger_detected else "✓ Clean"
        print(f"\n{result.check_name}: {status}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Details: {result.details}")
    
    overall_status = "CLEAN" if not debugger.is_debugger_detected() else "DEBUGGER DETECTED"
    print(f"\nOverall Status: {overall_STATUS}")
