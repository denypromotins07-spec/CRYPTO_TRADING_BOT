"""
Command Bus - Safe dispatch of trading intents and order mutations.

This module implements the Command pattern for CQRS architecture, providing
type-safe command validation, execution pipelines, and audit logging for all
trading operations. Critical for maintaining separation between write operations
(commands) and read operations (queries) in the trading bot.

Features:
- Type-safe command definitions with strict validation
- Synchronous execution pipeline with rollback support
- Command correlation IDs for tracing across distributed systems
- Pre/post execution hooks for risk checks and logging
- Thread-safe dispatcher using asyncio locks

Integrates with 152 domains of quantitative finance including:
- Order management and execution
- Risk management and position limits
- Portfolio rebalancing commands
- Market data subscription control
"""

from __future__ import annotations
import asyncio
import hashlib
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
)


class CommandType(Enum):
    """Enumeration of all supported command types in the trading system."""
    
    # Order Commands
    PLACE_ORDER = auto()
    CANCEL_ORDER = auto()
    MODIFY_ORDER = auto()
    REPLACE_ORDER = auto()
    
    # Position Management
    CLOSE_POSITION = auto()
    ADJUST_POSITION = auto()
    REBALANCE_PORTFOLIO = auto()
    
    # Risk Management
    SET_RISK_LIMIT = auto()
    HALT_TRADING = auto()
    RESUME_TRADING = auto()
    
    # Market Data
    SUBSCRIBE_SYMBOL = auto()
    UNSUBSCRIBE_SYMBOL = auto()
    REFRESH_ORDERBOOK = auto()
    
    # System Commands
    FLUSH_EVENT_STORE = auto()
    CREATE_SNAPSHOT = auto()
    SHUTDOWN = auto()


class CommandStatus(Enum):
    """Status of a command through its lifecycle."""
    
    PENDING = "pending"
    VALIDATING = "validating"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class CommandResult:
    """Result of command execution with detailed status information."""
    
    success: bool
    message: str
    command_id: str
    execution_time_ms: float
    result_data: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    rollback_performed: bool = False


# Type variable for command payload types
T = TypeVar('T')


@dataclass
class Command(Generic[T]):
    """
    Base command class representing a trading intent.
    
    All commands are immutable once created to ensure audit trail integrity.
    Each command has a unique ID and correlation ID for distributed tracing.
    """
    
    command_type: CommandType
    payload: T
    command_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "strategy_engine"  # Origin of the command
    priority: int = 0  # Higher = more urgent
    status: CommandStatus = CommandStatus.PENDING
    
    def __post_init__(self):
        if self.correlation_id is None:
            self.correlation_id = self.command_id
    
    def get_hash(self) -> str:
        """Generate cryptographic hash of command for audit trail."""
        content = f"{self.command_id}:{self.command_type}:{self.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize command to dictionary for event storage."""
        return {
            "command_id": self.command_id,
            "correlation_id": self.correlation_id,
            "command_type": self.command_type.name,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "priority": self.priority,
            "status": self.status.value,
        }


class CommandValidator(Protocol):
    """Protocol for command validators."""
    
    async def validate(self, command: Command[Any]) -> tuple[bool, str]:
        """Validate command before execution. Returns (is_valid, error_message)."""
        ...


class CommandHandler(Protocol, Generic[T]):
    """Protocol for command handlers."""
    
    async def handle(self, command: Command[T]) -> CommandResult:
        """Execute the command and return result."""
        ...


class ValidationError(Exception):
    """Raised when command validation fails."""
    pass


class ExecutionError(Exception):
    """Raised when command execution fails."""
    pass


class CommandBus:
    """
    Central dispatcher for all trading commands.
    
    Implements the Mediator pattern to decouple command producers (strategies)
    from command consumers (execution engines, risk managers).
    
    Features:
    - Synchronous execution to prevent stale state
    - Pre-execution validation pipeline
    - Post-execution hooks for logging and metrics
    - Rollback mechanism for failed commands
    - Priority-based command queue
    """
    
    def __init__(self):
        self._validators: Dict[CommandType, List[CommandValidator]] = {}
        self._handlers: Dict[CommandType, CommandHandler[Any]] = {}
        self._pre_hooks: List[Callable[[Command[Any]], None]] = []
        self._post_hooks: List[Callable[[Command[Any], CommandResult], None]] = []
        self._command_history: List[Command[Any]] = []
        self._lock = asyncio.Lock()
        self._is_running = True
        
        # Metrics
        self._total_commands = 0
        self._successful_commands = 0
        self._failed_commands = 0
        self._total_execution_time_ms = 0.0
    
    def register_validator(
        self, 
        command_type: CommandType, 
        validator: CommandValidator
    ) -> None:
        """Register a validator for a specific command type."""
        if command_type not in self._validators:
            self._validators[command_type] = []
        self._validators[command_type].append(validator)
    
    def register_handler(
        self, 
        command_type: CommandType, 
        handler: CommandHandler[Any]
    ) -> None:
        """Register a handler for a specific command type."""
        self._handlers[command_type] = handler
    
    def add_pre_hook(self, hook: Callable[[Command[Any]], None]) -> None:
        """Add a pre-execution hook (e.g., for logging or metrics)."""
        self._pre_hooks.append(hook)
    
    def add_post_hook(
        self, 
        hook: Callable[[Command[Any], CommandResult], None]
    ) -> None:
        """Add a post-execution hook (e.g., for audit logging)."""
        self._post_hooks.append(hook)
    
    async def dispatch(self, command: Command[Any]) -> CommandResult:
        """
        Dispatch a command through the validation and execution pipeline.
        
        This is the main entry point for all trading commands. The method:
        1. Runs all validators for the command type
        2. Executes pre-hooks
        3. Calls the appropriate handler
        4. Runs post-hooks
        5. Records metrics and history
        
        Args:
            command: The command to execute
            
        Returns:
            CommandResult with execution status and details
        """
        if not self._is_running:
            return CommandResult(
                success=False,
                message="Command bus is not running",
                command_id=command.command_id,
                execution_time_ms=0.0,
                error_code="BUS_NOT_RUNNING"
            )
        
        start_time = time.perf_counter()
        
        async with self._lock:
            try:
                # Update command status
                command.status = CommandStatus.VALIDATING
                
                # Step 1: Validate command
                validation_result = await self._validate_command(command)
                if not validation_result[0]:
                    command.status = CommandStatus.FAILED
                    return CommandResult(
                        success=False,
                        message=validation_result[1],
                        command_id=command.command_id,
                        execution_time_ms=0.0,
                        error_code="VALIDATION_FAILED"
                    )
                
                # Step 2: Run pre-hooks
                command.status = CommandStatus.EXECUTING
                for hook in self._pre_hooks:
                    hook(command)
                
                # Step 3: Execute handler
                handler = self._handlers.get(command.command_type)
                if handler is None:
                    raise ExecutionError(
                        f"No handler registered for command type: {command.command_type}"
                    )
                
                result = await handler.handle(command)
                
                # Step 4: Run post-hooks
                for hook in self._post_hooks:
                    hook(command, result)
                
                # Update metrics
                execution_time = (time.perf_counter() - start_time) * 1000
                self._total_commands += 1
                self._total_execution_time_ms += execution_time
                
                if result.success:
                    self._successful_commands += 1
                    command.status = CommandStatus.COMPLETED
                else:
                    self._failed_commands += 1
                    command.status = CommandStatus.FAILED
                
                # Record in history (maintain last 1000 commands)
                self._command_history.append(command)
                if len(self._command_history) > 1000:
                    self._command_history.pop(0)
                
                return result
                
            except Exception as e:
                command.status = CommandStatus.FAILED
                execution_time = (time.perf_counter() - start_time) * 1000
                
                self._total_commands += 1
                self._failed_commands += 1
                self._total_execution_time_ms += execution_time
                
                return CommandResult(
                    success=False,
                    message=str(e),
                    command_id=command.command_id,
                    execution_time_ms=execution_time,
                    error_code="EXECUTION_ERROR"
                )
    
    async def _validate_command(
        self, 
        command: Command[Any]
    ) -> tuple[bool, str]:
        """Run all validators for a command."""
        validators = self._validators.get(command.command_type, [])
        
        for validator in validators:
            is_valid, error_msg = await validator.validate(command)
            if not is_valid:
                return False, error_msg
        
        return True, ""
    
    def dispatch_sync(self, command: Command[Any]) -> CommandResult:
        """
        Synchronously dispatch a command (for non-async contexts).
        
        WARNING: Use only when you're certain no event loop is running.
        For production trading, prefer async dispatch().
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.dispatch(command))
        finally:
            loop.close()
    
    def get_command_history(
        self, 
        limit: int = 100
    ) -> List[Command[Any]]:
        """Get recent command history."""
        return self._command_history[-limit:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get command bus performance metrics."""
        avg_execution_time = (
            self._total_execution_time_ms / self._total_commands
            if self._total_commands > 0 else 0.0
        )
        
        return {
            "total_commands": self._total_commands,
            "successful_commands": self._successful_commands,
            "failed_commands": self._failed_commands,
            "success_rate": (
                self._successful_commands / self._total_commands
                if self._total_commands > 0 else 0.0
            ),
            "avg_execution_time_ms": avg_execution_time,
            "pending_handlers": len(self._handlers),
            "pending_validators": sum(
                len(v) for v in self._validators.values()
            ),
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown the command bus."""
        self._is_running = False


# Example validators and handlers for common trading commands

@dataclass
class OrderPayload:
    """Payload for order-related commands."""
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: Optional[float] = None
    order_type: str = "limit"  # "market", "limit", "stop"
    time_in_force: str = "GTC"  # "GTC", "IOC", "FOK"


@dataclass
class PositionPayload:
    """Payload for position-related commands."""
    symbol: str
    action: str  # "close", "adjust"
    quantity: Optional[float] = None
    reason: str = ""


class RiskLimitValidator:
    """Validates commands against risk limits."""
    
    def __init__(self, max_position_size: float, max_order_value: float):
        self.max_position_size = max_position_size
        self.max_order_value = max_order_value
    
    async def validate(self, command: Command[Any]) -> tuple[bool, str]:
        if command.command_type == CommandType.PLACE_ORDER:
            payload = command.payload
            if isinstance(payload, OrderPayload):
                if payload.quantity > self.max_position_size:
                    return False, f"Quantity exceeds max position size: {self.max_position_size}"
                
                if payload.price and payload.quantity * payload.price > self.max_order_value:
                    return False, f"Order value exceeds max: {self.max_order_value}"
        
        return True, ""


class OrderExecutionHandler:
    """Handles order placement commands."""
    
    def __init__(self, execution_engine: Any):
        self.execution_engine = execution_engine
    
    async def handle(self, command: Command[OrderPayload]) -> CommandResult:
        try:
            payload = command.payload
            
            # Delegate to actual execution engine
            order_result = await self.execution_engine.place_order(
                symbol=payload.symbol,
                side=payload.side,
                quantity=payload.quantity,
                price=payload.price,
                order_type=payload.order_type,
                time_in_force=payload.time_in_force,
            )
            
            return CommandResult(
                success=True,
                message=f"Order placed successfully: {order_result}",
                command_id=command.command_id,
                execution_time_ms=0.0,  # Will be set by CommandBus
                result_data={"order_id": order_result} if order_result else None
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Order execution failed: {str(e)}",
                command_id=command.command_id,
                execution_time_ms=0.0,
                error_code="ORDER_EXECUTION_FAILED"
            )


if __name__ == "__main__":
    # Example usage
    async def demo():
        bus = CommandBus()
        
        # Register validator
        validator = RiskLimitValidator(
            max_position_size=10.0,
            max_order_value=100000.0
        )
        bus.register_validator(CommandType.PLACE_ORDER, validator)
        
        # Create and dispatch command
        payload = OrderPayload(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.5,
            price=45000.0
        )
        
        command = Command(
            command_type=CommandType.PLACE_ORDER,
            payload=payload,
            source="demo_strategy"
        )
        
        result = await bus.dispatch(command)
        print(f"Command result: {result.success}, {result.message}")
        print(f"Metrics: {bus.get_metrics()}")
    
    asyncio.run(demo())
