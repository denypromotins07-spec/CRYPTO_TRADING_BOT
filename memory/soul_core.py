"""
=============================================================================
ZAID PERSONAL CRYPTO TRADING BOT - SOUL CORE MEMORY SYSTEM
=============================================================================
Memory and Self-Learning Foundation for Continuous Algorithmic Evolution

This module implements the core memory system where the bot stores its lessons,
experiences, and learned patterns in the SOUL.md file. It provides thread-safe
read/write operations with automatic versioning and backup.

Domains Integrated:
- Machine Learning Memory Systems
- Experience Replay
- Pattern Recognition Storage
- Meta-Learning Architecture
- Knowledge Graph Construction
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import hashlib
import threading
import shutil
from enum import Enum


class MemoryType(Enum):
    """Classification of memory types for organized storage."""
    TRADE_LESSON = "trade_lesson"
    MARKET_REGIME = "market_regime"
    PATTERN_RECOGNITION = "pattern_recognition"
    RISK_ADJUSTMENT = "risk_adjustment"
    EXECUTION_OPTIMIZATION = "execution_optimization"
    STRATEGY_REFINEMENT = "strategy_refinement"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class MemoryEntry:
    """
    Represents a single memory/lesson entry in the SOUL system.
    
    Attributes:
        entry_id: Unique identifier (SHA256 hash of content)
        timestamp: When the memory was created
        memory_type: Classification of the memory
        title: Short descriptive title
        content: Detailed memory content
        confidence: Confidence score (0.0 to 1.0)
        relevance_decay: Rate at which memory relevance decreases
        tags: Associated tags for retrieval
        related_entries: IDs of related memories
    """
    entry_id: str
    timestamp: str
    memory_type: str
    title: str
    content: str
    confidence: float = 1.0
    relevance_decay: float = 0.001  # Per hour
    tags: List[str] = field(default_factory=list)
    related_entries: List[str] = field(default_factory=list)
    access_count: int = 0
    last_accessed: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create instance from dictionary."""
        return cls(**data)
    
    def get_relevance_score(self, current_time: datetime) -> float:
        """
        Calculate current relevance score based on time decay.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        created = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
        hours_elapsed = (current_time - created).total_seconds() / 3600
        decay_factor = self.relevance_decay * hours_elapsed
        base_relevance = max(0.0, self.confidence - decay_factor)
        
        # Boost by access count (diminishing returns)
        access_boost = min(0.2, 0.01 * self.access_count)
        
        return min(1.0, base_relevance + access_boost)


@dataclass
class SoulState:
    """
    Represents the current state of the bot's learning system.
    
    Attributes:
        total_memories: Count of all stored memories
        memories_by_type: Breakdown by memory type
        avg_confidence: Average confidence across all memories
        learning_rate: Current adaptive learning rate
        last_update: Timestamp of last update
    """
    total_memories: int = 0
    memories_by_type: Dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    learning_rate: float = 0.01
    last_update: str = ""
    
    def to_markdown_summary(self) -> str:
        """Generate markdown summary of soul state."""
        return f"""## Soul State Summary

- **Total Memories**: {self.total_memories}
- **Average Confidence**: {self.avg_confidence:.4f}
- **Learning Rate**: {self.learning_rate:.6f}
- **Last Update**: {self.last_update}

### Memories by Type
""" + "\n".join([f"- {k}: {v}" for k, v in self.memories_by_type.items()])


class SoulCore:
    """
    Core memory system for the trading bot's self-learning capabilities.
    Implements thread-safe read/write operations with automatic backup.
    
    This is the foundation of the bot's continuous evolution, storing
    lessons learned from trades, market patterns, and strategy refinements.
    """
    
    def __init__(self, soul_file_path: str = "./memory/SOUL.md"):
        """
        Initialize the Soul Core memory system.
        
        Args:
            soul_file_path: Path to the SOUL.md file
        """
        self.soul_path = Path(soul_file_path)
        self._lock = threading.RLock()
        self._memories: Dict[str, MemoryEntry] = {}
        self._state = SoulState()
        self._backup_dir = self.soul_path.parent / "backups"
        
        # Ensure directories exist
        self.soul_path.parent.mkdir(parents=True, exist_ok=True)
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing memories
        self._load_memories()
    
    def _generate_entry_id(self, content: str, timestamp: str) -> str:
        """Generate unique ID for memory entry using SHA256."""
        combined = f"{timestamp}:{content}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _load_memories(self) -> None:
        """Load memories from SOUL.md file."""
        if not self.soul_path.exists():
            self._initialize_soul_file()
            return
        
        try:
            content = self.soul_path.read_text(encoding='utf-8')
            self._parse_markdown_memories(content)
        except Exception as e:
            print(f"Warning: Could not load SOUL.md: {e}")
            self._initialize_soul_file()
    
    def _initialize_soul_file(self) -> None:
        """Create new SOUL.md file with header."""
        header = """# ZAID Personal Crypto Trading Bot - SOUL Memory

## System Information
- **Created**: {created}
- **Version**: 1.0.0
- **Purpose**: Continuous algorithmic evolution through experience storage

---

## Memories

<!-- Memories are stored below in structured format -->

---

## State Summary

<!-- Auto-generated state summary -->

"""
        initial_content = header.format(created=datetime.now(timezone.utc).isoformat())
        self.soul_path.write_text(initial_content, encoding='utf-8')
        self._memories = {}
        self._update_state()
    
    def _parse_markdown_memories(self, content: str) -> None:
        """Parse memory entries from markdown content."""
        self._memories = {}
        
        # Simple parsing: look for JSON blocks between markers
        import re
        pattern = r'<!-- MEMORY_START -->(.*?)<!-- MEMORY_END -->'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match.strip())
                entry = MemoryEntry.from_dict(data)
                self._memories[entry.entry_id] = entry
            except (json.JSONDecodeError, KeyError):
                continue
        
        self._update_state()
    
    def _create_backup(self) -> Optional[Path]:
        """Create timestamped backup of SOUL.md."""
        if not self.soul_path.exists():
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self._backup_dir / f"SOUL_backup_{timestamp}.md"
        
        try:
            shutil.copy2(self.soul_path, backup_path)
            return backup_path
        except Exception as e:
            print(f"Warning: Backup creation failed: {e}")
            return None
    
    def _update_state(self) -> None:
        """Update internal state based on current memories."""
        self._state.total_memories = len(self._memories)
        
        # Count by type
        type_counts: Dict[str, int] = {}
        total_confidence = 0.0
        
        for entry in self._memories.values():
            mem_type = entry.memory_type
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
            total_confidence += entry.confidence
        
        self._state.memories_by_type = type_counts
        self._state.avg_confidence = (
            total_confidence / len(self._memories) 
            if self._memories else 0.0
        )
        self._state.last_update = datetime.now(timezone.utc).isoformat()
    
    def add_memory(
        self,
        memory_type: MemoryType,
        title: str,
        content: str,
        confidence: float = 1.0,
        tags: Optional[List[str]] = None,
        related_ids: Optional[List[str]] = None
    ) -> str:
        """
        Add a new memory to the SOUL system.
        
        Args:
            memory_type: Type classification of the memory
            title: Short descriptive title
            content: Detailed memory content
            confidence: Initial confidence score
            tags: Associated tags
            related_ids: IDs of related memories
            
        Returns:
            entry_id: The unique ID of the created memory
        """
        with self._lock:
            timestamp = datetime.now(timezone.utc).isoformat()
            entry_id = self._generate_entry_id(content, timestamp)
            
            entry = MemoryEntry(
                entry_id=entry_id,
                timestamp=timestamp,
                memory_type=memory_type.value,
                title=title,
                content=content,
                confidence=min(1.0, max(0.0, confidence)),
                tags=tags or [],
                related_entries=related_ids or []
            )
            
            self._memories[entry_id] = entry
            self._update_state()
            self._save_memories()
            
            return entry_id
    
    def get_memory(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a memory by ID.
        
        Args:
            entry_id: The unique ID of the memory
            
        Returns:
            MemoryEntry if found, None otherwise
        """
        with self._lock:
            entry = self._memories.get(entry_id)
            if entry:
                entry.access_count += 1
                entry.last_accessed = datetime.now(timezone.utc).isoformat()
            return entry
    
    def search_memories(
        self,
        query: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """
        Search memories with various filters.
        
        Args:
            query: Text search query
            memory_type: Filter by memory type
            tags: Filter by tags (must have all specified tags)
            min_confidence: Minimum confidence threshold
            limit: Maximum number of results
            
        Returns:
            List of matching MemoryEntry objects
        """
        with self._lock:
            results = []
            current_time = datetime.now(timezone.utc)
            
            for entry in self._memories.values():
                # Apply filters
                if memory_type and entry.memory_type != memory_type.value:
                    continue
                
                if min_confidence > 0 and entry.confidence < min_confidence:
                    continue
                
                if tags and not all(tag in entry.tags for tag in tags):
                    continue
                
                if query:
                    query_lower = query.lower()
                    if (query_lower not in entry.title.lower() and 
                        query_lower not in entry.content.lower()):
                        continue
                
                results.append((entry.get_relevance_score(current_time), entry))
            
            # Sort by relevance score descending
            results.sort(key=lambda x: x[0], reverse=True)
            
            return [entry for _, entry in results[:limit]]
    
    def update_confidence(self, entry_id: str, new_confidence: float) -> bool:
        """
        Update confidence score for a memory.
        
        Args:
            entry_id: The memory ID
            new_confidence: New confidence value (0.0 to 1.0)
            
        Returns:
            True if successful, False if memory not found
        """
        with self._lock:
            if entry_id not in self._memories:
                return False
            
            entry = self._memories[entry_id]
            entry.confidence = min(1.0, max(0.0, new_confidence))
            self._update_state()
            self._save_memories()
            return True
    
    def _save_memories(self) -> None:
        """Save memories to SOUL.md file with backup."""
        # Create backup before saving
        self._create_backup()
        
        # Build markdown content
        lines = [
            "# ZAID Personal Crypto Trading Bot - SOUL Memory\n",
            "## System Information",
            f"- **Created**: {self._state.last_update or 'N/A'}",
            "- **Version**: 1.0.0",
            "- **Purpose**: Continuous algorithmic evolution through experience storage\n",
            "---\n",
            "## Memories\n",
            ""
        ]
        
        # Add each memory as JSON block
        for entry in sorted(
            self._memories.values(),
            key=lambda x: x.timestamp,
            reverse=True
        ):
            lines.append("<!-- MEMORY_START -->")
            lines.append(json.dumps(entry.to_dict(), indent=2))
            lines.append("<!-- MEMORY_END -->\n")
        
        lines.append("---\n")
        lines.append("## State Summary\n")
        lines.append(self._state.to_markdown_summary())
        
        # Write atomically
        temp_path = self.soul_path.with_suffix('.tmp')
        temp_path.write_text('\n'.join(lines), encoding='utf-8')
        temp_path.replace(self.soul_path)
    
    def get_state(self) -> SoulState:
        """Get current soul state."""
        with self._lock:
            return self._state
    
    def export_memories_json(self, output_path: str) -> Path:
        """
        Export all memories to JSON file.
        
        Args:
            output_path: Path for output JSON file
            
        Returns:
            Path to created file
        """
        with self._lock:
            output = Path(output_path)
            data = {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "state": asdict(self._state),
                "memories": [
                    entry.to_dict() 
                    for entry in self._memories.values()
                ]
            }
            
            output.write_text(json.dumps(data, indent=2), encoding='utf-8')
            return output


# Singleton instance
_soul_core: Optional[SoulCore] = None


def get_soul_core(soul_file_path: str = "./memory/SOUL.md") -> SoulCore:
    """
    Get or create the singleton SoulCore instance.
    
    Args:
        soul_file_path: Path to SOUL.md file
        
    Returns:
        SoulCore instance
    """
    global _soul_core
    if _soul_core is None:
        _soul_core = SoulCore(soul_file_path)
    return _soul_core


if __name__ == "__main__":
    # Test the Soul Core system
    soul = get_soul_core()
    
    # Add a test memory
    entry_id = soul.add_memory(
        memory_type=MemoryType.TRADE_LESSON,
        title="BTC Support Level Test",
        content="Learned that BTC finds strong support at $42,000 during Asian trading hours.",
        confidence=0.85,
        tags=["BTC", "support", "asian_session"]
    )
    
    print(f"Added memory with ID: {entry_id}")
    print(f"Current state: {soul.get_state().total_memories} memories")
    
    # Search for memories
    results = soul.search_memories(query="BTC", limit=5)
    print(f"Found {len(results)} memories matching 'BTC'")
