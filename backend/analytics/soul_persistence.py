"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 9
File: backend/analytics/soul_persistence.py

Permanently archives daily SOUL.md learnings from trading sessions.
Distills 4-hour trading sessions into concise markdown updates.

Features:
- Automatic session learning extraction
- Markdown-formatted knowledge persistence
- Cumulative wisdom aggregation
- Pattern recognition across sessions
- Cross-platform file handling (Windows/Linux/macOS)

Design Patterns: Repository, Builder, Memento
"""

from __future__ import annotations
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LearningCategory(Enum):
    """Categories of trading learnings."""
    MARKET_STRUCTURE = "market_structure"
    ORDER_FLOW = "order_flow"
    LIQUIDITY = "liquidity"
    VOLATILITY = "volatility"
    TIMING = "timing"
    RISK_MANAGEMENT = "risk_management"
    EXECUTION = "execution"
    PSYCHOLOGY = "psychology"
    STRATEGY = "strategy"
    MACRO = "macro"


@dataclass
class TradingInsight:
    """A single trading insight/learning."""
    category: LearningCategory
    title: str
    description: str
    confidence: float  # 0.0 to 1.0
    timestamp: float
    symbols_involved: List[str] = field(default_factory=list)
    supporting_data: Optional[Dict[str, Any]] = None
    
    def to_markdown(self) -> str:
        """Convert insight to markdown format."""
        emoji = {
            LearningCategory.MARKET_STRUCTURE: "📊",
            LearningCategory.ORDER_FLOW: "📈",
            LearningCategory.LIQUIDITY: "💧",
            LearningCategory.VOLATILITY: "📉",
            LearningCategory.TIMING: "⏱️",
            LearningCategory.RISK_MANAGEMENT: "🛡️",
            LearningCategory.EXECUTION: "⚡",
            LearningCategory.PSYCHOLOGY: "🧠",
            LearningCategory.STRATEGY: "🎯",
            LearningCategory.MACRO: "🌍",
        }.get(self.category, "📝")
        
        return f"""### {emoji} {self.title}

**Category:** {self.category.value.replace('_', ' ').title()}
**Confidence:** {self.confidence:.0%}
**Symbols:** {', '.join(self.symbols_involved) if self.symbols_involved else 'N/A'}

{self.description}

---
"""


@dataclass
class SessionSummary:
    """Summary of a trading session."""
    session_id: str
    date: str
    start_time: float
    end_time: float
    duration_hours: float
    total_trades: int
    total_pnl: float
    win_rate: float
    key_learnings: List[TradingInsight]
    mistakes_made: List[str]
    improvements_identified: List[str]
    market_conditions: str
    notes: str


class SoulPersistence:
    """
    Manages the permanent archiving of trading session learnings.
    
    Creates and maintains SOUL.md files that capture the distilled
    wisdom from each trading session, building a cumulative knowledge base.
    """
    
    SOUL_FILENAME = "SOUL.md"
    
    def __init__(self, soul_dir: Path):
        """
        Initialize the soul persistence manager.
        
        Args:
            soul_dir: Directory for storing SOUL.md files
        """
        self.soul_dir = soul_dir
        self.soul_dir.mkdir(parents=True, exist_ok=True)
        
        self._current_session_insights: List[TradingInsight] = []
        self._session_summaries: List[SessionSummary] = []
        
        logger.info(f"SoulPersistence initialized at {soul_dir}")
    
    def add_insight(
        self,
        category: LearningCategory,
        title: str,
        description: str,
        confidence: float,
        symbols: Optional[List[str]] = None,
    ) -> TradingInsight:
        """
        Add a new trading insight.
        
        Args:
            category: Category of the insight
            title: Short title
            description: Detailed description
            confidence: Confidence level (0.0 to 1.0)
            symbols: Related trading symbols
            
        Returns:
            Created TradingInsight
        """
        insight = TradingInsight(
            category=category,
            title=title,
            description=description,
            confidence=min(max(confidence, 0.0), 1.0),
            timestamp=time.time(),
            symbols_involved=symbols or [],
        )
        
        self._current_session_insights.append(insight)
        logger.debug(f"Added insight: {title}")
        
        return insight
    
    def record_session_summary(self, summary: SessionSummary) -> None:
        """Record a completed session summary."""
        self._session_summaries.append(summary)
        self._current_session_insights = []  # Reset for next session
    
    def _generate_daily_header(self, date: str) -> str:
        """Generate the header for a daily SOUL entry."""
        return f"""# 📒 Trading Soul - {date}

> *"The market teaches those who are willing to learn."*

---

"""
    
    def _generate_session_section(self, summary: SessionSummary) -> str:
        """Generate markdown for a session summary."""
        pnl_emoji = "✅" if summary.total_pnl >= 0 else "❌"
        
        return f"""## Session: {summary.session_id}

| Metric | Value |
|--------|-------|
| Duration | {summary.duration_hours:.2f} hours |
| Trades | {summary.total_trades} |
| PnL | {pnl_emoji} {summary.total_pnl:+.2f} USDT |
| Win Rate | {summary.win_rate:.1%} |
| Market Conditions | {summary.market_conditions} |

### Key Learnings

""" + "\n".join(insight.to_markdown() for insight in summary.key_learnings) + f"""
### Mistakes & Corrections

{chr(10).join(f"- ❌ {m}" for m in summary.mistakes_made)}

### Improvements Identified

{chr(10).join(f"- ✅ {i}" for i in summary.improvements_identified)}

### Notes

{summary.notes}

---

"""
    
    def write_daily_soul(self, date: Optional[str] = None) -> Path:
        """
        Write the daily SOUL.md file.
        
        Args:
            date: Date string (default: today)
            
        Returns:
            Path to the written file
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        soul_path = self.soul_dir / self.SOUL_FILENAME
        
        # Check if file exists and has content
        existing_content = ""
        if soul_path.exists():
            with open(soul_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        
        # Generate new content
        header = self._generate_daily_header(date)
        
        sessions_content = ""
        for summary in self._session_summaries:
            if summary.date == date:
                sessions_content += self._generate_session_section(summary)
        
        new_content = header + sessions_content
        
        # Append to existing or create new
        if existing_content:
            # Check if today's section already exists
            if f"# 📒 Trading Soul - {date}" not in existing_content:
                final_content = existing_content + "\n\n" + new_content
            else:
                # Replace today's section
                final_content = self._update_existing_soul(
                    existing_content, date, new_content
                )
        else:
            final_content = new_content
        
        # Write atomically
        temp_path = soul_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
            f.flush()
            os.fsync(f.fileno())
        
        temp_path.rename(soul_path)
        
        logger.info(f"Daily SOUL written: {soul_path}")
        
        return soul_path
    
    def _update_existing_soul(
        self,
        existing: str,
        date: str,
        new_section: str,
    ) -> str:
        """Update an existing SOUL.md with new content."""
        # Find the start of today's section
        marker = f"# 📒 Trading Soul - {date}"
        
        if marker in existing:
            # Find where today's section starts and ends
            lines = existing.split('\n')
            new_lines = []
            skip_until_next = False
            
            for line in lines:
                if line.startswith("# 📒 Trading Soul - "):
                    if date in line:
                        skip_until_next = True
                        continue
                    else:
                        skip_until_next = False
                
                if skip_until_next:
                    if line.startswith("## Session:") or line.startswith("---"):
                        skip_until_next = False
                    else:
                        continue
                
                new_lines.append(line)
            
            # Insert new section after the header
            existing_content = '\n'.join(new_lines)
            parts = existing_content.split('\n\n', 1)
            
            if len(parts) > 1:
                return parts[0] + '\n\n' + new_section + '\n\n' + parts[1]
            return existing_content + '\n\n' + new_section
        
        return existing + '\n\n' + new_section
    
    def append_to_cumulative_soul(self, insights: List[TradingInsight]) -> Path:
        """
        Append insights to the cumulative SOUL.md.
        
        Args:
            insights: List of insights to append
            
        Returns:
            Path to the cumulative SOUL file
        """
        cumulative_path = self.soul_dir / "CUMULATIVE_SOUL.md"
        
        # Group insights by category
        by_category: Dict[LearningCategory, List[TradingInsight]] = {}
        for insight in insights:
            by_category.setdefault(insight.category, []).append(insight)
        
        # Generate content
        content = f"\n## Updates - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        for category, category_insights in sorted(
            by_category.items(),
            key=lambda x: -sum(i.confidence for i in x[1])
        ):
            content += f"### {category.value.replace('_', ' ').title()}\n\n"
            
            for insight in sorted(category_insights, key=lambda x: -x.confidence):
                content += f"- **{insight.title}** ({insight.confidence:.0%}): "
                content += f"{insight.description[:100]}...\n"
            
            content += "\n"
        
        # Append to file
        with open(cumulative_path, 'a', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Appended to cumulative SOUL: {cumulative_path}")
        
        return cumulative_path
    
    def get_session_insights(self) -> List[TradingInsight]:
        """Get all insights from the current session."""
        return self._current_session_insights.copy()
    
    def export_session_json(self) -> Dict[str, Any]:
        """Export session data as JSON-serializable dict."""
        return {
            'insights': [
                {
                    'category': i.category.value,
                    'title': i.title,
                    'description': i.description,
                    'confidence': i.confidence,
                    'timestamp': i.timestamp,
                    'symbols': i.symbols_involved,
                }
                for i in self._current_session_insights
            ],
            'summaries': [
                {
                    'session_id': s.session_id,
                    'date': s.date,
                    'duration_hours': s.duration_hours,
                    'total_trades': s.total_trades,
                    'total_pnl': s.total_pnl,
                    'win_rate': s.win_rate,
                    'learning_count': len(s.key_learnings),
                }
                for s in self._session_summaries
            ],
        }


# Example usage
if __name__ == "__main__":
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        soul = SoulPersistence(Path(tmpdir))
        
        # Add some insights
        soul.add_insight(
            category=LearningCategory.LIQUIDITY,
            title="BTC Liquidity Pool Detection",
            description=(
                "Identified that major liquidity pools form at round number levels "
                "(50k, 55k, etc.) with 2-3x more stop losses than expected. "
                "Consider targeting these levels for entries."
            ),
            confidence=0.85,
            symbols=["BTCUSDT"],
        )
        
        soul.add_insight(
            category=LearningCategory.TIMING,
            title="NY Session Overlap Volatility",
            description=(
                "Highest probability trades occur during NY-London overlap (13:00-16:00 UTC). "
                "Avoid trading outside this window except for clear breakouts."
            ),
            confidence=0.75,
            symbols=["BTCUSDT", "ETHUSDT"],
        )
        
        # Create session summary
        summary = SessionSummary(
            session_id="session_20241125_001",
            date="2024-11-25",
            start_time=time.time() - 14400,
            end_time=time.time(),
            duration_hours=4.0,
            total_trades=12,
            total_pnl=850.0,
            win_rate=0.75,
            key_learnings=soul.get_session_insights(),
            mistakes_made=[
                "Entered too early on ETH breakout without confirmation",
                "Position size was too large on first trade",
            ],
            improvements_identified=[
                "Wait for candle close confirmation",
                "Use half position size for first trade of session",
            ],
            market_conditions="Trending bullish, moderate volatility",
            notes="Good recovery after initial mistakes",
        )
        
        soul.record_session_summary(summary)
        
        # Write daily SOUL
        path = soul.write_daily_soul("2024-11-25")
        print(f"Written to: {path}")
        
        # Export
        export = soul.export_session_json()
        print(f"Exported {len(export['insights'])} insights")
