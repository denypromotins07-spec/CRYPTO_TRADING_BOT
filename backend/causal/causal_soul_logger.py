#!/usr/bin/env python3
"""
Causal SOUL Logger for Causal Discovery Tracking

This module logs causal discoveries and rejections to SOUL.md in the
ZAID PERSONAL CRYPTO TRADING BOT. Tracks all causal inference results,
spurious correlation traps identified, and model updates.

Features:
- Structured logging of causal discoveries
- Spurious correlation trap documentation
- Temporal tracking of causal structure evolution
- Integration with trading bot's SOUL.md knowledge base
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class CausalDiscovery:
    """Record of a causal discovery event."""
    timestamp: str
    discovery_id: str
    source_variable: str
    target_variable: str
    effect_strength: float
    confidence_level: float
    method_used: str
    is_confirmed: bool
    validation_status: str
    notes: str


@dataclass
class SpuriousCorrelationTrap:
    """Record of an identified spurious correlation."""
    timestamp: str
    trap_id: str
    variable_a: str
    variable_b: str
    observed_correlation: float
    true_relationship: str  # "none", "confounded", "reverse_causation"
    confounder_identified: Optional[str]
    detection_method: str
    lesson_learned: str


@dataclass
class ModelUpdate:
    """Record of a causal model update."""
    timestamp: str
    update_id: str
    update_type: str  # "add_edge", "remove_edge", "modify_strength"
    affected_variables: List[str]
    reason: str
    triggered_by: str


class CausalSOULLogger:
    """
    Logger for causal discoveries that updates SOUL.md.
    
    Maintains a comprehensive record of all causal inference activities,
    helping the bot learn from both successes and failures.
    """
    
    def __init__(self, soul_md_path: str = "SOUL.md"):
        self.soul_md_path = Path(soul_md_path)
        self.discoveries: List[CausalDiscovery] = []
        self.spurious_traps: List[SpuriousCorrelationTrap] = []
        self.model_updates: List[ModelUpdate] = []
        
        # Ensure SOUL.md exists
        if not self.soul_md_path.exists():
            self._initialize_soul_md()
    
    def _initialize_soul_md(self) -> None:
        """Create initial SOUL.md file with causal section."""
        initial_content = """# ZAID PERSONAL CRYPTO TRADING BOT - SOUL.md

## System Overview
This document contains the accumulated knowledge and learning of the causal trading bot.

---

## Causal Knowledge Base

### Discoveries
*No causal discoveries recorded yet.*

### Spurious Correlation Traps Identified
*No spurious correlations identified yet.*

### Model Evolution
*No model updates recorded yet.*

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| Total Discoveries | 0 |
| Confirmed Causal | 0 |
| Spurious Traps Avoided | 0 |
| Model Updates | 0 |

---

## Recent Activity Log

*No recent activity.*

"""
        self.soul_md_path.write_text(initial_content)
        logger.info(f"Initialized SOUL.md at {self.soul_md_path}")
    
    def _generate_id(self, prefix: str, *args) -> str:
        """Generate unique ID based on content and timestamp."""
        content = f"{prefix}:{datetime.now().isoformat()}:{':'.join(str(a) for a in args)}"
        return f"{prefix}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
    
    def log_discovery(
        self,
        source: str,
        target: str,
        effect_strength: float,
        confidence: float,
        method: str,
        is_confirmed: bool = False,
        validation_status: str = "pending",
        notes: str = ""
    ) -> CausalDiscovery:
        """
        Log a new causal discovery.
        
        Args:
            source: Cause variable
            target: Effect variable
            effect_strength: Magnitude of causal effect
            confidence: Confidence level (0-1)
            method: Method used for discovery
            is_confirmed: Whether discovery has been validated
            validation_status: Current validation status
            notes: Additional context
            
        Returns:
            The created CausalDiscovery object
        """
        discovery = CausalDiscovery(
            timestamp=datetime.now().isoformat(),
            discovery_id=self._generate_id("DISC", source, target),
            source_variable=source,
            target_variable=target,
            effect_strength=effect_strength,
            confidence_level=confidence,
            method_used=method,
            is_confirmed=is_confirmed,
            validation_status=validation_status,
            notes=notes
        )
        
        self.discoveries.append(discovery)
        self._update_soul_md()
        
        logger.info(
            f"Logged causal discovery: {source} -> {target} "
            f"(strength={effect_strength:.3f}, confidence={confidence:.3f})"
        )
        
        return discovery
    
    def log_spurious_trap(
        self,
        var_a: str,
        var_b: str,
        observed_corr: float,
        true_relationship: str,
        confounder: Optional[str] = None,
        detection_method: str = "causal_test",
        lesson: str = ""
    ) -> SpuriousCorrelationTrap:
        """
        Log identification of a spurious correlation trap.
        
        This is critical for the bot to avoid repeating the same mistakes.
        
        Args:
            var_a: First variable in spurious correlation
            var_b: Second variable
            observed_corr: Observed (misleading) correlation
            true_relationship: Actual relationship type
            confounder: Identified confounding variable if any
            detection_method: How the spuriousness was detected
            lesson: Key takeaway for future avoidance
            
        Returns:
            The created SpuriousCorrelationTrap object
        """
        trap = SpuriousCorrelationTrap(
            timestamp=datetime.now().isoformat(),
            trap_id=self._generate_id("TRAP", var_a, var_b),
            variable_a=var_a,
            variable_b=var_b,
            observed_correlation=observed_corr,
            true_relationship=true_relationship,
            confounder_identified=confounder,
            detection_method=detection_method,
            lesson_learned=lesson
        )
        
        self.spurious_traps.append(trap)
        self._update_soul_md()
        
        logger.warning(
            f"Logged spurious correlation trap: {var_a} <-> {var_b} "
            f"(observed={observed_corr:.3f}, true={true_relationship})"
        )
        
        return trap
    
    def log_model_update(
        self,
        update_type: str,
        affected_vars: List[str],
        reason: str,
        triggered_by: str
    ) -> ModelUpdate:
        """
        Log a change to the causal model.
        
        Args:
            update_type: Type of update performed
            affected_vars: Variables affected by the update
            reason: Why the update was made
            triggered_by: What triggered this update
            
        Returns:
            The created ModelUpdate object
        """
        update = ModelUpdate(
            timestamp=datetime.now().isoformat(),
            update_id=self._generate_id("UPD", update_type, *affected_vars),
            update_type=update_type,
            affected_variables=affected_vars,
            reason=reason,
            triggered_by=triggered_by
        )
        
        self.model_updates.append(update)
        self._update_soul_md()
        
        logger.info(f"Logged model update: {update_type} affecting {affected_vars}")
        
        return update
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of causal learning."""
        confirmed_count = sum(1 for d in self.discoveries if d.is_confirmed)
        
        return {
            "total_discoveries": len(self.discoveries),
            "confirmed_causal": confirmed_count,
            "pending_validation": len(self.discoveries) - confirmed_count,
            "spurious_traps_identified": len(self.spurious_traps),
            "model_updates": len(self.model_updates),
            "unique_variables": len(set(
                [d.source_variable for d in self.discoveries] +
                [d.target_variable for d in self.discoveries]
            ))
        }
    
    def _update_soul_md(self) -> None:
        """Update the SOUL.md file with current state."""
        stats = self.get_summary_stats()
        
        content = f"""# ZAID PERSONAL CRYPTO TRADING BOT - SOUL.md

## System Overview
This document contains the accumulated knowledge and learning of the causal trading bot.
**Last Updated:** {datetime.now().isoformat()}

---

## Causal Knowledge Base

### Recent Causal Discoveries

"""
        # Add recent discoveries (last 20)
        if self.discoveries:
            content += "| Timestamp | Relationship | Strength | Confidence | Status |\n"
            content += "|-----------|--------------|----------|------------|--------|\n"
            
            for disc in self.discoveries[-20:]:
                arrow = "→" if disc.is_confirmed else "⇢"
                content += (
                    f"| {disc.timestamp[-14:-7]} | "
                    f"{disc.source_variable} {arrow} {disc.target_variable} | "
                    f"{disc.effect_strength:.3f} | "
                    f"{disc.confidence_level:.3f} | "
                    f"{disc.validation_status} |\n"
                )
        else:
            content += "*No causal discoveries recorded yet.*\n"
        
        content += "\n### Spurious Correlation Traps Identified\n\n"
        
        # Add spurious traps (last 10)
        if self.spurious_traps:
            content += "| Timestamp | Variables | Observed Corr | True Relation | Confounder |\n"
            content += "|-----------|-----------|---------------|---------------|------------|\n"
            
            for trap in self.spurious_traps[-10:]:
                confounder_str = trap.confounder_identified or "None"
                content += (
                    f"| {trap.timestamp[-14:-7]} | "
                    f"{trap.variable_a} ↔ {trap.variable_b} | "
                    f"{trap.observed_correlation:.3f} | "
                    f"{trap.true_relationship} | "
                    f"{confounder_str} |\n"
                )
        else:
            content += "*No spurious correlations identified yet.*\n"
        
        content += "\n### Model Evolution\n\n"
        
        # Add recent model updates (last 10)
        if self.model_updates:
            content += "| Timestamp | Update Type | Affected Variables | Trigger |\n"
            content += "|-----------|-------------|-------------------|---------|\n"
            
            for update in self.model_updates[-10:]:
                vars_str = ", ".join(update.affected_variables[:3])
                if len(update.affected_variables) > 3:
                    vars_str += "..."
                content += (
                    f"| {update.timestamp[-14:-7]} | "
                    f"{update.update_type} | "
                    f"{vars_str} | "
                    f"{update.triggered_by} |\n"
                )
        else:
            content += "*No model updates recorded yet.*\n"
        
        content += f"""
---

## Performance Metrics

| Metric | Value |
|--------|-------|
| Total Discoveries | {stats['total_discoveries']} |
| Confirmed Causal | {stats['confirmed_causal']} |
| Spurious Traps Avoided | {stats['spurious_traps_identified']} |
| Model Updates | {stats['model_updates']} |
| Unique Variables | {stats['unique_variables']} |

---

## Lessons Learned from Spurious Correlations

"""
        # Add key lessons
        if self.spurious_traps:
            for i, trap in enumerate(self.spurious_traps[-5:], 1):
                content += f"{i}. **{trap.variable_a} vs {trap.variable_b}:** {trap.lesson_learned}\n"
        else:
            content += "*No lessons recorded yet.*\n"
        
        content += """
---

## Recent Activity Log

"""
        # Combine all recent events chronologically
        all_events = []
        for d in self.discoveries[-5:]:
            all_events.append((d.timestamp, f"🔬 Discovery: {d.source_variable} → {d.target_variable}"))
        for t in self.spurious_traps[-5:]:
            all_events.append((t.timestamp, f"⚠️ Trap: {t.variable_a} ↔ {t.variable_b}"))
        for u in self.model_updates[-5:]:
            all_events.append((u.timestamp, f"🔄 Update: {u.update_type}"))
        
        all_events.sort(key=lambda x: x[0], reverse=True)
        
        if all_events:
            for ts, event in all_events[:15]:
                content += f"- [{ts[-14:-7]}] {event}\n"
        else:
            content += "*No recent activity.*\n"
        
        content += "\n---\n\n*Generated by CausalSOULLogger for ZAID PERSONAL CRYPTO TRADING BOT*\n"
        
        self.soul_md_path.write_text(content)
        logger.debug(f"Updated SOUL.md with {len(self.discoveries)} discoveries, "
                    f"{len(self.spurious_traps)} traps, {len(self.model_updates)} updates")


# Singleton instance for global access
_soul_logger: Optional[CausalSOULLogger] = None


def get_soul_logger(soul_md_path: str = "SOUL.md") -> CausalSOULLogger:
    """Get or create the singleton SOUL logger instance."""
    global _soul_logger
    if _soul_logger is None:
        _soul_logger = CausalSOULLogger(soul_md_path)
    return _soul_logger


# Example usage
if __name__ == "__main__":
    logger_instance = get_soul_logger()
    
    # Log some example discoveries
    logger_instance.log_discovery(
        source="BTC_order_flow",
        target="ETH_price_change",
        effect_strength=0.15,
        confidence=0.92,
        method="Granger causality + PC algorithm",
        is_confirmed=True,
        validation_status="validated",
        notes="Strong leading indicator during high volatility periods"
    )
    
    # Log a spurious trap
    logger_instance.log_spurious_trap(
        var_a="Twitter_sentiment",
        var_b="BTC_returns",
        observed_corr=0.45,
        true_relationship="confounded",
        confounder="Market-wide momentum",
        detection_method="Backdoor adjustment",
        lesson="Sentiment correlates with returns only because both respond to price momentum"
    )
    
    # Log a model update
    logger_instance.log_model_update(
        update_type="remove_edge",
        affected_vars=["RSI", "Future_returns"],
        reason="Failed out-of-sample validation",
        triggered_by="alpha_validator"
    )
    
    # Print summary
    stats = logger_instance.get_summary_stats()
    print("\n=== Causal SOUL Summary ===")
    print(f"Total Discoveries: {stats['total_discoveries']}")
    print(f"Confirmed Causal: {stats['confirmed_causal']}")
    print(f"Spurious Traps: {stats['spurious_traps_identified']}")
    print(f"\nSOUL.md updated at: {logger_instance.soul_md_path}")
