# ZAID PERSONAL CRYPTO TRADING BOT - SOUL LOG
## Microstructure Event Audit Trail

This file contains an immutable record of all microstructure
events that affected trading decisions.

---

{"details": {"action_taken": "quotes_pulled", "confidence": 0.85, "order_ids": ["SPOOF-001", "SPOOF-002"]}, "event_id": "SOUL-1784989435879419893-000001", "event_type": "spoofing_detected", "pnl_impact_estimate": 0.0, "price": 50000.0, "quantity": 100.0, "severity": "CRITICAL", "symbol": "BTC/USDT", "timestamp_iso": "2026-07-25T14:23:55.879428Z", "timestamp_ns": 1784989435879427721}
{"details": {"detection_method": "fake_liquidity_filter", "estimated_impact_bps": 25.0, "wall_size": 5000.0}, "event_id": "SOUL-1784989435880079815-000002", "event_type": "fake_liquidity_wall_avoided", "pnl_impact_estimate": 625000.0, "price": 50000.0, "quantity": 50.0, "severity": "WARNING", "symbol": "BTC/USDT", "timestamp_iso": "2026-07-25T14:23:55.880085Z", "timestamp_ns": 1784989435880084712}
{"details": {"quantity_ahead": 50000, "queue_position": 150, "reason": "queue_position_too_far_back"}, "event_id": "SOUL-1784989435880250687-000003", "event_type": "fill_miss", "pnl_impact_estimate": 0.0, "price": 3000.0, "quantity": 10.0, "severity": "INFO", "symbol": "ETH/USDT", "timestamp_iso": "2026-07-25T14:23:55.880255Z", "timestamp_ns": 1784989435880254354}
{"details": {"action_taken": "spread_widened", "toxicity_score": 0.75, "vpin": 0.65}, "event_id": "SOUL-1784989435880403398-000004", "event_type": "adverse_selection_prevented", "pnl_impact_estimate": 0.0, "price": 100.0, "quantity": 0.0, "severity": "WARNING", "symbol": "SOL/USDT", "timestamp_iso": "2026-07-25T14:23:55.880407Z", "timestamp_ns": 1784989435880406746}
