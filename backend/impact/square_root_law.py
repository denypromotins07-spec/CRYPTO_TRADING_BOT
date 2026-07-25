#!/usr/bin/env python3
"""Square Root Law Market Impact Model."""
from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Dict, Optional, Tuple, List
from enum import Enum
import math
from threading import Lock

class ImpactRegime(Enum):
    LINEAR = "linear"
    SQUARE_ROOT = "square_root"
    POWER_LAW = "power_law"

@dataclass(slots=True)
class ImpactParameters:
    volatility: float
    avg_daily_volume: float
    participation_rate_limit: float
    gamma: float
    alpha: float
    temporary_ratio: float
    
    def validate(self) -> bool:
        return (0 < self.volatility < 2.0 and self.avg_daily_volume > 0 and
                0 < self.participation_rate_limit <= 1.0 and 0.3 <= self.gamma <= 0.8 and
                self.alpha > 0 and 0 <= self.temporary_ratio <= 1.0)

@dataclass(slots=True)
class ImpactResult:
    trade_size: float
    total_impact_bps: float
    temporary_impact_bps: float
    permanent_impact_bps: float
    impact_regime: ImpactRegime
    participation_rate: float
    expected_slippage_pct: float
    confidence_lower: float
    confidence_upper: float

class SquareRootImpactModel:
    DEFAULT_PARAMS: Dict[str, ImpactParameters] = {
        "BTC": ImpactParameters(0.04, 50000.0, 0.05, 0.5, 0.8, 0.6),
        "ETH": ImpactParameters(0.05, 500000.0, 0.05, 0.5, 0.9, 0.55),
        "SOL": ImpactParameters(0.08, 5000000.0, 0.03, 0.55, 1.2, 0.5),
        "USDT": ImpactParameters(0.001, 100000000.0, 0.10, 0.3, 0.1, 0.8),
    }
    
    def __init__(self, asset: str = "BTC", custom_params: Optional[ImpactParameters] = None):
        self.asset = asset.upper()
        self._lock = Lock()
        if custom_params is not None:
            if not custom_params.validate():
                raise ValueError("Invalid impact parameters")
            self.params = custom_params
        else:
            p = self.DEFAULT_PARAMS.get(self.asset, self.DEFAULT_PARAMS["BTC"])
            # Use field values instead of __dict__ for slots compatibility
            self.params = ImpactParameters(
                p.volatility, p.avg_daily_volume, p.participation_rate_limit,
                p.gamma, p.alpha, p.temporary_ratio
            )
        self.small_order_threshold = 0.001
        self.large_order_threshold = 0.05
    
    def calculate_impact(self, trade_size: float, side: str = "buy", current_price: float = 0.0) -> ImpactResult:
        with self._lock:
            if trade_size <= 0:
                return ImpactResult(0.0, 0.0, 0.0, 0.0, ImpactRegime.LINEAR, 0.0, 0.0, 0.0, 0.0)
            
            participation_rate = trade_size / self.params.avg_daily_volume
            regime = self._determine_regime(participation_rate)
            raw_impact = self._calculate_raw_impact(trade_size, regime, participation_rate)
            
            temporary_impact = raw_impact * self.params.temporary_ratio
            permanent_impact = raw_impact * (1.0 - self.params.temporary_ratio)
            impact_bps = abs(raw_impact) * 10000.0
            
            return ImpactResult(
                trade_size=trade_size, total_impact_bps=impact_bps,
                temporary_impact_bps=temporary_impact * 10000.0,
                permanent_impact_bps=permanent_impact * 10000.0,
                impact_regime=regime, participation_rate=participation_rate,
                expected_slippage_pct=abs(raw_impact) * 100.0,
                confidence_lower=impact_bps * 0.7, confidence_upper=impact_bps * 1.3,
            )
    
    def _determine_regime(self, participation_rate: float) -> ImpactRegime:
        if participation_rate < self.small_order_threshold:
            return ImpactRegime.LINEAR
        elif participation_rate < self.large_order_threshold:
            return ImpactRegime.SQUARE_ROOT
        return ImpactRegime.POWER_LAW
    
    def _calculate_raw_impact(self, trade_size: float, regime: ImpactRegime, participation_rate: float) -> float:
        sigma, alpha, gamma = self.params.volatility, self.params.alpha, self.params.gamma
        if regime == ImpactRegime.LINEAR:
            return alpha * sigma * participation_rate
        elif regime == ImpactRegime.SQUARE_ROOT:
            return alpha * sigma * math.sqrt(participation_rate)
        else:
            adjusted_gamma = min(gamma + 0.2, 0.9)
            return alpha * sigma * (participation_rate ** adjusted_gamma)
    
    def calculate_optimal_order_size(self, max_impact_bps: float, side: str = "buy") -> float:
        with self._lock:
            max_impact_decimal = max_impact_bps / 10000.0
            ratio = max_impact_decimal / (self.params.alpha * self.params.volatility)
            optimal_size = self.params.avg_daily_volume * (ratio ** 2)
            max_allowed = self.params.avg_daily_volume * self.params.participation_rate_limit
            return min(optimal_size, max_allowed)
    
    def estimate_total_cost(self, total_quantity: float, num_slices: int = 10) -> Dict[str, float]:
        with self._lock:
            slice_size = total_quantity / num_slices
            total_temporary, total_permanent = 0.0, 0.0
            for i in range(num_slices):
                result = self.calculate_impact(slice_size, "buy")
                total_temporary += result.temporary_impact_bps
                total_permanent += result.permanent_impact_bps * (num_slices - i) / num_slices
            single_cost = self.calculate_impact(total_quantity, "buy").total_impact_bps
            return {
                "total_temporary_bps": total_temporary, "total_permanent_bps": total_permanent,
                "total_cost_bps": total_temporary + total_permanent,
                "cost_per_slice_bps": (total_temporary + total_permanent) / num_slices,
                "savings_vs_single_bps": single_cost - (total_temporary + total_permanent),
            }

class MultiAssetImpactTracker:
    def __init__(self, assets: List[str]):
        self.models: Dict[str, SquareRootImpactModel] = {a: SquareRootImpactModel(a) for a in assets}
    
    def get_impact(self, asset: str, trade_size: float, side: str = "buy") -> Optional[ImpactResult]:
        model = self.models.get(asset.upper())
        return model.calculate_impact(trade_size, side) if model else None

if __name__ == "__main__":
    model = SquareRootImpactModel("BTC")
    print("Square Root Law Impact Model initialized")
    for size in [0.1, 1.0, 10.0, 100.0]:
        r = model.calculate_impact(size, "buy")
        print(f"{size} BTC: {r.total_impact_bps:.2f} bps ({r.impact_regime.value})")
