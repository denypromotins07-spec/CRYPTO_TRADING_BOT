#!/usr/bin/env python3
"""
Dashboard Provisioner for Grafana

This module auto-generates Grafana JSON configurations for the trading bot,
implementing the Proxy design pattern for dashboard management.

Key Features:
- Auto-generated dashboard JSON from templates
- Dynamic panel creation based on metrics
- Threshold-based alerting configuration
- Time range and refresh rate customization
- Export to Grafana-compatible format

Designed for the ZAID Personal Crypto Trading Bot to provide
immediate observability through pre-configured dashboards.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class PanelType(Enum):
    """Types of Grafana panels."""
    GRAPH = "graph"
    STAT = "stat"
    GAUGE = "gauge"
    TABLE = "table"
    HEATMAP = "heatmap"
    BARGAUGE = "bargauge"
    TEXT = "text"
    ALERTLIST = "alertlist"


class QueryType(Enum):
    """Types of data source queries."""
    PROMETHEUS = "prometheus"
    INFLUXDB = "influxdb"
    ELASTICSEARCH = "elasticsearch"
    MYSQL = "mysql"
    POSTGRES = "postgres"


@dataclass
class DashboardConfig:
    """Configuration for dashboard generation."""
    title: str
    uid: str
    refresh_interval: str = "5s"
    time_range: str = "now-1h"
    timezone: str = "browser"
    tags: List[str] = field(default_factory=lambda: ["zaidd", "crypto", "trading"])
    editable: bool = True
    version: int = 1


@dataclass
class PanelConfig:
    """Configuration for a single panel."""
    title: str
    panel_type: PanelType
    query: str
    query_type: QueryType = QueryType.PROMETHEUS
    grid_pos: Optional[Dict[str, int]] = None
    thresholds: Optional[List[Tuple[float, str]]] = None
    unit: str = "short"
    decimals: int = 2
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'panel_type': self.panel_type.value,
            'query': self.query,
            'query_type': self.query_type.value,
            'grid_pos': self.grid_pos,
            'thresholds': self.thresholds,
            'unit': self.unit,
            'decimals': self.decimals,
            'description': self.description
        }


class DashboardProvisioner:
    """
    Auto-generates Grafana dashboard configurations.
    
    Implements the Proxy pattern to manage dashboard lifecycle
    and provide dynamic panel generation.
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None) -> None:
        self.config = config or DashboardConfig(
            title="ZAID Trading Bot Dashboard",
            uid="zaidd-bot-main"
        )
        self._panels: List[PanelConfig] = []
        self._variables: Dict[str, str] = {}
        self._annotations: List[Dict[str, Any]] = []
        self._alerts: List[Dict[str, Any]] = []
    
    def add_panel(self, panel: PanelConfig) -> 'DashboardProvisioner':
        """Add a panel to the dashboard."""
        # Auto-calculate grid position if not provided
        if panel.grid_pos is None:
            panel.grid_pos = self._calculate_grid_pos(len(self._panels))
        
        self._panels.append(panel)
        return self
    
    def add_stat_panel(
        self,
        title: str,
        query: str,
        unit: str = "short",
        decimals: int = 2,
        thresholds: Optional[List[Tuple[float, str]]] = None
    ) -> 'DashboardProvisioner':
        """Add a stat panel (single value display)."""
        return self.add_panel(PanelConfig(
            title=title,
            panel_type=PanelType.STAT,
            query=query,
            unit=unit,
            decimals=decimals,
            thresholds=thresholds
        ))
    
    def add_graph_panel(
        self,
        title: str,
        query: str,
        unit: str = "short",
        decimals: int = 2,
        height: int = 6
    ) -> 'DashboardProvisioner':
        """Add a graph panel (time series)."""
        grid_pos = {'x': 0, 'y': len(self._panels) * height, 'w': 24, 'h': height}
        return self.add_panel(PanelConfig(
            title=title,
            panel_type=PanelType.GRAPH,
            query=query,
            unit=unit,
            decimals=decimals,
            grid_pos=grid_pos
        ))
    
    def add_gauge_panel(
        self,
        title: str,
        query: str,
        min_val: float = 0,
        max_val: float = 100,
        unit: str = "percent"
    ) -> 'DashboardProvisioner':
        """Add a gauge panel."""
        return self.add_panel(PanelConfig(
            title=title,
            panel_type=PanelType.GAUGE,
            query=query,
            unit=unit
        ))
    
    def add_table_panel(
        self,
        title: str,
        query: str,
        columns: Optional[List[str]] = None
    ) -> 'DashboardProvisioner':
        """Add a table panel."""
        return self.add_panel(PanelConfig(
            title=title,
            panel_type=PanelType.TABLE,
            query=query
        ))
    
    def add_variable(self, name: str, query: str, label: str = "") -> 'DashboardProvisioner':
        """Add a template variable."""
        self._variables[name] = query
        return self
    
    def add_annotation(
        self,
        name: str,
        query: str,
        color: str = "#E02F44"
    ) -> 'DashboardProvisioner':
        """Add an annotation query."""
        self._annotations.append({
            'name': name,
            'query': query,
            'color': color
        })
        return self
    
    def add_alert(
        self,
        name: str,
        condition: str,
        threshold: float,
        severity: str = "warning"
    ) -> 'DashboardProvisioner':
        """Add an alert rule."""
        self._alerts.append({
            'name': name,
            'condition': condition,
            'threshold': threshold,
            'severity': severity
        })
        return self
    
    def _calculate_grid_pos(self, index: int) -> Dict[str, int]:
        """Calculate grid position for a panel."""
        # Default layout: 2 columns, each panel is 12 units wide
        row = index // 2
        col = index % 2
        
        return {
            'x': col * 12,
            'y': row * 6,
            'w': 12,
            'h': 6
        }
    
    def generate_dashboard_json(self) -> Dict[str, Any]:
        """Generate the complete Grafana dashboard JSON."""
        dashboard = {
            'dashboard': {
                'id': None,
                'uid': self.config.uid,
                'title': self.config.title,
                'tags': self.config.tags,
                'timezone': self.config.timezone,
                'editable': self.config.editable,
                'version': self.config.version,
                'refresh': self.config.refresh_interval,
                'schemaVersion': 38,
                'fiscalYearStartMonth': 0,
                'graphTooltip': 0,
                'links': [],
                'liveNow': False,
                
                # Time settings
                'time': {
                    'from': self.config.time_range.split('to')[0].strip(),
                    'to': 'now'
                },
                
                # Template variables
                'templating': {
                    'list': [
                        {
                            'name': name,
                            'type': 'query',
                            'datasource': {'type': 'prometheus', 'uid': 'prometheus'},
                            'query': query,
                            'label': name.replace('_', ' ').title(),
                            'current': {},
                            'hide': 0,
                            'includeAll': False,
                            'multi': False,
                            'options': [],
                            'refresh': 2,
                            'regex': '',
                            'skipUrlSync': False,
                            'sort': 0
                        }
                        for name, query in self._variables.items()
                    ]
                },
                
                # Annotations
                'annotations': {
                    'list': [
                        {
                            'builtIn': 1,
                            'datasource': {'type': 'grafana', 'uid': '-- Grafana --'},
                            'enable': True,
                            'hide': True,
                            'iconColor': 'rgba(0, 211, 255, 1)',
                            'name': 'Annotations & Alerts',
                            'target': {
                                'limit': 100,
                                'matchAny': False,
                                'tags': [],
                                'type': 'dashboard'
                            },
                            'type': 'dashboard'
                        }
                    ] + [
                        {
                            'datasource': {'type': 'prometheus', 'uid': 'prometheus'},
                            'enable': True,
                            'expr': ann['query'],
                            'iconColor': ann['color'],
                            'name': ann['name'],
                            'step': '60s'
                        }
                        for ann in self._annotations
                    ]
                },
                
                # Panels
                'panels': self._generate_panels()
            },
            'overwrite': True,
            'message': f'Auto-generated by ZAID Bot at {time.strftime("%Y-%m-%d %H:%M:%S")}'
        }
        
        return dashboard
    
    def _generate_panels(self) -> List[Dict[str, Any]]:
        """Generate panel configurations."""
        panels = []
        
        for i, panel_config in enumerate(self._panels):
            panel = self._create_panel(panel_config, i)
            panels.append(panel)
        
        return panels
    
    def _create_panel(self, config: PanelConfig, index: int) -> Dict[str, Any]:
        """Create a single panel configuration."""
        base_panel = {
            'id': index + 1,
            'title': config.title,
            'description': config.description,
            'gridPos': config.grid_pos or self._calculate_grid_pos(index),
            'type': config.panel_type.value,
            'targets': [
                {
                    'datasource': {'type': 'prometheus', 'uid': 'prometheus'},
                    'expr': config.query,
                    'legendFormat': config.title,
                    'refId': 'A'
                }
            ],
            'fieldConfig': {
                'defaults': {
                    'unit': config.unit,
                    'decimals': config.decimals,
                    'custom': {}
                },
                'overrides': []
            },
            'options': {}
        }
        
        # Add thresholds for stat/gauge panels
        if config.thresholds and config.panel_type in (PanelType.STAT, PanelType.GAUGE):
            steps = []
            for value, color in sorted(config.thresholds, key=lambda x: x[0]):
                steps.append({'value': value, 'color': color})
            
            base_panel['fieldConfig']['defaults']['thresholds'] = {
                'mode': 'absolute',
                'steps': steps
            }
        
        # Panel-specific options
        if config.panel_type == PanelType.GAUGE:
            base_panel['options'] = {
                'showThresholdLabels': False,
                'showThresholdMarkers': True
            }
        elif config.panel_type == PanelType.STAT:
            base_panel['options'] = {
                'colorMode': 'value',
                'graphMode': 'area',
                'justifyMode': 'auto',
                'orientation': 'auto',
                'reduceOptions': {
                    'calcs': ['lastNotNull'],
                    'fields': '',
                    'values': False
                },
                'textMode': 'auto'
            }
        elif config.panel_type == PanelType.GRAPH:
            base_panel['options'] = {
                'legend': {
                    'displayMode': 'list',
                    'placement': 'bottom',
                    'showLegend': True
                },
                'tooltip': {
                    'mode': 'single',
                    'sort': 'none'
                }
            }
            base_panel['fieldConfig']['defaults']['custom'] = {
                'drawStyle': 'line',
                'lineInterpolation': 'linear',
                'lineWidth': 1,
                'fillOpacity': 10,
                'gradientMode': 'none',
                'spanNulls': False,
                'showPoints': 'auto',
                'pointSize': 5,
                'stacking': {'mode': 'none', 'group': 'A'},
                'axisPlacement': 'auto',
                'axisLabel': '',
                'scaleDistribution': {'type': 'linear'},
                'hideFrom': {'tooltip': False, 'viz': False, 'legend': False},
                'thresholdsStyle': {'mode': 'off'}
            }
        elif config.panel_type == PanelType.TABLE:
            base_panel['options'] = {
                'showHeader': True,
                'sortBy': []
            }
        
        return base_panel
    
    def export_json(self, indent: int = 2) -> str:
        """Export dashboard as JSON string."""
        dashboard = self.generate_dashboard_json()
        return json.dumps(dashboard, indent=indent)
    
    def export_file(self, filepath: str) -> None:
        """Export dashboard to a file."""
        with open(filepath, 'w') as f:
            f.write(self.export_json())
    
    def get_preview(self) -> str:
        """Get a text preview of the dashboard."""
        lines = [
            f"Dashboard: {self.config.title}",
            f"UID: {self.config.uid}",
            f"Refresh: {self.config.refresh_interval}",
            f"Time Range: {self.config.time_range}",
            f"\nPanels ({len(self._panels)}):",
        ]
        
        for i, panel in enumerate(self._panels):
            lines.append(f"  {i+1}. [{panel.panel_type.value}] {panel.title}")
            lines.append(f"     Query: {panel.query[:60]}...")
        
        if self._variables:
            lines.append(f"\nVariables ({len(self._variables)}):")
            for name in self._variables:
                lines.append(f"  - ${name}")
        
        if self._alerts:
            lines.append(f"\nAlerts ({len(self._alerts)}):")
            for alert in self._alerts:
                lines.append(f"  - {alert['name']}: {alert['condition']} > {alert['threshold']}")
        
        return '\n'.join(lines)


def create_trading_dashboard() -> DashboardProvisioner:
    """Create a pre-configured trading dashboard."""
    provisioner = DashboardProvisioner(DashboardConfig(
        title="ZAID Trading Bot - Main Dashboard",
        uid="zaidd-trading-main",
        refresh_interval="5s",
        time_range="now-1h"
    ))
    
    # PnL Panel
    provisioner.add_stat_panel(
        title="Current PnL (USDT)",
        query='bot_pnl_usdt{pair="BTC/USDT"}',
        unit="currencyUSD",
        decimals=2,
        thresholds=[
            (-100, "#E02F44"),  # Red for losses
            (0, "#FF9830"),     # Orange for break-even
            (50, "#73BF69"),    # Green for profits
        ]
    )
    
    # Latency Panel
    provisioner.add_stat_panel(
        title="Order Execution Latency (μs)",
        query='histogram_quantile(0.99, rate(bot_order_latency_us_bucket[1m]))',
        unit="µs",
        decimals=0,
        thresholds=[
            (0, "#73BF69"),      # Green < 1ms
            (1000, "#FF9830"),   # Orange 1-5ms
            (5000, "#E02F44"),   # Red > 5ms
        ]
    )
    
    # Active Positions
    provisioner.add_stat_panel(
        title="Active Positions",
        query='count(bot_position_open)',
        unit="short",
        decimals=0
    )
    
    # Win Rate Gauge
    provisioner.add_gauge_panel(
        title="Win Rate (%)",
        query='sum(bot_trade_won) / (sum(bot_trade_won) + sum(bot_trade_lost)) * 100',
        min_val=0,
        max_val=100,
        unit="percent"
    )
    
    # PnL Graph
    provisioner.add_graph_panel(
        title="PnL Over Time",
        query='bot_pnl_usdt',
        unit="currencyUSD",
        decimals=2
    )
    
    # Latency Graph
    provisioner.add_graph_panel(
        title="Latency Distribution",
        query='histogram_quantile(0.95, rate(bot_order_latency_us_bucket[1m]))',
        unit="µs",
        decimals=0
    )
    
    # Volume Graph
    provisioner.add_graph_panel(
        title="Trading Volume",
        query='sum(rate(bot_volume_usdt[5m]))',
        unit="currencyUSD",
        decimals=2
    )
    
    # Variables
    provisioner.add_variable("pair", 'label_values(bot_pnl_usdt, pair)', "Trading Pair")
    provisioner.add_variable("exchange", 'label_values(bot_metrics, exchange)', "Exchange")
    
    # Annotations
    provisioner.add_annotation(
        "Trade Executions",
        'bot_trade_executed > 0',
        "#73BF69"
    )
    
    provisioner.add_annotation(
        "Circuit Breaker Events",
        'bot_circuit_breaker_triggered > 0',
        "#E02F44"
    )
    
    # Alerts
    provisioner.add_alert(
        "High Latency",
        "latency_p99",
        5000,
        "critical"
    )
    
    provisioner.add_alert(
        "PnL Drawdown",
        "pnl_drawdown",
        0.05,
        "warning"
    )
    
    return provisioner


if __name__ == '__main__':
    # Example usage
    print("Creating Trading Dashboard...")
    
    dashboard = create_trading_dashboard()
    
    print("\nDashboard Preview:")
    print(dashboard.get_preview())
    
    print("\n\nExporting JSON...")
    json_output = dashboard.export_json()
    print(f"Generated {len(json_output)} bytes of JSON")
    
    # Save to file
    output_file = "zaidd_dashboard.json"
    dashboard.export_file(output_file)
    print(f"Saved to {output_file}")
    
    print("\nDashboard Provisioner test complete.")
