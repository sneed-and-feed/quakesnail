"""
QuakeSnail Studio: Zero-Install Single-File WebGL2 3D Globe & EEW Tactical Workstation.

Provides offline scenario bundling, interactive WebGL2 3D globe visualization,
Wadati-Benioff subduction cross-sections, expanding P/S wavefront rings, H3 ShakeMap
intensity contours, multi-station waveform helicorders, and real-time municipality
countdown telemetry with strictly zero population felt counters.
"""

from __future__ import annotations

from quakesnail.studio.bundler import (
    export_portable_studio,
    generate_studio_html,
    get_offline_scenarios,
    resolve_studio_path,
)

__all__ = [
    "export_portable_studio",
    "generate_studio_html",
    "get_offline_scenarios",
    "resolve_studio_path",
]
