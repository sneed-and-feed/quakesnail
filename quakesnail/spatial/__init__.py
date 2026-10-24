"""
QuakeSnail Spatial & Municipality Intensity Engine.

Provides multi-tier municipality spatial indexing, fault/regime-aware anisotropic
ShakeMap GMPE modeling (PGA, PGV, MMI, JMA Shindo), and real-time arrival countdowns.
"""

from quakesnail.spatial.catalog import (
    Municipality,
    MunicipalityCatalog,
    MunicipalityIntensity,
    ShorelineRegion,
    get_default_catalog,
)
from quakesnail.spatial.countdown import (
    ArrivalCountdownEngine,
    ArrivalCountdownResult,
)
from quakesnail.spatial.shakemap import (
    GroundMotionResult,
    ShakeMapEngine,
    ShakeMapGridPoint,
)

__all__ = [
    "Municipality",
    "MunicipalityCatalog",
    "MunicipalityIntensity",
    "ShorelineRegion",
    "get_default_catalog",
    "ShakeMapEngine",
    "GroundMotionResult",
    "ShakeMapGridPoint",
    "ArrivalCountdownEngine",
    "ArrivalCountdownResult",
]
