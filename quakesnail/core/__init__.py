"""QuakeSnail Core Engine Package.

Provides high-performance 3-stage seismic processing:
- Stage 1: Waveform picking, biquad filtering, Beltrami spike protection, 3-point Morse gating, 1-sample deferral.
- Stage 2: Wavefront phase association with so(C) triad tie-breaking and S-wave split elimination.
- Stage 3: Hypocenter inversion with Fibonacci seeding, damped Huber-IRLS, and analytical covariance.
- 3D pseudo-bending raytracing across PREM, subducting slabs, and mantle plumes.
- SO(6) Kelvin-Mandel stiffness rotors for TTI anisotropic velocity modeling.
- Live and deterministic replay feeds maintaining active heap < 35 MB.
"""

from .picking import WaveformPicker, Pick, GateClassification
from .association import WavefrontAssociator, AssociatedCluster
from .inversion import HypocenterInverter, HypocenterEstimate
from .raytrace import SphericalRaytracer, RayPath, prem_vp_kms, prem_vs_kms
from .stiffness import StiffnessRotor, compute_so6_rotor
from .streaming import StreamingClient, ReplayFeed, StationTrace
from .._quakesnail_native import py_haversine_distance_km as haversine_distance_km

__all__ = [
    "WaveformPicker",
    "Pick",
    "GateClassification",
    "WavefrontAssociator",
    "AssociatedCluster",
    "HypocenterInverter",
    "HypocenterEstimate",
    "SphericalRaytracer",
    "RayPath",
    "prem_vp_kms",
    "prem_vs_kms",
    "StiffnessRotor",
    "compute_so6_rotor",
    "StreamingClient",
    "ReplayFeed",
    "StationTrace",
    "haversine_distance_km",
]
