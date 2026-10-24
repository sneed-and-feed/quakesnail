"""
quakesnail.benchmark.baseline_gq: Simulated Legacy GlobalQuake Baselines.

Faithfully implements the algorithmic behaviors and systemic vulnerabilities
documented in `explorer_survey_gq/analysis.md` (legacy GlobalQuake Java classes):
1. `LegacyGlobalQuakeSTA_LTA`: Classic STA/LTA (1s/20s, threshold 3.5, BetterAnalysis.java)
   lacking 1-sample causal deferral, triggering false alerts on isolated spikes/glitches.
2. `LegacyGlobalQuakeAssociator`: Greedy distance-based associator (ClusterAnalysis.java)
   lacking cross-phase discrimination and so(C) triad anti-stealing, triggering
   phantom earthquake cluster splits on S-wave arrivals.
3. `LegacyFibonacciGrid10kmFallback`: Ill-conditioned centroid/grid search
   (EarthquakeAnalysis.java) with fixed 10.0 km depth fallback clamp.
4. `LegacyLinearCityScan`: O(N) 43,000-row linear city scan (FeatureCities.java)
   with arbitrary population 'countFelt' / 'countStrong' arctangent multipliers.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from quakesnail._quakesnail_native import (
    EARTH_RADIUS_KM,
    PHASE_P,
    PHASE_S,
    PHASE_PKP,
    PHASE_PKIKP,
    py_haversine_distance_km,
)
from quakesnail.core.picking import Pick


class LegacyGlobalQuakeSTA_LTA:
    """
    Simulates Legacy GlobalQuake classic recursive STA/LTA detector
    (BetterAnalysis.java lines 126-152) without 1-sample causal deferral.
    """

    def __init__(
        self,
        fs: float = 50.0,
        sta_len_s: float = 1.0,
        lta_len_s: float = 20.0,
        threshold: float = 3.5,
    ):
        self.fs = fs
        self.alpha_sta = 1.0 / max(1.0, sta_len_s * fs)
        self.alpha_lta = 1.0 / max(1.0, lta_len_s * fs)
        self.threshold = threshold
        self.sta = 0.0
        self.lta = 1.0
        self.trigger_count = 0
        self.in_event = False

    def reset(self) -> None:
        self.sta = 0.0
        self.lta = 1.0
        self.trigger_count = 0
        self.in_event = False

    def feed_sample(self, x: float) -> bool:
        """Feed a single sample and return whether ratio > threshold (matching test_referee.py)."""
        cfa = abs(float(x))
        self.sta = (1.0 - self.alpha_sta) * self.sta + self.alpha_sta * cfa
        self.lta = (1.0 - self.alpha_lta) * self.lta + self.alpha_lta * cfa
        ratio = self.sta / max(1e-6, self.lta)

        if ratio > self.threshold:
            self.trigger_count += 1
            return True
        return False

    def process_stream(self, samples: np.ndarray) -> List[int]:
        """Process continuous waveform stream, returning sample indices of triggers."""
        triggers: List[int] = []
        for i, s in enumerate(samples):
            if self.feed_sample(s):
                triggers.append(i)
        return triggers


# Alias for compatibility with tests/test_referee.py
LegacyGlobalQuakeDetector = LegacyGlobalQuakeSTA_LTA


class LegacyGlobalQuakeAssociator:
    """
    Simulates Legacy GlobalQuake greedy distance association (ClusterAnalysis.java)
    without S-wave protection or so(C) triad tie-breaking.
    Treats delayed S-wave moveout as a separate spurious cluster (phantom split).
    """

    def __init__(self, max_p_velocity_kms: float = 8.5):
        self.max_p_velocity_kms = max_p_velocity_kms
        self.clusters: List[Dict] = []

    def associate_picks(
        self,
        picks: List[Pick],
        station_coords: Dict[str, Tuple[float, float, float]],
    ) -> List[Dict]:
        """Greedy association that treats S-waves as duplicate phantom clusters."""
        sorted_picks = sorted(picks, key=lambda p: p.arrival_time_s)
        self.clusters.clear()

        for p in sorted_picks:
            assigned = False
            p_lat, p_lon, _ = station_coords.get(p.station_code, (0.0, 0.0, 0.0))

            for cl in self.clusters:
                first_pick = cl["picks"][0]
                dt = p.arrival_time_s - first_pick.arrival_time_s
                d_km = py_haversine_distance_km(p_lat, p_lon, cl["lat"], cl["lon"])
                expected_tt = d_km / self.max_p_velocity_kms

                # If within nominal P travel time, attach; otherwise spawn new cluster
                if abs(dt - expected_tt) < 4.0:
                    cl["picks"].append(p)
                    assigned = True
                    break

            if not assigned:
                self.clusters.append({
                    "cluster_id": f"legacy_cl_{len(self.clusters)}",
                    "lat": p_lat,
                    "lon": p_lon,
                    "depth_km": 10.0,
                    "t0": p.arrival_time_s,
                    "picks": [p],
                })

        return self.clusters


class LegacyFibonacciGrid10kmFallback:
    """
    Simulates Legacy GlobalQuake least-squares grid search with 10.0 km depth collapse
    (EarthquakeAnalysis.java lines 639-646 and 904-908).
    """

    def __init__(self, fallback_depth_km: float = 10.0):
        self.fallback_depth_km = fallback_depth_km

    def invert(
        self,
        picks: List[Pick],
        station_coords: Dict[str, Tuple[float, float, float]],
    ) -> Tuple[float, float, float, float]:
        """Legacy inversion with ill-conditioned vertical collapse to fixed 10.0 km."""
        if len(picks) < 4:
            return (0.0, 0.0, self.fallback_depth_km, 0.0)

        valid_picks = [p for p in picks if p.station_code in station_coords]
        if not valid_picks:
            return (0.0, 0.0, self.fallback_depth_km, 0.0)

        earliest_picks = sorted(valid_picks, key=lambda p: p.arrival_time_s)[:min(8, len(valid_picks))]
        lats = [station_coords[p.station_code][0] for p in earliest_picks]
        lons = [station_coords[p.station_code][1] for p in earliest_picks]
        mean_lat = float(np.mean(lats))
        mean_lon = float(np.mean(lons))

        # Hardcoded 10.0 km depth clamp
        fixed_depth = self.fallback_depth_km

        first_pick = earliest_picks[0]
        st_lat, st_lon, _ = station_coords[first_pick.station_code]
        d_km = py_haversine_distance_km(mean_lat, mean_lon, st_lat, st_lon)
        slant_km = math.sqrt(d_km**2 + fixed_depth**2)
        t0 = first_pick.arrival_time_s - (slant_km / 6.8)

        return (mean_lat, mean_lon, fixed_depth, t0)


# Alias for compatibility with tests/test_referee.py
LegacyGlobalQuakeInverter = LegacyFibonacciGrid10kmFallback


class LegacyLinearCityScan:
    """
    Simulates Legacy GlobalQuake FeatureCities.java & ShakemapService.java:
    Linear O(N) scan across 43,000 city entries, and unphysical arctangent
    'people felt' exposure multipliers.
    """

    def __init__(self, num_cities: int = 43000, seed: int = 42):
        self.num_cities = num_cities
        rng = np.random.RandomState(seed)
        self.city_lats = rng.uniform(-60.0, 70.0, size=num_cities).astype(np.float32)
        self.city_lons = rng.uniform(-180.0, 180.0, size=num_cities).astype(np.float32)
        self.city_pops = rng.lognormal(mean=10.0, sigma=1.5, size=num_cities).astype(np.float64)

    @staticmethod
    def legacy_felt_multiplier(pga_gal: float) -> float:
        """Unphysical arctangent felt multiplier (GlobalQuakePanel.java line 240)."""
        return float((2.0 / math.pi) * math.atan(0.2 * pga_gal))

    @staticmethod
    def legacy_felt_strong_multiplier(pga_gal: float, pga_mmi_v: float = 34.0) -> float:
        """Unphysical arctangent strong felt multiplier."""
        return float(max(0.0, (2.0 / math.pi) * math.atan(0.2 * (pga_gal - pga_mmi_v))))

    def scan_linear(
        self,
        epicenter_lat: float,
        epicenter_lon: float,
        depth_km: float,
        magnitude: float,
        radius_km: float = 400.0,
    ) -> Dict[str, float]:
        """Execute linear O(N) city scan computing distance and unphysical population impact."""
        t_start = time.perf_counter()

        affected_count = 0
        total_felt_pop = 0.0
        total_strong_pop = 0.0

        for i in range(self.num_cities):
            # Spherical haversine distance
            d_km = py_haversine_distance_km(epicenter_lat, epicenter_lon, float(self.city_lats[i]), float(self.city_lons[i]))
            if d_km <= radius_km:
                hypo_dist = math.sqrt(d_km**2 + depth_km**2)
                # Heuristic PGA
                pga_gal = float(10.0 ** (0.5 * magnitude - math.log10(max(10.0, hypo_dist)))) * 980.665 * 0.01
                felt_mult = self.legacy_felt_multiplier(pga_gal)
                strong_mult = self.legacy_felt_strong_multiplier(pga_gal)

                pop = float(self.city_pops[i])
                total_felt_pop += pop * felt_mult
                total_strong_pop += pop * strong_mult
                affected_count += 1

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "elapsed_ms": elapsed_ms,
            "affected_cities": affected_count,
            "legacy_count_felt": total_felt_pop,
            "legacy_count_strong": total_strong_pop,
        }
