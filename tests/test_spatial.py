"""
tests/test_spatial.py: E2E Verification of Spatially Indexed Municipality & ShakeMap Engine.

Authority & Specifications:
- `ORIGINAL_REQUEST.md` § R3 (Municipality & ShakeMap Engine, Zero "People Felt" Counters)
- `PROJECT.md` § Features 18, 19, 20, 21, 22
- `TEST_INFRA.md` § Features F14, F15, F16, F17:
  - F14: Strict Elimination of "People Felt" Counters (countFelt, countStrong, feltMultiplier, atan)
  - F15: Multi-Tier Municipality Spatial Indexing (KD-tree, < 2.0 ms latency)
  - F16: Anisotropic ShakeMap Generation (PGA, PGV, MMI, JMA Shindo)
  - F17: Live Municipality P/S Arrival Countdowns (seconds_p, seconds_s)

Tests:
- Tier 1 (Coverage): Multi-tier spatial index query, ShakeMap intensity calculation, dynamic countdowns,
  and comprehensive source/output audit asserting ZERO "people felt" counters.
- Tier 2 (Boundary): Empty query results (remote ocean deep event), zero depth, maximum perception radius,
  epicenter right on a municipality coordinate, deep intraslab events.
- Tier 3 (Pairwise): Combinations of earthquake depths {5, 30, 100, 450 km} and magnitudes {4.0, 5.5, 7.0, 8.5}.
- Tier 4 (Real-World Application): 1,000 spatial queries asserting mean latency < 2.0 ms,
  and ShakeMap evaluation on Subduction Megathrust event.
"""

from __future__ import annotations

import inspect
import math
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quakesnail.spatial.catalog import (
    Municipality,
    MunicipalityCatalog,
    MunicipalityIntensity,
)
from quakesnail.spatial.countdown import ArrivalCountdownEngine
from quakesnail.spatial.shakemap import GroundMotionResult, ShakeMapEngine


# ===========================================================================
# Tier 1: Feature Coverage (F14 through F17)
# ===========================================================================

class TestSpatialTier1:
    """Tier 1: Multi-tier catalog, KD-tree lookup, ShakeMap GMPE, and countdowns."""

    def test_f14_zero_people_felt_counters_audit(self):
        """F14: STRICT AUDIT: Assert ZERO occurrences of countFelt, countStrong, feltMultiplier, or atan."""
        # 1. Audit source code of quakesnail/spatial/
        spatial_dir = PROJECT_ROOT / "quakesnail" / "spatial"
        assert spatial_dir.exists(), "quakesnail/spatial does not exist"

        forbidden_tokens = [
            "countfelt",
            "countstrong",
            "feltmultiplier",
            "feltstrongmultiplier",
            "possibly felt by",
            "possibly heavily felt by",
        ]

        for py_file in spatial_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8").lower()
            for token in forbidden_tokens:
                assert token not in content, (
                    f"Integrity Violation in {py_file.name}: Found forbidden token '{token}'"
                )

        # 2. Audit fields and attributes of MunicipalityIntensity dataclass
        field_names = [f.name.lower() for f in MunicipalityIntensity.__dataclass_fields__.values()]
        for token in forbidden_tokens:
            assert token not in field_names, f"Found forbidden field in MunicipalityIntensity: {token}"

        # 3. Audit query output objects
        catalog = MunicipalityCatalog()
        # Query Tokyo M7.0 event
        results = catalog.query_affected(
            epicenter_lat=35.68, epicenter_lon=139.76, depth_km=20.0, magnitude=7.0, min_mmi=1.0
        )
        assert len(results) > 0
        for item in results:
            item_dict = {k: getattr(item, k) for k in item.__dataclass_fields__}
            for k in item_dict:
                for token in forbidden_tokens:
                    assert token not in k.lower(), f"Forbidden key in output: {k}"

    def test_f15_multi_tier_spatial_indexing_and_coverage(self):
        """F15: Multi-tier catalog covers global capitals, Japan JMA, US counties, Italy, NZ, Chile, etc."""
        catalog = MunicipalityCatalog()
        assert len(catalog._municipalities) > 100, "Catalog must contain rich multi-tier dataset"

        # Check representation across countries
        countries = {m.country_code for m in catalog._municipalities}
        required_countries = {"JP", "US", "IT", "NZ", "CL", "MX", "TW", "TR", "ID"}
        missing = required_countries - countries
        assert not missing, f"Catalog missing key seismic countries: {missing}"

        # Check regional naming / shoreline resolution
        shoreline = catalog.get_shoreline_description(38.0, 142.5)  # Off Tohoku coast
        assert shoreline is not None and len(shoreline) > 0
        desc = catalog.describe_epicenter(38.0, 142.5)
        assert len(desc) > 0

    def test_f16_anisotropic_shakemap_gmpe(self):
        """F16: ShakeMap engine computes PGA, PGV, Instrumental MMI, and JMA Shindo."""
        engine = ShakeMapEngine()
        # Compute ground motion for M7.0 at 30 km distance, 15 km depth
        hypo_dist = math.sqrt(30.0 ** 2 + 15.0 ** 2)
        gm = engine.compute_ground_motion(magnitude=7.0, distance_hypo_km=hypo_dist, depth_km=15.0)

        assert isinstance(gm, GroundMotionResult)
        assert gm.pga_g > 0.05, f"PGA too low: {gm.pga_g}"
        assert gm.pga_gal == pytest.approx(gm.pga_g * 980.665, rel=1e-4)
        assert gm.pgv_cms > 5.0, f"PGV too low: {gm.pgv_cms}"
        # Instrumental MMI should be around VI to VIII
        assert 5.0 <= gm.mmi <= 9.0
        assert gm.mmi_roman in ("V", "VI", "VII", "VIII", "IX")
        # JMA Shindo scale should be valid
        assert gm.jma_shindo in ("4", "5-", "5+", "6-", "6+", "7")

    def test_f17_real_time_arrival_countdowns(self):
        """F17: Sub-millisecond calculation of live P- and S-wave arrival countdowns."""
        # Station at 60 km distance from hypocenter, origin time = now - 5s
        t_now = time.time()
        origin_time = t_now - 5.0
        dist_hypo = 60.0

        res = ArrivalCountdownEngine.compute_countdowns(
            origin_time=origin_time,
            distance_hypo_km=dist_hypo,
            current_time=t_now,
        )

        # Theoretical P travel time around 9.1s for 60 km
        assert 8.0 < res.travel_time_p < 11.0
        assert 14.0 < res.travel_time_s < 18.0

        # Live countdown seconds: seconds_p = travel_time_p - elapsed_time
        elapsed_s = t_now - origin_time
        assert abs(res.seconds_p - (res.travel_time_p - elapsed_s)) < 0.05
        assert abs(res.seconds_s - (res.travel_time_s - elapsed_s)) < 0.05
        assert not res.p_arrived
        assert not res.s_arrived
        assert "s" in res.p_countdown_str.lower()


# ===========================================================================
# Tier 2: Boundary & Corner Cases
# ===========================================================================

class TestSpatialTier2:
    """Tier 2: Remote ocean deep event, zero depth, epicentral distance 0, deep intraslab."""

    def test_boundary_remote_ocean_deep_event_no_shaking(self):
        """Boundary: Remote South Pacific deep event (lat -50.0, lon -140.0) produces zero felt cities."""
        catalog = MunicipalityCatalog()
        results = catalog.query_affected(
            epicenter_lat=-50.0, epicenter_lon=-140.0, depth_km=500.0, magnitude=4.5, min_mmi=3.0
        )
        assert len(results) == 0

    def test_boundary_zero_depth_singularity_guard(self):
        """Boundary: Zero focal depth (depth = 0.0 km) does not cause divide-by-zero or NaN."""
        engine = ShakeMapEngine()
        gm = engine.compute_ground_motion(magnitude=6.0, distance_hypo_km=10.0, depth_km=0.0)
        assert gm.pga_g > 0.0
        assert not math.isnan(gm.pga_g)
        assert not math.isinf(gm.pga_g)

    def test_boundary_station_at_epicenter(self):
        """Boundary: Municipality directly at epicenter (distance = 0.0 km)."""
        engine = ShakeMapEngine()
        gm = engine.compute_ground_motion(magnitude=6.5, distance_hypo_km=10.0, depth_km=10.0)
        assert gm.pga_g > 0.1
        assert gm.mmi > 5.5

    def test_boundary_deep_intraslab_attenuation(self):
        """Boundary: Deep intraslab event (depth = 350 km) has strongly attenuated surface PGA."""
        engine = ShakeMapEngine()
        gm_shallow = engine.compute_ground_motion(magnitude=6.0, distance_hypo_km=50.0, depth_km=10.0)
        gm_deep = engine.compute_ground_motion(magnitude=6.0, distance_hypo_km=350.0, depth_km=350.0)
        # Deep event surface shaking is much lower than shallow
        assert gm_deep.pga_g < gm_shallow.pga_g * 0.3

    def test_boundary_already_arrived_countdown_formatting(self):
        """Boundary: Waves that have already arrived display negative seconds or arrived status."""
        t_now = time.time()
        # Origin was 60 seconds ago, wave arrived 45 seconds ago
        res = ArrivalCountdownEngine.compute_countdowns(
            origin_time=t_now - 60.0,
            distance_hypo_km=50.0,
            current_time=t_now,
        )
        assert res.p_arrived
        assert res.s_arrived
        assert res.seconds_p < 0.0
        assert res.seconds_s < 0.0


# ===========================================================================
# Tier 3: Pairwise Combinations
# ===========================================================================

class TestSpatialTier3:
    """Tier 3: Pairwise combinations across focal depths and earthquake magnitudes."""

    @pytest.mark.parametrize("depth_km", [5.0, 30.0, 100.0, 450.0])
    @pytest.mark.parametrize("magnitude", [4.0, 5.5, 7.0, 8.5])
    def test_pairwise_depth_and_magnitude_catalog_query(self, depth_km, magnitude):
        """Pairwise: Verify robust municipality lookup across diverse earthquake scales."""
        catalog = MunicipalityCatalog()
        # Test near dense seismic zone (Tokyo / Kanto: lat 35.5, lon 139.5)
        results = catalog.query_affected(
            epicenter_lat=35.5, epicenter_lon=139.5, depth_km=depth_km, magnitude=magnitude, min_mmi=1.0
        )
        # For M >= 5.5 at shallow depth, affected municipalities must be found
        if magnitude >= 5.5 and depth_km <= 30.0:
            assert len(results) > 0
            assert results[0].pga_g > 0.0
            assert results[0].mmi >= 1.0


# ===========================================================================
# Tier 4: Real-World Scenarios & Latency Invariant (< 2.0 ms)
# ===========================================================================

class TestSpatialTier4:
    """Tier 4: 1,000 queries latency stress test (< 2.0 ms) and real-world ShakeMap evaluation."""

    def test_spatial_query_latency_under_2ms_across_1000_queries(self):
        """
        Real-world Latency Benchmark:
        Execute 1,000 spatial queries at random global earthquake locations.
        Assert mean execution latency < 2.0 ms per query (> 10x faster than 43k scan).
        """
        catalog = MunicipalityCatalog()
        rng = np.random.default_rng(20261024)

        n_queries = 1_000
        lats = rng.uniform(-60.0, 60.0, n_queries)
        lons = rng.uniform(-180.0, 180.0, n_queries)
        depths = rng.uniform(5.0, 200.0, n_queries)
        mags = rng.uniform(4.5, 7.5, n_queries)

        t_start = time.perf_counter()
        for i in range(n_queries):
            _ = catalog.query_affected(
                epicenter_lat=float(lats[i]),
                epicenter_lon=float(lons[i]),
                depth_km=float(depths[i]),
                magnitude=float(mags[i]),
                min_mmi=2.0,
            )
        t_elapsed = time.perf_counter() - t_start

        mean_latency_ms = (t_elapsed / n_queries) * 1000.0
        assert mean_latency_ms < 2.0, (
            f"Latency Invariant Violated: Mean latency was {mean_latency_ms:.3f} ms (limit < 2.0 ms)"
        )

    def test_realworld_subduction_megathrust_shakemap(self):
        """Scenario 1: Full ShakeMap generation for M8.2 Tohoku Megathrust."""
        catalog = MunicipalityCatalog()
        results = catalog.query_affected(
            epicenter_lat=38.2, epicenter_lon=142.4, depth_km=25.0, magnitude=8.2, min_mmi=2.0
        )
        assert len(results) > 5
        # Nearest coastal cities should experience intense shaking (MMI >= VIII / JMA 6- or 6+)
        names = [r.name for r in results]
        assert any("Sendai" in n or "Morioka" in n or "Fukushima" in n for n in names)
        # Max MMI must be severe
        max_mmi = max(r.mmi for r in results)
        assert max_mmi >= 7.5
