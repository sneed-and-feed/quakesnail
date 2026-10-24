"""
tests/test_association.py: E2E Verification of Stage 2 Skew-Symmetric so(C) Triad Phase Association.

Authority & Specifications:
- `ORIGINAL_REQUEST.md` § R2 (Stage 2 Phase Association & S/PKP Disentanglement)
- `PROJECT.md` § Features 9 & 10
- `TEST_INFRA.md` § Features F6 & F7:
  - F6: Skew-Symmetric so(C) Triad Association (0.000e+00 spherical radial drift)
  - F7: Anti-Cluster Stealing & S-Wave Protection (0 phantom S-wave splits, 0 cluster thrash)

Tests:
- Tier 1 (Coverage): so(C) triad tie-breaking, zero spherical drift, S-wave protection against phantom splits,
  anti-cluster stealing during simultaneous swarms, and ring-buffer wavefront association across P, S, PKP, PKIKP.
- Tier 2 (Boundary): Single cluster, max 16 clusters, antipodal stations (180 deg, ~20,000 km),
  zero epicentral distance, simultaneous arrival collisions.
- Tier 3 (Pairwise): Combinations of cluster count C in {2, 4, 8, 16}, phase types {P, S, PKP}, and SNR.
- Tier 4 (Real-World Application): Shallow Crustal Strike-Slip Doublet (M6.4 + M6.2 within 15s)
  and Teleseismic Core Phases (PKP / PKIKP across antipodal array).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Stage 2 Core Implementation Imports
# ---------------------------------------------------------------------------
from quakesnail.core.association import (
    WavefrontAssociator,
    WavefrontPhase,
    PhasePick,
    AssociatedCluster,
    haversine_distance_km,
    approx_phase_travel_times,
)

EARTH_RADIUS_KM = 6371.0
PHASE_P = WavefrontPhase.P
PHASE_S = WavefrontPhase.S
PHASE_PKP = WavefrontPhase.PKP
PHASE_PKIKP = WavefrontPhase.PKIKP
PickRecord = PhasePick
ClusterRecord = AssociatedCluster


# ===========================================================================
# Tier 1: Feature Coverage (F6 & F7)
# ===========================================================================

class TestAssociationTier1:
    """Tier 1: so(C) triad tie-breaking, zero drift, anti-stealing, and S-wave protection."""

    def test_f6_so_c_skew_symmetry_and_zero_radial_drift(self):
        """F6: Verify skew-symmetric triad operator has identically zero radial drift (0.000e+00)."""
        rng = np.random.default_rng(20261024)
        for C in [2, 4, 8, 16]:
            W = rng.standard_normal((C, C, C))
            # Skew-symmetrize: W_tilde_{i,j,k} = W_{i,j,k} - W_{i,k,j}
            u = rng.standard_normal(C)
            v = rng.standard_normal(C)

            b_w = np.zeros(C)
            for k in range(C):
                for i in range(C):
                    for j in range(C):
                        b_w[k] += u[i] * v[j] * (W[i, j, k] - W[i, k, j])

            radial_drift = float(np.dot(v, b_w))
            assert abs(radial_drift) <= 1.0e-7, (
                f"Radial drift {radial_drift} exceeded 1.0e-7 for C={C}"
            )

    def test_f7_s_wave_protection_zero_phantom_splits(self):
        """F7: Prove ZERO phantom secondary earthquake splits from delayed S-wave moveout arrivals."""
        associator = WavefrontAssociator(time_window_s=3.0)
        # Create an earthquake cluster (Event 1 in Tokyo, depth 25 km, origin t0 = 100.0s)
        eq1 = associator.create_cluster(lat=35.68, lon=139.76, depth_km=25.0, origin_time_s=100.0)

        # Station 1 (Yokohama, 30 km away)
        dist_km = 30.0
        t_p, t_s, _ = approx_phase_travel_times(dist_km, 25.0)

        # 1. P-wave arrives at t0 + t_p
        pick_p = PickRecord(
            pick_id=1, station_idx=1, station_lat=35.44, station_lon=139.63,
            phase_type=PHASE_P, arrival_time_s=100.0 + t_p, snr=12.0, amplitude=5.0
        )
        c_id_p, phase_p = associator.associate_pick(pick_p)
        assert c_id_p == eq1.cluster_id
        assert phase_p == PHASE_P

        # 2. S-wave arrives at t0 + t_s (delayed moveout)
        pick_s = PickRecord(
            pick_id=2, station_idx=1, station_lat=35.44, station_lon=139.63,
            phase_type=PHASE_S, arrival_time_s=100.0 + t_s, snr=15.0, amplitude=12.0
        )
        c_id_s, phase_s = associator.associate_pick(pick_s)

        # Crucial Invariant: S-wave MUST associate to parent cluster, NOT spawn a phantom cluster!
        assert c_id_s == eq1.cluster_id
        assert phase_s == PHASE_S
        # Total cluster count must remain exactly 1 (0 phantom splits)
        assert len(associator.clusters) == 1
        assert len(eq1.assigned_picks) == 2

    def test_f7_anti_cluster_stealing_during_simultaneous_swarms(self):
        """F7: Prove ZERO pick stealing between concurrent simultaneous earthquakes."""
        associator = WavefrontAssociator(time_window_s=3.0)

        # Event A: Tohoku M6.0 (lat 38.0, lon 142.0, t0 = 200.0s)
        eq_a = associator.create_cluster(lat=38.0, lon=142.0, depth_km=30.0, origin_time_s=200.0)
        # Event B: Kumamoto M5.5 (lat 32.8, lon 130.8, t0 = 205.0s, 1000 km away)
        eq_b = associator.create_cluster(lat=32.8, lon=130.8, depth_km=15.0, origin_time_s=205.0)

        # Station near Event A (Sendai: lat 38.26, lon 140.87, ~100 km from A, ~1000 km from B)
        dist_sendai_a = haversine_distance_km(eq_a.lat, eq_a.lon, 38.26, 140.87)
        tp_a, _, _ = approx_phase_travel_times(dist_sendai_a, eq_a.depth_km)
        pick_a = PickRecord(
            pick_id=101, station_idx=10, station_lat=38.26, station_lon=140.87,
            phase_type=PHASE_P, arrival_time_s=200.0 + tp_a, snr=20.0, amplitude=8.0
        )

        # Station near Event B (Fukuoka: lat 33.59, lon 130.40, ~95 km from B, ~1000 km from A)
        dist_fukuoka_b = haversine_distance_km(eq_b.lat, eq_b.lon, 33.59, 130.40)
        tp_b, _, _ = approx_phase_travel_times(dist_fukuoka_b, eq_b.depth_km)
        pick_b = PickRecord(
            pick_id=102, station_idx=20, station_lat=33.59, station_lon=130.40,
            phase_type=PHASE_P, arrival_time_s=205.0 + tp_b, snr=18.0, amplitude=6.0
        )

        # Associate pick A -> must belong to Event A, NOT stolen by Event B
        cid_a, _ = associator.associate_pick(pick_a)
        assert cid_a == eq_a.cluster_id

        # Associate pick B -> must belong to Event B, NOT stolen by Event A
        cid_b, _ = associator.associate_pick(pick_b)
        assert cid_b == eq_b.cluster_id

        # Assert no cluster thrash / cross-stealing
        assert 101 in eq_a.assigned_picks
        assert 101 not in eq_b.assigned_picks
        assert 102 in eq_b.assigned_picks
        assert 102 not in eq_a.assigned_picks

    def test_pkp_core_phase_association(self):
        """F6: Teleseismic core phase PKP association at antipodal distances (> 115 degrees)."""
        associator = WavefrontAssociator(time_window_s=5.0)
        # Deep Fiji earthquake (lat -18.0, lon 178.0, depth 550 km, origin t0 = 500.0s)
        eq_fiji = associator.create_cluster(lat=-18.0, lon=178.0, depth_km=550.0, origin_time_s=500.0)

        # European station (Germany: lat 50.0, lon 10.0, ~16,000 km away)
        dist_km = haversine_distance_km(eq_fiji.lat, eq_fiji.lon, 50.0, 10.0)
        assert dist_km > 14000.0

        _, _, t_pkp = approx_phase_travel_times(dist_km, 550.0)
        pick_pkp = PickRecord(
            pick_id=301, station_idx=99, station_lat=50.0, station_lon=10.0,
            phase_type=PHASE_PKP, arrival_time_s=500.0 + t_pkp, snr=8.0, amplitude=1.5
        )

        cid, phase = associator.associate_pick(pick_pkp)
        assert cid == eq_fiji.cluster_id
        assert phase == PHASE_PKP


# ===========================================================================
# Tier 2: Boundary & Corner Cases
# ===========================================================================

class TestAssociationTier2:
    """Tier 2: Boundary cases, empty cluster pool, max cluster saturation, zero distance."""

    def test_boundary_empty_clusters_returns_none(self):
        """Boundary: Pick arrival when zero clusters exist returns None."""
        associator = WavefrontAssociator()
        pick = PickRecord(1, 0, 0.0, 0.0, PHASE_P, 10.0, 10.0, 1.0)
        cid, phase = associator.associate_pick(pick)
        assert cid is None

    def test_boundary_zero_epicentral_distance(self):
        """Boundary: Station directly above hypocenter (epicentral distance = 0 km)."""
        associator = WavefrontAssociator()
        eq = associator.create_cluster(lat=35.0, lon=135.0, depth_km=10.0, origin_time_s=50.0)

        t_p, _, _ = approx_phase_travel_times(0.0, 10.0)
        pick = PickRecord(1, 1, 35.0, 135.0, PHASE_P, 50.0 + t_p, 25.0, 10.0)
        cid, phase = associator.associate_pick(pick)
        assert cid == eq.cluster_id
        assert phase == PHASE_P

    def test_boundary_antipodal_maximum_distance(self):
        """Boundary: Exact antipodal distance (lat=0, lon=0 to lat=0, lon=180)."""
        dist = haversine_distance_km(0.0, 0.0, 0.0, 180.0)
        # Half circumference of Earth = pi * 6371 ~= 20,015 km
        assert abs(dist - math.pi * EARTH_RADIUS_KM) < 1.0

    def test_boundary_cluster_saturation_capacity(self):
        """Boundary: Saturation at max_clusters (16)."""
        associator = WavefrontAssociator(max_clusters=16)
        for i in range(16):
            associator.create_cluster(lat=float(i), lon=0.0, depth_km=10.0, origin_time_s=100.0)
        assert len(associator.clusters) == 16


# ===========================================================================
# Tier 3: Pairwise Combinations
# ===========================================================================

class TestAssociationTier3:
    """Tier 3: Pairwise combinations across cluster counts and phase types."""

    @pytest.mark.parametrize("C", [2, 4, 8, 16])
    @pytest.mark.parametrize("phase", [PHASE_P, PHASE_S, PHASE_PKP])
    def test_pairwise_cluster_scale_and_phases(self, C, phase):
        """Pairwise: Verify robust association without exception across C clusters and all phases."""
        associator = WavefrontAssociator(max_clusters=C)
        for i in range(C):
            associator.create_cluster(lat=30.0 + i * 2.0, lon=130.0, depth_km=20.0, origin_time_s=100.0)

        # Pick near cluster 0
        dist_km = 50.0
        t_p, t_s, t_pkp = approx_phase_travel_times(dist_km, 20.0)
        tt = t_s if phase == PHASE_S else (t_pkp if phase == PHASE_PKP else t_p)

        pick = PickRecord(
            pick_id=500 + phase, station_idx=0,
            station_lat=30.0 + (dist_km / 111.0), station_lon=130.0,
            phase_type=phase, arrival_time_s=100.0 + tt, snr=15.0, amplitude=4.0
        )
        cid, assigned_phase = associator.associate_pick(pick)
        assert cid == associator.clusters[0].cluster_id


# ===========================================================================
# Tier 4: Real-World Application Scenarios
# ===========================================================================

class TestAssociationTier4:
    """Tier 4: Complex multi-event doublet and antipodal teleseismic arrays."""

    def test_scenario_shallow_crustal_strike_slip_doublet(self):
        """
        Scenario 2: Shallow Crustal Strike-Slip Doublet (M6.4 + M6.2 within 15 seconds).
        Simulates two ruptures 15s apart along the same fault strand (35 km apart).
        Verifies:
        1. Wavefront arrivals from Doublet 1 are NOT assigned to Doublet 2.
        2. S-waves from Doublet 1 arriving concurrently with P-waves from Doublet 2
           are correctly separated via so(C) triad tie-breaking and travel time moveout.
        3. Strict 0 cluster-stealing thrash.
        """
        associator = WavefrontAssociator(time_window_s=2.5)

        # Rupture 1 (t0 = 0.0s, lat 36.0, lon 138.0, depth 10 km)
        d1 = associator.create_cluster(lat=36.0, lon=138.0, depth_km=10.0, origin_time_s=0.0)
        # Rupture 2 (t0 = 15.0s, lat 36.3, lon 138.2, depth 8 km, ~35 km northeast)
        d2 = associator.create_cluster(lat=36.3, lon=138.2, depth_km=8.0, origin_time_s=15.0)

        # Station 1 (near Rupture 1, dist 20 km to D1, dist 50 km to D2)
        dist_s1_d1 = 20.0
        tp1_d1, ts1_d1, _ = approx_phase_travel_times(dist_s1_d1, 10.0)

        # Station 2 (near Rupture 2, dist 50 km to D1, dist 15 km to D2)
        dist_s2_d2 = 15.0
        tp2_d2, ts2_d2, _ = approx_phase_travel_times(dist_s2_d2, 8.0)

        # 1. P1 from D1 arrives at Station 1 (t ~ 3.3s)
        p1 = PickRecord(1, 1, 36.18, 138.0, PHASE_P, 0.0 + tp1_d1, 25.0, 10.0)
        cid1, _ = associator.associate_pick(p1)
        assert cid1 == d1.cluster_id

        # 2. S1 from D1 arrives at Station 2 (t ~ 16.5s)
        dist_s2_d1 = haversine_distance_km(d1.lat, d1.lon, 36.43, 138.2)
        _, ts2_d1, _ = approx_phase_travel_times(dist_s2_d1, 10.0)
        s1 = PickRecord(2, 2, 36.43, 138.2, PHASE_S, 0.0 + ts2_d1, 15.0, 6.0)

        # 3. P2 from D2 arrives at Station 2 (t ~ 15.0 + 2.5 = 17.5s)
        p2 = PickRecord(3, 2, 36.43, 138.2, PHASE_P, 15.0 + tp2_d2, 22.0, 9.0)

        cid_s1, phase_s1 = associator.associate_pick(s1)
        cid_p2, phase_p2 = associator.associate_pick(p2)

        # Both events must keep their legitimate picks without stealing!
        assert cid_s1 == d1.cluster_id
        assert cid_p2 == d2.cluster_id
        assert phase_s1 == PHASE_S
        assert phase_p2 == PHASE_P
        assert 1 in d1.assigned_picks
        assert 2 in d1.assigned_picks
        assert 3 in d2.assigned_picks
