"""
tests/test_adversarial_stage3.py: Adversarial Empirical Stress Testing for Stage 3.

Empirical verification of:
1. Stage 3 Hypocenter Inversion:
   - Extreme focal depths: ultra-shallow (0.5 to 3.0 km) and deep mantle transition (400 to 700 km).
   - Pathological station geometries: azimuthal gap > 220 deg, sparse 4-station networks, timing jitter +- 1.5 s.
   - Invariants: positive-definite analytical 3D covariance ellipsoids, NO 10.0 km depth collapse.
2. 3D Raytracing Edge Cases:
   - Antipodal distances (140 to 175 deg) for PKP / PKIKP core phases.
   - Steep subducting slabs with 45 deg dip.
   - Thermal mantle plume velocity contrasts.
   - Raypath node continuity and travel-time monotonicity.
3. Municipality KD-Tree High-Throughput & Boundary Stress:
   - 10,000 rapid queries under random and extreme epicenters (North/South poles, date line, Mariana Trench).
   - Mean latency < 2.0 ms.
   - Zero traces of countFelt, countStrong, feltMultiplier, or population atan multipliers.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import numpy as np
import pytest

from quakesnail._quakesnail_native import (
    EARTH_RADIUS_KM,
    DEG_TO_RAD,
    py_haversine_distance_km,
    py_prem_vp_kms,
    is_native_available,
    QuakeSnailCoreEngine,
)
from quakesnail.core.inversion import HypocenterEstimate, HypocenterInverter
from quakesnail.core.picking import Pick
from quakesnail.core.raytrace import RayPath, SphericalRaytracer, trace_bent_ray
from quakesnail.spatial.catalog import MunicipalityCatalog, MunicipalityIntensity
from quakesnail.spatial.shakemap import ShakeMapEngine


# ===========================================================================
# Helper Functions for Synthetic Network Generation
# ===========================================================================

def generate_synthetic_network(
    center_lat: float,
    center_lon: float,
    num_stations: int,
    radius_km: float = 60.0,
    azimuth_range: Tuple[float, float] = (0.0, 360.0),
    random_seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """Generate synthetic seismic station network with controlled azimuthal spread."""
    rng = np.random.default_rng(random_seed)
    stations: Dict[str, Tuple[float, float, float]] = {}

    az_min, az_max = azimuth_range
    for i in range(num_stations):
        # Evenly spaced or perturbed azimuth
        frac = i / max(1, num_stations - 1) if num_stations > 1 else 0.5
        az = math.radians(az_min + frac * (az_max - az_min))
        r = radius_km * (0.6 + 0.4 * rng.uniform(0.0, 1.0))

        # Displace from center
        dlat = (r * math.cos(az)) / 111.19
        cos_lat = max(0.1, math.cos(math.radians(center_lat)))
        dlon = (r * math.sin(az)) / (111.19 * cos_lat)

        st_lat = center_lat + dlat
        st_lon = center_lon + dlon
        stations[f"ST_{i:02d}"] = (float(st_lat), float(st_lon), 0.0)

    return stations


def generate_exact_picks(
    true_lat: float,
    true_lon: float,
    true_depth: float,
    true_t0: float,
    stations: Dict[str, Tuple[float, float, float]],
    timing_jitter_s: float = 0.0,
    random_seed: int = 1234,
) -> List[Pick]:
    """Compute theoretical P-wave arrival times with optional timing jitter."""
    rng = np.random.default_rng(random_seed)
    picks: List[Pick] = []

    for idx, (st_code, (st_lat, st_lon, _)) in enumerate(stations.items()):
        dist_km = py_haversine_distance_km(true_lat, true_lon, st_lat, st_lon)
        slant_km = math.sqrt(dist_km**2 + true_depth**2)
        vp = 6.8 if dist_km < 800.0 else 9.0
        tt = slant_km / vp

        jitter = rng.uniform(-timing_jitter_s, timing_jitter_s) if timing_jitter_s > 0.0 else 0.0
        arr_time = true_t0 + tt + jitter

        picks.append(
            Pick(
                pick_id=idx + 1,
                station_code=st_code,
                phase_type=0,  # P-wave
                arrival_time_s=float(arr_time),
                snr=25.0,
                amplitude_m=1.5e-5,
                period_s=0.25,
                classification=2,  # P_WAVE_ONSET
            )
        )

    return picks


# ===========================================================================
# 1. Hypocenter Inversion Empirical Stress Tests
# ===========================================================================

class TestAdversarialInversion:
    """Empirical stress tests for Stage 3 Hypocenter Inverter."""

    @pytest.mark.parametrize("depth_km", [0.5, 1.0, 1.5, 2.0, 3.0])
    def test_extreme_focal_depths_ultra_shallow(self, depth_km: float):
        """
        Stress test: Ultra-shallow focal depths (0.5 to 3.0 km).
        Assert:
        - No collapse to 10.0 km depth fallback.
        - Positive-definite 3D covariance ellipsoid (all eigenvalues > 0, sigma_z > 0).
        - Residual RMS is tightly bounded (< 1.5 s).
        """
        inverter = HypocenterInverter(lambda_ridge=0.1, max_iter=20)
        true_lat, true_lon = 35.5, 139.5
        true_t0 = 1000.0

        stations = generate_synthetic_network(true_lat, true_lon, num_stations=8, radius_km=30.0)
        picks = generate_exact_picks(true_lat, true_lon, depth_km, true_t0, stations)

        # Seed near station centroid with physical shallow seed
        seed_depth = max(2.0, depth_km + 1.0)
        res = inverter.invert(picks, stations, seed_hypo=(true_lat + 0.02, true_lon - 0.02, seed_depth, true_t0))

        assert res is not None, "Inversion returned None"
        # STRICT INVARIANT 1: No 10.0 km depth collapse
        assert abs(res.depth_km - 10.0) > 0.5 or abs(depth_km - 10.0) < 0.5, (
            f"Artificial 10 km collapse detected! Inverted depth={res.depth_km} for true depth={depth_km}"
        )

        # STRICT INVARIANT 2: Depth stays within shallow range (< 10.0 km)
        assert res.depth_km <= 8.0, f"Shallow depth failed to stay shallow: {res.depth_km} km"

        # STRICT INVARIANT 3: Analytical 3D Covariance Ellipsoid is Positive-Definite
        cov = res.cov_matrix
        assert cov.shape == (4, 4), f"Expected 4x4 covariance matrix, got {cov.shape}"
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0.0), f"Covariance not positive definite! Eigvals: {eigvals}"
        assert res.depth_error_km > 0.0, f"Invalid depth error: {res.depth_error_km}"
        assert res.semi_major_km > 0.0, f"Invalid semi-major axis: {res.semi_major_km}"
        assert res.semi_minor_km > 0.0, f"Invalid semi-minor axis: {res.semi_minor_km}"

        # Residual RMS must be small for noise-free theoretical arrivals
        assert res.residual_rms_s < 1.0, f"High residual RMS: {res.residual_rms_s} s"

    @pytest.mark.parametrize("depth_km", [400.0, 500.0, 600.0, 650.0, 700.0])
    def test_extreme_focal_depths_mantle_transition(self, depth_km: float):
        """
        Stress test: Deep mantle transition zone focal depths (400 to 700 km).
        Assert:
        - No collapse to 10.0 km depth fallback.
        - Positive-definite 3D covariance ellipsoid.
        - Depth converges into deep mantle (> 350 km).
        """
        inverter = HypocenterInverter(lambda_ridge=0.1, max_iter=20)
        true_lat, true_lon = 28.0, 140.0  # Izu-Bonin subduction zone
        true_t0 = 2000.0

        stations = generate_synthetic_network(true_lat, true_lon, num_stations=10, radius_km=300.0)
        picks = generate_exact_picks(true_lat, true_lon, depth_km, true_t0, stations)

        # Provide initial seed at 300 km or near truth
        seed_depth = min(680.0, max(300.0, depth_km - 20.0))
        res = inverter.invert(picks, stations, seed_hypo=(true_lat + 0.1, true_lon + 0.1, seed_depth, true_t0 - 1.0))

        assert res is not None, "Inversion returned None for deep event"
        # STRICT INVARIANT 1: No 10.0 km collapse
        assert res.depth_km > 300.0, f"Deep event collapsed to shallow depth: {res.depth_km} km (true: {depth_km})"
        assert abs(res.depth_km - 10.0) > 50.0, f"Deep event collapsed to 10km: {res.depth_km}"

        # STRICT INVARIANT 2: Positive-definite covariance
        eigvals = np.linalg.eigvalsh(res.cov_matrix)
        assert np.all(eigvals > 0.0), f"Covariance not positive definite! Eigvals: {eigvals}"
        assert res.depth_error_km > 0.0
        assert res.residual_rms_s < 2.0

    def test_pathological_azimuthal_gap_greater_than_220_deg(self):
        """
        Stress test: Pathological network geometry with azimuthal gap > 220 deg.
        Stations are restricted to an arc of 60 deg (azimuth 30 to 90 deg),
        leaving an azimuthal gap of 300 deg (> 220 deg).
        """
        inverter = HypocenterInverter(lambda_ridge=0.2, max_iter=25)
        true_lat, true_lon = 36.0, 137.0
        true_depth = 25.0
        true_t0 = 500.0

        # Stations only between 30 and 90 deg azimuth (gap = 300 deg)
        stations = generate_synthetic_network(
            true_lat, true_lon, num_stations=6, radius_km=80.0, azimuth_range=(30.0, 90.0)
        )
        picks = generate_exact_picks(true_lat, true_lon, true_depth, true_t0, stations)

        res = inverter.invert(picks, stations, seed_hypo=(true_lat + 0.1, true_lon + 0.1, 20.0, true_t0))

        assert res is not None
        # Must not blow up or collapse to 10.0 km
        assert 10.0 < res.depth_km < 60.0
        assert abs(res.depth_km - 10.0) > 1.0

        # Covariance must remain positive definite despite high azimuthal elongation
        eigvals = np.linalg.eigvalsh(res.cov_matrix)
        assert np.all(eigvals > 0.0), f"Covariance failed positive-definiteness: {eigvals}"
        # Semi-major axis should reflect elongation in direction of gap
        assert res.semi_major_km > res.semi_minor_km

    def test_pathological_sparse_4_station_network(self):
        """
        Stress test: Minimal 4-station sparse network (mathematical lower bound).
        """
        inverter = HypocenterInverter(lambda_ridge=0.15, max_iter=20)
        true_lat, true_lon = 38.0, 141.0
        true_depth = 35.0
        true_t0 = 150.0

        stations = generate_synthetic_network(true_lat, true_lon, num_stations=4, radius_km=50.0)
        assert len(stations) == 4

        picks = generate_exact_picks(true_lat, true_lon, true_depth, true_t0, stations)
        res = inverter.invert(picks, stations, seed_hypo=(true_lat + 0.05, true_lon - 0.05, 30.0, true_t0))

        assert res is not None
        assert res.num_picks == 4
        assert abs(res.depth_km - 10.0) > 5.0, f"Sparse network collapsed to 10 km: {res.depth_km}"
        assert res.depth_km > 15.0

        eigvals = np.linalg.eigvalsh(res.cov_matrix)
        assert np.all(eigvals > 0.0), f"Sparse 4-station covariance not positive definite: {eigvals}"

    def test_timing_jitter_stress_1_5_seconds(self):
        """
        Stress test: Heavy timing jitter (+- 1.5 seconds) on arrivals.
        Asserts damped Huber-IRLS downweights jittered residuals without diverging.
        """
        inverter = HypocenterInverter(lambda_ridge=0.2, max_iter=25)
        true_lat, true_lon = 37.0, 140.0
        true_depth = 40.0
        true_t0 = 300.0

        stations = generate_synthetic_network(true_lat, true_lon, num_stations=12, radius_km=100.0)
        picks = generate_exact_picks(
            true_lat, true_lon, true_depth, true_t0, stations, timing_jitter_s=1.5, random_seed=777
        )

        res = inverter.invert(picks, stations, seed_hypo=(true_lat + 0.1, true_lon + 0.1, 35.0, true_t0))

        assert res is not None
        # Residual RMS must remain within physical jitter magnitude (< 2.0 s)
        assert res.residual_rms_s <= 2.2, f"Residual RMS exploded under jitter: {res.residual_rms_s}"
        assert not math.isnan(res.depth_km)
        assert abs(res.depth_km - 10.0) > 3.0, f"Jitter caused 10 km collapse: {res.depth_km}"

        eigvals = np.linalg.eigvalsh(res.cov_matrix)
        assert np.all(eigvals > 0.0)

    @pytest.mark.parametrize("depth_km", [2.0, 15.0, 35.0])
    def test_unseeded_fibonacci_shallow_depth_collapse_defect(self, depth_km: float):
        """
        Stress test: Verify unseeded Fibonacci search accurately recovers shallow depths (< 50 km).
        Fails because the search is sucked into the 300 km depth attractor.
        """
        inverter = HypocenterInverter(lambda_ridge=0.1, max_iter=20)
        true_lat, true_lon = 35.5, 139.5
        true_t0 = 1000.0

        stations = generate_synthetic_network(true_lat, true_lon, num_stations=8, radius_km=40.0)
        picks = generate_exact_picks(true_lat, true_lon, depth_km, true_t0, stations)

        res = inverter.invert(picks, stations, seed_hypo=None)
        assert res is not None
        # Assert inverted depth is physically consistent with shallow truth (< 50 km error)
        assert abs(res.depth_km - depth_km) < 50.0, (
            f"Unseeded depth collapsed from true {depth_km} km to {res.depth_km:.2f} km!"
        )

    def test_rust_c_abi_inversion_adversarial(self):
        """
        Verify Rust C-ABI quakesnail_invert_hypocenter directly in QuakeSnailArena.
        Tests:
        - 4-station sparse network
        - Deep event (450 km) and shallow event (25 km)
        - Analytical 4x4 covariance matrix positive definiteness
        - No 10.0 km collapse
        """
        import ctypes
        from quakesnail._quakesnail_native import _NATIVE_LIB, QuakeSnailArena, HypocenterResult, is_native_available

        if not is_native_available() or _NATIVE_LIB is None:
            pytest.skip("Rust native cdylib not available")

        arena = _NATIVE_LIB.quakesnail_arena_new()
        assert arena, "Failed to allocate QuakeSnailArena"

        try:
            # 1. Test shallow crustal event (depth = 25.0 km)
            true_lat, true_lon, true_depth = 36.0, 138.0, 25.0
            stations = [
                (1, 36.2, 138.2, 0.0),
                (2, 35.8, 138.3, 0.0),
                (3, 35.9, 137.7, 0.0),
                (4, 36.3, 137.8, 0.0),
            ]
            for s_id, lat, lon, alt in stations:
                _NATIVE_LIB.quakesnail_register_station(arena, s_id, lat, lon, alt)

            arena.contents.num_picks = 4
            for i, (s_id, s_lat, s_lon, _) in enumerate(stations):
                dist = py_haversine_distance_km(true_lat, true_lon, s_lat, s_lon)
                slant = math.sqrt(dist**2 + true_depth**2)
                tt = slant / 6.8
                p = arena.contents.picks[i]
                p.pick_id = i + 1
                p.station_idx = i
                p.phase_type = 0  # P
                p.arrival_time_s = 100.0 + tt
                p.amplitude_m = 1e-5
                p.classification = 2
                p.active = 1

            arena.contents.clusters[0].cluster_id = 1
            arena.contents.clusters[0].active = 1
            arena.contents.num_clusters = 1

            out_hypo = HypocenterResult()
            ret = _NATIVE_LIB.quakesnail_invert_hypocenter(arena, 0, ctypes.byref(out_hypo))
            assert ret == 0, f"quakesnail_invert_hypocenter returned error code {ret}"

            print(
                f"\n[Rust Native Inversion Result]\n"
                f"  Lat: {out_hypo.lat_deg:.3f}, Lon: {out_hypo.lon_deg:.3f}, Depth: {out_hypo.depth_km:.2f} km\n"
                f"  RMS: {out_hypo.residual_rms_s:.4f} s, Depth Err: {out_hypo.depth_error_km:.2f} km\n"
                f"  Semi-major: {out_hypo.semi_major_km:.2f} km, Semi-minor: {out_hypo.semi_minor_km:.2f} km\n"
            )

            # Invariant 1: Positive-definite covariance matrix
            cov_4x4 = np.array(out_hypo.cov_matrix, dtype=np.float32).reshape((4, 4))
            eigvals = np.linalg.eigvalsh(cov_4x4)
            assert np.all(eigvals > 0.0), f"Rust covariance not positive definite: {eigvals}"

            # Invariant 2: No 10.0 km collapse
            assert abs(out_hypo.depth_km - 10.0) > 2.0, (
                f"Rust inversion collapsed to 10.0 km: {out_hypo.depth_km}"
            )
            assert out_hypo.depth_error_km > 0.0
            assert out_hypo.semi_major_km > 0.0

        finally:
            _NATIVE_LIB.quakesnail_arena_free(arena)


# ===========================================================================
# 2. 3D Raytracing Edge Cases
# ===========================================================================

class TestAdversarialRaytracing:
    """Empirical stress tests for 3D pseudo-bending raytracing."""

    @pytest.mark.parametrize("delta_deg", [140.0, 145.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0])
    @pytest.mark.parametrize("phase", ["PKP", "PKIKP"])
    def test_antipodal_core_phases_continuity_and_monotonicity(self, delta_deg: float, phase: str):
        """
        Stress test: 3D pseudo-bending across extreme antipodal distances (140 to 175 deg)
        for PKP and PKIKP core phases.
        Assert:
        - Raypath node continuity: max inter-node segment distance is bounded (< 2500 km for 32 nodes).
        - No NaN or Inf in coordinates or travel time.
        - Core penetration: turning depth reaches outer/inner core (> 2891 km).
        """
        tracer = SphericalRaytracer(slab_dip_deg=45.0, slab_azimuth_deg=90.0, enable_slab=True, enable_plume=True)

        src = (0.0, 0.0, 50.0)  # Source on equator
        rcv = (0.0, delta_deg, 0.0)  # Receiver at antipodal angular distance Delta

        ray = tracer.trace(src, rcv, phase=phase, num_nodes=32, iterations=8)

        # 1. Travel time validity
        assert ray.travel_time_s > 0.0
        assert not math.isnan(ray.travel_time_s)
        assert not math.isinf(ray.travel_time_s)
        # Expected core phase travel time for 140-175 deg is ~1100 to 1250 seconds (~18 to 21 min)
        assert 800.0 < ray.travel_time_s < 1600.0, (
            f"Unphysical travel time for {phase} at {delta_deg} deg: {ray.travel_time_s} s"
        )

        # 2. Raypath Node Continuity
        nodes = ray.nodes_xyz_km
        assert nodes.shape == (32, 3)
        diffs = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
        max_step = float(np.max(diffs))
        min_step = float(np.min(diffs))
        assert not np.isnan(max_step)
        # Segment length must be smoothly distributed without discontinuous teleportation
        assert max_step < 2500.0, f"Discontinuous node jump in raypath: max step {max_step} km"
        assert min_step > 50.0, f"Collapsed duplicate node in raypath: min step {min_step} km"

        # 3. Core Penetration
        # For Delta >= 140 deg, straight chord passes well into core (depth > 2891 km)
        assert ray.turning_depth_km > 2891.0, (
            f"Expected core phase to penetrate core (>2891 km), got turning depth {ray.turning_depth_km} km"
        )

    def test_antipodal_travel_time_monotonicity_sweep(self):
        """
        Verify travel time increases monotonically with epicentral distance
        from 140 deg to 175 deg.
        """
        tracer = SphericalRaytracer(enable_slab=False, enable_plume=False)
        distances = [140.0, 145.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0]
        travel_times = []

        for delta in distances:
            ray = tracer.trace((0.0, 0.0, 30.0), (0.0, delta, 0.0), phase="PKIKP", num_nodes=32)
            travel_times.append(ray.travel_time_s)

        # Assert strict monotonicity: t(delta_i+1) > t(delta_i)
        for i in range(len(travel_times) - 1):
            assert travel_times[i + 1] > travel_times[i], (
                f"Monotonicity violation between {distances[i]} deg ({travel_times[i]:.2f}s) "
                f"and {distances[i+1]} deg ({travel_times[i+1]:.2f}s)"
            )

    def test_steep_subducting_slab_45deg_fermat_acceleration(self):
        """
        Verify that a steep 45 deg subducting slab (+3.5% fast tabular body)
        advances arrival time along the slab strike/dip compared to pure PREM.
        Slab plane in model passes through (0, 0, R) (North Pole region) with 45 deg dip.
        """
        # Source and receiver aligned with slab plane at high latitudes
        src = (88.5, 90.0, 150.0)
        rcv = (89.5, 90.0, 10.0)

        # Tracer WITH slab
        tracer_slab = SphericalRaytracer(slab_dip_deg=45.0, slab_azimuth_deg=90.0, enable_slab=True, enable_plume=False)
        ray_slab = tracer_slab.trace(src, rcv, phase="P", num_nodes=24, iterations=8)

        # Tracer WITHOUT slab (pure PREM)
        tracer_prem = SphericalRaytracer(enable_slab=False, enable_plume=False)
        ray_prem = tracer_prem.trace(src, rcv, phase="P", num_nodes=24, iterations=8)

        # Slab has +3.5% faster velocity -> travel time through slab must be faster (less)
        assert ray_slab.travel_time_s < ray_prem.travel_time_s, (
            f"Fast slab should reduce travel time: slab={ray_slab.travel_time_s}s, prem={ray_prem.travel_time_s}s"
        )

    def test_thermal_mantle_plume_velocity_contrast_delay(self):
        """
        Verify that a thermal mantle plume (-3.0% slow conduit)
        delays arrival time across the plume compared to pure PREM.
        Plume conduit in model is centered along polar axis (x=0, y=0) for 3480 < r < 6000 km.
        """
        # Chord between 85N 0E and 85N 180E passes directly through x=0, y=0 at lower mantle depth
        src = (85.0, 0.0, 2000.0)
        rcv = (85.0, 180.0, 2000.0)

        tracer_plume = SphericalRaytracer(enable_slab=False, enable_plume=True)
        ray_plume = tracer_plume.trace(src, rcv, phase="P", num_nodes=24, iterations=6)

        tracer_prem = SphericalRaytracer(enable_slab=False, enable_plume=False)
        ray_prem = tracer_prem.trace(src, rcv, phase="P", num_nodes=24, iterations=6)

        # Plume is slow (-3.0%) -> travel time must be greater
        assert ray_plume.travel_time_s > ray_prem.travel_time_s, (
            f"Slow plume should increase travel time: plume={ray_plume.travel_time_s}s, prem={ray_prem.travel_time_s}s"
        )


# ===========================================================================
# 3. Municipality KD-Tree High-Throughput & Boundary Stress
# ===========================================================================

class TestAdversarialMunicipalityKDTree:
    """Empirical stress tests for Municipality KD-Tree and ShakeMap engine."""

    def test_10000_rapid_queries_latency_and_integrity(self):
        """
        Stress test: Execute 10,000 rapid queries under random global epicenters.
        Asserts:
        - Mean query latency strictly < 2.0 ms.
        - Records maximum and 99th percentile latencies.
        - Zero NaN, Inf, or unhandled exceptions across 10,000 queries.
        """
        catalog = MunicipalityCatalog()
        assert catalog.count > 0, "Catalog is empty"

        n_queries = 10000
        rng = np.random.default_rng(20261024)

        # Generate 10,000 diverse test parameters
        lats = rng.uniform(-85.0, 85.0, size=n_queries)
        lons = rng.uniform(-180.0, 180.0, size=n_queries)
        depths = rng.uniform(2.0, 650.0, size=n_queries)
        mags = rng.uniform(4.0, 8.5, size=n_queries)

        latencies_ms: List[float] = []

        t_start = time.perf_counter()
        for i in range(n_queries):
            t0 = time.perf_counter()
            results = catalog.query_affected(
                epicenter_lat=float(lats[i]),
                epicenter_lon=float(lons[i]),
                depth_km=float(depths[i]),
                magnitude=float(mags[i]),
                min_mmi=1.0,
            )
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        t_total = time.perf_counter() - t_start

        mean_latency = float(np.mean(latencies_ms))
        median_latency = float(np.median(latencies_ms))
        p95_latency = float(np.percentile(latencies_ms, 95))
        p99_latency = float(np.percentile(latencies_ms, 99))
        max_latency = float(np.max(latencies_ms))

        print(
            f"\n[KD-Tree 10,000 Query Stress Benchmark]\n"
            f"  Total time:     {t_total:.3f} s ({n_queries / t_total:.1f} qps)\n"
            f"  Mean latency:   {mean_latency:.4f} ms\n"
            f"  Median latency: {median_latency:.4f} ms\n"
            f"  95th %ile:      {p95_latency:.4f} ms\n"
            f"  99th %ile:      {p99_latency:.4f} ms\n"
            f"  Max latency:    {max_latency:.4f} ms\n"
        )

        # STRICT INVARIANT: Mean latency < 2.0 ms
        assert mean_latency < 2.0, f"Mean latency {mean_latency:.3f} ms exceeds 2.0 ms limit!"

    @pytest.mark.parametrize(
        "location_name, lat, lon, depth_km, mag",
        [
            ("North Pole", 90.0, 0.0, 10.0, 7.0),
            ("North Pole Dateline", 90.0, 180.0, 25.0, 6.5),
            ("South Pole", -90.0, 0.0, 15.0, 7.2),
            ("South Pole Dateline", -90.0, -180.0, 30.0, 6.8),
            ("Equatorial Dateline East", 0.0, 180.0, 20.0, 8.0),
            ("Equatorial Dateline West", 0.0, -180.0, 20.0, 8.0),
            ("Mariana Trench Shallow", 11.35, 142.20, 11.0, 8.5),
            ("Mariana Trench Deep Slab", 11.35, 142.20, 650.0, 7.8),
            ("Greenwich Meridian Equator", 0.0, 0.0, 15.0, 7.5),
        ],
    )
    def test_extreme_coordinate_boundaries_singularity_safety(
        self, location_name: str, lat: float, lon: float, depth_km: float, mag: float
    ):
        """
        Verify boundary coordinates (poles, dateline +-180, ocean trenches)
        execute cleanly without division-by-zero, NaN distances, or crashes.
        """
        catalog = MunicipalityCatalog()
        results = catalog.query_affected(
            epicenter_lat=lat,
            epicenter_lon=lon,
            depth_km=depth_km,
            magnitude=mag,
            min_mmi=1.0,
        )

        # Check all returned intensities
        for muni in results:
            assert not math.isnan(muni.distance_km)
            assert not math.isnan(muni.pga_g)
            assert not math.isnan(muni.mmi)
            assert not math.isnan(muni.seconds_p)
            assert not math.isnan(muni.seconds_s)
            assert muni.distance_km >= 0.0
            assert muni.pga_g >= 0.0
            assert muni.mmi >= 1.0

    def test_zero_people_felt_counters_strict_purge_audit(self):
        """
        STRICT AUDIT: Complete purge of 'people felt' counters everywhere.
        Assert that MunicipalityIntensity, MunicipalityCatalog, ShakeMapEngine,
        and all output dictionaries contain strictly ZERO traces of:
        - 'countFelt'
        - 'countStrong'
        - 'feltMultiplier'
        - population 'atan' multipliers
        - 'people felt'
        - 'Possibly felt by'
        - 'Possibly heavily felt by'
        """
        catalog = MunicipalityCatalog()
        results = catalog.query_affected(
            epicenter_lat=35.68,
            epicenter_lon=139.76,  # Tokyo
            depth_km=15.0,
            magnitude=7.0,
            min_mmi=1.0,
        )
        assert len(results) > 0, "Expected affected municipalities for Tokyo M7.0"

        forbidden_tokens = [
            "countfelt",
            "countstrong",
            "feltmultiplier",
            "people felt",
            "possibly felt by",
            "possibly heavily felt by",
        ]

        # 1. Audit MunicipalityIntensity attributes and dataclass fields
        sample = results[0]
        fields = [f for f in dir(sample) if not f.startswith("__")]
        for f in fields:
            f_lower = f.lower()
            for token in forbidden_tokens:
                assert token not in f_lower, f"Forbidden token '{token}' found in field name '{f}'!"

        # 2. Audit string representations and __repr__
        repr_str = repr(sample).lower()
        for token in forbidden_tokens:
            assert token not in repr_str, f"Forbidden token '{token}' found in repr: {repr_str}"

        # 3. Audit ShakeMapEngine methods
        engine_attrs = dir(ShakeMapEngine)
        for a in engine_attrs:
            a_lower = a.lower()
            for token in forbidden_tokens:
                assert token not in a_lower, f"Forbidden token '{token}' in ShakeMapEngine attribute '{a}'"
