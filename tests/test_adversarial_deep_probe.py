"""
tests/test_adversarial_deep_probe.py: Deep Adversarial Stress Probes for QuakeSnail Iteration 2 Gate Review.
Written by challenger_r2_glitch to independently verify and probe edge cases in the seismic engine.
"""

import math
import sys
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quakesnail._quakesnail_native import (
    CASCADE_DIM,
    PHASE_P,
    PHASE_S,
    PHASE_PKP,
    py_haversine_distance_km,
    py_evaluate_so_c_triad,
    is_native_available,
    get_native_lib,
)
from quakesnail.core.picking import WaveformPicker, GateClassification, Pick, StationFilterState
from quakesnail.core.association import WavefrontAssociator, AssociatedCluster
from quakesnail.core.streaming import StreamingClient, WaveformPacket


def test_probe_1_negative_and_repetitive_adc_spikes():
    """Probe 1: Verify negative spikes (-1e6), alternating spikes, and repetitive spikes do not trigger ringing."""
    fs = 50.0
    dt = 1.0 / fs
    picker = WaveformPicker(sample_rate_hz=fs)
    st = "STA_NEG_SPIKE"
    picker.register_station(st)

    rng = np.random.default_rng(12345)
    # Warm up 500 samples
    for i in range(500):
        picker.feed_sample(st, float(rng.normal(0, 1.0)), i * dt)

    t = 500 * dt
    false_picks = []

    # 1. Negative spike -1e6
    _, pk = picker.feed_sample(st, -1e6, t)
    t += dt
    evt, pk = picker.feed_sample(st, 0.0, t)
    assert evt == GateClassification.GLINT_REJECTED, f"Negative spike return not rejected: {evt}"

    for _ in range(50):
        t += dt
        _, pk = picker.feed_sample(st, float(rng.normal(0, 1.0)), t)
        if pk:
            false_picks.append((t, pk.snr, "neg_spike_tail"))

    # 2. Alternating spikes: +1e6, return, -1e6, return
    t += dt
    picker.feed_sample(st, 1e6, t)
    t += dt
    evt1, _ = picker.feed_sample(st, 0.0, t)
    assert evt1 == GateClassification.GLINT_REJECTED

    t += dt
    picker.feed_sample(st, -1e6, t)
    t += dt
    evt2, _ = picker.feed_sample(st, 0.0, t)
    assert evt2 == GateClassification.GLINT_REJECTED

    for _ in range(50):
        t += dt
        _, pk = picker.feed_sample(st, float(rng.normal(0, 1.0)), t)
        if pk:
            false_picks.append((t, pk.snr, "alt_spike_tail"))

    # 3. Series of 20 random isolated spikes separated by 10-25 samples
    for k in range(20):
        spike_val = float(rng.choice([-1e6, 1e6, -5e5, 5e5]))
        t += dt
        picker.feed_sample(st, spike_val, t)
        t += dt
        evt, _ = picker.feed_sample(st, 0.0, t)
        assert evt == GateClassification.GLINT_REJECTED

        gap = rng.integers(10, 25)
        for _ in range(gap):
            t += dt
            _, pk = picker.feed_sample(st, float(rng.normal(0, 1.0)), t)
            if pk:
                false_picks.append((t, pk.snr, f"rep_spike_{k}_tail"))

    print(f"\n[PROBE 1] Total false picks after negative & repetitive spikes: {len(false_picks)}")
    assert len(false_picks) == 0, f"Detected {len(false_picks)} false picks after spike glitches: {false_picks}"


def test_probe_2_priming_noise_levels_and_dc_offset():
    """Probe 2: Verify priming on varying noise levels (1e-6 to 100.0) and huge initial DC offset."""
    fs = 50.0
    dt = 1.0 / fs

    for noise_std in [1e-6, 1e-4, 0.01, 1.0, 100.0]:
        picker = WaveformPicker(sample_rate_hz=fs)
        st = f"STA_NOISE_{noise_std}"
        picker.register_station(st)
        rng = np.random.default_rng(int(noise_std * 1000) + 42)

        picks = []
        for i in range(200):
            t = i * dt
            val = float(rng.normal(0, noise_std))
            _, pk = picker.feed_sample(st, val, t)
            if pk:
                picks.append((t, pk.snr))

        assert len(picks) == 0, f"Noise std {noise_std} caused {len(picks)} false triggers during startup: {picks}"

    # DC offset test: 5000 counts constant offset + 1.0 noise
    picker = WaveformPicker(sample_rate_hz=fs)
    st_dc = "STA_DC_OFFSET"
    picker.register_station(st_dc)
    rng = np.random.default_rng(999)
    dc_picks = []
    for i in range(200):
        t = i * dt
        val = 5000.0 + float(rng.normal(0, 1.0))
        _, pk = picker.feed_sample(st_dc, val, t)
        if pk:
            dc_picks.append((t, pk.snr))

    print(f"\n[PROBE 2] DC offset false picks during first 200 samples: {len(dc_picks)}")
    assert len(dc_picks) == 0, f"DC offset caused {len(dc_picks)} false triggers: {dc_picks}"


def test_probe_3_nan_inf_sanitization_and_recovery():
    """Probe 3: Inject NaNs and Infs, verify state is unsullied, and verify subsequent genuine P-wave is detected."""
    fs = 50.0
    dt = 1.0 / fs
    picker = WaveformPicker(sample_rate_hz=fs)
    st = "STA_NAN_RECOVER"
    picker.register_station(st)

    # Warm up 150 samples
    for i in range(150):
        picker.feed_sample(st, 0.0, i * dt)

    t = 150 * dt
    # Inject toxic values: NaN, +Inf, -Inf, NaN, Inf
    toxic_vals = [float("nan"), float("inf"), float("-inf"), float("nan"), float("inf")]
    for val in toxic_vals:
        t += dt
        evt, pk = picker.feed_sample(st, val, t)
        assert evt == GateClassification.NOMINAL
        assert pk is None

    # Check filter state variables
    state = picker.states[st]
    for b in range(3):
        assert math.isfinite(state.d1[b]), f"Band {b} d1 is non-finite: {state.d1[b]}"
        assert math.isfinite(state.d2[b]), f"Band {b} d2 is non-finite: {state.d2[b]}"
        assert math.isfinite(state.sta[b]), f"Band {b} sta is non-finite: {state.sta[b]}"
        assert math.isfinite(state.lta[b]), f"Band {b} lta is non-finite: {state.lta[b]}"
        assert math.isfinite(state.ratio[b]), f"Band {b} ratio is non-finite: {state.ratio[b]}"

    # Feed 10 baseline samples
    for _ in range(10):
        t += dt
        picker.feed_sample(st, 0.0, t)

    # Now inject a synthetic genuine P-wave onset (persistent impulsive wave of 20 counts)
    p_wave_picks = []
    for step in range(25):
        t += dt
        # Multi-sample step onset (P-wave)
        val = 20.0 * math.sin(2.0 * math.pi * 3.0 * step * dt)
        evt, pk = picker.feed_sample(st, val, t)
        if pk:
            p_wave_picks.append(pk)

    print(f"\n[PROBE 3] Picks detected on genuine P-wave after NaN bombardment: {len(p_wave_picks)}")
    assert len(p_wave_picks) > 0, "Picker failed to recover and detect genuine P-wave after NaN/Inf ingestion!"


def test_probe_4_swarm_jitter_and_density_resilience():
    """Probe 4: Test 8-event swarm with arrival time jitter (+/- 0.05s) across 32 stations."""
    rng = np.random.RandomState(888)
    main_lat, main_lon = 36.0, 138.0

    stations: Dict[str, Tuple[float, float, float]] = {}
    for i in range(32):
        code = f"STA{i:02d}"
        r_km = rng.uniform(20.0, 500.0)
        theta = rng.uniform(0.0, 2.0 * math.pi)
        st_lat = main_lat + (r_km * math.cos(theta)) / 111.19
        st_lon = main_lon + (r_km * math.sin(theta)) / (111.19 * math.cos(math.radians(main_lat)))
        stations[code] = (st_lat, st_lon, float(rng.uniform(10.0, 500.0)))

    # 8 rapid events
    events = [
        {"id": "EV0", "lat": main_lat, "lon": main_lon, "depth": 15.0, "t0": 0.0},
        {"id": "EV1", "lat": main_lat + 0.08, "lon": main_lon - 0.05, "depth": 12.0, "t0": 4.5},
        {"id": "EV2", "lat": main_lat - 0.06, "lon": main_lon + 0.07, "depth": 18.0, "t0": 9.2},
        {"id": "EV3", "lat": main_lat + 0.12, "lon": main_lon + 0.10, "depth": 14.0, "t0": 16.0},
        {"id": "EV4", "lat": main_lat - 0.10, "lon": main_lon - 0.08, "depth": 20.0, "t0": 22.8},
        {"id": "EV5", "lat": main_lat + 0.04, "lon": main_lon - 0.12, "depth": 11.0, "t0": 31.4},
        {"id": "EV6", "lat": main_lat - 0.03, "lon": main_lon + 0.15, "depth": 16.0, "t0": 40.0},
        {"id": "EV7", "lat": main_lat + 0.15, "lon": main_lon + 0.02, "depth": 13.0, "t0": 50.1},
    ]

    vp = 7.0
    vs = vp / 1.732
    all_picks: List[Pick] = []
    pick_counter = 1

    for ev in events:
        ev_id = ev["id"]
        ev_lat = ev["lat"]
        ev_lon = ev["lon"]
        ev_z = ev["depth"]
        ev_t0 = ev["t0"]

        for code, (st_lat, st_lon, _) in stations.items():
            d_km = py_haversine_distance_km(ev_lat, ev_lon, st_lat, st_lon)
            slant_km = math.sqrt(d_km**2 + ev_z**2)
            # Add Gaussian jitter +/- 0.05s
            jitter_p = float(rng.normal(0.0, 0.03))
            jitter_s = float(rng.normal(0.0, 0.03))
            t_p = ev_t0 + slant_km / vp + jitter_p
            t_s = ev_t0 + slant_km / vs + jitter_s

            all_picks.append(Pick(
                pick_id=f"{ev_id}_P_{code}_{pick_counter}",
                station_code=code,
                arrival_time_s=t_p,
                phase_type=PHASE_P,
                amplitude_m=1e-4,
            ))
            pick_counter += 1

            all_picks.append(Pick(
                pick_id=f"{ev_id}_S_{code}_{pick_counter}",
                station_code=code,
                arrival_time_s=t_s,
                phase_type=PHASE_S,
                amplitude_m=3e-4,
            ))
            pick_counter += 1

    all_picks.sort(key=lambda p: p.arrival_time_s)

    associator = WavefrontAssociator(velocity_p_kms=vp, vp_vs_ratio=1.732)
    for code, coords in stations.items():
        associator.register_station(code, *coords)

    for ev in events:
        cl = AssociatedCluster(
            cluster_id=f"EV_{ev['id']}",
            lat_deg=ev["lat"],
            lon_deg=ev["lon"],
            depth_km=ev["depth"],
            origin_time_s=ev["t0"],
            picks=[],
            created_at_s=ev["t0"],
        )
        associator.clusters.append(cl)

    for p in all_picks:
        associator.associate_pick(p)

    active_clusters = [c for c in associator.clusters if c.active]
    stealing_errors = 0
    total_assigned = 0
    for cl in active_clusters:
        cl_prefix = cl.cluster_id.replace("EV_", "")
        for p in cl.picks:
            total_assigned += 1
            if not p.pick_id.startswith(cl_prefix):
                stealing_errors += 1
                print(f"Stolen pick: {p.pick_id} assigned to cluster {cl.cluster_id} (pick arrival: {p.arrival_time_s})")

    accuracy = (total_assigned - stealing_errors) / total_assigned
    print(f"\n[PROBE 4] Swarm with timing jitter: {stealing_errors} stolen out of {total_assigned} assigned picks ({accuracy*100.0:.2f}%)")
    # Even under 30ms random arrival jitter across overlapping P/S wavefronts, accuracy must be >= 98%
    assert accuracy >= 0.98, f"Swarm association accuracy degraded below 98%: {accuracy*100.0:.2f}%"
    assert stealing_errors <= 10, f"Excessive cluster stealing under jitter: {stealing_errors} picks"


def test_probe_5_float32_spherical_radial_drift_1000_trials():
    """Probe 5: Exhaustive 1,000-trial stress test of float32 spherical radial drift with ill-conditioned vectors."""
    rng = np.random.default_rng(42)
    max_drift_f32 = 0.0
    C = CASCADE_DIM

    for trial in range(1000):
        # Scale vectors across dynamic ranges: 1e-4 to 1e4
        scale = float(10.0 ** rng.uniform(-4, 4))
        W = rng.standard_normal((C, C, C), dtype=np.float32) * scale
        u = rng.standard_normal(C, dtype=np.float32)
        v = rng.standard_normal(C, dtype=np.float32)
        # Normalize v to unit sphere to measure pure radial drift
        v = v / float(np.linalg.norm(v))
        u = u / float(np.linalg.norm(u))

        bw_f32 = py_evaluate_so_c_triad(W, u, v)
        # Drift is projection along v, normalized by ||W||
        norm_W = float(np.linalg.norm(W))
        if norm_W > 1e-12:
            drift = abs(float(np.dot(v, bw_f32))) / norm_W
            if drift > max_drift_f32:
                max_drift_f32 = drift

    print(f"\n[PROBE 5] Maximum normalized float32 radial drift across 1,000 trials: {max_drift_f32:.6e}")
    assert max_drift_f32 <= 1.0e-5, f"Float32 normalized radial drift exceeded 1.0e-5: {max_drift_f32:.6e}"


def test_probe_6_multi_station_streaming_heap_envelope():
    """Probe 6: Stream 500,000 samples across 8 active stations simultaneously, measuring heap growth."""
    tracemalloc.start()
    client = StreamingClient(offline_mode=True)
    num_stations = 8
    samples_per_station = 62_500  # Total 500,000 samples
    batch_size = 2_500
    n_batches = samples_per_station // batch_size

    for s_idx in range(num_stations):
        client.register_station(f"MULTI_STA_{s_idx:02d}", 35.0 + s_idx * 0.1, 139.0 + s_idx * 0.1, 100.0)

    rng = np.random.RandomState(777)
    fs = 50.0
    dt = 1.0 / fs

    for b in range(n_batches):
        for s_idx in range(num_stations):
            raw_chunk = rng.normal(0.0, 1.0, size=batch_size).astype(np.float32)
            packet = WaveformPacket(
                station_code=f"MULTI_STA_{s_idx:02d}",
                timestamp_s=b * batch_size * dt,
                samples=raw_chunk,
                sample_rate_hz=fs,
            )
            client.process_packet(packet)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    current_mb = current / (1024 * 1024)
    print(f"\n[PROBE 6] Multi-Station Streaming: 8 stations, 500k samples | Current: {current_mb:.2f} MB | Peak: {peak_mb:.2f} MB (< 35.0 MB)")
    assert peak_mb < 35.0, f"Peak heap {peak_mb:.2f} MB exceeded 35.0 MB limit!"
