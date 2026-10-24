"""
tests/test_adversarial_challenger.py: Empirical Adversarial Stress Test Suite for QuakeSnail.

Adversarial Stress Verification for Stage 1 Picker and Stage 2 Associator:
- Challenge 1: Glitch Torture Test (50k-100k Spikes, ADC Saturation 10^6, Steps, Drops, NaN/Inf)
- Challenge 2: Multi-Earthquake Swarm & Overlapping S-Wave Moveout (Mw 7.2 + 5 Aftershocks)
- Challenge 3: Continuous 1,000,000-Sample Streaming Leak Test under tracemalloc (< 35 MB)
- Challenge 4: Float32 Spherical Radial Drift vs Mathematical Theorem Invariant (0.000e+00)
"""

from __future__ import annotations

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
)
from quakesnail.core.picking import WaveformPicker, GateClassification, Pick
from quakesnail.core.association import WavefrontAssociator, AssociatedCluster
from quakesnail.core.streaming import StreamingClient, ReplayFeed, WaveformPacket


# ===========================================================================
# 1. Glitch Torture Stress Tests
# ===========================================================================

class TestChallengerGlitchTorture:
    """Empirical torture testing of Stage 1 Picker against extreme telemetry corruption."""

    def test_challenge_1a_biquad_ringing_avalanche_on_adc_saturation_spike(self):
        """
        Challenge 1a: Single-sample ADC saturation spike (10^6 counts) injected into warm picker.
        Empirical observation:
        The 1-sample causal deferral correctly marks sample t+1 as GLINT_REJECTED.
        HOWEVER, because the raw 10^6 spike was pushed directly into the recursive IIR biquad
        filter prior to causal classification, the biquad states (d1, d2) ring with huge amplitude (>10^5).
        This keeps STA/LTA ratio > 16.0 for multiple samples, producing multiple FALSE P_WAVE_ONSET picks.
        """
        fs = 50.0
        dt = 1.0 / fs
        picker = WaveformPicker(sample_rate_hz=fs)
        station_code = "STA_RING"
        picker.register_station(station_code)

        # Warm up filter with 1000 samples of calm noise (noise std = 1.0)
        rng = np.random.default_rng(42)
        for i in range(1000):
            picker.feed_sample(station_code, float(rng.normal(0, 1.0)), i * dt)

        picks_before = len(picker.picks)

        # Inject single-sample ADC saturation spike (10^6) at t=20.0s
        t_spike = 1000 * dt
        evt_spike, pk_spike = picker.feed_sample(station_code, 1e6, t_spike)

        # Return to baseline at t=20.02s
        evt_ret, pk_ret = picker.feed_sample(station_code, 0.0, t_spike + dt)
        assert evt_ret == GateClassification.GLINT_REJECTED, "Spike return was not marked as GLINT_REJECTED"

        # Observe subsequent 50 samples of calm noise
        false_picks_on_ringing = []
        for i in range(2, 52):
            t_curr = t_spike + i * dt
            evt, pk = picker.feed_sample(station_code, float(rng.normal(0, 1.0)), t_curr)
            if pk:
                false_picks_on_ringing.append((t_curr, pk.snr, pk.amplitude_m))

        print(f"\n[CHALLENGE 1a] False triggers caused by biquad ringing after single 10^6 spike: {len(false_picks_on_ringing)}")
        for t_p, snr, amp in false_picks_on_ringing:
            print(f"  False pick at t={t_p:.2f}s | SNR={snr:.2f} | Amp={amp:.1f}")

        # Strict requirement: Exactly 0 false earthquake triggers from isolated spikes!
        assert len(false_picks_on_ringing) == 0, (
            f"VULNERABILITY CONFIRMED: Single 10^6 ADC spike produced {len(false_picks_on_ringing)} "
            f"false triggers due to unshielded biquad ringing!"
        )

    def test_challenge_1b_premature_priming_false_picks_on_calm_noise(self):
        """
        Challenge 1b: Verify whether WaveformPicker produces false picks on calm noise during startup.
        Empirical observation:
        is_primed is set to True after only 20 samples (0.4s). Because LTA has a 20s time constant
        (1000 samples) and starts at 0, LTA has only integrated 2% of steady state, so STA/LTA ratio
        is artificially inflated to 5-7x (> 3.5 threshold), emitting multiple false picks on calm noise.
        """
        fs = 50.0
        dt = 1.0 / fs
        picker = WaveformPicker(sample_rate_hz=fs)
        station_code = "STA_CALM"
        picker.register_station(station_code)

        rng = np.random.default_rng(20261024)
        startup_picks = []
        for i in range(200):  # First 4 seconds
            t = i * dt
            val = float(rng.normal(0, 0.01))
            evt, pk = picker.feed_sample(station_code, val, t)
            if pk:
                startup_picks.append((t, pk.snr))

        print(f"\n[CHALLENGE 1b] Startup false picks on calm noise (first 4s): {len(startup_picks)}")
        for t, snr in startup_picks[:5]:
            print(f"  Startup false pick at t={t:.2f}s | SNR={snr:.2f}")

        assert len(startup_picks) == 0, (
            f"VULNERABILITY CONFIRMED: Premature priming (impulse_counter >= 20) caused "
            f"{len(startup_picks)} false triggers on calm noise!"
        )

    def test_challenge_1c_nan_inf_state_poisoning(self):
        """
        Challenge 1c: Feeding NaN or Inf floats permanently corrupts StationFilterState.
        Empirical observation:
        Neither picking.py nor filter.rs validates or sanitizes incoming floats against NaN/Inf.
        A single NaN sample permanently poisons d1, d2, sta, lta with NaN, killing the station forever.
        """
        fs = 50.0
        dt = 1.0 / fs
        picker = WaveformPicker(sample_rate_hz=fs)
        station_code = "STA_POISON"
        picker.register_station(station_code)

        # Warm up
        for i in range(100):
            picker.feed_sample(station_code, 0.0, i * dt)

        # Inject single NaN sample
        t_nan = 100 * dt
        picker.feed_sample(station_code, float("nan"), t_nan)

        # Feed 50 normal samples
        for i in range(1, 51):
            picker.feed_sample(station_code, 1.0, t_nan + i * dt)

        state = picker.states[station_code]
        is_poisoned = math.isnan(state.d1[0]) or math.isnan(state.sta[0]) or math.isnan(state.ratio[0])

        print(f"\n[CHALLENGE 1c] State after NaN injection: d1={state.d1[0]}, sta={state.sta[0]}, ratio={state.ratio[0]}")
        assert not is_poisoned, (
            "VULNERABILITY CONFIRMED: StationFilterState permanently poisoned by single NaN sample!"
        )

    def test_challenge_1d_rectangular_steps_and_dropouts(self):
        """
        Challenge 1d: Rectangular step telemetry glitches (+/- 500 counts) and dropouts.
        Evaluates resilience against step transitions without infinite ringing.
        """
        fs = 50.0
        dt = 1.0 / fs
        picker = WaveformPicker(sample_rate_hz=fs)
        station_code = "STA_STEP"
        picker.register_station(station_code)

        rng = np.random.default_rng(42)
        # Warm up
        for i in range(500):
            picker.feed_sample(station_code, float(rng.normal(0, 0.1)), i * dt)

        step_picks = 0
        t = 500 * dt
        baseline = 0.0
        for _ in range(200):
            baseline += rng.choice([-200.0, 200.0])
            for _ in range(50):
                val = baseline + float(rng.normal(0, 0.1))
                _, pk = picker.feed_sample(station_code, val, t)
                t += dt
                if pk:
                    step_picks += 1

        print(f"\n[CHALLENGE 1d] Picks during 200 rectangular steps: {step_picks}")
        # Steps generate transient step edges; assert finite response
        assert step_picks < 300, f"Excessive triggers on rectangular steps: {step_picks}"


# ===========================================================================
# 2. Multi-Earthquake Swarm & Overlapping S-Wave Moveout Test
# ===========================================================================

class TestChallengerSwarmAndSWaveMoveout:
    """Empirical torture testing of Stage 2 Wavefront Associator."""

    def test_challenge_2a_swarm_cluster_stealing_and_misassociation(self):
        """
        Challenge 2a: Multi-earthquake swarm (Mw 7.2 mainshock + 5 rapid aftershocks within 60s).
        Empirical observation:
        tie_break_margin in WavefrontAssociator is hardcoded as:
            triad_tensor[0, c1, c2] - triad_tensor[0, c2, c1] = 0.001 * (c1 - c2)
        This is an arbitrary index subtraction that has zero physical connection to arrival residuals.
        Consequently, 60 out of 384 picks are stolen by incorrect clusters.
        """
        rng = np.random.RandomState(42)
        main_lat, main_lon = 35.5, 139.0

        stations: Dict[str, Tuple[float, float, float]] = {}
        for i in range(32):
            code = f"STA{i:02d}"
            r_km = rng.uniform(20.0, 600.0)
            theta = rng.uniform(0.0, 2.0 * math.pi)
            st_lat = main_lat + (r_km * math.cos(theta)) / 111.19
            st_lon = main_lon + (r_km * math.sin(theta)) / (111.19 * math.cos(math.radians(main_lat)))
            stations[code] = (st_lat, st_lon, float(rng.uniform(10.0, 500.0)))

        events = [
            {"id": "MAIN", "lat": main_lat, "lon": main_lon, "depth": 18.0, "t0": 0.0, "mw": 7.2},
            {"id": "AFT1", "lat": main_lat + 0.12, "lon": main_lon - 0.08, "depth": 15.0, "t0": 11.2, "mw": 5.4},
            {"id": "AFT2", "lat": main_lat - 0.15, "lon": main_lon + 0.10, "depth": 22.0, "t0": 23.5, "mw": 4.8},
            {"id": "AFT3", "lat": main_lat + 0.05, "lon": main_lon + 0.18, "depth": 12.0, "t0": 34.0, "mw": 5.8},
            {"id": "AFT4", "lat": main_lat - 0.08, "lon": main_lon - 0.12, "depth": 16.0, "t0": 45.1, "mw": 4.5},
            {"id": "AFT5", "lat": main_lat + 0.20, "lon": main_lon + 0.02, "depth": 14.0, "t0": 58.7, "mw": 5.1},
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
                t_p = ev_t0 + slant_km / vp
                t_s = ev_t0 + slant_km / vs

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
        for cl in active_clusters:
            cl_prefix = cl.cluster_id.replace("EV_", "")
            for p in cl.picks:
                if not p.pick_id.startswith(cl_prefix):
                    stealing_errors += 1

        print(f"\n[CHALLENGE 2a] Cluster stealing errors in 6-event swarm: {stealing_errors} / {len(all_picks)} picks")
        assert stealing_errors == 0, (
            f"VULNERABILITY CONFIRMED: Cluster stealing occurred on {stealing_errors} picks!"
        )

    def test_challenge_2b_float32_spherical_radial_drift_precision(self):
        """
        Challenge 2b: Measure actual numerical spherical radial drift under IEEE-754 float32.
        Theorem 2a claims 0.000e+00 spherical radial drift: <v, B_W(u, v)> = 0.
        In double precision (float64), residual is <= 1.0e-15.
        However, in float32 (as used in quakesnail_rs and _quakesnail_native.py),
        roundoff error produces drift of 4.8e-06.
        """
        rng = np.random.default_rng(20261024)
        max_drift_f32 = 0.0
        max_drift_f64 = 0.0

        C = CASCADE_DIM
        for _ in range(50):
            W = rng.standard_normal((C, C, C), dtype=np.float32)
            u = rng.standard_normal(C, dtype=np.float32)
            v = rng.standard_normal(C, dtype=np.float32)

            bw_f32 = py_evaluate_so_c_triad(W, u, v)
            drift_f32 = abs(float(np.dot(v, bw_f32)))
            max_drift_f32 = max(max_drift_f32, drift_f32)

            # Float64 exact computation
            W64 = W.astype(np.float64)
            u64 = u.astype(np.float64)
            v64 = v.astype(np.float64)
            bw_f64 = np.zeros(C, dtype=np.float64)
            for k in range(C):
                for i in range(C):
                    for j in range(C):
                        bw_f64[k] += u64[i] * v64[j] * (W64[i, j, k] - W64[i, k, j])
            drift_f64 = abs(float(np.dot(v64, bw_f64)))
            max_drift_f64 = max(max_drift_f64, drift_f64)

        print(f"\n[CHALLENGE 2b] Max radial drift: float32 = {max_drift_f32:.6e} | float64 = {max_drift_f64:.6e}")
        # Document discrepancy between claimed 0.000e+00 and float32 reality
        assert max_drift_f64 <= 1.0e-12, "Float64 theoretical drift violated"
        assert max_drift_f32 <= 1.0e-5, f"Float32 drift exceeded 1.0e-5: {max_drift_f32}"


# ===========================================================================
# 3. 1,000,000-Sample Streaming Leak Test under tracemalloc
# ===========================================================================

class TestChallengerStreamingMemoryLeak:
    """Empirical memory leak testing under continuous 1,000,000-sample streaming."""

    def test_challenge_3_streaming_client_1m_samples_tracemalloc_under_35mb(self):
        """
        Challenge 3: Stream 1,000,000 continuous samples through StreamingClient and ReplayFeed.
        Requirement: Active heap strictly remains < 35.0 MB.
        Result: StreamingClient uses circular buffers and strictly respects < 35.0 MB (measured ~0.76 MB).
        """
        tracemalloc.start()
        client = StreamingClient(offline_mode=True)
        station_code = "STREAM_1M"
        client.register_station(station_code, 35.0, 139.0, 100.0)

        total_samples = 1_000_000
        batch_size = 25_000
        n_batches = total_samples // batch_size
        rng = np.random.RandomState(42)

        t_start = 0.0
        fs = 50.0
        dt = 1.0 / fs

        for b in range(n_batches):
            raw_chunk = rng.normal(0.0, 1.0, size=batch_size).astype(np.float32)
            packet = WaveformPacket(
                station_code=station_code,
                timestamp_s=t_start + b * batch_size * dt,
                samples=raw_chunk,
                sample_rate_hz=fs,
            )
            client.process_packet(packet)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        current_mb = current / (1024 * 1024)
        print(f"\n[CHALLENGE 3 FINAL] Total samples: {total_samples:,} | Current Heap: {current_mb:.2f} MB | Peak Heap: {peak_mb:.2f} MB (< 35.0 MB)")

        assert peak_mb < 35.0, f"Peak memory {peak_mb:.2f} MB exceeded 35.0 MB limit!"
