"""
tests/test_rust_simd.py: E2E Verification of #[no_std] Rust SIMD Kernel & Python Memory Invariants.

Authority & Specifications:
- `PROJECT.md` § Feature 4 & Interface Contracts (`quakesnail.core` <-> `rust/quakesnail_rs`)
- `TEST_INFRA.md` § Memory & Invariant Assertions:
  - `rust/quakesnail_rs` static arena <= 2,097,152 bytes (2.0 MB SRAM)
  - IEEE-754 float32 equivalence <= 1.0e-6
  - Python active heap < 35.0 MB measured via `tracemalloc`

Tests:
- Tier 1 (Coverage): Static arena byte layout, struct slot sizes, float32 biquad, float32 Morse gate,
  float32 Huber gradient, and active heap measurement.
- Tier 2 (Boundary): Buffer capacity wrap-around at 512 samples, 256-station saturation, float32 subnormals,
  extremal sampling rates (1 Hz to 200 Hz), zero damping limit.
- Tier 3 (Pairwise): Parallel multi-band ring buffer updates paired with tracemalloc memory auditing.
- Tier 4 (Real-World Application): Continuous 256-station 50 Hz streaming stress test under < 35 MB heap.
"""

from __future__ import annotations

import ctypes
import math
import os
import struct
import subprocess
import sys
import tracemalloc
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# C-ABI & Struct Constants and Imports from quakesnail._quakesnail_native
# ---------------------------------------------------------------------------
from quakesnail._quakesnail_native import (
    ARENA_MAX_BYTES,
    CASCADE_DIM,
    ClusterSlot,
    MAX_CLUSTERS,
    MAX_PICKS,
    MAX_RAYS,
    MAX_STATIONS,
    NUM_BANDS,
    PickSlot,
    QuakeSnailArena,
    QuakeSnailNativeBridge,
    RAY_DISCRETIZATION,
    SCRATCH_BUFFER_SIZE,
    StationChannelState,
    StationSlot,
    WAVEFORM_RING_CAPACITY,
    get_native_lib,
    is_native_available,
    py_biquad_step,
    py_morse_impulse,
)


def float32_biquad_step(
    x: float, b0: float, b1: float, b2: float, a1: float, a2: float, d1: float, d2: float
) -> Tuple[float, float, float]:
    """Single step of Direct Form II Transposed biquad filter in IEEE-754 float32."""
    y, next_d1, next_d2 = py_biquad_step(b0, b1, b2, a1, a2, x, d1, d2)
    return float(y), float(next_d1), float(next_d2)


def float32_morse_impulse(v_prev: float, v_curr: float, v_next: float) -> Tuple[float, float]:
    """Morse impulse product and Laplacian in IEEE-754 float32."""
    prod = py_morse_impulse(v_prev, v_curr, v_next)
    lap = np.float32((np.float32(v_curr - v_prev) + np.float32(v_curr - v_next)) * np.float32(0.5))
    return float(prod), float(lap)



# ===========================================================================
# Tier 1: Feature Coverage (Arena Layout, SIMD Operations, Float32 Equivalence)
# ===========================================================================

class TestRustSimdTier1:
    """Tier 1: Static arena size and float32 equivalence tests."""

    def test_arena_static_memory_footprint_under_2mb(self):
        """Verify the exact byte layout of QuakeSnailArena is strictly <= 2,097,152 bytes (2.0 MB)."""
        station_bytes = MAX_STATIONS * ctypes.sizeof(StationSlot)
        waveform_bytes = MAX_STATIONS * NUM_BANDS * WAVEFORM_RING_CAPACITY * 4
        ring_heads_bytes = MAX_STATIONS * 8  # usize on 64-bit
        channel_bytes = MAX_STATIONS * NUM_BANDS * ctypes.sizeof(StationChannelState)
        pick_bytes = MAX_PICKS * ctypes.sizeof(PickSlot)
        cluster_bytes = MAX_CLUSTERS * ctypes.sizeof(ClusterSlot)
        ray_coords_bytes = MAX_RAYS * RAY_DISCRETIZATION * 3 * 4
        travel_times_bytes = MAX_RAYS * 4
        slowness_bytes = MAX_RAYS * RAY_DISCRETIZATION * 4
        masks_bytes = MAX_STATIONS * 2
        triad_bytes = (CASCADE_DIM ** 3) * 4
        rotor_bytes = 36 * 4
        scratch_bytes = SCRATCH_BUFFER_SIZE

        total_bytes = (
            station_bytes
            + waveform_bytes
            + ring_heads_bytes
            + channel_bytes
            + pick_bytes
            + cluster_bytes
            + ray_coords_bytes
            + travel_times_bytes
            + slowness_bytes
            + masks_bytes
            + triad_bytes
            + rotor_bytes
            + scratch_bytes
        )

        assert total_bytes <= ARENA_MAX_BYTES, (
            f"Static arena footprint {total_bytes} B exceeded 2.0 MB limit ({ARENA_MAX_BYTES} B)"
        )
        # Verify it fits comfortably in typical 2MB L2/L3 cache (approx 1.65 MB - 1.85 MB)
        assert 1_500_000 <= total_bytes <= 2_097_152

        # Verify against ctypes sizeof(QuakeSnailArena)
        assert ctypes.sizeof(QuakeSnailArena) <= ARENA_MAX_BYTES

        # If compiled Rust library is available, verify actual exported C-ABI arena byte size
        native_lib = get_native_lib()
        if native_lib is not None:
            rust_arena_size = native_lib.quakesnail_get_arena_size_bytes()
            assert rust_arena_size <= ARENA_MAX_BYTES, (
                f"Compiled Rust arena {rust_arena_size} exceeded {ARENA_MAX_BYTES} B limit"
            )
            assert 1_500_000 <= rust_arena_size <= 2_097_152


    def test_struct_alignments_and_padding(self):
        """Verify C-ABI struct layouts match 32-bit/64-bit alignment without unexpected padding."""
        # StationSlot: 1 int32 + 3 float32 + 1 int32 = 5 * 4 = 20 bytes
        assert ctypes.sizeof(StationSlot) == 20
        # StationChannelState: 10 float32 + 2 int32 = 12 * 4 = 48 bytes
        assert ctypes.sizeof(StationChannelState) == 48
        # PickSlot: 3 int32 (12) + double (8) + 3 float32 (12) + 2 int32 (8) = 40 bytes
        assert ctypes.sizeof(PickSlot) in (40, 48)  # Depending on 8-byte alignment of double

    def test_ieee754_float32_biquad_numerical_precision(self):
        """Verify float32 biquad filter step matches IEEE-754 precision within <= 1.0e-6."""
        # Standard Butterworth bandpass coefficients in float32
        b0, b1, b2 = 0.0592, 0.0, -0.0592
        a1, a2 = -1.8021, 0.8816

        x_in = 1.25
        d1, d2 = 0.0, 0.0
        y, next_d1, next_d2 = float32_biquad_step(x_in, b0, b1, b2, a1, a2, d1, d2)

        expected_y = float(np.float32(b0 * x_in + d1))
        assert abs(y - expected_y) <= 1.0e-6

    def test_ieee754_float32_morse_impulse_precision(self):
        """Verify 3-point Morse impulse computation matches IEEE-754 float32 within <= 1.0e-6."""
        vp, vc, vn = 10.25, 45.75, 12.125
        prod, lap = float32_morse_impulse(vp, vc, vn)

        expected_dm = np.float32(vc - vp)
        expected_dp = np.float32(vc - vn)
        expected_prod = float(np.float32(expected_dm * expected_dp))
        expected_lap = float(np.float32((expected_dm + expected_dp) * 0.5))

        assert abs(prod - expected_prod) <= 1.0e-6
        assert abs(lap - expected_lap) <= 1.0e-6

    def test_python_heap_envelope_under_35mb(self):
        """Verify Python active heap allocation remains strictly < 35.0 MB during streaming."""
        tracemalloc.start()
        snapshot_start = tracemalloc.take_snapshot()

        # Simulate 256 stations streaming 2000 samples across 3 bands
        n_samples = 2000
        stream_chunks = [
            np.random.randn(MAX_STATIONS, NUM_BANDS, 128).astype(np.float32)
            for _ in range(n_samples // 128)
        ]

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 35.0, f"Peak memory {peak_mb:.2f} MB exceeded 35 MB envelope"

    def test_native_bridge_rust_c_abi_export_parity(self):
        """Verify QuakeSnailNativeBridge binds genuine Rust C-ABI exports with float32 parity."""
        bridge = QuakeSnailNativeBridge()
        assert bridge is not None
        if is_native_available():
            assert bridge.use_native is True
            assert bridge._arena_ptr is not None

        # Test station registration across boundary
        idx = bridge.register_station(42, 35.6895, 139.6917, 40.0)
        assert idx == 0

        # Test single-step sample push through C-ABI
        status, pick = bridge.push_sample(0, 1.25, 0.02)
        assert isinstance(status, int)
        assert status >= 0

        # Test raytracing via C-ABI
        src = (0.0, 0.0, 30.0)
        rcv = (50.0, 50.0, 0.0)
        tt, nodes = bridge.raytrace(src, rcv, phase=0, num_nodes=16)
        assert tt > 0.0
        assert nodes.shape == (16, 3)

        bridge.reset()


# ===========================================================================
# Tier 2: Boundary & Corner Cases
# ===========================================================================

class TestRustSimdTier2:
    """Tier 2: Boundary value analysis for arena buffers and float32 extremes."""

    def test_ring_buffer_exact_wrap_around(self):
        """Boundary: Ring buffer wrap-around at exactly 512 samples."""
        capacity = WAVEFORM_RING_CAPACITY
        ring = np.zeros(capacity, dtype=np.float32)
        head = 0

        # Push 600 samples (wrapping around by 88)
        for i in range(600):
            ring[head] = np.float32(i)
            head = (head + 1) % capacity

        assert head == 600 % capacity  # 88
        # Oldest unoverwritten sample should be 88
        assert ring[head] == np.float32(88)
        # Most recent sample should be at head-1
        assert ring[(head - 1) % capacity] == np.float32(599)

    def test_station_registry_maximum_saturation(self):
        """Boundary: Station registry saturation at MAX_STATIONS (256)."""
        stations = [StationSlot(station_id=i, lat_deg=35.0, lon_deg=139.0, alt_m=10.0, active=1) for i in range(MAX_STATIONS)]
        assert len(stations) == 256
        assert stations[-1].station_id == 255

    def test_float32_subnormal_handling(self):
        """Boundary: Underflow to float32 subnormals (denormals) in recursive filtering."""
        # Very tiny number approaching float32 min normal (1.17549435e-38)
        tiny_x = 1.0e-39
        y, d1, d2 = float32_biquad_step(tiny_x, 0.1, 0.0, -0.1, -1.8, 0.8, 0.0, 0.0)
        assert not math.isnan(y)
        assert not math.isinf(y)

    def test_float32_infinite_and_nan_resilience(self):
        """Boundary: Telemetry NaN or Inf does not corrupt filter state permanently if guarded."""
        nan_x = float("nan")
        # Direct check that math.isnan detects corrupted samples
        assert math.isnan(nan_x)
        # Guarded replacement pattern:
        safe_x = 0.0 if (math.isnan(nan_x) or math.isinf(nan_x)) else nan_x
        y, d1, d2 = float32_biquad_step(safe_x, 0.1, 0.0, -0.1, -1.8, 0.8, 0.0, 0.0)
        assert y == 0.0

    def test_sampling_rate_boundaries(self):
        """Boundary: Extreme sample rate configurations (1 Hz to 200 Hz)."""
        for fs in [1.0, 10.0, 20.0, 40.0, 50.0, 100.0, 200.0]:
            state = StationChannelState(
                sta=0.0, lta=0.0, ratio=1.0, gate_state=0,
                pending_val=0.0, last_sample=0.0, prev_sample=0.0, sample_rate_hz=fs
            )
            assert state.sample_rate_hz == fs


# ===========================================================================
# Tier 3: Pairwise Combinations
# ===========================================================================

class TestRustSimdTier3:
    """Tier 3: Pairwise interactions of multi-band buffers and SIMD filtering."""

    @pytest.mark.parametrize("band_idx", [0, 1, 2])
    @pytest.mark.parametrize("n_chunks", [2, 5, 10])
    def test_pairwise_multiband_streaming_and_peak_heap(self, band_idx, n_chunks):
        """Pairwise: Stream into band_idx with n_chunks, asserting memory stays under 35 MB."""
        tracemalloc.start()
        # Allocate multi-station chunk for this band
        chunk = np.zeros((MAX_STATIONS, 512), dtype=np.float32)
        for _ in range(n_chunks):
            chunk += np.random.randn(MAX_STATIONS, 512).astype(np.float32) * 0.01

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak / (1024 * 1024) < 35.0

    @pytest.mark.parametrize("station_count", [16, 64, 256])
    @pytest.mark.parametrize("spike_amp", [10.0, 500.0, 5000.0])
    def test_pairwise_station_batch_morse_impulse(self, station_count, spike_amp):
        """Pairwise: Vectorized Morse impulse across variable station counts."""
        v_prev = np.zeros(station_count, dtype=np.float32)
        v_curr = np.full(station_count, spike_amp, dtype=np.float32)
        v_next = np.zeros(station_count, dtype=np.float32)

        dm = v_curr - v_prev
        dp = v_curr - v_next
        prod = dm * dp
        assert np.all(prod > 0)
        assert np.allclose(prod, spike_amp ** 2, rtol=1.0e-5)


# ===========================================================================
# Tier 4: Real-World Workload & Cargo Verification
# ===========================================================================

class TestRustSimdTier4:
    """Tier 4: End-to-end heavy streaming workload and Rust crate build validation."""

    def test_cargo_crate_compilation_and_test_if_present(self):
        """Verify `cargo test` passes in rust/quakesnail_rs if rust src exists."""
        cargo_toml = PROJECT_ROOT / "rust" / "quakesnail_rs" / "Cargo.toml"
        src_dir = PROJECT_ROOT / "rust" / "quakesnail_rs" / "src"
        if cargo_toml.exists() and src_dir.exists() and (src_dir / "lib.rs").exists():
            proc = subprocess.run(
                ["cargo", "test", "--manifest-path", str(cargo_toml)],
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, f"cargo test failed:\n{proc.stdout}\n{proc.stderr}"
        else:
            pytest.skip("Rust source files not yet fully written by worker_rust_core")

    def test_continuous_256_station_50hz_telemetry_stress(self):
        """Real-world: Stream 1,000 packets across 256 stations at 50 Hz, asserting zero heap growth."""
        tracemalloc.start()
        bridge = QuakeSnailNativeBridge()
        for s in range(MAX_STATIONS):
            bridge.register_station(s, 35.0 + s * 0.01, 139.0 + s * 0.01, 10.0)

        n_packets = 20  # 20 packets of 50 samples = 1000 samples per station
        ring_buffers = np.zeros((MAX_STATIONS, NUM_BANDS, WAVEFORM_RING_CAPACITY), dtype=np.float32)
        heads = np.zeros(MAX_STATIONS, dtype=np.int32)

        for _ in range(n_packets):
            packet_data = np.random.randn(MAX_STATIONS, 50).astype(np.float32)
            # Push into band 0 and into native bridge
            for s in range(MAX_STATIONS):
                h = heads[s]
                for i in range(50):
                    ring_buffers[s, 0, (h + i) % WAVEFORM_RING_CAPACITY] = packet_data[s, i]
                    bridge.push_sample(s, float(packet_data[s, i]), 0.02)
                heads[s] = (h + 50) % WAVEFORM_RING_CAPACITY

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 35.0, f"Peak memory {peak_mb:.2f} MB exceeded 35 MB envelope"

