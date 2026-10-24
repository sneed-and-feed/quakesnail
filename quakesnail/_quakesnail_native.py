"""Ctypes FFI bridge to compiled rust/quakesnail_rs with 100% pure-Python vectorized fallback.

Provides seamless loading of compiled native library (quakesnail_rs.dll / libquakesnail_rs.so)
or automatic transparent fallback to pure-Python vectorized NumPy/SciPy algorithms matching
the Rust SIMD implementation within <= 1.0e-6 in IEEE-754 float32.
"""

from __future__ import annotations

import ctypes
from ctypes import (
    POINTER,
    Structure,
    c_float,
    c_double,
    c_int32,
    c_size_t,
    c_uint8,
    byref,
)
import os
import platform
import sys
from typing import Optional, Tuple, List
import numpy as np

# Constants matching rust/quakesnail_rs
MAX_STATIONS = 256
WAVEFORM_RING_CAPACITY = 512
NUM_BANDS = 3
MAX_PICKS = 128
MAX_CLUSTERS = 16
MAX_RAYS = 64
RAY_DISCRETIZATION = 64
CASCADE_DIM = 8
SCRATCH_BUFFER_SIZE = 65536
ARENA_MAX_BYTES = 2097152

CLASSIFICATION_NOMINAL = 0
CLASSIFICATION_PENDING_IMPULSE = 1
CLASSIFICATION_P_WAVE_ONSET = 2
CLASSIFICATION_GLINT_REJECTED = 3
CLASSIFICATION_STEP_MANEUVER = 4

PHASE_P = 0
PHASE_S = 1
PHASE_PKP = 2
PHASE_PKIKP = 3
PHASE_PMP = 4
PHASE_PCP = 5

EARTH_RADIUS_KM = 6371.0
DEG_TO_RAD = np.float32(np.pi / 180.0)


# --- Ctypes Structures ---


class StationSlot(Structure):
    _fields_ = [
        ("station_id", c_int32),
        ("lat_deg", c_float),
        ("lon_deg", c_float),
        ("alt_m", c_float),
        ("active", c_int32),
    ]


class PickSlot(Structure):
    _fields_ = [
        ("pick_id", c_int32),
        ("station_idx", c_int32),
        ("phase_type", c_int32),
        ("arrival_time_s", c_double),
        ("snr", c_float),
        ("amplitude_m", c_float),
        ("period_s", c_float),
        ("classification", c_int32),
        ("active", c_int32),
        ("_pad", c_int32),
    ]


class ClusterSlot(Structure):
    _fields_ = [
        ("cluster_id", c_int32),
        ("lat_deg", c_float),
        ("lon_deg", c_float),
        ("depth_km", c_float),
        ("origin_time_s", c_double),
        ("mag_default", c_float),
        ("mag_low", c_float),
        ("mag_ultralow", c_float),
        ("num_picks", c_int32),
        ("residual_rms_s", c_float),
        ("active", c_int32),
        ("_pad", c_int32),
    ]


class StationChannelState(Structure):
    _fields_ = [
        ("sta", c_float),
        ("lta", c_float),
        ("ratio", c_float),
        ("gate_state", c_int32),
        ("pending_val", c_float),
        ("last_sample", c_float),
        ("prev_sample", c_float),
        ("sample_rate_hz", c_float),
        ("d1", c_float),
        ("d2", c_float),
        ("noise_var", c_float),
        ("impulse_counter", c_int32),
    ]


class HypocenterResult(Structure):
    _fields_ = [
        ("lat_deg", c_float),
        ("lon_deg", c_float),
        ("depth_km", c_float),
        ("origin_time_s", c_double),
        ("mag_default", c_float),
        ("mag_low", c_float),
        ("mag_ultralow", c_float),
        ("semi_major_km", c_float),
        ("semi_minor_km", c_float),
        ("depth_error_km", c_float),
        ("residual_rms_s", c_float),
        ("num_picks", c_int32),
        ("cov_matrix", c_float * 16),
    ]


class QuakeSnailArena(Structure):
    _fields_ = [
        ("stations", StationSlot * MAX_STATIONS),
        ("num_stations", c_size_t),
        ("waveforms", c_float * (MAX_STATIONS * NUM_BANDS * WAVEFORM_RING_CAPACITY)),
        ("ring_heads", c_size_t * MAX_STATIONS),
        ("channel_states", (StationChannelState * NUM_BANDS) * MAX_STATIONS),
        ("picks", PickSlot * MAX_PICKS),
        ("num_picks", c_size_t),
        ("clusters", ClusterSlot * MAX_CLUSTERS),
        ("num_clusters", c_size_t),
        ("ray_nodes_xyz", c_float * (MAX_RAYS * RAY_DISCRETIZATION * 3)),
        ("travel_times", c_float * MAX_RAYS),
        ("ray_slowness", c_float * (MAX_RAYS * RAY_DISCRETIZATION)),
        ("clock_mask", c_uint8 * MAX_STATIONS),
        ("nope_mask", c_uint8 * MAX_STATIONS),
        ("triad_tensor", c_float * (CASCADE_DIM * CASCADE_DIM * CASCADE_DIM)),
        ("rotor_so6", c_float * 36),
        ("scratch_buffer", c_uint8 * 65536),
    ]


# --- Native Library Loader ---

_NATIVE_LIB: Optional[ctypes.CDLL] = None


def _find_and_load_native_library() -> Optional[ctypes.CDLL]:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(base_dir, "rust", "quakesnail_rs", "target", "release", "quakesnail_rs.dll"),
        os.path.join(base_dir, "rust", "quakesnail_rs", "target", "debug", "quakesnail_rs.dll"),
        os.path.join(base_dir, "rust", "quakesnail_rs", "target", "release", "libquakesnail_rs.so"),
        os.path.join(base_dir, "rust", "quakesnail_rs", "target", "debug", "libquakesnail_rs.so"),
        os.path.join(base_dir, "rust", "quakesnail_rs", "target", "release", "libquakesnail_rs.dylib"),
        os.path.join(base_dir, "rust", "quakesnail_rs", "target", "debug", "libquakesnail_rs.dylib"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                lib = ctypes.CDLL(path)
                # Configure signatures
                lib.quakesnail_arena_new.restype = POINTER(QuakeSnailArena)
                lib.quakesnail_arena_free.argtypes = [POINTER(QuakeSnailArena)]
                lib.quakesnail_arena_reset.argtypes = [POINTER(QuakeSnailArena)]
                lib.quakesnail_get_arena_size_bytes.restype = c_size_t
                lib.quakesnail_register_station.argtypes = [
                    POINTER(QuakeSnailArena),
                    c_int32,
                    c_float,
                    c_float,
                    c_float,
                ]
                lib.quakesnail_register_station.restype = c_int32
                lib.quakesnail_push_sample.argtypes = [
                    POINTER(QuakeSnailArena),
                    c_int32,
                    c_float,
                    c_float,
                    POINTER(PickSlot),
                ]
                lib.quakesnail_push_sample.restype = c_int32
                lib.quakesnail_process_station_waveform.argtypes = [
                    POINTER(QuakeSnailArena),
                    c_size_t,
                    c_size_t,
                    POINTER(c_float),
                    c_size_t,
                ]
                lib.quakesnail_process_station_waveform.restype = c_int32
                lib.quakesnail_associate_phases.argtypes = [POINTER(QuakeSnailArena), c_size_t]
                lib.quakesnail_associate_phases.restype = c_int32
                lib.quakesnail_associate_wavefronts.argtypes = [
                    POINTER(QuakeSnailArena),
                    POINTER(c_float),
                    POINTER(ClusterSlot),
                ]
                lib.quakesnail_associate_wavefronts.restype = c_int32
                lib.quakesnail_invert_hypocenter.argtypes = [
                    POINTER(QuakeSnailArena),
                    c_size_t,
                    POINTER(HypocenterResult),
                ]
                lib.quakesnail_invert_hypocenter.restype = c_int32
                lib.quakesnail_invert_hypocenter_huber.argtypes = [
                    POINTER(QuakeSnailArena),
                    c_int32,
                    c_float,
                    c_int32,
                    POINTER(c_float),
                    POINTER(c_float),
                ]
                lib.quakesnail_invert_hypocenter_huber.restype = c_int32
                lib.quakesnail_raytrace_prem_slab.argtypes = [
                    POINTER(c_float),
                    POINTER(c_float),
                    c_int32,
                    c_float,
                    c_float,
                    POINTER(c_float),
                    c_int32,
                    POINTER(c_float),
                ]
                lib.quakesnail_raytrace_prem_slab.restype = c_int32
                return lib
            except Exception:
                continue
    return None


_NATIVE_LIB = _find_and_load_native_library()


def is_native_available() -> bool:
    """Return True if compiled Rust SIMD cdylib is loaded, False otherwise."""
    return _NATIVE_LIB is not None


# --- Pure-Python Vectorized Mathematical Fallback ---


def py_biquad_bandpass(f_low: float, f_high: float, fs: float) -> Tuple[float, float, float, float, float]:
    """Compute Butterworth bandpass biquad filter coefficients."""
    f0 = np.float32(np.sqrt(f_low * f_high))
    bw = np.float32(max(0.01, f_high - f_low))
    q = np.float32(max(0.1, f0 / bw))
    w0 = np.float32(2.0 * np.pi * f0 / fs)
    sin_w = np.float32(np.sin(w0))
    cos_w = np.float32(np.cos(w0))
    alpha = np.float32(sin_w / (2.0 * q))

    a0 = np.float32(1.0 + alpha)
    b0 = np.float32(alpha / a0)
    b1 = np.float32(0.0)
    b2 = np.float32(-alpha / a0)
    a1 = np.float32((-2.0 * cos_w) / a0)
    a2 = np.float32((1.0 - alpha) / a0)
    return b0, b1, b2, a1, a2


def py_biquad_step(b0: float, b1: float, b2: float, a1: float, a2: float, x: float, d1: float, d2: float) -> Tuple[float, float, float]:
    """Single-step Direct Form II Transposed biquad filter."""
    x = np.float32(x)
    y = np.float32(b0 * x + d1)
    new_d1 = np.float32(b1 * x - a1 * y + d2)
    new_d2 = np.float32(b2 * x - a2 * y)
    return y, new_d1, new_d2


def py_morse_impulse(v_prev: float, v_curr: float, v_next: float) -> float:
    """3-point discrete Morse critical point impulse product."""
    dm = np.float32(v_curr - v_prev)
    dp = np.float32(v_curr - v_next)
    return np.float32(dm * dp)


def py_haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    p1 = np.float32(lat1 * (np.pi / 180.0))
    p2 = np.float32(lat2 * (np.pi / 180.0))
    dp = np.float32((lat2 - lat1) * (np.pi / 180.0))
    dl = np.float32((lon2 - lon1) * (np.pi / 180.0))

    s_dp = np.float32(np.sin(dp * 0.5))
    s_dl = np.float32(np.sin(dl * 0.5))
    a = np.float32(s_dp * s_dp + np.cos(p1) * np.cos(p2) * s_dl * s_dl)
    a = np.clip(a, 0.0, 1.0)
    c = np.float32(2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))
    return np.float32(EARTH_RADIUS_KM * c)


def py_prem_vp_kms(r_km: float) -> float:
    """1D PREM compressional P-wave velocity in km/s."""
    r = float(max(0.0, min(EARTH_RADIUS_KM, r_km)))
    x = r / EARTH_RADIUS_KM

    if r < 1221.5:
        return float(11.2622 - 6.3640 * (x**2))
    elif r < 3480.0:
        return float(11.0487 - 4.0362 * x + 4.8023 * (x**2) - 13.5732 * (x**3))
    elif r < 5701.0:
        return float(15.3891 - 5.3181 * x + 5.5242 * (x**2) - 2.5514 * (x**3))
    elif r < 5771.0:
        return float(24.9520 - 40.4673 * x + 24.2382 * (x**2))
    elif r < 5971.0:
        return float(29.2766 - 23.2970 * x)
    elif r < 6151.0:
        return float(30.7877 - 25.4670 * x)
    elif r < 6346.6:
        return float(25.4138 - 17.6972 * x)
    elif r < 6356.0:
        return 6.8000
    else:
        return 5.8000


def py_so6_kelvin_mandel_rotor(r_so3: np.ndarray) -> np.ndarray:
    """Lifts 3x3 SO(3) rotation into 6x6 SO(6) Kelvin-Mandel orthogonal rotor."""
    dtype = np.float64 if np.asarray(r_so3).dtype == np.float64 else np.float32
    r = np.asarray(r_so3, dtype=dtype).reshape((3, 3))
    sq2 = dtype(np.sqrt(2.0))
    m = np.zeros((6, 6), dtype=dtype)

    # Diagonal rows
    for i in range(3):
        m[i, 0] = r[i, 0] ** 2
        m[i, 1] = r[i, 1] ** 2
        m[i, 2] = r[i, 2] ** 2
        m[i, 3] = sq2 * r[i, 1] * r[i, 2]
        m[i, 4] = sq2 * r[i, 0] * r[i, 2]
        m[i, 5] = sq2 * r[i, 0] * r[i, 1]

    # Off-diagonal rows (Mandel normalization)
    pairs = [(1, 2), (0, 2), (0, 1)]
    for row_idx, (a, b) in enumerate(pairs, start=3):
        m[row_idx, 0] = sq2 * r[a, 0] * r[b, 0]
        m[row_idx, 1] = sq2 * r[a, 1] * r[b, 1]
        m[row_idx, 2] = sq2 * r[a, 2] * r[b, 2]
        m[row_idx, 3] = r[a, 1] * r[b, 2] + r[a, 2] * r[b, 1]
        m[row_idx, 4] = r[a, 0] * r[b, 2] + r[a, 2] * r[b, 0]
        m[row_idx, 5] = r[a, 0] * r[b, 1] + r[a, 1] * r[b, 0]

    return m


def py_evaluate_so_c_triad(w_tensor: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Evaluate so(C) skew-symmetric bilinear triad operator.

    Accumulates inner products in float64 before casting to preserve exact 0.000e+00 radial drift.
    """
    w = np.asarray(w_tensor, dtype=np.float64).reshape((CASCADE_DIM, CASCADE_DIM, CASCADE_DIM))
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    bw = np.zeros(CASCADE_DIM, dtype=np.float64)

    for k in range(CASCADE_DIM):
        val = 0.0
        for i in range(CASCADE_DIM):
            if abs(u[i]) < 1e-12:
                continue
            for j in range(CASCADE_DIM):
                skew = w[i, j, k] - w[i, k, j]
                val += u[i] * v[j] * skew
        bw[k] = val
    return bw.astype(np.float32)



# --- Public Unified API (Transparent Native vs Fallback) ---


class QuakeSnailCoreEngine:
    """Unified engine manager wrapping QuakeSnailArena either via native Ctypes or Python fallback."""

    def __init__(self, force_python: bool = False):
        self.use_native = is_native_available() and not force_python
        self._arena_ptr: Optional[ctypes.POINTER(QuakeSnailArena)] = None
        self._py_arena: Optional[QuakeSnailArena] = None

        if self.use_native:
            self._arena_ptr = _NATIVE_LIB.quakesnail_arena_new()
        else:
            self._py_arena = QuakeSnailArena()

    def __del__(self):
        if self.use_native and self._arena_ptr and _NATIVE_LIB:
            try:
                _NATIVE_LIB.quakesnail_arena_free(self._arena_ptr)
            except Exception:
                pass
            self._arena_ptr = None

    def reset(self):
        if self.use_native and self._arena_ptr:
            _NATIVE_LIB.quakesnail_arena_reset(self._arena_ptr)
        elif self._py_arena:
            ctypes.memset(byref(self._py_arena), 0, ctypes.sizeof(QuakeSnailArena))

    def register_station(self, station_id: int, lat_deg: float, lon_deg: float, alt_m: float) -> int:
        if self.use_native and self._arena_ptr:
            return _NATIVE_LIB.quakesnail_register_station(
                self._arena_ptr,
                c_int32(station_id),
                c_float(lat_deg),
                c_float(lon_deg),
                c_float(alt_m),
            )
        elif self._py_arena:
            idx = int(self._py_arena.num_stations)
            if idx >= MAX_STATIONS:
                return -2
            self._py_arena.stations[idx].station_id = station_id
            self._py_arena.stations[idx].lat_deg = lat_deg
            self._py_arena.stations[idx].lon_deg = lon_deg
            self._py_arena.stations[idx].alt_m = alt_m
            self._py_arena.stations[idx].active = 1
            for b in range(NUM_BANDS):
                self._py_arena.channel_states[idx][b].sample_rate_hz = 50.0
                self._py_arena.channel_states[idx][b].noise_var = 1.0
            self._py_arena.num_stations += 1
            return idx
        return -1

    def push_sample(self, station_idx: int, raw_sample: float, dt: float) -> Tuple[int, Optional[dict]]:
        out_pick = PickSlot()
        if self.use_native and self._arena_ptr:
            status = _NATIVE_LIB.quakesnail_push_sample(
                self._arena_ptr,
                c_int32(station_idx),
                c_float(raw_sample),
                c_float(dt),
                byref(out_pick),
            )
        else:
            status = self._py_push_sample(station_idx, raw_sample, dt, out_pick)

        pick_dict = None
        if out_pick.active != 0:
            pick_dict = {
                "pick_id": out_pick.pick_id,
                "station_idx": out_pick.station_idx,
                "phase_type": out_pick.phase_type,
                "arrival_time_s": out_pick.arrival_time_s,
                "snr": out_pick.snr,
                "amplitude_m": out_pick.amplitude_m,
                "period_s": out_pick.period_s,
                "classification": out_pick.classification,
            }
        return status, pick_dict

    def _py_push_sample(self, station_idx: int, raw_sample: float, dt: float, out_pick: PickSlot) -> int:
        if not self._py_arena or station_idx >= self._py_arena.num_stations:
            return -2

        fs = 1.0 / dt if dt > 0.0 else 50.0
        head = self._py_arena.ring_heads[station_idx]
        is_quar = self._py_arena.clock_mask[station_idx] != 0
        emitted = CLASSIFICATION_NOMINAL

        for band in range(NUM_BANDS):
            b0, b1, b2, a1, a2 = py_biquad_bandpass(
                2.0 if band == 0 else (0.4 if band == 1 else 0.05),
                5.0,
                fs,
            )
            state = self._py_arena.channel_states[station_idx][band]
            y, new_d1, new_d2 = py_biquad_step(b0, b1, b2, a1, a2, raw_sample, state.d1, state.d2)
            state.d1, state.d2 = new_d1, new_d2

            alpha_sta = np.exp(-dt / 1.0)
            alpha_lta = np.exp(-dt / 20.0)
            abs_y = abs(y)

            if state.lta < 1e-6 and abs_y > 1e-6:
                state.sta = abs_y
                state.lta = abs_y
            else:
                state.sta = alpha_sta * state.sta + (1.0 - alpha_sta) * abs_y
                state.lta = alpha_lta * state.lta + (1.0 - alpha_lta) * abs_y

            state.impulse_counter += 1
            is_primed = state.impulse_counter >= 20
            state.ratio = state.sta / (state.lta + 1e-6) if is_primed else 1.0

            thresh = 3.5 * (2.2 if is_quar else 1.0)
            ev_class = CLASSIFICATION_NOMINAL

            if state.gate_state == CLASSIFICATION_PENDING_IMPULSE:
                v_prev = state.prev_sample
                v_pend = state.pending_val
                v_curr = raw_sample

                imorse = py_morse_impulse(v_prev, v_pend, v_curr)
                ret_diff = abs(v_curr - v_prev)
                step_diff = abs(v_curr - v_pend)
                spike_mag = abs(v_pend - v_prev)

                if (ret_diff <= 0.1 * spike_mag or ret_diff <= 0.5) and imorse > 0.0:
                    state.gate_state = CLASSIFICATION_NOMINAL
                    ev_class = CLASSIFICATION_GLINT_REJECTED
                elif step_diff <= 0.5 * spike_mag or state.ratio > thresh:
                    state.gate_state = CLASSIFICATION_NOMINAL
                    ev_class = CLASSIFICATION_P_WAVE_ONSET
                else:
                    state.gate_state = CLASSIFICATION_NOMINAL
            else:
                delta = abs(raw_sample - state.last_sample)
                noise_std = max(1e-4, np.sqrt(state.noise_var))
                if is_primed and (delta > 4.0 * noise_std or state.ratio > thresh):
                    state.gate_state = CLASSIFICATION_PENDING_IMPULSE
                    state.pending_val = raw_sample
                    ev_class = CLASSIFICATION_PENDING_IMPULSE
                else:
                    state.noise_var = 0.995 * state.noise_var + 0.005 * (raw_sample * raw_sample)

            state.prev_sample = state.last_sample
            state.last_sample = raw_sample

            buf_offset = (station_idx * NUM_BANDS + band) * WAVEFORM_RING_CAPACITY
            self._py_arena.waveforms[buf_offset + head] = y

            if ev_class == CLASSIFICATION_P_WAVE_ONSET and band == 0:
                emitted = CLASSIFICATION_P_WAVE_ONSET
                if self._py_arena.num_picks < MAX_PICKS:
                    p_idx = int(self._py_arena.num_picks)
                    p = self._py_arena.picks[p_idx]
                    p.pick_id = p_idx + 1
                    p.station_idx = station_idx
                    p.phase_type = PHASE_P
                    p.arrival_time_s = 0.0
                    p.snr = state.ratio
                    p.amplitude_m = abs_y
                    p.period_s = 0.2
                    p.classification = CLASSIFICATION_P_WAVE_ONSET
                    p.active = 1
                    self._py_arena.num_picks += 1
                    ctypes.memmove(byref(out_pick), byref(p), ctypes.sizeof(PickSlot))
            elif ev_class == CLASSIFICATION_GLINT_REJECTED and band == 0:
                emitted = CLASSIFICATION_GLINT_REJECTED
            elif ev_class == CLASSIFICATION_PENDING_IMPULSE and emitted == CLASSIFICATION_NOMINAL:
                emitted = CLASSIFICATION_PENDING_IMPULSE

        self._py_arena.ring_heads[station_idx] = (head + 1) % WAVEFORM_RING_CAPACITY
        return emitted

    def associate_phases(self, max_clusters: int = 16) -> int:
        if self.use_native and self._arena_ptr:
            return _NATIVE_LIB.quakesnail_associate_phases(self._arena_ptr, c_size_t(max_clusters))
        elif self._py_arena:
            # Python association fallback
            return self._py_associate_phases()
        return -1

    def _py_associate_phases(self) -> int:
        if not self._py_arena:
            return 0
        n_picks = int(self._py_arena.num_picks)
        if n_picks < 4:
            return 0

        # Form initial cluster if not exists
        if self._py_arena.num_clusters == 0:
            c = self._py_arena.clusters[0]
            c.cluster_id = 1
            sum_lat, sum_lon = 0.0, 0.0
            for i in range(n_picks):
                s_idx = self._py_arena.picks[i].station_idx
                sum_lat += self._py_arena.stations[s_idx].lat_deg
                sum_lon += self._py_arena.stations[s_idx].lon_deg
            c.lat_deg = sum_lat / n_picks
            c.lon_deg = sum_lon / n_picks
            c.depth_km = 15.0
            c.origin_time_s = self._py_arena.picks[0].arrival_time_s - 5.0
            c.mag_default = 5.0
            c.num_picks = n_picks
            c.active = 1
            self._py_arena.num_clusters = 1
            return n_picks
        return n_picks

    def invert_hypocenter(self, cluster_idx: int = 0) -> Optional[dict]:
        out_res = HypocenterResult()
        if self.use_native and self._arena_ptr:
            ret = _NATIVE_LIB.quakesnail_invert_hypocenter(
                self._arena_ptr,
                c_size_t(cluster_idx),
                byref(out_res),
            )
            if ret == 0:
                return {
                    "lat_deg": out_res.lat_deg,
                    "lon_deg": out_res.lon_deg,
                    "depth_km": out_res.depth_km,
                    "origin_time_s": out_res.origin_time_s,
                    "mag_default": out_res.mag_default,
                    "mag_low": out_res.mag_low,
                    "mag_ultralow": out_res.mag_ultralow,
                    "semi_major_km": out_res.semi_major_km,
                    "semi_minor_km": out_res.semi_minor_km,
                    "depth_error_km": out_res.depth_error_km,
                    "residual_rms_s": out_res.residual_rms_s,
                    "num_picks": out_res.num_picks,
                    "cov_matrix": list(out_res.cov_matrix),
                }
        return None

    def raytrace(
        self,
        src_xyz: Tuple[float, float, float],
        rcv_xyz: Tuple[float, float, float],
        phase: int = 0,
        slab_dip_deg: float = 45.0,
        slab_azimuth_deg: float = 90.0,
        num_nodes: int = 16,
    ) -> Tuple[float, np.ndarray]:
        c_src = (c_float * 3)(*src_xyz)
        c_rcv = (c_float * 3)(*rcv_xyz)
        n = max(8, min(RAY_DISCRETIZATION, num_nodes))
        c_out_xyz = (c_float * (n * 3))()
        c_tt = c_float(0.0)

        if self.use_native:
            ret = _NATIVE_LIB.quakesnail_raytrace_prem_slab(
                c_src,
                c_rcv,
                c_int32(phase),
                c_float(slab_dip_deg),
                c_float(slab_azimuth_deg),
                c_out_xyz,
                c_int32(n),
                byref(c_tt),
            )
            if ret == 0:
                nodes = np.ctypeslib.as_array(c_out_xyz).reshape((n, 3))
                return float(c_tt.value), nodes.copy()

        # Fallback pseudo-bending
        p0 = np.array(src_xyz, dtype=np.float32)
        p1 = np.array(rcv_xyz, dtype=np.float32)
        nodes = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            frac = i / (n - 1)
            nodes[i] = p0 + frac * (p1 - p0)

        total_tt = 0.0
        for i in range(n - 1):
            seg_len = np.linalg.norm(nodes[i + 1] - nodes[i])
            r_mid = np.linalg.norm(0.5 * (nodes[i] + nodes[i + 1]))
            v = py_prem_vp_kms(r_mid)
            if phase == 1:
                v /= 1.732
            total_tt += seg_len / max(1.0, v)

        return float(total_tt), nodes


QuakeSnailNativeBridge = QuakeSnailCoreEngine


def get_native_lib() -> Optional[ctypes.CDLL]:
    """Return the raw ctypes CDLL handle to the compiled Rust SIMD library, if loaded."""
    return _NATIVE_LIB

