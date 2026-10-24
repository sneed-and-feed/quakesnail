"""Stage 1: Waveform Picking, Multi-Band Filtering, and Telemetry Glitch Shielding.

Grounded in:
- Multi-band second-order biquads (2-5 Hz, 0.4-5 Hz, 0.01-5 Hz).
- Beltrami exterior-curl spike preservation metric.
- 3-point discrete Morse impulse gating: Delta_- v * Delta_+ v > 0.
- 1-sample causal deferral (pending_impulse -> glint_rejected vs p_wave_onset).
- Stationary microseism quarantine (H_clock (+) H_nope).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
import numpy as np


from quakesnail._quakesnail_native import (
    CLASSIFICATION_NOMINAL,
    CLASSIFICATION_PENDING_IMPULSE,
    CLASSIFICATION_P_WAVE_ONSET,
    CLASSIFICATION_GLINT_REJECTED,
    CLASSIFICATION_STEP_MANEUVER,
    PHASE_P,
    PHASE_S,
    PHASE_PKP,
    PHASE_PKIKP,
    QuakeSnailCoreEngine,
    py_biquad_bandpass,
    py_biquad_step,
    py_morse_impulse,
)


class TriggerStatus(IntEnum):
    NOMINAL = CLASSIFICATION_NOMINAL
    PENDING_IMPULSE = CLASSIFICATION_PENDING_IMPULSE
    P_WAVE_ONSET = CLASSIFICATION_P_WAVE_ONSET
    GLINT_REJECTED = CLASSIFICATION_GLINT_REJECTED
    STEP_MANEUVER = CLASSIFICATION_STEP_MANEUVER
    # Aliases
    NO_TRIGGER = CLASSIFICATION_NOMINAL
    PWAVE_ACCEPTED = CLASSIFICATION_P_WAVE_ONSET


GateClassification = TriggerStatus


class BiquadFilter:
    """Second-order section bandpass filter in Direct Form II Transposed."""

    def __init__(self, f_low: float, f_high: float, fs: float):
        self.f_low = float(f_low)
        self.f_high = float(f_high)
        self.fs = float(fs)
        f0 = math.sqrt(f_low * f_high)
        bw = max(f_high - f_low, 0.01)
        q = max(f0 / bw, 0.1)
        w0 = 2.0 * math.pi * f0 / fs
        sin_w = math.sin(w0)
        cos_w = math.cos(w0)
        alpha = sin_w / (2.0 * q)

        a0 = 1.0 + alpha
        self.b0 = alpha / a0
        self.b1 = 0.0
        self.b2 = -alpha / a0
        self.a1 = (-2.0 * cos_w) / a0
        self.a2 = (1.0 - alpha) / a0

        self.d1 = 0.0
        self.d2 = 0.0

    def step(self, x: float) -> float:
        x_val = float(x)
        y = self.b0 * x_val + self.d1
        self.d1 = self.b1 * x_val - self.a1 * y + self.d2
        self.d2 = self.b2 * x_val - self.a2 * y
        return float(y)

    def reset(self):
        self.d1 = 0.0
        self.d2 = 0.0


class DiscreteMorseFilter:
    """Discrete Morse impulse gating and Beltrami edge-preservation metric."""

    def __init__(self, kappa: float = 0.05):
        self.kappa = float(kappa)

    @staticmethod
    def impulse(v_prev: float, v_curr: float, v_next: float) -> float:
        return float(py_morse_impulse(v_prev, v_curr, v_next))

    @staticmethod
    def beltrami(v_curr: float, v_prev: float, kappa: float = 0.05) -> float:
        dv = abs(v_curr - v_prev)
        return float(dv / math.sqrt(1.0 + kappa * dv * dv))

    def evaluate_impulse(self, v_prev: float, v_curr: float, v_next: float) -> float:
        return self.impulse(v_prev, v_curr, v_next)

    def evaluate_beltrami(self, v_curr: float, v_prev: float) -> float:
        return self.beltrami(v_curr, v_prev, self.kappa)


def beltrami_metric(v_curr: float, v_prev: float, kappa: float = 0.05) -> float:
    return DiscreteMorseFilter.beltrami(v_curr, v_prev, kappa)


def morse_impulse(v_prev: float, v_curr: float, v_next: float) -> float:
    return DiscreteMorseFilter.impulse(v_prev, v_curr, v_next)


@dataclass
class Pick:
    pick_id: str
    station_code: str
    arrival_time_s: float
    phase_type: int = PHASE_P
    snr: float = 0.0
    amplitude_m: float = 0.0
    period_s: float = 0.2
    classification: GateClassification = GateClassification.NOMINAL
    uncertainty_s: float = 0.05
    band_idx: int = 0


@dataclass
class StationFilterState:
    sample_rate_hz: float = 50.0
    d1: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    d2: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    sta: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    lta: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    ratio: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    gate_state: int = CLASSIFICATION_NOMINAL
    pending_val: float = 0.0
    pending_time_s: float = 0.0
    last_sample: float = 0.0
    prev_sample: float = 0.0
    noise_var: float = 1.0
    impulse_counter: int = 0
    is_quarantined: bool = False
    ring_buffer: np.ndarray = field(default_factory=lambda: np.zeros(512, dtype=np.float32))
    ring_head: int = 0
    priming_buffer: List[List[float]] = field(default_factory=lambda: [[], [], []])



class WaveformPicker:
    """Stage 1 Waveform Picker & Glitch Shielding Engine."""

    NOMINAL = TriggerStatus.NOMINAL
    PENDING_IMPULSE = TriggerStatus.PENDING_IMPULSE
    P_WAVE_ONSET = TriggerStatus.P_WAVE_ONSET
    GLINT_REJECTED = TriggerStatus.GLINT_REJECTED
    STEP_MANEUVER = TriggerStatus.STEP_MANEUVER

    def __init__(
        self,
        sample_rate_hz: float = 50.0,
        use_native: bool = True,
        fs: Optional[float] = None,
        base_threshold: float = 4.5,
    ):
        if fs is not None:
            sample_rate_hz = fs
        self.sample_rate_hz = float(sample_rate_hz)
        self.dt = 1.0 / self.sample_rate_hz
        self.base_threshold = float(base_threshold)
        self.use_native = use_native
        self.engine = QuakeSnailCoreEngine(force_python=not use_native)

        self.stations: Dict[str, int] = {}  # station_code -> station_idx
        self.states: Dict[str, StationFilterState] = {}
        self.picks: List[Pick] = []
        self._next_pick_id = 1

        # Precompute default band coefficients
        self.band_coeffs = [
            py_biquad_bandpass(2.0, 5.0, self.sample_rate_hz),    # Band 0: Local/regional P
            py_biquad_bandpass(0.4, 5.0, self.sample_rate_hz),    # Band 1: Intermediate
            py_biquad_bandpass(0.05, 5.0, self.sample_rate_hz),   # Band 2: Long-period
        ]

        self._biquad = BiquadFilter(2.0, 5.0, self.sample_rate_hz)

    @property
    def biquad(self) -> BiquadFilter:
        return self._biquad

    def _default_state(self) -> StationFilterState:
        if "DEFAULT" not in self.stations:
            self.register_station("DEFAULT")
        return self.states["DEFAULT"]

    @property
    def sta(self) -> float:
        return float(self._default_state().sta[0])

    @sta.setter
    def sta(self, val: float):
        self._default_state().sta[0] = float(val)

    @property
    def lta(self) -> float:
        return float(self._default_state().lta[0])

    @lta.setter
    def lta(self, val: float):
        self._default_state().lta[0] = float(val)

    @property
    def ratio(self) -> float:
        return float(self._default_state().ratio[0])

    @ratio.setter
    def ratio(self, val: float):
        self._default_state().ratio[0] = float(val)

    def process_sample(
        self,
        raw_sample: float,
        is_quarantined: bool = False,
        station_code: str = "DEFAULT",
    ) -> int:
        if station_code not in self.stations:
            self.register_station(station_code)
        state = self.states[station_code]
        state.is_quarantined = is_quarantined
        evt, _ = self.feed_sample(station_code, raw_sample, state.impulse_counter * self.dt)
        return int(evt)

    def register_station(self, station_code: str, lat_deg: float = 0.0, lon_deg: float = 0.0, alt_m: float = 0.0) -> int:
        if station_code in self.stations:
            return self.stations[station_code]
        idx = len(self.stations)
        self.stations[station_code] = idx
        self.states[station_code] = StationFilterState(sample_rate_hz=self.sample_rate_hz)
        self.engine.register_station(idx, lat_deg, lon_deg, alt_m)
        return idx

    def beltrami_metric(self, v_curr: float, v_prev: float, kappa: float = 0.05) -> float:
        """Beltrami exterior-curl metric preserving sharp steps while preventing spike saturation."""
        dv = abs(v_curr - v_prev)
        return float(dv / np.sqrt(1.0 + kappa * dv * dv))

    def feed_sample(
        self,
        station_code: str,
        raw_sample: float,
        timestamp_s: float,
    ) -> Tuple[GateClassification, Optional[Pick]]:
        """Process a single raw sample through multi-band biquads and 1-sample causal deferral."""
        if not math.isfinite(raw_sample):
            return GateClassification.NOMINAL, None

        if station_code not in self.stations:
            self.register_station(station_code)

        state = self.states[station_code]
        raw_sample = float(raw_sample)

        alpha_sta = np.exp(-self.dt / 1.0)
        alpha_lta = np.exp(-self.dt / 20.0)

        def _step_biquads(s_in: float):
            filtered = []
            for b in range(3):
                b0, b1, b2, a1, a2 = self.band_coeffs[b]
                y, new_d1, new_d2 = py_biquad_step(
                    b0, b1, b2, a1, a2,
                    s_in,
                    state.d1[b],
                    state.d2[b],
                )
                state.d1[b] = new_d1
                state.d2[b] = new_d2
                filtered.append(y)

                abs_y = abs(y)
                if state.impulse_counter < 100:
                    state.priming_buffer[b].append(abs_y)
                    if state.impulse_counter >= 99:
                        med = float(np.median(state.priming_buffer[b])) if state.priming_buffer[b] else 1e-4
                        state.lta[b] = max(med, 1e-4)
                        state.sta[b] = max(med, 1e-4)
                        state.ratio[b] = 1.0
                    else:
                        state.sta[b] = max(abs_y, 1e-4)
                        state.lta[b] = max(abs_y, 1e-4)
                        state.ratio[b] = 1.0
                else:
                    state.sta[b] = alpha_sta * state.sta[b] + (1.0 - alpha_sta) * abs_y
                    state.lta[b] = alpha_lta * state.lta[b] + (1.0 - alpha_lta) * abs_y
                    state.ratio[b] = state.sta[b] / (state.lta[b] + 1e-6)

            state.impulse_counter += 1
            head = state.ring_head
            state.ring_buffer[head] = filtered[1]
            state.ring_head = (head + 1) % 512

            if state.impulse_counter % 64 == 0:
                state.is_quarantined = self._check_microseism(state.ring_buffer)

            return filtered

        is_primed = state.impulse_counter >= 100
        effective_thresh = self.base_threshold * (2.2 if state.is_quarantined else 1.0)
        emitted_event = GateClassification.NOMINAL
        emitted_pick = None

        if state.gate_state == CLASSIFICATION_PENDING_IMPULSE:
            v_prev = state.prev_sample
            v_pend = state.pending_val
            v_curr = raw_sample

            imorse = py_morse_impulse(v_prev, v_pend, v_curr)
            ret_diff = abs(v_curr - v_prev)
            step_diff = abs(v_curr - v_pend)
            spike_mag = abs(v_pend - v_prev)
            noise_std = max(1e-4, np.sqrt(state.noise_var))
            theta_persist = 8.0 * noise_std

            if (ret_diff <= max(theta_persist, 0.2 * spike_mag) or ret_diff <= 0.5) and imorse > 0.0:
                # Isolated 1-sample spike / bit-flip glitch -> Reject!
                # Glitch causal shielding: clamp/zero the pending spike fed to recursive IIR biquad state registers
                state.gate_state = CLASSIFICATION_NOMINAL
                emitted_event = GateClassification.GLINT_REJECTED
                _step_biquads(raw_sample)
            elif abs(v_curr - v_pend) > 6.0 * noise_std and abs(v_pend - v_prev) < 3.0 * noise_std:
                # If the previous deferred sample was just noise, and THIS sample is the real spike:
                _step_biquads(v_pend)
                state.gate_state = CLASSIFICATION_PENDING_IMPULSE
                state.pending_val = v_curr
                state.pending_time_s = timestamp_s
                emitted_event = GateClassification.PENDING_IMPULSE
            elif step_diff <= 0.5 * spike_mag or state.ratio[0] > effective_thresh:
                # Genuine P-wave impulsive onset!
                filtered_pend = _step_biquads(v_pend)
                _step_biquads(v_curr)
                state.gate_state = CLASSIFICATION_NOMINAL
                emitted_event = GateClassification.P_WAVE_ONSET
                pick = Pick(
                    pick_id=f"PK_{self._next_pick_id:04d}",
                    station_code=station_code,
                    arrival_time_s=state.pending_time_s,
                    phase_type=PHASE_P,
                    snr=float(state.ratio[0]),
                    amplitude_m=float(abs(filtered_pend[0])),
                    period_s=0.2,
                    classification=GateClassification.P_WAVE_ONSET,
                )
                self._next_pick_id += 1
                self.picks.append(pick)
                emitted_pick = pick
            else:
                state.gate_state = CLASSIFICATION_NOMINAL
                _step_biquads(v_pend)
                _step_biquads(v_curr)
        else:
            delta = abs(raw_sample - state.last_sample)
            noise_std = max(1e-4, np.sqrt(state.noise_var))

            if is_primed and delta > 6.0 * noise_std:
                # Enter PENDING_IMPULSE: defer raw sample update to biquad filter registers
                state.gate_state = CLASSIFICATION_PENDING_IMPULSE
                state.pending_val = raw_sample
                state.pending_time_s = timestamp_s
                emitted_event = GateClassification.PENDING_IMPULSE
            else:
                filtered = _step_biquads(raw_sample)
                state.noise_var = 0.995 * state.noise_var + 0.005 * (raw_sample * raw_sample)
                if is_primed and state.ratio[0] > effective_thresh:
                    emitted_event = GateClassification.P_WAVE_ONSET
                    pick = Pick(
                        pick_id=f"PK_{self._next_pick_id:04d}",
                        station_code=station_code,
                        arrival_time_s=timestamp_s,
                        phase_type=PHASE_P,
                        snr=float(state.ratio[0]),
                        amplitude_m=float(abs(filtered[0])),
                        period_s=0.2,
                        classification=GateClassification.P_WAVE_ONSET,
                    )
                    self._next_pick_id += 1
                    self.picks.append(pick)
                    emitted_pick = pick

        state.prev_sample = state.last_sample
        state.last_sample = raw_sample

        return emitted_event, emitted_pick


    def feed_waveform(
        self,
        station_code: str,
        waveform: np.ndarray,
        start_time_s: float = 0.0,
    ) -> List[Pick]:
        """Feed a contiguous chunk of waveform samples and return any detected picks."""
        picks_found = []
        t = start_time_s
        for s in waveform:
            _, pick = self.feed_sample(station_code, float(s), t)
            if pick:
                picks_found.append(pick)
            t += self.dt
        return picks_found

    def _check_microseism(self, buffer: np.ndarray) -> bool:
        """Evaluate cyclic autocorrelation across 0.1-0.3 Hz ocean swell."""
        energy = float(np.sum(buffer**2))
        if energy < 1e-6:
            return False

        # Test lags 10 to 40 (approx 0.2 to 0.8 s)
        max_r = 0.0
        for lag in range(10, 40):
            dot = float(np.sum(buffer[lag:] * buffer[:-lag]))
            r_k = dot / energy
            if r_k > max_r:
                max_r = r_k
        return max_r > 0.85


CausalDeferralPicker = WaveformPicker
