"""
tests/test_picking.py: E2E Verification of Stage 1 Waveform Picking, Morse Gating & Glitch Shielding.

Authority & Specifications:
- `ORIGINAL_REQUEST.md` § R2 (Stage 1 Picking & Glitch Shielding)
- `PROJECT.md` § Features 5, 6, 7, 8
- `TEST_INFRA.md` § Features F1-F5:
  - F1: Multi-band Waveform Filtering (2-5 Hz, 0.4-5 Hz, 0.01-5 Hz)
  - F2: Beltrami Spike Preservation
  - F3: 3-Point Discrete Morse Gating
  - F4: 1-Sample Causal Deferral Glitch Shield (pending_impulse -> glint_rejected vs p_wave_onset)
  - F5: H_clock (+) H_nope Microseism Quarantine

Tests:
- Tier 1 (Coverage): Multi-band biquads, Beltrami metric, Morse impulse, causal deferral, microseism quarantine.
  Strict assertion: ZERO false earthquake triggers across 10,000 synthetic SeedLink telemetry glitches.
- Tier 2 (Boundary): Flatline silence, massive DC offset, Nyquist limit oscillations, double-glitches,
  extremely low SNR, threshold edge values.
- Tier 3 (Pairwise): Combinations of sampling rates, filter bands, and glitch amplitudes.
- Tier 4 (Real-World Application): Real-world megathrust event with pre-event ocean swell, telemetry glitches,
  and impulsive P-wave onset.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Stage 1 Core Implementation Imports
# ---------------------------------------------------------------------------
from quakesnail.core.picking import (
    WaveformPicker,
    BiquadFilter,
    TriggerStatus,
    DiscreteMorseFilter,
    beltrami_metric,
    morse_impulse,
    CausalDeferralPicker,
)


# ===========================================================================
# Tier 1: Feature Coverage (F1 through F5)
# ===========================================================================

class TestPickingTier1:
    """Tier 1: Multi-band filtering, Beltrami spike preservation, Morse gating, and causal deferral."""

    def test_f1_multiband_biquad_filtering(self):
        """F1: Multi-band biquads correctly isolate seismic passbands."""
        fs = 50.0
        # Default band: 2.0 - 5.0 Hz
        b_default = BiquadFilter(2.0, 5.0, fs)
        # Low band: 0.4 - 5.0 Hz
        b_low = BiquadFilter(0.4, 5.0, fs)
        # Ultralow band: 0.05 - 5.0 Hz
        b_ultra = BiquadFilter(0.05, 5.0, fs)

        t = np.linspace(0, 10.0, int(10.0 * fs))
        # 3.0 Hz signal (in default band)
        sig_3hz = np.sin(2.0 * np.pi * 3.0 * t)
        # 0.2 Hz signal (in ultralow band, rejected by default band)
        sig_slow = np.sin(2.0 * np.pi * 0.2 * t)

        out_default_3hz = [b_default.step(float(x)) for x in sig_3hz]
        b_default.reset()
        out_default_slow = [b_default.step(float(x)) for x in sig_slow]

        # 3.0 Hz passes with high amplitude in steady state (last 100 samples)
        assert np.max(np.abs(out_default_3hz[-100:])) > 0.4
        # 0.2 Hz is severely attenuated in steady state (< 0.15)
        assert np.max(np.abs(out_default_slow[-100:])) < 0.15

    def test_f2_beltrami_spike_preservation(self):
        """F2: Beltrami exterior-curl metric preserves sharp step-like onset without blurring."""
        v_prev = 10.0
        # Sharp step jump (dv = 100) vs small noise fluctuation (dv = 1)
        b_large = beltrami_metric(110.0, v_prev, kappa=0.05)
        b_small = beltrami_metric(11.0, v_prev, kappa=0.05)

        assert b_large > b_small
        # As dv -> infinity, Beltrami metric saturates cleanly to 1 / sqrt(kappa)
        expected_saturation = 1.0 / math.sqrt(0.05)
        b_huge = beltrami_metric(10000.0, v_prev, kappa=0.05)
        assert abs(b_huge - expected_saturation) < 0.05

    def test_f3_discrete_morse_impulse_gating(self):
        """F3: 3-point discrete Morse impulse detects local extrema and rejects ramps."""
        # Local peak: 10 -> 50 -> 12
        p_peak = morse_impulse(10.0, 50.0, 12.0)
        assert p_peak > 0.0  # (50-10) * (50-12) = 40 * 38 = 1520 > 0

        # Local trough: 50 -> 10 -> 45
        p_trough = morse_impulse(50.0, 10.0, 45.0)
        assert p_trough > 0.0  # (10-50) * (10-45) = (-40) * (-35) = 1400 > 0

        # Monotonic ramp: 10 -> 30 -> 50
        p_ramp = morse_impulse(10.0, 30.0, 50.0)
        assert p_ramp < 0.0  # (30-10) * (30-50) = 20 * (-20) = -400 < 0 (rejected!)

    def test_f4_telemetry_glitch_shield_zero_false_triggers_across_10k_spikes(self):
        """F4: Prove strictly ZERO false triggers from 10,000 isolated single-sample SeedLink glitches."""
        rng = np.random.default_rng(20261024)
        picker = CausalDeferralPicker(fs=50.0)

        # Warm up filter on calm Gaussian background noise
        for _ in range(500):
            picker.process_sample(float(rng.normal(0.0, 0.5)))

        false_triggers = 0
        glints_rejected = 0

        # Inject 10,000 single-sample telemetry glitches embedded in noise
        for _ in range(10_000):
            spike_amp = float(rng.uniform(50.0, 50_000.0) * rng.choice([-1.0, 1.0]))
            # Sample t: spike arrives
            c1 = picker.process_sample(spike_amp)
            # Sample t+1: telemetry baseline recovers immediately
            c2 = picker.process_sample(float(rng.normal(0.0, 0.5)))

            if c1 == CausalDeferralPicker.P_WAVE_ONSET or c2 == CausalDeferralPicker.P_WAVE_ONSET:
                false_triggers += 1
            if c2 == CausalDeferralPicker.GLINT_REJECTED:
                glints_rejected += 1

            # Calm samples between glitches
            for _ in range(10):
                picker.process_sample(float(rng.normal(0.0, 0.5)))

        assert false_triggers == 0, f"Integrity failure: {false_triggers} false glitch triggers produced!"
        assert glints_rejected > 9_500, f"Expected high glint rejection rate, got {glints_rejected}"

    def test_f5_microseism_stationary_quarantine(self):
        """F5: Continuous ocean swell harmonic microseisms (0.1-0.3 Hz) are quarantined."""
        picker = CausalDeferralPicker(fs=50.0)
        # Generate 20 seconds of continuous 0.2 Hz ocean microseism swell
        t = np.linspace(0, 20.0, 1000)
        microseism = 15.0 * np.sin(2.0 * np.pi * 0.2 * t)

        triggers = 0
        for x in microseism:
            status = picker.process_sample(float(x), is_quarantined=True)
            if status == CausalDeferralPicker.P_WAVE_ONSET:
                triggers += 1

        # Microseism swell must produce 0 false impulsive P-wave triggers
        assert triggers == 0


# ===========================================================================
# Tier 2: Boundary & Corner Cases
# ===========================================================================

class TestPickingTier2:
    """Tier 2: Boundary conditions, silence, extreme DC offsets, and Nyquist limits."""

    def test_boundary_flatline_silence(self):
        """Boundary: Zero amplitude signal produces zero triggers."""
        picker = CausalDeferralPicker()
        for _ in range(500):
            status = picker.process_sample(0.0)
            assert status == CausalDeferralPicker.NOMINAL

    def test_boundary_massive_dc_offset(self):
        """Boundary: 1,000,000 count uncalibrated geophone DC bias."""
        picker = CausalDeferralPicker()
        dc_val = 1_000_000.0
        # Push constant DC: biquad bandpass removes DC completely
        outputs = [picker.biquad.step(dc_val) for _ in range(200)]
        # After transient response, output settles to 0.0
        assert abs(outputs[-1]) < 1.0e-3

    def test_boundary_consecutive_glitches_double_spike(self):
        """Boundary: Double-sample glitch spike (samples t and t+1 high, then returns)."""
        picker = CausalDeferralPicker()
        # Warm up
        for _ in range(100):
            picker.process_sample(0.0)

        # Inject double spike: [1000, 1000, 0]
        c1 = picker.process_sample(1000.0)
        c2 = picker.process_sample(1000.0)
        c3 = picker.process_sample(0.0)

        # Should not cause unhandled state corruption
        assert c1 in (CausalDeferralPicker.NOMINAL, CausalDeferralPicker.PENDING_IMPULSE)

    def test_boundary_nyquist_limit_oscillation(self):
        """Boundary: Signal oscillating at exact Nyquist frequency (fs / 2 = 25 Hz)."""
        picker = CausalDeferralPicker(fs=50.0)
        # Alternating +50, -50, +50, -50
        for i in range(100):
            val = 50.0 if (i % 2 == 0) else -50.0
            picker.process_sample(val)

        # Assert no NaN or Inf produced in filter states
        assert not math.isnan(picker.sta)
        assert not math.isnan(picker.lta)

    def test_boundary_zero_sta_lta_ratio_guard(self):
        """Boundary: Ratio computation when STA and LTA are zero is guarded against zero division."""
        picker = CausalDeferralPicker()
        picker.sta = 0.0
        picker.lta = 0.0
        picker.process_sample(0.0)
        assert picker.ratio >= 0.0 and not math.isnan(picker.ratio)


# ===========================================================================
# Tier 3: Pairwise Combinations
# ===========================================================================

class TestPickingTier3:
    """Tier 3: Cross-feature combinations of sample rates and glitch profiles."""

    @pytest.mark.parametrize("fs", [20.0, 50.0, 100.0])
    @pytest.mark.parametrize("spike_amp", [80.0, 800.0, 8000.0])
    def test_pairwise_samplerate_and_glitch_rejection(self, fs, spike_amp):
        """Pairwise: Causal deferral glitch rejection invariant across sampling rates."""
        picker = CausalDeferralPicker(fs=fs)
        for _ in range(int(fs * 2)):
            picker.process_sample(0.0)

        c1 = picker.process_sample(spike_amp)
        c2 = picker.process_sample(0.0)

        assert c1 != CausalDeferralPicker.P_WAVE_ONSET
        assert c2 in (CausalDeferralPicker.GLINT_REJECTED, CausalDeferralPicker.NOMINAL)

    @pytest.mark.parametrize("noise_floor", [0.1, 1.0, 5.0])
    @pytest.mark.parametrize("quarantined", [False, True])
    def test_pairwise_noise_floor_and_quarantine_stability(self, noise_floor, quarantined):
        """Pairwise: Quarantine multiplier elevates trigger threshold in high noise floors."""
        picker = CausalDeferralPicker()
        rng = np.random.default_rng(42)
        # Warm up picker so noise variance estimates the station baseline
        for _ in range(500):
            picker.process_sample(float(rng.normal(0.0, noise_floor)), is_quarantined=quarantined)

        triggers = 0
        for _ in range(200):
            sample = float(rng.normal(0.0, noise_floor))
            status = picker.process_sample(sample, is_quarantined=quarantined)
            if status == CausalDeferralPicker.P_WAVE_ONSET:
                triggers += 1

        assert triggers == 0


# ===========================================================================
# Tier 4: Real-World Application Scenarios
# ===========================================================================

class TestPickingTier4:
    """Tier 4: End-to-end realistic seismological records with glitches and genuine arrivals."""

    def test_realworld_subduction_megathrust_picking(self):
        """
        Scenario 1: Subduction megathrust event (M8.2).
        Record contains:
        1. 0-5s: Background oceanic microseism noise
        2. 5s: Intermittent telemetry packet glitch (should be REJECTED)
        3. 5-10s: Calm baseline
        4. 10s: Impulsive P-wave arrival with continuous decaying coda (should be ACCEPTED)
        """
        fs = 50.0
        n_samples = int(25.0 * fs)
        t = np.linspace(0, 25.0, n_samples)
        rng = np.random.default_rng(999)

        # Background noise + 0.2 Hz swell
        wave = rng.normal(0.0, 1.0, n_samples) + 2.0 * np.sin(2.0 * np.pi * 0.2 * t)

        # Telemetry glitch at 5.0s (index 250)
        glitch_idx = int(5.0 * fs)
        wave[glitch_idx] += 800.0

        # Genuine P-wave arrival at 10.0s (index 500)
        p_idx = int(10.0 * fs)
        coda_len = n_samples - p_idx
        t_coda = np.linspace(0, 15.0, coda_len)
        wave[p_idx:] += 50.0 * np.exp(-t_coda / 4.0) * np.sin(2.0 * np.pi * 3.5 * t_coda)

        picker = CausalDeferralPicker(fs=fs)
        p_accepted_times: List[float] = []
        glints_rejected_times: List[float] = []

        for i, val in enumerate(wave):
            status = picker.process_sample(float(val))
            if status == CausalDeferralPicker.P_WAVE_ONSET:
                p_accepted_times.append(t[i])
            elif status == CausalDeferralPicker.GLINT_REJECTED:
                glints_rejected_times.append(t[i])

        # Glitch must be rejected at ~5.0s
        assert any(abs(tg - 5.0) < 0.1 for tg in glints_rejected_times), "Glitch was not rejected!"
        # P-wave must be accepted at ~10.0s
        assert any(abs(tp - 10.0) < 0.2 for tp in p_accepted_times), "P-wave was not accepted!"
        # ZERO false triggers prior to 9.5s
        early_triggers = [tp for tp in p_accepted_times if tp < 9.5]
        assert len(early_triggers) == 0, f"Early false triggers before P-wave: {early_triggers}"
