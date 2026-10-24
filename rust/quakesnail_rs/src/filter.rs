//! Multi-band biquad filtering, Beltrami spike preservation, 3-point discrete Morse impulse gating,
//! 1-sample causal deferral, and microseism quarantine (H_clock (+) H_nope).

#![allow(dead_code)]

use crate::{
    StationChannelState, CLASSIFICATION_GLINT_REJECTED, CLASSIFICATION_NOMINAL,
    CLASSIFICATION_PENDING_IMPULSE, CLASSIFICATION_P_WAVE_ONSET,
    NUM_BANDS,
};

pub const PI: f32 = 3.14159265358979323846;

/// Second-order section (biquad) coefficients in Direct Form II Transposed:
/// y[n] = b0 * x[n] + d1[n-1]
/// d1[n] = b1 * x[n] - a1 * y[n] + d2[n-1]
/// d2[n] = b2 * x[n] - a2 * y[n]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiquadCoeffs {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

impl Default for BiquadCoeffs {
    fn default() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }
    }
}

impl BiquadCoeffs {
    /// Design Butterworth bandpass filter
    pub fn bandpass(f_low: f32, f_high: f32, fs: f32) -> Self {
        let f0 = (f_low * f_high).sqrt();
        let bw = (f_high - f_low).max(0.01);
        let q = (f0 / bw).max(0.1);
        let w0 = 2.0 * PI * f0 / fs;
        let sin_w = w0.sin();
        let cos_w = w0.cos();
        let alpha = sin_w / (2.0 * q);

        let a0 = 1.0 + alpha;
        let b0 = alpha / a0;
        let b1 = 0.0;
        let b2 = -alpha / a0;
        let a1 = (-2.0 * cos_w) / a0;
        let a2 = (1.0 - alpha) / a0;

        Self { b0, b1, b2, a1, a2 }
    }

    /// Direct Form II Transposed sample evaluation
    #[inline(always)]
    pub fn step(&self, x: f32, d1: &mut f32, d2: &mut f32) -> f32 {
        let y = self.b0 * x + *d1;
        *d1 = self.b1 * x - self.a1 * y + *d2;
        *d2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Standard 3 seismic monitoring filter bands:
/// Band 0: 2.0 - 5.0 Hz (local / regional P-wave onset)
/// Band 1: 0.4 - 5.0 Hz (intermediate / teleseismic P/S)
/// Band 2: 0.01 - 5.0 Hz (long-period teleseismic / core phases)
pub fn get_default_band_coeffs(band_idx: usize, fs: f32) -> BiquadCoeffs {
    match band_idx {
        0 => BiquadCoeffs::bandpass(2.0, 5.0, fs),
        1 => BiquadCoeffs::bandpass(0.4, 5.0, fs),
        2 => BiquadCoeffs::bandpass(0.05, 5.0, fs),
        _ => BiquadCoeffs::bandpass(2.0, 5.0, fs),
    }
}

/// Beltrami exterior-curl spike preservation metric:
/// S = |delta_v| / sqrt(1 + kappa * delta_v^2)
#[inline(always)]
pub fn beltrami_metric(v_t: f32, v_prev: f32, kappa: f32) -> f32 {
    let dv = (v_t - v_prev).abs();
    dv / (1.0 + kappa * dv * dv).sqrt()
}

/// 3-Point Discrete Morse Impulse Gate:
/// I_Morse = (v_t - v_{t-1}) * (v_t - v_{t+1})
/// Strict local extremum (peak/trough) iff I_Morse > 0.
#[inline(always)]
pub fn morse_impulse(v_prev: f32, v_curr: f32, v_next: f32) -> f32 {
    let delta_minus = v_curr - v_prev;
    let delta_plus = v_curr - v_next;
    delta_minus * delta_plus
}

/// Result of 1-sample causal deferral processing
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PickEvent {
    pub classification: i32,
    pub sample_val: f32,
    pub snr: f32,
    pub amplitude: f32,
    pub period_s: f32,
}

/// Process a single incoming raw waveform sample through the multi-band biquad,
/// Beltrami spike protector, STA/LTA tracker, 3-point discrete Morse impulse gate,
/// and 1-sample causal deferral state machine.
#[inline(always)]
fn step_biquad_and_sta_lta(
    s_in: f32,
    coeffs: &BiquadCoeffs,
    state: &mut StationChannelState,
    alpha_sta: f32,
    alpha_lta: f32,
) -> f32 {
    let y = coeffs.step(s_in, &mut state.d1, &mut state.d2);
    let abs_y = y.abs();

    if state.impulse_counter < 100 {
        state.impulse_counter += 1;
        let count = state.impulse_counter as f32;
        let alpha = 1.0 / count;
        if state.impulse_counter == 1 {
            state.sta = abs_y;
            state.lta = abs_y.max(1e-4);
        } else {
            state.lta = (1.0 - alpha) * state.lta + alpha * abs_y.max(1e-4);
            state.sta = state.lta;
        }
        state.ratio = 1.0;
    } else {
        state.impulse_counter += 1;
        state.sta = alpha_sta * state.sta + (1.0 - alpha_sta) * abs_y;
        state.lta = alpha_lta * state.lta + (1.0 - alpha_lta) * abs_y;
        state.ratio = state.sta / (state.lta + 1e-6);
    }
    y
}

pub fn process_channel_sample(
    raw_sample: f32,
    coeffs: &BiquadCoeffs,
    state: &mut StationChannelState,
    is_quarantined: bool,
) -> PickEvent {
    if !raw_sample.is_finite() {
        return PickEvent {
            classification: CLASSIFICATION_NOMINAL,
            sample_val: 0.0,
            snr: 1.0,
            amplitude: 0.0,
            period_s: 0.2,
        };
    }

    let fs = if state.sample_rate_hz > 0.0 { state.sample_rate_hz } else { 50.0 };
    let dt = 1.0 / fs;
    let alpha_sta = (-dt / 1.0).exp();
    let alpha_lta = (-dt / 20.0).exp();

    let is_primed = state.impulse_counter >= 100;
    let base_threshold: f32 = 3.5;
    let effective_threshold = if is_quarantined {
        base_threshold * 2.2
    } else {
        base_threshold
    };

    let mut event = PickEvent {
        classification: CLASSIFICATION_NOMINAL,
        sample_val: 0.0,
        snr: state.ratio,
        amplitude: 0.0,
        period_s: 0.2,
    };

    if state.gate_state == CLASSIFICATION_PENDING_IMPULSE {
        let v_prev = state.prev_sample;
        let v_pend = state.pending_val;
        let v_curr = raw_sample;

        let imorse = morse_impulse(v_prev, v_pend, v_curr);
        let return_diff = (v_curr - v_prev).abs();
        let step_diff = (v_curr - v_pend).abs();
        let spike_mag = (v_pend - v_prev).abs();

        if (return_diff <= 0.1 * spike_mag || return_diff <= 0.5) && imorse > 0.0 {
            // Glint rejected: do NOT push glitch spike into biquad
            state.gate_state = CLASSIFICATION_NOMINAL;
            event.classification = CLASSIFICATION_GLINT_REJECTED;
            let y = step_biquad_and_sta_lta(raw_sample, coeffs, state, alpha_sta, alpha_lta);
            event.sample_val = y;
            event.snr = state.ratio;
            event.amplitude = y.abs();
        } else if step_diff <= 0.5 * spike_mag || state.ratio > effective_threshold {
            let y_pend = step_biquad_and_sta_lta(v_pend, coeffs, state, alpha_sta, alpha_lta);
            let y_curr = step_biquad_and_sta_lta(v_curr, coeffs, state, alpha_sta, alpha_lta);
            state.gate_state = CLASSIFICATION_NOMINAL;
            event.classification = CLASSIFICATION_P_WAVE_ONSET;
            event.sample_val = y_pend;
            event.snr = state.ratio;
            event.amplitude = y_pend.abs().max(y_curr.abs());
        } else {
            state.gate_state = CLASSIFICATION_NOMINAL;
            let _ = step_biquad_and_sta_lta(v_pend, coeffs, state, alpha_sta, alpha_lta);
            let y = step_biquad_and_sta_lta(v_curr, coeffs, state, alpha_sta, alpha_lta);
            event.classification = CLASSIFICATION_NOMINAL;
            event.sample_val = y;
            event.snr = state.ratio;
            event.amplitude = y.abs();
        }
    } else {
        let delta = (raw_sample - state.last_sample).abs();
        let noise_std = state.noise_var.sqrt().max(1e-4);

        if is_primed && (delta > 4.0 * noise_std || state.ratio > effective_threshold) {
            state.gate_state = CLASSIFICATION_PENDING_IMPULSE;
            state.pending_val = raw_sample;
            event.classification = CLASSIFICATION_PENDING_IMPULSE;
            event.sample_val = raw_sample;
            event.snr = state.ratio;
        } else {
            let y = step_biquad_and_sta_lta(raw_sample, coeffs, state, alpha_sta, alpha_lta);
            state.noise_var = 0.995 * state.noise_var + 0.005 * (raw_sample * raw_sample);
            event.classification = CLASSIFICATION_NOMINAL;
            event.sample_val = y;
            event.snr = state.ratio;
            event.amplitude = y.abs();
        }
    }

    state.prev_sample = state.last_sample;
    state.last_sample = raw_sample;

    event
}

/// Compute cyclic autocorrelation over the waveform buffer to split
/// stationary microseisms (H_clock) from transient arrivals (H_nope).
pub fn check_microseism_autocorrelation(
    waveforms: &[f32],
    station_idx: usize,
    band_idx: usize,
    head: usize,
    capacity: usize,
) -> f32 {
    let offset = (station_idx * NUM_BANDS + band_idx) * capacity;
    let window_len = 128.min(capacity);

    // Compute energy
    let mut energy = 0.0f32;
    for i in 0..window_len {
        let idx = (head + capacity - i) % capacity;
        let s = waveforms[offset + idx];
        energy += s * s;
    }
    if energy < 1e-8 {
        return 0.0;
    }

    // Test lags k in [10, 40] (approx 0.25 - 1.0 s periods)
    let mut max_corr = 0.0f32;
    for lag in 10..40 {
        let mut dot = 0.0f32;
        for i in 0..window_len {
            let idx0 = (head + capacity - i) % capacity;
            let idx1 = (head + capacity - i - lag) % capacity;
            dot += waveforms[offset + idx0] * waveforms[offset + idx1];
        }
        let r_k = dot / energy;
        if r_k > max_corr {
            max_corr = r_k;
        }
    }

    max_corr
}
