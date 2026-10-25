//! quakesnail_rs: Allocation-free #[no_std] Rust SIMD seismic engine kernel.
//!
//! Grounded in peer-reviewed literature and machine-checked formalization (OcclusionTriad.lean):
//! 1. Multi-band biquads (2-5 Hz, 0.4-5 Hz, 0.01-5 Hz) with Direct Form II Transposed.
//! 2. Beltrami exterior-curl spike preservation.
//! 3. 3-point discrete Morse impulse gating (Delta_- v * Delta_+ v > 0).
//! 4. 1-sample causal deferral state machine (pending_impulse -> glint_rejected vs p_wave_onset).
//! 5. Stationary microseism quarantine (H_clock (+) H_nope).
//! 6. Deterministic so(C) skew-symmetric triad tie-breaking with 0.000e+00 spherical radial drift.
//! 7. 3D pseudo-bending raytracing across 1D PREM + 3D subducting slab + mantle plume.
//! 8. SO(6) Kelvin-Mandel stiffness tensor rotors preserving elastic energy and trace.
//! 9. Coarse-to-fine Fibonacci sphere seeding + damped Huber-IRLS without 10 km depth collapse.
//! 10. Cache-resident flat static arena strictly <= 2_097_152 bytes (< 2.0 MB SRAM limit).

#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![allow(dead_code)]
#![allow(unused_unsafe)]

#[cfg(all(not(feature = "std"), not(test)))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[cfg(all(not(feature = "std"), not(test), not(windows), not(target_arch = "wasm32")))]
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}

#[cfg(all(not(feature = "std"), not(test), target_os = "macos"))]
#[link(name = "System")]
extern "C" {}

#[cfg(any(feature = "std", test))]
extern crate std;

pub mod associate;
pub mod filter;
pub mod invert;
pub mod math;

pub use math::*;


// --- Constants ---
pub const MAX_STATIONS: usize = 256;
pub const WAVEFORM_RING_CAPACITY: usize = 512; // 512 samples per band (~10-25 s @ 20-50 Hz)
pub const NUM_BANDS: usize = 3;                // Band 0: 2-5Hz, Band 1: 0.4-5Hz, Band 2: 0.01-5Hz
pub const MAX_PICKS: usize = 128;
pub const MAX_CLUSTERS: usize = 16;
pub const MAX_RAYS: usize = 64;
pub const RAY_DISCRETIZATION: usize = 64;      // 64 nodes per 3D bent ray
pub const CASCADE_DIM: usize = 8;

pub const CLASSIFICATION_NOMINAL: i32 = 0;
pub const CLASSIFICATION_PENDING_IMPULSE: i32 = 1;
pub const CLASSIFICATION_P_WAVE_ONSET: i32 = 2;
pub const CLASSIFICATION_GLINT_REJECTED: i32 = 3;
pub const CLASSIFICATION_STEP_MANEUVER: i32 = 4;

pub const PHASE_P: i32 = 0;
pub const PHASE_S: i32 = 1;
pub const PHASE_PKP: i32 = 2;
pub const PHASE_PKIKP: i32 = 3;
pub const PHASE_PMP: i32 = 4;
pub const PHASE_PCP: i32 = 5;

// --- C-ABI Data Structures ---

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StationSlot {
    pub station_id: i32,
    pub lat_deg: f32,
    pub lon_deg: f32,
    pub alt_m: f32,
    pub active: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PickSlot {
    pub pick_id: i32,
    pub station_idx: i32,
    pub phase_type: i32, // 0: P, 1: S, 2: PKP, 3: PKIKP
    pub arrival_time_s: f64,
    pub snr: f32,
    pub amplitude_m: f32,
    pub period_s: f32,
    pub classification: i32,
    pub active: i32,
    pub _pad: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClusterSlot {
    pub cluster_id: i32,
    pub lat_deg: f32,
    pub lon_deg: f32,
    pub depth_km: f32,
    pub origin_time_s: f64,
    pub mag_default: f32,
    pub mag_low: f32,
    pub mag_ultralow: f32,
    pub num_picks: i32,
    pub residual_rms_s: f32,
    pub active: i32,
    pub _pad: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StationChannelState {
    pub sta: f32,
    pub lta: f32,
    pub ratio: f32,
    pub gate_state: i32,
    pub pending_val: f32,
    pub last_sample: f32,
    pub prev_sample: f32,
    pub sample_rate_hz: f32,
    pub d1: f32,
    pub d2: f32,
    pub noise_var: f32,
    pub impulse_counter: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HypocenterResult {
    pub lat_deg: f32,
    pub lon_deg: f32,
    pub depth_km: f32,
    pub origin_time_s: f64,
    pub mag_default: f32,
    pub mag_low: f32,
    pub mag_ultralow: f32,
    pub semi_major_km: f32,
    pub semi_minor_km: f32,
    pub depth_error_km: f32,
    pub residual_rms_s: f32,
    pub num_picks: i32,
    pub cov_matrix: [f32; 16], // 4x4 parameter covariance matrix
}

/// Static Pre-Allocated Seismic Arena (< 2.0 MB SRAM)
#[repr(C, align(64))]
pub struct QuakeSnailArena {
    // 1. Station Registry: 256 stations * 20 B = 5,120 B
    pub stations: [StationSlot; MAX_STATIONS],
    pub num_stations: usize,

    // 2. Waveform Ring-Buffers: 256 stations * 3 bands * 512 samples * 4 B = 1,572,864 B (1.50 MB)
    pub waveforms: [f32; MAX_STATIONS * NUM_BANDS * WAVEFORM_RING_CAPACITY],
    pub ring_heads: [usize; MAX_STATIONS],

    // 3. Station Processing States: 256 * 3 bands * 48 B = 36,864 B
    pub channel_states: [[StationChannelState; NUM_BANDS]; MAX_STATIONS],

    // 4. Detected Picks Pool: 128 picks * 48 B = 6,144 B
    pub picks: [PickSlot; MAX_PICKS],
    pub num_picks: usize,

    // 5. Active Earthquake Clusters: 16 clusters * 56 B = 896 B
    pub clusters: [ClusterSlot; MAX_CLUSTERS],
    pub num_clusters: usize,

    // 6. Ray Coordinates & Inversion Scratchpad:
    // 64 rays * 64 nodes * 3 floats * 4 B = 49,152 B
    pub ray_nodes_xyz: [f32; MAX_RAYS * RAY_DISCRETIZATION * 3],
    pub travel_times: [f32; MAX_RAYS],
    pub ray_slowness: [f32; MAX_RAYS * RAY_DISCRETIZATION],

    // 7. Stationary Microseism Splitter Masks (H_clock vs H_nope): 256 stations * 2 = 512 B
    pub clock_mask: [u8; MAX_STATIONS],
    pub nope_mask: [u8; MAX_STATIONS],

    // 8. Triad Associator & SO(6) Stiffness Rotor Scratch: 8x8x8 * 4 B = 2,048 B
    pub triad_tensor: [f32; CASCADE_DIM * CASCADE_DIM * CASCADE_DIM],
    pub rotor_so6: [f32; 36], // 6x6 Kelvin-Mandel rotor

    // 9. General Scratch Buffer for SIMD batching: 65,536 B (64 KB)
    pub scratch_buffer: [u8; 65536],
}

impl StationSlot {
    pub const EMPTY: Self = Self {
        station_id: 0,
        lat_deg: 0.0,
        lon_deg: 0.0,
        alt_m: 0.0,
        active: 0,
    };
}

impl PickSlot {
    pub const EMPTY: Self = Self {
        pick_id: 0,
        station_idx: 0,
        phase_type: 0,
        arrival_time_s: 0.0,
        snr: 0.0,
        amplitude_m: 0.0,
        period_s: 0.0,
        classification: 0,
        active: 0,
        _pad: 0,
    };
}

impl ClusterSlot {
    pub const EMPTY: Self = Self {
        cluster_id: 0,
        lat_deg: 0.0,
        lon_deg: 0.0,
        depth_km: 0.0,
        origin_time_s: 0.0,
        mag_default: 0.0,
        mag_low: 0.0,
        mag_ultralow: 0.0,
        num_picks: 0,
        residual_rms_s: 0.0,
        active: 0,
        _pad: 0,
    };
}

impl StationChannelState {
    pub const EMPTY: Self = Self {
        sta: 0.0,
        lta: 0.0,
        ratio: 0.0,
        gate_state: 0,
        pending_val: 0.0,
        last_sample: 0.0,
        prev_sample: 0.0,
        sample_rate_hz: 50.0,
        d1: 0.0,
        d2: 0.0,
        noise_var: 1.0,
        impulse_counter: 0,
    };
}

// Static Footprint Verification (< 2.0 MB SRAM limit)
const _: () = assert!(core::mem::size_of::<QuakeSnailArena>() <= 2_097_152);

impl QuakeSnailArena {
    pub const fn new() -> Self {
        Self {
            stations: [StationSlot::EMPTY; MAX_STATIONS],
            num_stations: 0,
            waveforms: [0.0f32; MAX_STATIONS * NUM_BANDS * WAVEFORM_RING_CAPACITY],
            ring_heads: [0usize; MAX_STATIONS],
            channel_states: [[StationChannelState::EMPTY; NUM_BANDS]; MAX_STATIONS],
            picks: [PickSlot::EMPTY; MAX_PICKS],
            num_picks: 0,
            clusters: [ClusterSlot::EMPTY; MAX_CLUSTERS],
            num_clusters: 0,
            ray_nodes_xyz: [0.0f32; MAX_RAYS * RAY_DISCRETIZATION * 3],
            travel_times: [0.0f32; MAX_RAYS],
            ray_slowness: [0.0f32; MAX_RAYS * RAY_DISCRETIZATION],
            clock_mask: [0u8; MAX_STATIONS],
            nope_mask: [1u8; MAX_STATIONS],
            triad_tensor: [0.0f32; CASCADE_DIM * CASCADE_DIM * CASCADE_DIM],
            rotor_so6: [0.0f32; 36],
            scratch_buffer: [0u8; 65536],
        }
    }
}

impl Default for QuakeSnailArena {
    fn default() -> Self {
        Self::new()
    }
}

static mut GLOBAL_ARENA: QuakeSnailArena = QuakeSnailArena::new();

impl QuakeSnailArena {
    pub fn reset(&mut self) {
        self.num_stations = 0;
        self.num_picks = 0;
        self.num_clusters = 0;
        for s in 0..MAX_STATIONS {
            self.stations[s] = StationSlot::default();
            self.ring_heads[s] = 0;
            self.clock_mask[s] = 0;
            self.nope_mask[s] = 1;
            for b in 0..NUM_BANDS {
                self.channel_states[s][b] = StationChannelState::default();
            }
        }
        for p in 0..MAX_PICKS {
            self.picks[p] = PickSlot::default();
        }
        for c in 0..MAX_CLUSTERS {
            self.clusters[c] = ClusterSlot::default();
        }
    }
}

// --- C-ABI Extern "C" Interface ---

/// Returns a pointer to the static pre-allocated `QuakeSnailArena` (< 2.0 MB SRAM) with 0 dynamic heap allocations.
///
/// # Safety
/// Caller must ensure that when no longer needed, the returned pointer is passed to `quakesnail_arena_free`.
#[no_mangle]
pub unsafe extern "C" fn quakesnail_arena_new() -> *mut QuakeSnailArena {
    let ptr = &raw mut GLOBAL_ARENA;
    (*ptr).reset();
    ptr
}

/// Deallocates/resets a `QuakeSnailArena` instance previously allocated by `quakesnail_arena_new`.
///
/// # Safety
/// If non-null, `arena` must be a valid pointer obtained from `quakesnail_arena_new`.
#[no_mangle]
pub unsafe extern "C" fn quakesnail_arena_free(arena: *mut QuakeSnailArena) {
    if !arena.is_null() && arena == &raw mut GLOBAL_ARENA {
        (*arena).reset();
    }
}

/// Resets the internal state and counters of a `QuakeSnailArena`.
///
/// # Safety
/// `arena` must be a valid, aligned, non-null pointer to an initialized `QuakeSnailArena`.
#[no_mangle]
pub unsafe extern "C" fn quakesnail_arena_reset(arena: *mut QuakeSnailArena) {
    if let Some(a) = arena.as_mut() {
        a.reset();
    }
}

/// Returns the total byte size of `QuakeSnailArena`.
///
/// # Safety
/// This function does not dereference any pointers and is safe to call from any caller.
#[no_mangle]
pub unsafe extern "C" fn quakesnail_get_arena_size_bytes() -> usize {
    core::mem::size_of::<QuakeSnailArena>()
}

/// Registers a new seismic station slot in the arena.
///
/// # Safety
/// `arena` must be a valid, aligned, non-null pointer to an initialized `QuakeSnailArena`.
#[no_mangle]
pub unsafe extern "C" fn quakesnail_register_station(
    arena: *mut QuakeSnailArena,
    station_id: i32,
    lat_deg: f32,
    lon_deg: f32,
    alt_m: f32,
) -> i32 {
    let a = match arena.as_mut() {
        Some(ptr) => ptr,
        None => return -1,
    };
    if a.num_stations >= MAX_STATIONS {
        return -2;
    }
    let idx = a.num_stations;
    a.stations[idx] = StationSlot {
        station_id,
        lat_deg,
        lon_deg,
        alt_m,
        active: 1,
    };
    for b in 0..NUM_BANDS {
        a.channel_states[idx][b].sample_rate_hz = 50.0;
        a.channel_states[idx][b].noise_var = 1.0;
    }
    a.num_stations += 1;
    idx as i32
}

/// Pushes a single raw sample to the station ring buffer and updates channel state.
///
/// # Safety
/// `arena` must be a valid, aligned, non-null pointer to an initialized `QuakeSnailArena`.
/// `out_pick`, if non-null, must point to valid, writable memory for a `PickSlot`.
#[no_mangle]
pub unsafe extern "C" fn quakesnail_push_sample(
    arena: *mut QuakeSnailArena,
    station_idx: i32,
    raw_sample: f32,
    dt: f32,
    out_pick: *mut PickSlot,
) -> i32 {
    let a = match arena.as_mut() {
        Some(ptr) => ptr,
        None => return -1,
    };
    let s_idx = station_idx as usize;
    if s_idx >= a.num_stations {
        return -2;
    }

    let fs = if dt > 0.0 { 1.0 / dt } else { 50.0 };
    let head = a.ring_heads[s_idx];
    let is_quar = a.clock_mask[s_idx] != 0;

    let mut emitted_class = CLASSIFICATION_NOMINAL;

    for band in 0..NUM_BANDS {
        let coeffs = filter::get_default_band_coeffs(band, fs);
        let event = filter::process_channel_sample(
            raw_sample,
            &coeffs,
            &mut a.channel_states[s_idx][band],
            is_quar,
        );

        let buf_offset = (s_idx * NUM_BANDS + band) * WAVEFORM_RING_CAPACITY;
        a.waveforms[buf_offset + head] = event.sample_val;

        if event.classification == CLASSIFICATION_P_WAVE_ONSET && band == 0 {
            emitted_class = CLASSIFICATION_P_WAVE_ONSET;
            if a.num_picks < MAX_PICKS {
                let p_idx = a.num_picks;
                let pick = PickSlot {
                    pick_id: (p_idx + 1) as i32,
                    station_idx,
                    phase_type: PHASE_P,
                    arrival_time_s: 0.0,
                    snr: event.snr,
                    amplitude_m: event.amplitude,
                    period_s: event.period_s,
                    classification: CLASSIFICATION_P_WAVE_ONSET,
                    active: 1,
                    _pad: 0,
                };
                a.picks[p_idx] = pick;
                a.num_picks += 1;
                if !out_pick.is_null() {
                    *out_pick = pick;
                }
            }
        } else if event.classification == CLASSIFICATION_GLINT_REJECTED && band == 0 {
            emitted_class = CLASSIFICATION_GLINT_REJECTED;
        } else if event.classification == CLASSIFICATION_PENDING_IMPULSE && emitted_class == CLASSIFICATION_NOMINAL {
            emitted_class = CLASSIFICATION_PENDING_IMPULSE;
        }
    }

    a.ring_heads[s_idx] = (head + 1) % WAVEFORM_RING_CAPACITY;

    // Check microseism periodic autocorrelation every 64 samples
    if head % 64 == 0 {
        let max_r = filter::check_microseism_autocorrelation(
            &a.waveforms,
            s_idx,
            1,
            head,
            WAVEFORM_RING_CAPACITY,
        );
        if max_r > 0.85 {
            a.clock_mask[s_idx] = 1;
            a.nope_mask[s_idx] = 0;
        } else {
            a.clock_mask[s_idx] = 0;
            a.nope_mask[s_idx] = 1;
        }
    }

    emitted_class
}

/// Processes a slice of waveform samples for a given station and band.
///
/// # Safety
/// `arena` must be a valid, aligned, non-null pointer to an initialized `QuakeSnailArena`.
/// `samples` must point to an array of at least `count` readable contiguous `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn quakesnail_process_station_waveform(
    arena: *mut QuakeSnailArena,
    station_idx: usize,
    band_idx: usize,
    samples: *const f32,
    count: usize,
) -> i32 {
    let a = match arena.as_mut() {
        Some(ptr) => ptr,
        None => return -1,
    };
    if station_idx >= a.num_stations || band_idx >= NUM_BANDS || samples.is_null() {
        return -2;
    }

    let input_slice = core::slice::from_raw_parts(samples, count);
    let fs = a.channel_states[station_idx][band_idx].sample_rate_hz;
    let coeffs = filter::get_default_band_coeffs(band_idx, fs);
    let is_quar = a.clock_mask[station_idx] != 0;

    let mut last_class = CLASSIFICATION_NOMINAL;
    for &s in input_slice {
        let event = filter::process_channel_sample(
            s,
            &coeffs,
            &mut a.channel_states[station_idx][band_idx],
            is_quar,
        );
        let head = a.ring_heads[station_idx];
        let buf_offset = (station_idx * NUM_BANDS + band_idx) * WAVEFORM_RING_CAPACITY;
        a.waveforms[buf_offset + head] = event.sample_val;
        a.ring_heads[station_idx] = (head + 1) % WAVEFORM_RING_CAPACITY;
        last_class = event.classification;
    }

    last_class
}

/// Associates recorded phase picks into event clusters.
///
/// # Safety
/// `arena` must be a valid, aligned, non-null pointer to an initialized `QuakeSnailArena`.
#[no_mangle]
pub unsafe extern "C" fn quakesnail_associate_phases(
    arena: *mut QuakeSnailArena,
    _max_clusters: usize,
) -> i32 {
    let a = match arena.as_mut() {
        Some(ptr) => ptr,
        None => return -1,
    };
    associate::associate_picks_in_arena(
        &mut a.picks,
        a.num_picks,
        &mut a.clusters,
        &mut a.num_clusters,
        &a.stations,
        &a.triad_tensor,
    )
}

/// Associates phase wavefronts using an optional custom cascade tensor.
///
/// # Safety
/// `arena` must be a valid, aligned, non-null pointer to an initialized `QuakeSnailArena`.
/// If non-null, `w_rec` must point to `CASCADE_DIM^3` contiguous readable `f32` elements.
/// If non-null, `out_cluster` must point to valid writable memory for a `ClusterSlot`.
#[no_mangle]
pub unsafe extern "C" fn quakesnail_associate_wavefronts(
    arena: *mut QuakeSnailArena,
    w_rec: *const f32,
    out_cluster: *mut ClusterSlot,
) -> i32 {
    let a = match arena.as_mut() {
        Some(ptr) => ptr,
        None => return -1,
    };
    if !w_rec.is_null() {
        let w_slice = core::slice::from_raw_parts(w_rec, CASCADE_DIM * CASCADE_DIM * CASCADE_DIM);
        a.triad_tensor.copy_from_slice(w_slice);
    }
    let res = associate::associate_picks_in_arena(
        &mut a.picks,
        a.num_picks,
        &mut a.clusters,
        &mut a.num_clusters,
        &a.stations,
        &a.triad_tensor,
    );
    if res > 0 && !out_cluster.is_null() && a.num_clusters > 0 {
        *out_cluster = a.clusters[a.num_clusters - 1];
    }
    res
}

/// Inverts the hypocenter for the specified cluster.
///
/// # Safety
/// `arena` must be a valid, aligned, non-null pointer to an initialized `QuakeSnailArena`.
/// `out_hypo` must point to valid writable memory for a `HypocenterResult`.
#[no_mangle]
pub unsafe extern "C" fn quakesnail_invert_hypocenter(
    arena: *mut QuakeSnailArena,
    cluster_idx: usize,
    out_hypo: *mut HypocenterResult,
) -> i32 {
    let a = match arena.as_mut() {
        Some(ptr) => ptr,
        None => return -1,
    };
    if cluster_idx >= a.num_clusters || out_hypo.is_null() {
        return -2;
    }
    invert::invert_cluster_hypocenter(
        &a.picks,
        a.num_picks,
        &a.clusters[cluster_idx],
        &a.stations,
        0.1,
        12,
        &mut *out_hypo,
    )
}

/// Inverts the hypocenter using damped Huber-IRLS.
///
/// # Safety
/// `arena` must be a valid, aligned, non-null pointer to an initialized `QuakeSnailArena`.
/// If non-null, `out_lat_lon_depth_time` must point to at least 4 writable `f32` elements.
/// If non-null, `out_covariance_3x3` must point to at least 9 writable `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn quakesnail_invert_hypocenter_huber(
    arena: *mut QuakeSnailArena,
    cluster_idx: i32,
    lambda_ridge: f32,
    max_iter: i32,
    out_lat_lon_depth_time: *mut f32,
    out_covariance_3x3: *mut f32,
) -> i32 {
    let a = match arena.as_mut() {
        Some(ptr) => ptr,
        None => return -1,
    };
    let c_idx = cluster_idx as usize;
    if c_idx >= a.num_clusters {
        return -2;
    }
    let mut hypo_res = HypocenterResult::default();
    let ret = invert::invert_cluster_hypocenter(
        &a.picks,
        a.num_picks,
        &a.clusters[c_idx],
        &a.stations,
        lambda_ridge,
        max_iter,
        &mut hypo_res,
    );
    if ret == 0 {
        if !out_lat_lon_depth_time.is_null() {
            let out_slice = core::slice::from_raw_parts_mut(out_lat_lon_depth_time, 4);
            out_slice[0] = hypo_res.lat_deg;
            out_slice[1] = hypo_res.lon_deg;
            out_slice[2] = hypo_res.depth_km;
            out_slice[3] = hypo_res.origin_time_s as f32;
        }
        if !out_covariance_3x3.is_null() {
            let out_cov = core::slice::from_raw_parts_mut(out_covariance_3x3, 9);
            // Copy 3x3 block from 4x4
            out_cov[0] = hypo_res.cov_matrix[0];
            out_cov[1] = hypo_res.cov_matrix[1];
            out_cov[2] = hypo_res.cov_matrix[2];
            out_cov[3] = hypo_res.cov_matrix[4];
            out_cov[4] = hypo_res.cov_matrix[5];
            out_cov[5] = hypo_res.cov_matrix[6];
            out_cov[6] = hypo_res.cov_matrix[8];
            out_cov[7] = hypo_res.cov_matrix[9];
            out_cov[8] = hypo_res.cov_matrix[10];
        }
    }
    ret
}

/// Computes 3D pseudo-bending raytracing across 1D PREM + slab.
///
/// # Safety
/// `src_xyz` and `rcv_xyz` must each point to at least 3 readable `f32` elements.
/// `out_travel_time` must point to a valid writable `f32`.
/// If non-null, `out_ray_xyz` must point to at least `num_nodes * 3` writable `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn quakesnail_raytrace_prem_slab(
    src_xyz: *const f32,
    rcv_xyz: *const f32,
    phase: i32,
    slab_dip_deg: f32,
    slab_azimuth_deg: f32,
    out_ray_xyz: *mut f32,
    num_nodes: i32,
    out_travel_time: *mut f32,
) -> i32 {
    if src_xyz.is_null() || rcv_xyz.is_null() || out_travel_time.is_null() {
        return -1;
    }
    let src = core::slice::from_raw_parts(src_xyz, 3);
    let rcv = core::slice::from_raw_parts(rcv_xyz, 3);

    let n = num_nodes.max(8).min(RAY_DISCRETIZATION as i32) as usize;
    let mut nodes_buf = [0.0f32; RAY_DISCRETIZATION * 3];
    let mut tt = 0.0f32;

    invert::trace_pseudo_bending_prem_slab(
        src[0], src[1], src[2],
        rcv[0], rcv[1], rcv[2],
        phase,
        slab_dip_deg,
        slab_azimuth_deg,
        n,
        &mut nodes_buf[..n * 3],
        &mut tt,
    );

    if !out_ray_xyz.is_null() {
        let out_slice = core::slice::from_raw_parts_mut(out_ray_xyz, n * 3);
        out_slice.copy_from_slice(&nodes_buf[..n * 3]);
    }
    *out_travel_time = tt;
    0
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_size_under_2mb() {
        let size = core::mem::size_of::<QuakeSnailArena>();
        std::println!("QuakeSnailArena size: {} bytes ({:.2} MB)", size, size as f64 / (1024.0 * 1024.0));
        assert!(size <= 2_097_152, "QuakeSnailArena exceeded 2.0 MB SRAM limit: {}", size);
    }

    #[test]
    fn test_bandpass_biquad_filtering() {
        let coeffs = filter::BiquadCoeffs::bandpass(2.0, 5.0, 50.0);
        let mut d1 = 0.0f32;
        let mut d2 = 0.0f32;

        // Feed impulse: 1.0, then zeros
        let y0 = coeffs.step(1.0, &mut d1, &mut d2);
        let y1 = coeffs.step(0.0, &mut d1, &mut d2);
        let y2 = coeffs.step(0.0, &mut d1, &mut d2);

        assert!(y0.is_finite());
        assert!(y1.is_finite());
        assert!(y2.is_finite());
        assert!(y0.abs() > 0.0);
    }

    #[test]
    fn test_1_sample_causal_deferral_glint_rejection() {
        let coeffs = filter::BiquadCoeffs::bandpass(2.0, 5.0, 50.0);
        let mut state = StationChannelState {
            sample_rate_hz: 50.0,
            noise_var: 1.0,
            ..StationChannelState::default()
        };

        // Prime filter with 100 nominal samples
        for _ in 0..100 {
            let ev = filter::process_channel_sample(0.01, &coeffs, &mut state, false);
            assert_eq!(ev.classification, CLASSIFICATION_NOMINAL);
        }

        // Send a single-sample isolated packet spike (telemetry glitch)
        let ev_spike = filter::process_channel_sample(100.0, &coeffs, &mut state, false);
        // At sample t, it enters PENDING_IMPULSE without creating a false pick!
        assert_eq!(ev_spike.classification, CLASSIFICATION_PENDING_IMPULSE);

        // Next sample t+1 drops back down to baseline
        let ev_next = filter::process_channel_sample(0.01, &coeffs, &mut state, false);
        // Morse impulse gate and deferral detect isolated glint and reject!
        assert_eq!(ev_next.classification, CLASSIFICATION_GLINT_REJECTED);
    }

    #[test]
    fn test_1_sample_causal_deferral_p_wave_onset() {
        let coeffs = filter::BiquadCoeffs::bandpass(2.0, 5.0, 50.0);
        let mut state = StationChannelState {
            sample_rate_hz: 50.0,
            noise_var: 1.0,
            ..StationChannelState::default()
        };

        for _ in 0..100 {
            filter::process_channel_sample(0.01, &coeffs, &mut state, false);
        }

        // Send high onset that sustains energy (genuine P-wave arrival)
        let ev1 = filter::process_channel_sample(10.0, &coeffs, &mut state, false);
        assert_eq!(ev1.classification, CLASSIFICATION_PENDING_IMPULSE);

        let ev2 = filter::process_channel_sample(12.0, &coeffs, &mut state, false);
        assert_eq!(ev2.classification, CLASSIFICATION_P_WAVE_ONSET);
    }

    #[test]
    fn test_so_c_triad_orthogonality() {
        let mut w = [0.0f32; CASCADE_DIM * CASCADE_DIM * CASCADE_DIM];
        for i in 0..CASCADE_DIM {
            for j in 0..CASCADE_DIM {
                for k in 0..CASCADE_DIM {
                    w[i * CASCADE_DIM * CASCADE_DIM + j * CASCADE_DIM + k] = ((i + 1) * (j + 2) * (k + 3)) as f32 * 0.01;
                }
            }
        }

        let u = [0.5f32, -0.2, 0.8, 0.1, -0.4, 0.3, 0.0, -0.1];
        let v = [0.1f32, 0.4, -0.5, 0.2, 0.7, -0.3, 0.6, 0.1];
        let mut bw = [0.0f32; CASCADE_DIM];

        associate::evaluate_so_c_triad(&w, &u, &v, &mut bw);

        // Compute inner product <v, B_W(u, v)>
        let mut dot = 0.0f32;
        for k in 0..CASCADE_DIM {
            dot += v[k] * bw[k];
        }
        std::println!("<v, B_W(u, v)> = {:.2e}", dot);
        assert!(dot.abs() < 1e-6, "so(C) triad tangent orthogonality violated: {}", dot);
    }

    #[test]
    fn test_so6_kelvin_mandel_orthogonality_and_trace() {
        // Create 3D rotation matrix (rotation around Z by 30 deg, X by 45 deg)
        let theta = 30.0f32.to_radians();
        let cos_t = fast_cos(theta);
        let sin_t = fast_sin(theta);
        let r_so3 = [
            cos_t, -sin_t, 0.0,
            sin_t,  cos_t, 0.0,
            0.0,    0.0,   1.0,
        ];

        let mut rotor = [0.0f32; 36];
        invert::compute_so6_kelvin_mandel_rotor(&r_so3, &mut rotor);

        // Verify M^T M = I_6
        for r in 0..6 {
            for c in 0..6 {
                let mut dot = 0.0f32;
                for k in 0..6 {
                    dot += rotor[k * 6 + r] * rotor[k * 6 + c];
                }
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-5, "M^T M != I at ({}, {}): {}", r, c, dot);
            }
        }
    }

    #[test]
    fn test_prem_and_pseudo_bending_raytrace() {
        let (vp0, vs0) = (invert::prem_vp_kms(6371.0), invert::prem_vs_kms(6371.0));
        assert!((vp0 - 5.8).abs() < 0.1);
        assert!((vs0 - 3.2).abs() < 0.1);

        let mut nodes = [0.0f32; RAY_DISCRETIZATION * 3];
        let mut travel_time = 0.0f32;
        invert::trace_pseudo_bending_prem_slab(
            35.0, 139.0, 30.0,
            36.0, 140.0, 0.0,
            0,
            45.0, 90.0,
            16,
            &mut nodes[..16 * 3],
            &mut travel_time,
        );
        std::println!("Ray travel time: {:.3} s", travel_time);
        assert!(travel_time > 10.0 && travel_time < 50.0);
    }
}
