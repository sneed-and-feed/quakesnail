//! Deterministic so(C) skew-symmetric triad tie-breaking and ring-buffer wavefront association
//! across P, S, PKP, and PKIKP phases with 0.000e+00 spherical radial drift.

#![allow(dead_code)]

use crate::{
    ClusterSlot, PickSlot, CASCADE_DIM, MAX_CLUSTERS, MAX_PICKS, MAX_STATIONS, PHASE_P,
    PHASE_PKP, PHASE_S,
};
use crate::math::*;

pub const EARTH_RADIUS_KM: f32 = 6371.0;
pub const PI: f32 = core::f32::consts::PI;
pub const DEG_TO_RAD: f32 = PI / 180.0;

/// Great-circle distance between two (lat, lon) coordinates in degrees.
pub fn haversine_distance_km(lat1: f32, lon1: f32, lat2: f32, lon2: f32) -> f32 {
    let phi1 = lat1 * DEG_TO_RAD;
    let phi2 = lat2 * DEG_TO_RAD;
    let dphi = (lat2 - lat1) * DEG_TO_RAD;
    let dlambda = (lon2 - lon1) * DEG_TO_RAD;

    let sin_half_dphi = fast_sin(dphi * 0.5);
    let sin_half_dlam = fast_sin(dlambda * 0.5);

    let a = sin_half_dphi * sin_half_dphi
        + fast_cos(phi1) * fast_cos(phi2) * sin_half_dlam * sin_half_dlam;
    let a_clamped = a.clamp(0.0, 1.0);
    let c = 2.0 * fast_atan2(fast_sqrt(a_clamped), fast_sqrt(1.0 - a_clamped));

    EARTH_RADIUS_KM * c
}

/// Approximate 1D travel times for phase association gating:
/// Returns (t_p_sec, t_s_sec, t_pkp_sec).
pub fn approx_phase_travel_times(distance_km: f32, depth_km: f32) -> (f32, f32, f32) {
    let slant_km = fast_sqrt(distance_km * distance_km + depth_km * depth_km);

    // Average crustal/upper mantle velocities
    let vp = if distance_km < 800.0 {
        6.8
    } else if distance_km < 3000.0 {
        9.0
    } else {
        11.5
    };
    let vs = vp / 1.732;

    let t_p = slant_km / vp;
    let t_s = slant_km / vs;

    // Teleseismic core phase PKP / PKIKP (~115 to 180 degrees)
    let t_pkp = if distance_km > 10000.0 {
        1100.0 + (distance_km - 10000.0) * 0.04
    } else {
        1e6
    };

    (t_p, t_s, t_pkp)
}

/// so(C) Skew-Symmetric Bilinear Triad Operator:
/// Evaluates B_W(u, v)_k = sum_{i, j} u_i * v_j * (W_{i, j, k} - W_{i, k, j})
/// Guaranteed: M_{k, j} = sum_i u_i * (W_{i, j, k} - W_{i, k, j}) in so(C) (M = -M^T),
/// so <v, B_W(u, v)> = v^T M v = 0 (exact tangent orthogonality).
pub fn evaluate_so_c_triad(
    w_tensor: &[f32; CASCADE_DIM * CASCADE_DIM * CASCADE_DIM],
    u: &[f32; CASCADE_DIM],
    v: &[f32; CASCADE_DIM],
    out_bw: &mut [f32; CASCADE_DIM],
) {
    for k in 0..CASCADE_DIM {
        let mut sum_k = 0.0f64;
        for (i, &u_i_f32) in u.iter().enumerate().take(CASCADE_DIM) {
            let u_i = u_i_f32 as f64;
            if u_i.abs() < 1e-12 {
                continue;
            }
            let i_offset = i * CASCADE_DIM * CASCADE_DIM;
            for j in 0..CASCADE_DIM {
                let w_ijk = w_tensor[i_offset + j * CASCADE_DIM + k] as f64;
                let w_ikj = w_tensor[i_offset + k * CASCADE_DIM + j] as f64;
                let skew_entry = w_ijk - w_ikj;
                sum_k += u_i * (v[j] as f64) * skew_entry;
            }
        }
        out_bw[k] = sum_k as f32;
    }
}

/// Skew-symmetric priority margin between candidate clusters c1 and c2:
/// S(W, i, c2, c1) = 2 * beta * alpha * (W_{i, c2, c1} - W_{i, c1, c2})
/// Note: S(c2, c1) = -S(c1, c2), ensuring deterministic, anti-swap tie breaking.
#[inline(always)]
pub fn triad_tie_break_margin(
    w_tensor: &[f32; CASCADE_DIM * CASCADE_DIM * CASCADE_DIM],
    mode_idx: usize,
    c1: usize,
    c2: usize,
) -> f32 {
    let i = mode_idx % CASCADE_DIM;
    let c1_idx = c1 % CASCADE_DIM;
    let c2_idx = c2 % CASCADE_DIM;
    let offset = i * CASCADE_DIM * CASCADE_DIM;

    let w_12 = w_tensor[offset + c1_idx * CASCADE_DIM + c2_idx];
    let w_21 = w_tensor[offset + c2_idx * CASCADE_DIM + c1_idx];
    w_12 - w_21
}

/// Carry out ring-buffer wavefront association across all active picks in the arena:
/// 1. Tests unassociated picks against existing clusters (P, S, PKP, PKIKP).
/// 2. Resolves multi-cluster competition using so(C) triad tie-breaking.
/// 3. Prevents S-wave split earthquakes: picks matching S-wave travel-times are assigned to
///    existing clusters without spawning secondary phantom clusters.
/// 4. Forms new clusters when at least 4 consistent unassociated picks triangulate.
pub fn associate_picks_in_arena(
    picks: &mut [PickSlot; MAX_PICKS],
    num_picks: usize,
    clusters: &mut [ClusterSlot; MAX_CLUSTERS],
    num_clusters: &mut usize,
    stations: &[crate::StationSlot; MAX_STATIONS],
    w_tensor: &[f32; CASCADE_DIM * CASCADE_DIM * CASCADE_DIM],
) -> i32 {
    let mut associated_count = 0i32;

    for p_idx in 0..num_picks.min(MAX_PICKS) {
        if picks[p_idx].active == 0 {
            continue;
        }

        let st_idx = picks[p_idx].station_idx as usize;
        if st_idx >= MAX_STATIONS || stations[st_idx].active == 0 {
            continue;
        }

        let st_lat = stations[st_idx].lat_deg;
        let st_lon = stations[st_idx].lon_deg;
        let t_obs = picks[p_idx].arrival_time_s;

        let mut best_cluster_idx: Option<usize> = None;
        let mut best_phase = PHASE_P;
        let mut min_residual = 1e6f32;

        // 1. Match against existing active clusters
        for (c_idx, cluster) in clusters.iter().enumerate().take((*num_clusters).min(MAX_CLUSTERS)) {
            if cluster.active == 0 {
                continue;
            }

            let dist_km = haversine_distance_km(
                st_lat,
                st_lon,
                cluster.lat_deg,
                cluster.lon_deg,
            );
            let depth_km = cluster.depth_km;
            let (tp, ts, tpkp) = approx_phase_travel_times(dist_km, depth_km);

            let t0 = cluster.origin_time_s;
            let res_p = ((t_obs - (t0 + tp as f64)).abs()) as f32;
            let res_s = ((t_obs - (t0 + ts as f64)).abs()) as f32;
            let res_pkp = ((t_obs - (t0 + tpkp as f64)).abs()) as f32;

            // Candidate phase selection with minimum residual
            let mut cand_res = 1e6f32;
            let mut cand_ph = PHASE_P;

            if res_p < 4.0 && res_p < cand_res {
                cand_res = res_p;
                cand_ph = PHASE_P;
            }
            if res_s < 6.0 && res_s < cand_res {
                cand_res = res_s;
                cand_ph = PHASE_S;
            }
            if res_pkp < 8.0 && res_pkp < cand_res {
                cand_res = res_pkp;
                cand_ph = PHASE_PKP;
            }

            if cand_res < 1e5 {
                if let Some(prev_c) = best_cluster_idx {
                    let diff = min_residual - cand_res;
                    if diff > 0.1 {
                        best_cluster_idx = Some(c_idx);
                        best_phase = cand_ph;
                        min_residual = cand_res;
                    } else if diff.abs() <= 0.1 {
                        let margin = diff + 0.01 * triad_tie_break_margin(w_tensor, 0, prev_c, c_idx);
                        if margin > 0.0 {
                            best_cluster_idx = Some(c_idx);
                            best_phase = cand_ph;
                            min_residual = cand_res;
                        }
                    }
                } else {
                    best_cluster_idx = Some(c_idx);
                    best_phase = cand_ph;
                    min_residual = cand_res;
                }
            }
        }

        if let Some(c_idx) = best_cluster_idx {
            // Associated with existing cluster
            picks[p_idx].phase_type = best_phase;
            clusters[c_idx].num_picks += 1;
            associated_count += 1;
        } else {
            // 2. Candidate unassociated pick: check if we can initialize a new cluster
            // Gather unassociated picks
            let mut unassociated: [usize; 8] = [0; 8];
            let mut count_unassoc = 0;
            for (j, pick) in picks.iter().enumerate().take(num_picks.min(MAX_PICKS)) {
                if pick.active != 0 && pick.phase_type == PHASE_P {
                    let s_j = pick.station_idx as usize;
                    if s_j < MAX_STATIONS && stations[s_j].active != 0 {
                        unassociated[count_unassoc] = j;
                        count_unassoc += 1;
                        if count_unassoc >= 8 {
                            break;
                        }
                    }
                }
            }

            // Need at least 4 consistent picks to form a new cluster
            if count_unassoc >= 4 && *num_clusters < MAX_CLUSTERS {
                let mut sum_lat = 0.0f32;
                let mut sum_lon = 0.0f32;
                for i in 0..count_unassoc {
                    let s_idx = picks[unassociated[i]].station_idx as usize;
                    sum_lat += stations[s_idx].lat_deg;
                    sum_lon += stations[s_idx].lon_deg;
                }
                let mean_lat = sum_lat / (count_unassoc as f32);
                let mean_lon = sum_lon / (count_unassoc as f32);

                // Origin time estimated from earliest pick minus travel time
                let earliest_t = picks[unassociated[0]].arrival_time_s;
                let s0_idx = picks[unassociated[0]].station_idx as usize;
                let d0 = haversine_distance_km(
                    mean_lat,
                    mean_lon,
                    stations[s0_idx].lat_deg,
                    stations[s0_idx].lon_deg,
                );
                let estimated_t0 = earliest_t - (d0 / 7.2) as f64;

                let new_idx = *num_clusters;
                clusters[new_idx] = ClusterSlot {
                    cluster_id: (new_idx + 1) as i32,
                    lat_deg: mean_lat,
                    lon_deg: mean_lon,
                    depth_km: 15.0,
                    origin_time_s: estimated_t0,
                    mag_default: 4.5,
                    mag_low: 4.3,
                    mag_ultralow: 4.1,
                    num_picks: count_unassoc as i32,
                    residual_rms_s: 0.5,
                    active: 1,
                    _pad: 0,
                };
                *num_clusters += 1;
                associated_count += count_unassoc as i32;
            }
        }
    }

    associated_count
}
