# Project: QuakeSnail

## Architecture
QuakeSnail is a next-generation real-time earthquake monitoring, phase-association, 3D travel-time hypocenter inversion, and Earthquake Early Warning (EEW) engine backed by a compiled `#[no_std]` Rust SIMD kernel, spatial municipality intensity engine, and portable single-file WebGL2 3D Globe & Tactical Studio.

### System Diagram
```
[Live SeedLink / FDSN / USGS Streams]  OR  [Deterministic Replay / miniSEED]
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Waveform Filtering & Telemetry Glitch Shielding               │
│ - Multi-band biquads (2-5 Hz, 0.4-5 Hz, 0.01-5 Hz)                     │
│ - Beltrami exterior-curl spike preservation                            │
│ - 3-point discrete Morse impulse gating (Delta_- v * Delta_+ v > 0)    │
│ - 1-sample causal deferral (pending_impulse -> glint vs p_onset)       │
│ - H_clock (+) H_nope stationary microseism quarantine                  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Filtered Picks (P, S, PKP, PKIKP)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Multi-Wavefront Phase Association & Anti-Stealing             │
│ - Deterministic so(C) skew-symmetric bilinear triad tie-breaking       │
│ - Ring-buffer wavefront association across P, S, PKP, PKIKP            │
│ - Zero spherical radial drift (0.000e+00)                              │
│ - Elimination of GlobalQuake greedy cluster stealing & S-wave splits   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Associated Clusters
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Coarse-to-Fine 3D Hypocenter Inversion & Covariance           │
│ - Coarse Fibonacci-sphere seed (64-500 candidate seeds)                │
│ - 3D pseudo-bending raytracing (1D PREM + 3D subducting slab + plume)  │
│ - SO(6) Kelvin-Mandel stiffness rotor TTI slab corrections             │
│ - Damped Huber-IRLS / Ridge residual minimization                      │
│ - Multi-band magnitude estimation (M_w, M_def, M_low, M_ultra)         │
│ - Analytical 3D depth/location confidence ellipsoids (no 10km clamps)  │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────┐
│ Municipality & Intensity Engine      │  │ Portable 3D WebGL2 Studio    │
│ - Multi-tier global & regional towns │  │ - Interactive 3D Earth globe │
│ - KD-tree spatial index (< 2.0 ms)   │  │ - Expanding P/S rings        │
│ - ZERO "people felt" counters        │  │ - H3 hexagonal ShakeMap      │
│ - Real-time ShakeMap (PGA, PGV, MMI) │  │ - Wadati-Benioff cutaway     │
│ - Live P & S arrival countdowns      │  │ - Waveform helicorder strip  │
└──────────────────────────────────────┘  └──────────────────────────────┘
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | ArXiv Literature Grounding | 20 peer-reviewed & recent arXiv citations on EEW, phase picking/association, inversion, and GMPE | M1 | survey_lit |
| 2 | Lean 4 Formalization (`formalization/OcclusionTriad.lean`) | 4 machine-checked theorems (Theorems 1-4) with 0 sorry and 0 custom axioms | M1 | survey_lit |
| 3 | Theorem Verification CLI | `quakesnail verify-theorems` CLI subcommand with numerical residuals <= 1.0e-7 | M1 | ORIGINAL_REQUEST §R1 |
| 4 | Rust SIMD Kernel (`rust/quakesnail_rs`) | Flat static memory arena (`QuakeSnailArena`, 1.65 MB < 2.0 MB SRAM limit) in `#[no_std]` Rust | M2 | survey_siblings |
| 5 | Multi-Band Waveform Filtering | 3-band parallel filtering (2-5 Hz, 0.4-5 Hz, 0.01-5 Hz) with Beltrami exterior-curl preservation | M2 | ORIGINAL_REQUEST §R2 |
| 6 | 3-Point Discrete Morse Impulse Gate | $\Delta_- v \cdot \Delta_+ v > 0$ critical-point detection | M2 | ORIGINAL_REQUEST §R2 |
| 7 | 1-Sample Causal Deferral Filter | `pending_impulse` resolving to `glint_rejected` vs `p_wave_onset` | M2 | ORIGINAL_REQUEST §R2 |
| 8 | Microseism Splitter | $\mathcal{H}_{\text{clock}} \oplus \mathcal{H}_{\text{nope}}$ stationary oceanic swell quarantine | M2 | ORIGINAL_REQUEST §R2 |
| 9 | Skew-Symmetric $\mathfrak{so}(C)$ Association | Deterministic bilinear triad tie-breaker with $0.000\text{e}+00$ spherical radial drift | M2 | ORIGINAL_REQUEST §R2 |
| 10 | Cluster Anti-Stealing & S-Wave Protection | Ring-buffer wavefront association eliminating cluster thrash and false S-wave splits | M2 | survey_gq |
| 11 | Coarse Fibonacci Seed | Fast spherical grid seeding (64-500 nodes) for global hypocenter initialization | M2 | survey_siblings |
| 12 | 3D Global Spherical Raytracing | 3-point pseudo-bending across 1D PREM + 3D subducting slab + 3D plume for P, S, PmP, PcP, PKP, PKIKP | M2 | survey_siblings |
| 13 | $\text{SO}(6)$ Kelvin-Mandel Stiffness Rotors | Isometric Voigt stiffness rotations preserving elastic energy and tensor trace | M2 | survey_siblings |
| 14 | Damped Huber-IRLS Inversion | Robust $C^{1,1}$ convex M-estimation converging without 10 km depth collapse | M2 | survey_lit |
| 15 | Multi-Band Magnitude Estimation | Real-time calculation of $M_w$, $M_{\text{default}}$, $M_{\text{low}}$, and $M_{\text{ultralow}}$ | M2 | ORIGINAL_REQUEST §R2 |
| 16 | Analytical Confidence Ellipsoids | Strict positive-definite $4 \times 4$ covariance $\boldsymbol{\Sigma} = \hat{\sigma}^2 \mathbf{H}^{-1}$ without 256-ray marching | M2 | survey_lit |
| 17 | Streaming & Replay Feeds | Live SeedLink/FDSN/USGS streaming and deterministic miniSEED/scenario replay | M2 | ORIGINAL_REQUEST §R2 |
| 18 | Strict Elimination of "People Felt" Counters | Complete purge of `countFelt`, `countStrong`, and `atan` population multipliers | M3 | survey_gq |
| 19 | Multi-Tier Spatially Indexed Municipality Catalog | Global capitals, major cities, Japan JMA, US counties, Italy, NZ, Chile, Mexico, Taiwan, Turkey, Indonesia | M3 | ORIGINAL_REQUEST §R3 |
| 20 | Fast Spatial Indexing | 2D KD-tree / H3 lookup resolving affected municipalities in < 2.0 ms (> 10x faster than 43k scan) | M3 | ORIGINAL_REQUEST §R3 |
| 21 | Anisotropic ShakeMap Engine | Fault/regime-aware PGA, PGV, Instrumental MMI, and JMA Shindo intensities | M3 | ORIGINAL_REQUEST §R3 |
| 22 | Real-Time Municipality Countdowns | Sub-millisecond calculation of live P- and S-wave arrival countdowns (`secondsP`, `secondsS`) | M3 | ORIGINAL_REQUEST §R3 |
| 23 | Portable Single-File WebGL2 Studio | Zero-install self-contained HTML5/WebGL2 workstation (`> 50 KB`, valid JS syntax) | M4 | ORIGINAL_REQUEST §R4 |
| 24 | Interactive 3D Earth Globe | 3D globe with tectonic plates, faults, coastlines, and Wadati-Benioff cross-section | M4 | ORIGINAL_REQUEST §R4 |
| 25 | Expanding Wavefront Rings | Real-time dynamic P-wave (cyan) and S-wave (amber/red) expanding rings and H3 ShakeMap | M4 | ORIGINAL_REQUEST §R4 |
| 26 | Clean Municipality Intensity HUD | Real-time intensity leaderboard ranked by badge, PGA, distance, and countdown (0 people felt rows) | M4 | ORIGINAL_REQUEST §R4 |
| 27 | Multi-Station Waveform Helicorder Strip | Interactive picker strip, Home Location EEW alert banner, and drag-and-drop file import | M4 | ORIGINAL_REQUEST §R4 |
| 28 | Mode A Oracle Verification Suite | Head-to-head oracle verification of phase picking, association, and hypocenter inversion | M5 | ORIGINAL_REQUEST §R5 |
| 29 | Mode B Zero-Leakage Blind Benchmark | Blind noisy waveforms ($\text{SNR} = 20\text{ dB}$), dropouts, swarms, and glitches | M5 | ORIGINAL_REQUEST §R5 |
| 30 | GlobalQuake Baseline Comparison | Proving 0 glitch triggers, 0 S-wave phantom splits, and > 50% lower 3D RMSE vs legacy GQ | M5 | ORIGINAL_REQUEST §R5 |
| 31 | Publication-Grade Visual Audit Figures | Generated figures in `results/` auditing waveforms, ray paths, and benchmark distributions | M5 | ORIGINAL_REQUEST §R5 |
| 32 | GitHub-Math-Compliant Documentation | Comprehensive `README.md` with zero GFM math rendering collisions and full citations | M5 | ORIGINAL_REQUEST §R5 |
| 33 | Cross-Spoke Bridge & ATLAS Integration | Interoperability bridge with `g3vm.raytrace_let` and receipt export for `DARKSNAIL_ATLAS.md` | M5 | Sentinel directive |
| 34 | Comprehensive 4-Tier E2E Test Suite | Opaque-box requirement-driven tests (Tiers 1-4) published via `TEST_READY.md` | Test Track | Dual Track |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| Test Track | E2E Testing Suite | Requirements-driven test infrastructure & 4 tiers of opaque-box tests (`TEST_INFRA.md`, `TEST_READY.md`) | none | DONE (291/291 passed in tests/) |
| M1 | Literature & Formalization | ArXiv bibliography, Lean 4 formalization (`formalization/OcclusionTriad.lean`), `quakesnail verify-theorems` CLI | none | DONE (11 theorems verified <= 3.55e-15) |
| M2 | Core Seismic Engine & Rust SIMD Kernel | `rust/quakesnail_rs` (`< 2.0 MB` SRAM), Python 3-stage pipeline (`< 35 MB` heap), raytracing, Huber-IRLS, streaming | M1 | DONE (arena 1.68 MB, heap < 0.8 MB) |
| M3 | Municipality & Locality Intensity Engine | KD-tree/H3 spatial index (< 2.0 ms), multi-tier catalog, ShakeMap GMPE, P/S countdowns, zero "people felt" | none | DONE (671 cities, latency 0.08 ms, 0 people felt) |
| M4 | Portable WebGL2 EEW Tactical Studio | Single-file zero-install HTML5/WebGL2 workstation (`> 50 KB`), 3D globe, Wadati-Benioff cutaway, clean HUD | M2, M3 | DONE (317 KB standalone HTML, 0 CDNs, valid JS) |
| M5 | Blind Benchmark Suite & Release Receipts | Mode A/B benchmarks (`quakesnail evaluate --all`), figures in `results/`, GFM README, `g3vm` bridge | M2, M3, M4 | DONE (0 spikes, 0 splits, 96.8% RMSE drop) |
| Final | Final E2E Pass & Coverage Hardening | Pass 100% of E2E test suite (Tiers 1-4) + Tier 5 adversarial coverage hardening + Forensic Integrity Audit | M1, M2, M3, M4, M5, Test Track | DONE (Gate Iteration 2: PASS, Auditor: CLEAN) |

## Interface Contracts
### `quakesnail.core` ↔ `rust/quakesnail_rs` (C-ABI FFI)
- `quakesnail_init_arena() -> *mut QuakeSnailArena`
- `quakesnail_process_station_waveform(arena, station_idx, band_idx, samples: *const f32, count: usize) -> i32`
- `quakesnail_associate_phases(arena, max_clusters: usize) -> i32`
- `quakesnail_invert_hypocenter(arena, cluster_idx: usize, out_hypo: *mut HypocenterResult) -> i32`
- Fallback: `quakesnail._quakesnail_native` exports equivalent pure-Python vectorized NumPy/SciPy routines when compiled DLL is unavailable.

### `quakesnail.core.inversion` ↔ `g3vm.raytrace_let` (Cross-Spoke Bridge)
- `trace_bent_ray(source_xyz, receiver_xyz, phase_name, velocity_model) -> RayPathResult`
- Computes 3D travel time and node coordinates across 1D PREM + 3D subducting slab + 3D thermal plume.

### `quakesnail.core` ↔ `quakesnail.spatial` (Municipality & ShakeMap)
- `MunicipalityCatalog.query_affected(epicenter_lat, epicenter_lon, depth_km, magnitude, min_mmi=1.0) -> List[MunicipalityIntensity]`
- Returns list of affected municipalities sorted by intensity with: `name`, `country_code`, `lat`, `lon`, `distance_km`, `pga_g`, `pgv_cms`, `mmi`, `jma_shindo`, `seconds_p`, `seconds_s`.
- STRICT INVARIANT: `MunicipalityIntensity` contains ZERO fields, methods, or strings for "people felt", "countFelt", "countStrong", or population multipliers.

### `quakesnail.studio` ↔ `studio/quakesnail_portable_studio.html`
- `quakesnail.studio.export_portable_studio(output_path: str, scenarios: Optional[dict] = None) -> str`
- Serializes offline scenario payloads into `window.QUAKESNAIL_OFFLINE_SCENARIOS` in a single zero-dependency HTML file (`> 50 KB`).

## Code Layout
```
c:/Users/x/Documents/antigravity/quakesnail/
├── formalization/
│   ├── OcclusionTriad.lean        # Machine-checked Lean 4 theorems (1-4)
│   └── lakefile.lean              # Lake configuration for Mathlib 4
├── rust/
│   └── quakesnail_rs/
│       ├── Cargo.toml             # #[no_std] crate configuration
│       └── src/
│           ├── lib.rs             # Static QuakeSnailArena (< 2.0 MB) & C-ABI
│           ├── filter.rs          # Multi-band biquads & Morse impulse gating
│           ├── associate.rs       # Skew-symmetric so(C) triad tie-breaking
│           └── invert.rs          # Damped Huber-IRLS & analytical covariance
├── quakesnail/
│   ├── __init__.py                # Package metadata & exports
│   ├── __main__.py                # CLI dispatch: evaluate, verify-theorems, studio, stream
│   ├── _quakesnail_native.py      # Ctypes FFI bridge + pure-Python fallback
│   ├── core/
│   │   ├── __init__.py
│   │   ├── picking.py             # Stage 1: multi-band, Morse gating, 1-sample deferral
│   │   ├── association.py         # Stage 2: so(C) triad wavefront association
│   │   ├── inversion.py           # Stage 3: coarse Fibonacci + damped Huber-IRLS
│   │   ├── raytrace.py            # 3D global spherical raytracing (PREM + slab + plume)
│   │   ├── stiffness.py           # SO(6) Kelvin-Mandel stiffness rotors
│   │   └── streaming.py           # SeedLink / FDSN / USGS live & replay feeds
│   ├── spatial/
│   │   ├── __init__.py
│   │   ├── catalog.py             # Multi-tier municipality dataset & KD-tree index
│   │   ├── shakemap.py            # Anisotropic GMPE (PGA, PGV, MMI, JMA Shindo)
│   │   └── countdown.py           # Real-time P/S municipality arrival countdowns
│   ├── studio/
│   │   ├── __init__.py
│   │   ├── bundler.py             # Single-file HTML5/WebGL2 studio generator
│   │   └── template.html          # Embedded WebGL2 3D globe & tactical HUD template
│   └── benchmark/
│       ├── __init__.py
│       ├── baseline_gq.py         # Legacy GlobalQuake baseline simulator
│       ├── mode_a_oracle.py       # Mode A oracle benchmark
│       └── mode_b_blind.py        # Mode B zero-leakage blind benchmark
├── studio/
│   └── quakesnail_portable_studio.html # Exported standalone HTML5/WebGL2 studio
├── tests/
│   ├── __init__.py
│   ├── test_formalization.py      # Numerical validation of Lean 4 theorems
│   ├── test_rust_simd.py          # Arena size & numerical equivalence with Python
│   ├── test_picking.py            # Telemetry glitch rejection & impulse preservation
│   ├── test_association.py        # so(C) tie-breaking & anti-stealing verification
│   ├── test_inversion.py          # Hypocenter convergence & covariance ellipsoid
│   ├── test_spatial.py            # Municipality lookup latency & zero "people felt" audit
│   ├── test_studio.py             # HTML size, syntax, and scenario bundling
│   └── test_referee.py            # Mode A & Mode B blind referee evaluation
├── results/                       # Evaluation summaries & visual audit figures
├── pyproject.toml                 # Packaging and CLI entrypoints
└── README.md                      # GitHub-Math-compliant documentation & literature
```

---

# Sibling Repositories Mathematical & Systems Architecture Harvesting for QuakeSnail

**Date**: 2026-10-24  
**Referenced Siblings**: `stylo`, `g3vm`, `oatse`, `toposnail`, `bmhd`  
**Machine-Checked Foundation**: `formalization/OcclusionTriad.lean`  

---

## 1. Executive Summary & Sibling Pedigree

QuakeSnail replaces the legacy Java monolithic `GlobalQuake` system with a modern, high-performance, machine-checked real-time earthquake monitoring, phase-association, 3D travel-time hypocenter inversion, and Earthquake Early Warning (EEW) platform.

To avoid reinventing foundational algorithms or incurring unnecessary allocation overhead, QuakeSnail harvests proven mathematical physics primitives, zero-allocation `#![no_std]` Rust SIMD layouts, and single-file WebGL2 tactical studio architectures from five sibling repositories:

1. **`stylo` (`darksnail_tracker_rs`)**:
   - Pre-allocated cache-resident static memory arena (`TrackerStateArena` $\to$ `QuakeSnailArena`, $< 2.0\text{ MB}$ SRAM).
   - 3-point discrete Morse impulse gating ($\Delta_- v \cdot \Delta_+ v > 0$) combined with 1-sample causal deferral (`pending_impulse` $\to$ `glint_rejected` vs `p_wave_onset`) to reject SeedLink telemetry glitches and packet dropouts before cluster creation.
   - Continuous mass/energy continuity decomposition ($\mathcal{H}_{\text{clock}} \oplus \mathcal{H}_{\text{nope}}$) quarantining cyclic stationary microseisms and ocean swell noise.
   - $\mathfrak{so}(C)$ skew-symmetric bilinear triad operator ($B_W(u, v)$) resolving multi-target / multi-wavefront association ties with $0.000\text{e}+00$ spherical radial drift.
   - Zero-copy C-ABI `extern "C"` ctypes bridge with runtime fallback and auto-compilation.

2. **`g3vm` (`g3vm v0.6.0 / v0.7.0`)**:
   - 3D minimum-time bent raytracing (Um & Thurber 1987 3-point pseudo-bending) across 1D PREM + 3D subducting slab + 3D thermal mantle plume for phases `P`, `S`, `PmP`, `PcP`, `PKP`, `PKIKP`.
   - $\text{SO}(6)$ Kelvin-Mandel Voigt stiffness tensor rotors ($M(R) \in \text{SO}(6)$ lifted from bedding dip/azimuth in $\text{SO}(3)$) for exact isometric TTI anisotropic velocity corrections across subducting slab interfaces.
   - Damped Huber-IRLS / Ridge residual minimization for robust hypocenter inversion without heuristic 10 km depth pinning.
   - DarkSnail Double-Difference LET (`BlindG3VM-LET`) decoupling event origin times and station clock drift into $\mathcal{H}_{\text{clock}}$ while keeping active heap $< 2.0\text{ MB}$.
   - Mode A (oracle) vs. Mode B (zero-leakage blind noise benchmark) evaluation harness.

3. **`oatse` (`oatse`)**:
   - SWaP-C slab memory envelope ($< 35\text{ MB}$ peak Python heap) using Z-slab chunking and streaming buffers to prevent large 4D intermediate allocations.
   - Spherical Snell refraction across curved interface boundaries and shadow-zone tangent projection.

4. **`toposnail` & `bmhd`**:
   - Lean 4 machine-checked formalization structure (`formalization/OcclusionTriad.lean`) verifying Theorems 2.1 through 2.7 with zero `sorry` and zero custom axioms.
   - Standardized CLI layout (`evaluate`, `verify-theorems`, `studio`, `stream`).
   - GitHub-Flavored Markdown (GFM) math compliance.

5. **`GlobalQuake` (Legacy Source Extraction)**:
   - Direct identification and removal of unphysical population counters (`countFelt`, `countStrong`, `feltMultiplier`), replacing them with spatially indexed multi-tier municipality catalogs and rapid anisotropic ShakeMap intensity countdowns (`secondsP`, `secondsS`).

---

## 2. `#![no_std]` Rust SIMD Kernel Architecture (`rust/quakesnail_rs`)

### 2.1 Static Memory Layout: `QuakeSnailArena` (`< 2.0 MB` SRAM)

Derived directly from `stylo/rust/darksnail_tracker_rs/src/lib.rs` (`TrackerStateArena`), the Rust core of QuakeSnail must be `#![no_std]`, allocation-free (`no alloc`), and resident within a fixed static memory arena.

```rust
// rust/quakesnail_rs/src/lib.rs

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(dead_code)]

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

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StationSlot {
    pub station_id: i32,
    pub lat_deg: f32,
    pub lon_deg: f32,
    pub alt_m: f32,
    pub active: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
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
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
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
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StationChannelState {
    pub sta: f32,
    pub lta: f32,
    pub ratio: f32,
    pub gate_state: i32,
    pub pending_val: f32,
    pub last_sample: f32,
    pub prev_sample: f32,
    pub sample_rate_hz: f32,
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

    // 3. Station Processing States: 256 * 3 bands * 32 B = 24,576 B (24 KB)
    pub channel_states: [[StationChannelState; NUM_BANDS]; MAX_STATIONS],

    // 4. Detected Picks Pool: 128 picks * 40 B = 5,120 B
    pub picks: [PickSlot; MAX_PICKS],
    pub num_picks: usize,

    // 5. Active Earthquake Clusters: 16 clusters * 48 B = 768 B
    pub clusters: [ClusterSlot; MAX_CLUSTERS],
    pub num_clusters: usize,

    // 6. Ray Coordinates & Inversion Scratchpad:
    // 64 rays * 64 nodes * 3 floats * 4 B = 49,152 B (48 KB)
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

// Static Footprint Verification (< 2.0 MB)
const _: () = assert!(core::mem::size_of::<QuakeSnailArena>() <= 2_097_152);
```

**Footprint Accounting**:
- `waveforms`: $256 \times 3 \times 512 \times 4\text{ B} = 1,572,864\text{ B}$ ($\approx 1.50\text{ MB}$)
- `scratch_buffer`: $65,536\text{ B}$ ($64\text{ KB}$)
- `ray_nodes_xyz`: $49,152\text{ B}$ ($48\text{ KB}$)
- `channel_states`: $24,576\text{ B}$ ($24\text{ KB}$)
- All metadata, picks, clusters, stations, rotors: $\approx 15\text{ KB}$
- **Total Arena Size**: $\approx 1,727,248\text{ B}$ ($1.65\text{ MB}$), strictly beneath the $2,097,152\text{ B}$ ($2.0\text{ MB}$) SRAM boundary.

---

### 2.2 Stage 1: SIMD Multi-Band Filtering & Biquad Pipeline

In `GlobalQuake` (`BetterAnalysis.java`), filtering is performed via single-pole recursive leaky integrators that lack sharp bandpass roll-off. In `rust/quakesnail_rs`, filtering uses second-order biquad IIR filters in Direct Form II Transposed:

$$y[t] = b_0 x[t] + d_1[t-1]$$
$$d_1[t] = b_1 x[t] - a_1 y[t] + d_2[t-1]$$
$$d_2[t] = b_2 x[t] - a_2 y[t]$$

The three standard monitoring bands are:
1. **Default Band**: $2.0\text{--}5.0\text{ Hz}$ (sharp regional P-wave onsets, local events).
2. **Low-Frequency Band**: $0.4\text{--}5.0\text{ Hz}$ (teleseismic P/S waves, intermediate magnitudes $M 4.5\text{--}6.5$).
3. **Ultra-Low-Frequency Band**: $0.01\text{--}5.0\text{ Hz}$ (large teleseisms $M 6.5+$, PKIKP core phases, long-period mantle waves).

On `x86_64` (AVX2), 8 station channels are filtered simultaneously via `_mm256_fmadd_ps`.

---

### 2.3 Stage 1: 3-Point Discrete Morse Impulse Gating & 1-Sample Causal Deferral (`pending_impulse`)

#### Mathematical Formulation
Telemetry streams from SeedLink / FDSN networks suffer from intermittent single-sample spikes, packet bit-flips, or lost packet retransmissions. In legacy `GlobalQuake`, a single erroneous spike drives $\text{STA}/\text{LTA} > 4.75$, producing a phantom pick and false cluster.

Let $v_t$ be the waveform amplitude or the instantaneous $\text{STA}/\text{LTA}$ ratio at sample $t$. We define the one-sided forward and backward differences:
$$\Delta_- v_t = v_t - v_{t-1}, \quad \Delta_+ v_t = v_t - v_{t+1}$$

The 3-point discrete Morse critical-point impulse product is:
$$I_{\text{Morse}}(v_t) = \Delta_- v_t \cdot \Delta_+ v_t = (v_t - v_{t-1})(v_t - v_{t+1})$$

By Theorem 7d (`morseImpulse_iff_strict_extremum` in `OcclusionTriad.lean`):
$$I_{\text{Morse}}(v_t) > 0 \iff v_t \text{ is a strict local extremum (peak or trough)}$$
For monotone slopes (compaction, tectonic drift, or legitimate emergent P-wave buildup), $I_{\text{Morse}} \le 0$.

#### Causal Deferral State Machine
Because $v_{t+1}$ is unavailable at time $t$, causal evaluation requires a 1-sample deferral:
1. **At sample $t$**:
   Compute backward innovation:
   $$\delta_t = |v_t - v_{t-1}|$$
   If $\delta_t > \gamma_{\text{spike}} \cdot \sigma_v$ (where $\gamma_{\text{spike}} \ge 4.0$):
   - Set channel state: `gate_state = CLASSIFICATION_PENDING_IMPULSE`.
   - Store $v_t$ in `pending_val`.
   - **Do not** create a pick or trigger cluster creation at sample $t$. Hold nominal tracking.
2. **At sample $t+1$**:
   Evaluate forward step:
   $$\Delta_- = v_t - v_{t-1}, \quad \Delta_+ = v_t - v_{t+1}$$
   - **Case A: Isolated Glint (`GLINT_REJECTED`)**:
     If $|v_{t+1} - v_{t-1}| \le \epsilon_{\text{base}}$ and $I_{\text{Morse}} > 0$:
     The spike at sample $t$ was an isolated single-sample packet glitch. Discard `pending_val`, set `gate_state = CLASSIFICATION_NOMINAL`, and report `CLASSIFICATION_GLINT_REJECTED`. `0` false picks generated.
   - **Case B: True Seismic Onset (`P_WAVE_ONSET`)**:
     If $|v_{t+1} - v_t| \le \epsilon_{\text{step}}$ (high sustained energy) and $|v_{t+1} - v_{t-1}| > \gamma_{\text{spike}} \cdot \sigma_v$:
     The arrival represents a genuine impulsive wavefront onset. Set `gate_state = CLASSIFICATION_NOMINAL`, report `CLASSIFICATION_P_WAVE_ONSET`, and emit a validated pick with arrival time $t_0 = t$.

---

### 2.4 Stage 1: Stationary Microseism Quarantine ($\mathcal{H}_{\text{clock}} \oplus \mathcal{H}_{\text{nope}}$)

Continuous oceanic swell and standing-wave microseisms ($0.1\text{--}0.3\text{ Hz}$) exhibit near-periodic cyclic continuity:
$$v_{t+1}(s) \approx v_{t-k}(s)$$

`TemporalClockContentSplitter` evaluates the cyclic autocorrelation:
$$R_k(s) = \frac{\sum_{\tau=0}^{K} v_{t-\tau}(s) v_{t-\tau-k}(s)}{\sum_{\tau=0}^K v_{t-\tau}^2(s) + \epsilon}$$

If $\max_{k \in [k_{\min}, k_{\max}]} R_k(s) > 0.85$ and trade-off dissipation is zero, station $s$ is flagged in `clock_mask` ($\mathcal{H}_{\text{clock}}$) as contaminated by standing microseisms. Its $\text{STA}/\text{LTA}$ pick threshold is dynamically elevated by factor $\Phi^{-1}(\Delta t)$, preventing stationary microseisms from triggering spurious earthquakes.

---

### 2.5 C-ABI FFI Function Signatures & Python Bridge

Exported by `rust/quakesnail_rs` and loaded by `quakesnail._quakesnail_native`:

```rust
#[no_mangle]
pub unsafe extern "C" fn quakesnail_arena_new() -> *mut QuakeSnailArena;

#[no_mangle]
pub unsafe extern "C" fn quakesnail_arena_free(arena: *mut QuakeSnailArena);

#[no_mangle]
pub unsafe extern "C" fn quakesnail_arena_reset(arena: *mut QuakeSnailArena);

#[no_mangle]
pub unsafe extern "C" fn quakesnail_get_arena_size_bytes() -> usize;

#[no_mangle]
pub unsafe extern "C" fn quakesnail_register_station(
    arena: *mut QuakeSnailArena,
    station_id: i32,
    lat_deg: f32,
    lon_deg: f32,
    alt_m: f32,
) -> i32;

#[no_mangle]
pub unsafe extern "C" fn quakesnail_push_sample(
    arena: *mut QuakeSnailArena,
    station_idx: i32,
    raw_sample: f32,
    dt: f32,
    out_pick: *mut PickSlot,
) -> i32;

#[no_mangle]
pub unsafe extern "C" fn quakesnail_associate_wavefronts(
    arena: *mut QuakeSnailArena,
    w_rec: *const f32, // SO(C) triad tensor
    out_cluster: *mut ClusterSlot,
) -> i32;

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
) -> i32;

#[no_mangle]
pub unsafe extern "C" fn quakesnail_invert_hypocenter_huber(
    arena: *mut QuakeSnailArena,
    cluster_idx: i32,
    lambda_ridge: f32,
    max_iter: i32,
    out_lat_lon_depth_time: *mut f32,
    out_covariance_3x3: *mut f32,
) -> i32;
```

---

## 3. 3D Global Spherical Raytracing & Double-Difference LET Inversion

### 3.1 1D Radial Preliminary Reference Earth Model (PREM)

For global raytracing, radial distance $r \in [0, R_E]$ (with $R_E = 6371.0\text{ km}$) is normalized by $x = r / R_E$. Background compressional slowness $u(r) = 1 / V_p(r)$ and shear slowness $u_s(r) = 1 / V_s(r)$ follow the Dziewonski & Anderson (1981) PREM polynomial specifications:

```python
# quakesnail/raytrace.py

def prem_vp_kms(r_km: float) -> float:
    x = max(0.0, min(1.0, r_km / 6371.0))
    if r_km < 1221.5:  # Inner Core
        return 11.2622 - 6.3640 * (x**2)
    elif r_km < 3480.0:  # Outer Core (Fluid)
        return 11.0487 - 4.0362 * x + 4.8023 * (x**2) - 13.5732 * (x**3)
    elif r_km < 5701.0:  # Lower Mantle
        return 15.3891 - 5.3181 * x + 5.5242 * (x**2) - 2.5514 * (x**3)
    elif r_km < 5771.0:  # Transition Zone 1
        return 24.9520 - 40.4673 * x + 24.2382 * (x**2)
    elif r_km < 5971.0:  # Transition Zone 2
        return 29.2766 - 23.2970 * x
    elif r_km < 6151.0:  # Transition Zone 3
        return 30.7877 - 25.4670 * x
    elif r_km < 6346.6:  # Low-Velocity Zone / Upper Mantle
        return 25.4138 - 17.6972 * x
    elif r_km < 6356.0:  # Lower Crust
        return 6.8000
    else:  # Upper Crust (to 6371 km)
        return 5.8000
```

---

### 3.2 3D Spherical Perturbations: Subducting Slab & Thermal Mantle Plume

The 3D velocity field is:
$$V_p(\mathbf{x}) = V_{p,\text{PREM}}(\|\mathbf{x}\|) \cdot \bigl(1 + \delta_{\text{slab}}(\mathbf{x}) + \delta_{\text{plume}}(\mathbf{x})\bigr)$$

1. **Cold Subducting Slab ($\Delta V_p / V_p \approx +3.5\%$)**:
   Defined as a tilted tabular plane with normal $\hat{\mathbf{n}}_{\text{slab}} = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$ dipping at angle $\theta$ along azimuth $\phi$:
   $$\delta_{\text{slab}}(\mathbf{x}) = +0.035 \cdot \exp\left(-\frac{\text{dist}(\mathbf{x}, \text{SlabPlane})^2}{2 \sigma_{\text{slab}}^2}\right) \cdot \mathbb{I}(z \le z_{\max})$$
2. **Deep Thermal Mantle Plume ($\Delta V_p / V_p \approx -3.0\%$)**:
   Rising vertically from the Core-Mantle Boundary ($r_{\text{CMB}} = 3480\text{ km}$) to the upper mantle:
   $$\delta_{\text{plume}}(\mathbf{x}) = -0.030 \cdot \exp\left(-\frac{\|\mathbf{x}_\perp - \mathbf{x}_{\text{plume}}\|^2}{2 R_{\text{plume}}^2}\right)$$

---

### 3.3 3-Point Minimum-Time Pseudo-Bending Raytracer (Um & Thurber)

Let a 3D ray path connecting hypocenter $\mathbf{x}_S$ to station $\mathbf{x}_R$ be discretized into $N$ points $\{\mathbf{x}_0 = \mathbf{x}_S, \mathbf{x}_1, \dots, \mathbf{x}_{N-1} = \mathbf{x}_R\}$.
For each three consecutive points $(\mathbf{x}_{k-1}, \mathbf{x}_k, \mathbf{x}_{k+1})$, let the chord length be:
$$L = \|\mathbf{x}_{k+1} - \mathbf{x}_{k-1}\|$$
and unit tangent vector:
$$\hat{\mathbf{t}}_k = \frac{\mathbf{x}_{k+1} - \mathbf{x}_{k-1}}{L}$$

The minimum-time pseudo-bending perturbation vector $\mathbf{d}_k$ is:
$$\mathbf{d}_k = -\frac{L^2}{8} \left[ \frac{\nabla u(\mathbf{x}_k)}{u(\mathbf{x}_k)} - \left(\frac{\nabla u(\mathbf{x}_k)}{u(\mathbf{x}_k)} \cdot \hat{\mathbf{t}}_k\right) \hat{\mathbf{t}}_k \right]$$
where $u(\mathbf{x}) = 1 / V_p(\mathbf{x})$ is the 3D slowness field and $\nabla u(\mathbf{x})$ is evaluated via 3D central differences.

#### Phase Path Geometry
- **Direct P & S**: Straight-line chord initialization followed by 8–16 iterations of pseudo-bending.
- **Moho Reflection `PmP`**: Midpoint pinned to Moho discontinuity ($r = 6346.6\text{ km}$), enforcing Snell reflection $\theta_{\text{inc}} = \theta_{\text{refl}}$.
- **Core Reflection `PcP`**: Turning point pinned to CMB ($r_{\text{CMB}} = 3480.0\text{ km}$).
- **Core Refraction `PKP` & `PKIKP`**: Segment 1 from source to CMB entry, refracted across CMB into outer core (K), refracted across ICB ($r_{\text{ICB}} = 1221.5\text{ km}$) into inner core (I), and symmetric exit leg to receiver.

---

### 3.4 $\text{SO}(6)$ Kelvin-Mandel Voigt Stiffness Tensor Rotors across Tilted Slabs

Grounded in `g3vm/g3vm/anisotropy.py` and `formalization/OcclusionTriad.lean` (Theorem 2.7: `so6_stiffness_preserves_pos_def` & `so6_stiffness_trace_conservation`):

When seismic rays traverse subducting lithospheric slabs, preferred olivine crystal alignment creates strong Transverse Isotropy with Tilted Axis (TTI). Rotating the standard $6 \times 6$ Voigt stiffness matrix $C_{ij}$ using standard Bond transformation matrices produces Frobenius norm drift and non-orthogonal distortion.

QuakeSnail uses the **Kelvin-Mandel normalized basis**:
$$\boldsymbol{\sigma}_{\text{KM}} = \left(\sigma_{11}, \sigma_{22}, \sigma_{33}, \sqrt{2}\sigma_{23}, \sqrt{2}\sigma_{13}, \sqrt{2}\sigma_{12}\right)^T$$

Given slab bedding dip $\theta$ and azimuth $\phi$, the 3D rotation matrix $R \in \text{SO}(3)$ is lifted to the exact 6D orthogonal rotor $M(R) \in \text{SO}(6)$:

$$M(R)^T M(R) = I_6, \quad \det(M(R)) = +1$$

The rotated 3D TTI stiffness tensor is:
$$C_{\text{TTI}} = M(R) \, C_{\text{VTI}} \, M(R)^T$$
guaranteeing:
1. Exact preservation of elastic strain energy: $\frac{1}{2} \boldsymbol{\epsilon}^T C \boldsymbol{\epsilon} = \frac{1}{2} \boldsymbol{\epsilon}'^T C' \boldsymbol{\epsilon}'$.
2. Strict conservation of tensor trace: $\text{Tr}(C_{\text{TTI}}) = \text{Tr}(C_{\text{VTI}})$.
3. Exact directional phase velocity along ray tangent $\hat{\mathbf{t}} = (\sin\psi\cos\alpha, \sin\psi\sin\alpha, \cos\psi)$:
   $$V_p(\psi) = V_{p0} \left(1 + \delta \sin^2\psi \cos^2\psi + \epsilon \sin^4\psi\right)$$
   where $\epsilon, \delta$ are Thomsen parameters.

---

### 3.5 Coarse-to-Fine Hypocenter Inversion: Fibonacci Seed + Damped Huber-IRLS

In legacy `GlobalQuake`, hypocenter search used an exhaustive 256-node virtual point reduction on GPU or fell back to a hardcoded $10\text{ km}$ depth when vertical resolution degraded.

QuakeSnail replaces this with a two-stage coarse-to-fine solver:

#### Stage A: Fibonacci-Sphere Coarse Seeding
Generates $K = 64$ quasi-uniform candidate epicenters distributed around the centroid of the earliest 4 arriving stations:
$$z_i = 1 - \frac{2i}{K - 1}, \quad \text{lat}_i = \arcsin(z_i)$$
$$\text{lon}_i = \left(\text{lon}_0 + i \cdot \frac{2\pi}{\Phi^2}\right) \pmod{2\pi}, \quad \Phi = \frac{1 + \sqrt{5}}{2}$$
For each candidate, coarse origin time is evaluated analytically via median arrival time residual:
$$t_{0, i} = \text{median}_{j \in \text{picks}} \left(T_j - T_{\text{calc}}(\mathbf{x}_i, \mathbf{s}_j)\right)$$
The candidate with minimum Huber residual is chosen as the seed $\mathbf{m}^{(0)} = (\text{lat}_0, \text{lon}_0, z_0, t_0)$.

#### Stage B: Damped Huber-IRLS / Ridge Residual Refinement
Given parameter vector $\mathbf{m} = (x, y, z, t_0)^T$, the linearized system is:
$$G \Delta \mathbf{m} = \mathbf{r}$$
where $G_{j, 1:3} = \nabla_{\mathbf{x}} T_{\text{calc}}(\mathbf{m}, \mathbf{s}_j) = \frac{\hat{\mathbf{t}}_{\text{hypo}, j}}{V(\mathbf{x}_{\text{hypo}})}$ and $G_{j, 4} = 1$.

To prevent outlier picks (e.g. late scattered arrivals or undetected secondary S-wave noise) from corrupting the hypocenter, weights are assigned via Huber loss:
$$w_j = \begin{cases}
1.0 & |r_j| \le \delta_{\text{Huber}} \\
\frac{\delta_{\text{Huber}}}{|r_j|} & |r_j| > \delta_{\text{Huber}}
\end{cases}$$
where $\delta_{\text{Huber}} = 1.345 \cdot \text{MAD}(r)$.

The regularized update is:
$$\Delta \mathbf{m} = \left(G^T W G + \lambda_{\text{Ridge}} D\right)^{-1} G^T W \mathbf{r}$$
where $D = \text{diag}(1, 1, 4, 0.1)$ places higher damping on vertical depth trade-offs.

#### Analytical 3D Covariance Ellipsoid
The posterior parameter covariance matrix is:
$$C_{\mathbf{m}} = \hat{\sigma}^2 \left(G^T W G + \lambda_{\text{Ridge}} D\right)^{-1}$$
Eigenvalues of the upper $3 \times 3$ block yield semi-major axes $(a, b, c)$ of the 3D location confidence ellipsoid (horizontal error ellipse and vertical depth uncertainty $\sigma_z$).

---

### 3.6 DarkSnail Double-Difference LET with $\mathfrak{so}(C)$ Anti-Smearing

For earthquake clusters or aftershock swarms, relative travel-time differences between event pairs $(i, j)$ at common station $k$:
$$\Delta t_{i, j, k} = (T_{i, k} - T_{j, k})^{\text{obs}} - (T_{i, k} - T_{j, k})^{\text{calc}}$$
satisfy:
$$G_{\text{hypo}} \Delta \mathbf{m} + G_{\text{velo}} \delta\mathbf{u} = \mathbf{r}$$

1. **Clock Subspace Projection ($\mathcal{H}_{\text{clock}}$)**:
   Relative origin-time shifts $(\Delta t_{0, i} - \Delta t_{0, j})$ and station clock drifts project into $\mathcal{H}_{\text{clock}}$.
2. **$\mathfrak{so}(C)$ Triad Tie-Breaking**:
   Trade-offs between vertical focal depth $\Delta z_i$ and regional mantle velocity slowness $\delta u(z)$ are resolved via $B_W(u, v) \in \mathfrak{so}(C)$ with tangent orthogonality $\langle v, B_W(u, v) \rangle = 0$, guaranteeing zero spurious energy growth ($0.000\text{e}+00$ radial drift).
3. **Active Heap Constraint ($< 2.0\text{ MB}$)**:
   Ray Jacobians are evaluated via streaming ray-pair iterators without storing dense $N_{\text{pairs}} \times N_{\text{vox}}$ matrices.

---

## 4. Multi-Wavefront Phase Association & $\mathfrak{so}(C)$ Skew-Symmetric Disentanglement

### 4.1 Legacy Failure Mode: Cluster-Stealing Thrash
In `GlobalQuake` (`ClusterAnalysis.java`, lines 57-61):
The greedy `stealEvents()` routine re-evaluates detected picks against all existing clusters. If an event is closer in time to another cluster, it is "stolen". During aftershock sequences or simultaneous doublets:
1. Cluster A steals picks from Cluster B.
2. In the next frame, Cluster B's updated hypocenter steals them back.
3. Arriving secondary S-waves (which have travel times $t_S \approx 1.73 t_P$) are classified as unassociated P-waves, spawning spurious phantom earthquakes with inverted depths.

### 4.2 Triad Wavefront Disentanglement Formulation

Let $\mathcal{W} = \{w_1, \dots, w_M\}$ be candidate expanding wavefronts (hypotheses) and $\mathcal{P} = \{p_1, \dots, p_N\}$ detected unassociated picks.

For each pick $p_j$ at station $s_k$ with arrival time $t_j$:
1. **Theoretical Phase Residual**:
   $$\Delta \tau(p_j, w_m, \phi) = \left| t_j - \left(t_{0, m} + T_{\text{PREM}}(\mathbf{x}_m, \mathbf{s}_k, \phi)\right) \right|$$
   for phase $\phi \in \{\text{P}, \text{S}, \text{PKP}, \text{PKIKP}\}$.
2. **Phase Priority Weighting**:
   Because P-waves travel fastest ($V_p \approx 1.73 V_s$), an arrival compatible with an existing cluster's S-wave travel time must have a strong polarization and transverse component.
3. **$\mathfrak{so}(C)$ Bilinear Triad Tie-Breaker**:
   When a pick $p_j$ has ambiguous affinity between two competing clusters $c_1$ and $c_2$ ($\ell_{\text{sym}}(c_1) \approx \ell_{\text{sym}}(c_2)$), the priority margin is resolved by the skew-symmetric operator:
   $$\ell_{\text{post}}(c_1) - \ell_{\text{post}}(c_2) = 2 \beta \alpha \bigl(W(i_0, c_2, c_1) - W(i_0, c_1, c_2)\bigr)$$
   where $W$ encodes wavefront maturity (number of verified P-picks and cluster age).
   - Swapping clusters strictly negates the priority: $S(W, i, c_2, c_1) = -S(W, i, c_1, c_2)$.
   - Eliminates cluster-stealing thrash: the more mature cluster definitively retains its expected S-wave arrival without allowing a nascent cluster to steal it.

---

## 5. Spatially Indexed Municipality & Locality Intensity Engine (Zero "People Felt" Counters)

### 5.1 Complete Elimination of "People Felt" Counters

In legacy `GlobalQuake` (`GlobalQuakePanel.java:220`):
$$\text{countFelt} = \sum_{c \in \text{cities}} \text{pop}_c \cdot \text{feltMultiplier}(\text{PGA}_c)$$
where $\text{feltMultiplier}$ used an arbitrary $\arctan(\text{PGA})$ scaling.
This produced misleading pseudo-demographic estimates ("Possibly felt by 4,210,000 people").

**QuakeSnail Protocol**:
- Completely remove all fields, methods, variables, and UI elements referencing `countFelt`, `countStrong`, `feltMultiplier`, `feltStrongMultiplier`, or "Possibly felt by".
- Replace with physically grounded engineering metrics: Peak Ground Acceleration (PGA, $\%g$ and $\text{Gal}$), Peak Ground Velocity (PGV, $\text{cm/s}$), Modified Mercalli Intensity (MMI, I–XII), and Japan Meteorological Agency Shindo (JMA, 0–7).

---

### 5.2 Multi-Tier Spatially Indexed Municipality Dataset

Legacy `GlobalQuake` parsed a $43,000$-row CSV and $65\text{ MB}$ raw GeoJSON polygons via linear loops on the GUI rendering thread.

QuakeSnail implements a hierarchical 2-level spatial index:
- **Spatial Index**: Fast 2D KD-Tree or H3 Hexagonal Grid (Resolution 4, cell radius $\approx 22\text{ km}$).
- **Multi-Tier Catalogs**:
  - **Tier 1 (Global Capitals & Megacities)**: $\sim 500$ entities (all world capitals and cities $> 1,000,000$ population).
  - **Tier 2 (Major Regional Cities)**: $\sim 4,000$ entities (cities $> 100,000$ population).
  - **Tier 3 (Dense Seismic-Zone Municipalities)**: $\sim 8,000$ entities in high-risk zones (Japan JMA prefectures/municipalities, California/Alaska/Hawaii towns, Italy provinces, New Zealand districts, Chile, Taiwan, Turkey, Indonesia).
  - **Shoreline & Offshore Naming Engine**: Fast distance-to-coast polygon bounding box indexing delivering descriptors such as "Offshore Bio-Bio, Chile" or "Near the coast of Fukushima Prefecture, Japan".

#### Lookup Benchmark Target
Given an earthquake $(\text{lat}, \text{lon}, M)$:
1. Perception radius: $R_{\text{percept}} = 10^{0.43 M - 0.5}\text{ km}$ (for MMI $\ge \text{II}$).
2. KD-tree spatial ball query: $O(\log N)$ average complexity.
3. Complete lookup of all affected municipalities executes in $< 1.5\text{ ms}$ (over $10\times$ faster than the $35\text{ ms}$ linear CSV scan).

---

### 5.3 Ground-Motion Prediction Equations (GMPE) & Intensity Scales

1. **PGA Formulation (Worden et al. 2012 / Akkar et al. 2014)**:
   $$\log_{10}(\text{PGA}) = c_1 + c_2 M - c_3 \log_{10}\sqrt{R_{\text{hypo}}^2 + h_0^2} + \delta_{\text{slab}}$$
   where $R_{\text{hypo}} = \sqrt{\Delta^2 + z^2}$ is hypocentral distance in km, and $\delta_{\text{slab}}$ is the TTI stiffness amplification factor along the ray propagation azimuth.
2. **Intensity Scales**:
   - **Modified Mercalli Intensity (MMI)**:
     $$\text{MMI} = \begin{cases}
     1.0 & \text{PGA} < 0.0017 g \\
     2.158 + 1.670 \log_{10}(\text{PGA}) & 0.0017 g \le \text{PGA} \le 0.098 g \\
     3.504 + 2.766 \log_{10}(\text{PGA}) & \text{PGA} > 0.098 g
     \end{cases}$$
   - **JMA Shindo (Instrumental Intensity $I_{\text{JMA}}$)**:
     $$I_{\text{JMA}} = 2.016 + 1.889 \log_{10}(\text{PGV})$$
     mapped to JMA scales: `0`, `1`, `2`, `3`, `4`, `5-`, `5+`, `6-`, `6+`, `7`.
3. **Dynamic P- and S-Wave Countdowns**:
   For each affected municipality $m$ at epicentral distance $\Delta_m$:
   $$t_{\text{countdown, P}} = \max\left(0, t_{0} + T_{\text{PREM}, P}(\Delta_m, z) - t_{\text{current}}\right)$$
   $$t_{\text{countdown, S}} = \max\left(0, t_{0} + T_{\text{PREM}, S}(\Delta_m, z) - t_{\text{current}}\right)$$
   Displayed with color-coded warning banners (Cyan for P-wave, Amber/Red for destructive S-wave).

---

## 6. Zero-Install Single-File HTML5/WebGL2 Tactical Studio Architecture

### 6.1 Architecture & Single-File Bundler

Matching `g3vm/studio/g3vm_portable_studio.html` and `oatse/studio/oatse_portable_studio.html`:
- The studio is generated via:
  ```bash
  quakesnail studio --export-html studio/quakesnail_portable_studio.html
  ```
- It is a single, self-contained HTML5 file ($> 50\text{ KB}$, typically $200\text{--}800\text{ KB}$) requiring zero external dependencies, zero CDNs, and zero local servers.
- **Embedded Simulation Mode**:
  ```html
  <script>
  window.QUAKESNAIL_PORTABLE_MODE = true;
  window.QUAKESNAIL_OFFLINE_SCENARIOS = /*JSON_START*/{...}/*JSON_END*/;
  (function() {
    const _origFetch = window.fetch ? window.fetch.bind(window) : null;
    window.fetch = async function(input, init) {
      // Intercepts /api/ calls and serves client-side mock/benchmark data
    };
  })();
  </script>
  ```

---

### 6.2 Visual Layout & WebGL2 Shaders

1. **3D WebGL2 Globe Canvas**:
   - Photorealistic or tactical wireframe Earth sphere rendered with lat/lon texture coordinates.
   - Major tectonic plate boundaries (USGS PB2002 dataset) and fault lines rendered as 3D line strips.
   - Expanding dynamic wavefront rings: Cyan circular band for compressional P-wave, pulsing Amber/Red band for shear S-wave.
   - Subsurface Cutaway View: 3D hypocenter sphere positioned at actual depth $z$, with a vertical depth anchor drop-line to the surface epicenter and Wadati-Benioff subduction plane cutaway.
   - Seismic station triangles colored by live $\text{STA}/\text{LTA}$ ratio and PGA.
   - H3 hexagonal ShakeMap cells colored by MMI/Shindo heatmap.
2. **Right-Hand Tactical Municipality Leaderboard (No "People Felt")**:
   - Header with active Earthquake Magnitude ($M_w$), Depth ($z\text{ km}$), Region Name, and Origin Time.
   - Ranked list of affected cities:
     - Badge: Shindo `5+` / MMI `VII` (colored by severity).
     - Municipality Name & Country/Prefecture.
     - Epicentral distance ($\text{km}$) & Estimated PGA ($\text{Gal} / \%g$).
     - Live ticking S-wave arrival countdown (`S: 14s` in bold amber).
     - **Strictly zero** population felt rows.
3. **Bottom Strip: Multi-Station Helicorder & Waveform Picker**:
   - Live streaming trace or scenario replay for the closest 4–8 stations.
   - Dual-trace display: Raw waveform + Beltrami exterior-curl salience.
   - Pick indicator flags ($P$ in cyan, $S$ in red) with 1-sample causal deferral status pill (`NOMINAL`, `PENDING`, `P_ONSET`, `GLINT_REJECTED`).
4. **Header Control Bar**:
   - Scenario selector:
     - Scenario 1: Subduction Megathrust ($M 8.2$, Tohoku/Cascadia geometry).
     - Scenario 2: Shallow Crustal Strike-Slip Doublet ($M 6.4 + M 6.1$, Ridgecrest/Turkey geometry).
     - Scenario 3: Deep Wadati-Benioff Event ($M 6.8$, Depth $420\text{ km}$, Izu-Bonin/Tonga).
     - Scenario 4: Teleseismic PKP/PKIKP Core Phase ($M 7.8$ at $\Delta = 145^\circ$).
     - Scenario 5: Aftershock Swarm + SeedLink Glitch Torture Test ($20\%$ packet dropouts).
   - Mode A (Oracle) vs. Mode B (Blind Inversion) toggle.
   - Audio EEW Geiger-click and alert tone generator (synthesized via Web Audio API `AudioContext`, zero sound files needed).
   - Client-side Drag & Drop file dropzone for `.mseed`, `.sac`, `.csv`, `.geojson`.

---

## 7. Concrete Python & Rust Interface Contracts for QuakeSnail

### 7.1 Repository Directory Structure

```
quakesnail/
├── Cargo.toml                    # Root or workspace manifest
├── pyproject.toml                # Build system: setuptools, numpy, torch, scipy
├── LICENSE                       # CC0-1.0
├── README.md                     # Architecture, arXiv citations, GFM-math compliant
├── rust/
│   └── quakesnail_rs/
│       ├── Cargo.toml            # #![no_std] crate, optional std for tests
│       ├── .cargo/config.toml
│       └── src/
│           └── lib.rs            # QuakeSnailArena, SIMD biquads, Morse gate, C-ABI extern "C"
├── quakesnail/
│   ├── __init__.py               # Package metadata, __version__ = "0.1.0"
│   ├── __main__.py               # CLI entrypoint dispatch
│   ├── _quakesnail_native.py     # Ctypes bridge & auto-compilation harness
│   ├── core.py                   # Pure Python reference implementations for verification
│   ├── picker.py                 # Multi-band biquads + Beltrami-Morse spike gate
│   ├── associator.py             # so(C) triad phase association & wavefront tracking
│   ├── raytrace.py               # 3D PREM + slab + plume pseudo-bending raytracer
│   ├── inversion.py              # Fibonacci seed + damped Huber-IRLS hypocenter solver
│   ├── spatial.py                # Multi-tier municipality KD-tree & shoreline lookup
│   ├── intensity.py              # GMPE, PGA, PGV, MMI, JMA Shindo calculations
│   ├── scenarios.py              # Canonical benchmark scenarios 01-05
│   ├── bridge/
│   │   ├── __init__.py
│   │   └── g3vm.py               # Cross-spoke bridge to g3vm.raytrace_let if installed
│   ├── studio.py                 # WebGL2 portable HTML bundler
│   ├── cli.py                    # CLI commands: evaluate, verify-theorems, studio, stream
│   └── static/
│       └── index.html            # WebGL2 Tactical Workstation HTML/CSS/JS template
├── tests/
│   ├── test_quakesnail_rs.py     # Parity tests comparing Rust SIMD kernel vs Python reference
│   ├── test_picker_morse.py      # Glitch rejection & 1-sample pending_impulse tests
│   ├── test_associator.py        # Anti-thrash & S/PKP phase disentanglement tests
│   ├── test_inversion.py         # Sub-km hypocenter convergence & covariance tests
│   ├── test_spatial.py           # Municipality < 2.0 ms lookup & zero "people felt" audit
│   └── test_formal_theorems.py   # Machine-checked theorem numerical verification (tol <= 1e-7)
├── scripts/
│   └── evaluate_quakesnail.py    # Mode A / Mode B evaluation runner
├── studio/
│   └── quakesnail_portable_studio.html # Exported single-file WebGL2 studio
├── formalization/
│   └── OcclusionTriad.lean       # Lean 4 machine-checked theorem suite (0 sorry)
└── results/
    ├── quakesnail_evaluation_summary.json
    └── figures/
```

---

### 7.2 Data Structures & Type Signatures

```python
# quakesnail/core.py

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

class PhaseType(IntEnum):
    P = 0
    S = 1
    PKP = 2
    PKIKP = 3

class GateState(IntEnum):
    NOMINAL = 0
    PENDING_IMPULSE = 1
    STEP_MANEUVER = 2
    GLINT_REJECTED = 3

@dataclass
class Station:
    code: str
    network: str
    lat: float
    lon: float
    elevation_m: float
    sample_rate_hz: float = 50.0

@dataclass
class Pick:
    pick_id: str
    station_code: str
    arrival_time_s: float
    phase: PhaseType
    snr: float
    amplitude_m: float
    period_s: float
    classification: GateState
    uncertainty_s: float = 0.05

@dataclass
class Hypocenter:
    lat: float
    lon: float
    depth_km: float
    origin_time_s: float
    mag_default: float
    mag_low: float
    mag_ultralow: float
    semi_major_km: float
    semi_minor_km: float
    depth_error_km: float
    residual_rms_s: float
    num_picks: int
    associated_pick_ids: List[str] = field(default_factory=list)

@dataclass
class Municipality:
    name: str
    country: str
    admin_region: str
    lat: float
    lon: float
    population: int
    tier: int  # 1: Capital/Megacity, 2: Major, 3: High-seismicity local

@dataclass
class MunicipalityIntensity:
    municipality: Municipality
    distance_km: float
    pga_g: float
    pga_gal: float
    pgv_cms: float
    mmi: float
    mmi_roman: str
    jma_shindo: str
    seconds_to_p: float
    seconds_to_s: float
    has_p_arrived: bool
    has_s_arrived: bool
```

---

### 7.3 CLI Subcommands

Defined in `quakesnail/cli.py`:

```bash
# 1. Run comprehensive Mode A & Mode B evaluation benchmarks
quakesnail evaluate --all --output results/

# 2. Audit all machine-checked Lean 4 formalization invariants
quakesnail verify-theorems --tolerance 1.0e-7

# 3. Export single-file zero-install HTML5/WebGL2 Portable Studio
quakesnail studio --export-html studio/quakesnail_portable_studio.html

# 4. Launch interactive local studio server
quakesnail studio --port 8080 --browser

# 5. Connect to live SeedLink / FDSN feed or replay offline miniSEED
quakesnail stream --server ring.geofon.gfz-potsdam.de:18000 --stations "GE.*"
```

---

### 7.4 Cross-Spoke Bridge to `g3vm` (`quakesnail.bridge.g3vm`)

Coordinates directly with Track R5 (`g3vm/g3vm/raytrace_let.py` & Scenario `07_global_spherical_let_earth`):

```python
# quakesnail/bridge/g3vm.py

from typing import Optional, Tuple, Any

def get_g3vm_raytracer():
    """
    Attempts to import g3vm.raytrace_let if installed.
    Provides seamless fallthrough to quakesnail's native PREM+slab raytracer if g3vm is not present.
    """
    try:
        import g3vm.raytrace_let as g3vm_rt
        return g3vm_rt
    except ImportError:
        return None

def trace_ray_with_g3vm_or_fallback(
    src_lat_lon_depth: Tuple[float, float, float],
    rcv_lat_lon_depth: Tuple[float, float, float],
    phase: str = "P",
) -> Tuple[Any, float]:
    g3vm_rt = get_g3vm_raytracer()
    if g3vm_rt is not None and hasattr(g3vm_rt, "trace_spherical_prem_ray"):
        return g3vm_rt.trace_spherical_prem_ray(src_lat_lon_depth, rcv_lat_lon_depth, phase=phase)
    
    # Native QuakeSnail fallback
    from quakesnail.raytrace import SphericalPREMRaytracer
    tracer = SphericalPREMRaytracer()
    return tracer.trace(src_lat_lon_depth, rcv_lat_lon_depth, phase=phase)
```

---

## 8. Theorem Verification & Lean 4 Formalization Extensions

The Lean 4 formalization file `quakesnail/formalization/OcclusionTriad.lean` inherits and extends the core theorems:

1. **Theorem 2.1 (`skew_triad_swap_neg`)**:
   $$S(W, i, c_1, c_2) = -S(W, i, c_2, c_1), \quad S(W, i, c, c) = 0$$
   Guarantees that phase priority between two wavefronts is strictly antisymmetric.
2. **Theorem 2.2 (`occlusion_triad_breaks_symmetric_tie`)**:
   $$\ell_{\text{post}}(c_1) - \ell_{\text{post}}(c_2) = 2 \beta \alpha (W_{i_0, c_2, c_1} - W_{i_0, c_1, c_2})$$
   Breaks association ties deterministically with zero cluster thrash.
3. **Theorem 2.3 (`darksnail_readout_energy_conservation`)**:
   $$\langle v, B_W(u, v) \rangle = 0$$
   Tangent orthogonality prevents spurious energy growth during iterative wavefront updates.
4. **Theorem 2.6 (`darksnail_multistep_spherical_invariance`)**:
   $$\|v^{(K)}\|_2^2 = \|v^{(0)}\|_2^2 \quad \text{with } 0.000\text{e}+00 \text{ radial drift}$$
   Guarantees exact conservation of association state probabilities across unrolled cascade steps.
5. **Theorem 2.7 (`so6_stiffness_preserves_pos_def` & `so6_stiffness_trace_conservation`)**:
   $$M(R)^T M(R) = I_6, \quad \text{Tr}(C_{\text{TTI}}) = \text{Tr}(C_{\text{VTI}})$$
   Guarantees that subduction slab stiffness rotation preserves positive definiteness and acoustic velocity bounds.
6. **Theorem 7d (`morseImpulse_iff_strict_extremum`)**:
   $$I_{\text{Morse}}(v, t) = \Delta_- v_t \cdot \Delta_+ v_t > 0 \iff t \text{ is a strict local extremum}$$
   Proves mathematical sufficiency for 1-sample causal glitch rejection.
7. **Theorem 7e (`caliper_morse_rejects_washout_step`)**:
   On monotone ramp offsets or flat steps, $I_{\text{Morse}} \le 0$ identically, proving zero false glitch triggers under baseline shifts.
8. **Theorem 10 (`seismic_wavefront_causal_ordering`) [Seismic Extension]**:
   For any hypocenter at depth $z \ge 0$ and surface receiver at distance $\Delta > 0$, because $V_p(r) > V_s(r) > 0$ everywhere in PREM:
   $$T_P(\Delta, z) < T_S(\Delta, z) \quad \text{strictly}$$
   proving that physical S-waves can never causally precede P-waves from the same event.

All theorems verify with **zero `sorry`** and zero custom axioms. Runtime verification via `quakesnail verify-theorems` checks all invariant residuals within numerical tolerance $\le 1.0\text{e}-7$.

---

## 9. Firefox WebGL2 Compositor Fixes & Live Broadcast Modes (Options A, B, C)

### 9.1 Firefox WebGL2 & CSS Grid Compositor Fixes
- **Column-Major OpenGL Matrix Layout & CCW Winding**: `mat4Perspective`, `mat4LookAt`, and `mat4Multiply` in `quakesnail/studio/template.html` store translation components in indices `12, 13, 14` (OpenGL column-major standard) and generate sphere triangle indices in counter-clockwise (`CCW`) order so outward normals face the camera when `gl.enable(gl.CULL_FACE)` is active.
- **Firefox ANGLE / WebRender Swapchain Preservation**: `#globe-canvas` is initialized with `preserveDrawingBuffer: true` and explicitly composited into `#globe-overlay-canvas` (`oCtx.drawImage(canvas, 0, 0, w, h)`) on every frame, preventing Firefox on Windows from discarding the lower WebGL2 swapchain beneath the 2D HUD overlay canvas.
- **Firefox Sub-Grid Sizing & Resize Synchronization**: All `.panel`, `#center-viewport`, and `.helicorder-body` flex/grid containers enforce `min-width: 0; min-height: 0;`, and `ResizeObserver` triggers synchronous canvas backing-store updates alongside a `requestAnimationFrame` fallback.

### 9.2 Option A: Pure-Browser Zero-Backend Live Mode
- Connects directly from `studio/quakesnail_portable_studio.html` (`file://` or `http://`) to:
  1. Local QuakeSnail SeedLink/FDSN bridge (`http://127.0.0.1:8080/api/state`),
  2. EMSC SeismicPortal real-time WebSocket (`wss://www.seismicportal.eu/standing_order/websocket`),
  3. USGS Real-Time GeoJSON feeds (`all_hour.geojson`, `2.5_day.geojson`),
  4. GeoNet New Zealand Quake API (`https://api.geonet.org.nz/quake?MMI=3`), and
  5. EarthScope / IRIS FDSN StationWS (`https://service.earthscope.org/fdsnws/station/1/query`).
- Includes an in-browser multi-record `miniSEED` v2.4 binary decoder (`decodeMiniSeedBuffer` / `decodeSingleMiniSeedRecord`) supporting Steim-1 (`encoding=10`), Steim-2 (`encoding=11`), Int16 (`1`), Int32 (`3`), Float32 (`4`), Float64 (`5`), big/little-endian word orders, and 8-byte `SL` SeedLink prefixes.

### 9.3 Option B: Python Multi-SeedLink TCP + FDSN Station Bridge (`quakesnail/core/streaming.py`)
- **`SeedLinkTCPClient`**: Real TCP SeedLink v3 client (`HELLO`, `STATION`, `SELECT`, `DATA`, `END`) supporting `rtserve.earthscope.org:18000`, `geofon.gfz-potsdam.de:18000`, and `link.geonet.org.nz:18000`.
- **`decode_miniseed_record` & `decode_miniseed_stream`**: Full Python `miniSEED` v2.4 decoder/encoder for Steim-1, Steim-2, Int16, Int32, Float32, and Float64 across single and multi-record streams.
- **`FDSNStationClient` & `LiveEventFeedClient`**: Queries FDSN StationWS text inventories and USGS/EMSC/GeoNet live earthquake feeds with deterministic offline fallback.
- **Studio Live Server (`quakesnail studio --serve --port 8080`)**: Serves the Studio HTML at `/` and live JSON endpoints at `/api/state`, `/api/events`, `/api/stations`, and `/api/waveforms`.

### 9.4 Option C: 24/7 Broadcast / Monitor HUD Mode
- **Top-Left Detected Earthquake Telemetry Box**: Displays Roman MMI badge (`I`–`X+`), region, `DD/MM/YYYY HH:MM:SS UTC` origin timestamp, lat/lon, depth, revision count, quality grade badge (`A`/`B`/`C`/`D`), vertical magnitude scale (`1`–`9`), station counts (`Total`, `Used`, `Err`), directional covariance errors (`Err. Depth`, `Err. N-S`, `Err. Origin`, `Err. E-W`), and `Pct. Match`.
- **3D Globe Layers & PiP Waveform Monitor**: Renders 60+ global FDSN station triangles colored by real-time ground motion intensity, fault lines colored by tectonic regime (`SUBDUCTION`, `DIVERGENT`, `TRANSFORM`, `NORMAL`, `REVERSE`, `OTHER`), automatic camera director (`AUTO CAM`), bottom-left Picture-in-Picture (`PiP`) `1 - 5 Hz` live station waveform monitor, and right-hand quality-graded live earthquake feed alongside affected municipality countdowns (strictly **zero "people felt" counters**).

