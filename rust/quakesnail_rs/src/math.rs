//! #![no_std] mathematical functions and SIMD routines for quakesnail_rs.

#[inline(always)]
pub fn fast_sqrt_scalar_f32(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if !x.is_finite() {
        return x;
    }
    // Scale subnormal inputs into normal IEEE-754 range before bit-halving
    let (x_norm, scale) = if x < f32::from_bits(0x0080_0000) {
        (x * 16777216.0f32, 1.0f32 / 4096.0f32)
    } else {
        (x, 1.0f32)
    };
    let mut guess = f32::from_bits((x_norm.to_bits() >> 1) + 0x1fc0_0000);
    for _ in 0..4 {
        guess = 0.5f32 * (guess + x_norm / guess);
    }
    guess * scale
}

#[inline(always)]
pub fn fast_sqrt_scalar_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if !x.is_finite() {
        return x;
    }
    let (x_norm, scale) = if x < f64::from_bits(0x0010_0000_0000_0000) {
        (x * 18014398509481984.0f64, 1.0f64 / 134217728.0f64)
    } else {
        (x, 1.0f64)
    };
    let mut guess = f64::from_bits((x_norm.to_bits() >> 1) + 0x1ff8_0000_0000_0000);
    for _ in 0..5 {
        guess = 0.5f64 * (guess + x_norm / guess);
    }
    guess * scale
}

#[inline(always)]
pub fn fast_sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if !x.is_finite() {
        return x;
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)))
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        vgetq_lane_f32::<0>(vsqrtq_f32(vdupq_n_f32(x)))
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(x)))
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        fast_sqrt_scalar_f32(x)
    }
}

#[inline(always)]
pub fn fast_sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if !x.is_finite() {
        return x;
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        _mm_cvtsd_f64(_mm_sqrt_sd(_mm_setzero_pd(), _mm_set_sd(x)))
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        vgetq_lane_f64::<0>(vsqrtq_f64(vdupq_n_f64(x)))
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        f64x2_extract_lane::<0>(f64x2_sqrt(f64x2_splat(x)))
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        fast_sqrt_scalar_f64(x)
    }
}

/// IEEE-754 #![no_std]-safe natural logarithm accurate to < 1 ULP across all positive f32 (including subnormals).
#[inline(always)]
pub fn fast_ln(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    if x.is_infinite() {
        return f32::INFINITY;
    }
    let (x_norm, sub_exp) = if x < f32::from_bits(0x0080_0000) {
        (x * 8388608.0f32, -23i32)
    } else {
        (x, 0i32)
    };
    let bits = x_norm.to_bits();
    let mut exp = ((bits >> 23) as i32) - 127 + sub_exp;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000) as f64;
    if m > core::f64::consts::SQRT_2 {
        m *= 0.5;
        exp += 1;
    }
    let y = (m - 1.0) / (m + 1.0);
    let y2 = y * y;
    let ln_m = 2.0
        * y
        * (1.0
            + y2 * (1.0 / 3.0
                + y2 * (1.0 / 5.0
                    + y2 * (1.0 / 7.0 + y2 * (1.0 / 9.0 + y2 * (1.0 / 11.0))))));
    ((exp as f64) * core::f64::consts::LN_2 + ln_m) as f32
}

#[inline(always)]
pub fn fast_log10(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.log10()
    }
    #[cfg(not(feature = "std"))]
    {
        fast_ln(x) * core::f32::consts::LOG10_E
    }
}

#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.exp()
    }
    #[cfg(not(feature = "std"))]
    {
        if x.is_nan() {
            return f32::NAN;
        }
        if x <= -88.0 {
            return 0.0;
        }
        if x >= 88.0 {
            return f32::INFINITY;
        }
        let x_d = x as f64;
        const INV_LN2: f64 = core::f64::consts::LOG2_E;
        const LN2_HI: f64 = core::f64::consts::LN_2;
        let t = x_d * INV_LN2;
        let k = if t >= 0.0 {
            (t + 0.5) as i32
        } else {
            (t - 0.5) as i32
        };
        let r = x_d - (k as f64) * LN2_HI;
        // Taylor series for e^r on [-0.35, 0.35]
        let er = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0 + r * (1.0 / 5040.0 + r / 40320.0)))))));
        let scale = f32::from_bits(((k + 127) as u32) << 23);
        (er as f32) * scale
    }
}

#[inline(always)]
pub fn fast_cos(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.cos()
    }
    #[cfg(not(feature = "std"))]
    {
        if !x.is_finite() {
            return f32::NAN;
        }
        const TWO_PI: f64 = core::f64::consts::TAU;
        const FRAC_PI_2: f64 = core::f64::consts::FRAC_PI_2;
        let mut rem = (x as f64) % TWO_PI;
        if rem < 0.0 {
            rem += TWO_PI;
        }
        let q = (rem / FRAC_PI_2) as i32;
        let t = rem - (q as f64) * FRAC_PI_2;
        let t2 = t * t;
        let s = t * (1.0 - t2 * (1.0 / 6.0 - t2 * (1.0 / 120.0 - t2 * (1.0 / 5040.0 - t2 * (1.0 / 362880.0 - t2 / 39916800.0)))));
        let c = 1.0 - t2 * (0.5 - t2 * (1.0 / 24.0 - t2 * (1.0 / 720.0 - t2 * (1.0 / 40320.0 - t2 * (1.0 / 3628800.0 - t2 / 479001600.0)))));
        let res = match q & 3 {
            0 => c,
            1 => -s,
            2 => -c,
            _ => s,
        };
        res as f32
    }
}

#[inline(always)]
pub fn fast_sin(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.sin()
    }
    #[cfg(not(feature = "std"))]
    {
        if !x.is_finite() {
            return f32::NAN;
        }
        const TWO_PI: f64 = core::f64::consts::TAU;
        const FRAC_PI_2: f64 = core::f64::consts::FRAC_PI_2;
        let mut rem = (x as f64) % TWO_PI;
        if rem < 0.0 {
            rem += TWO_PI;
        }
        let q = (rem / FRAC_PI_2) as i32;
        let t = rem - (q as f64) * FRAC_PI_2;
        let t2 = t * t;
        let s = t * (1.0 - t2 * (1.0 / 6.0 - t2 * (1.0 / 120.0 - t2 * (1.0 / 5040.0 - t2 * (1.0 / 362880.0 - t2 / 39916800.0)))));
        let c = 1.0 - t2 * (0.5 - t2 * (1.0 / 24.0 - t2 * (1.0 / 720.0 - t2 * (1.0 / 40320.0 - t2 * (1.0 / 3628800.0 - t2 / 479001600.0)))));
        let res = match q & 3 {
            0 => s,
            1 => c,
            2 => -s,
            _ => -c,
        };
        res as f32
    }
}

#[inline(always)]
pub fn abs_f32(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7fff_ffff)
}

#[inline(always)]
pub fn abs_f64(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0x7fff_ffff_ffff_ffff)
}

#[inline(always)]
pub fn fast_atan2(y: f32, x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        y.atan2(x)
    }
    #[cfg(not(feature = "std"))]
    {
        if y.is_nan() || x.is_nan() {
            return f32::NAN;
        }
        if x == 0.0 {
            if y > 0.0 {
                return core::f32::consts::FRAC_PI_2;
            } else if y < 0.0 {
                return -core::f32::consts::FRAC_PI_2;
            } else {
                return 0.0;
            }
        }
        let yd = y as f64;
        let xd = x as f64;
        let ay = abs_f64(yd);
        let ax = abs_f64(xd);
        let (inv, u) = if ay > ax { (true, ax / ay) } else { (false, ay / ax) };
        let (shifted, w) = if u > core::f64::consts::SQRT_2 - 1.0 {
            (true, (u - 1.0) / (u + 1.0))
        } else {
            (false, u)
        };
        let w2 = w * w;
        let mut a = w
            * (1.0
                - w2 * (1.0 / 3.0
                    - w2 * (1.0 / 5.0
                        - w2 * (1.0 / 7.0
                            - w2 * (1.0 / 9.0
                                - w2 * (1.0 / 11.0 - w2 * (1.0 / 13.0 - w2 / 15.0)))))));
        if shifted {
            a += core::f64::consts::FRAC_PI_4;
        }
        if inv {
            a = core::f64::consts::FRAC_PI_2 - a;
        }
        let res = if xd > 0.0 {
            if yd >= 0.0 { a } else { -a }
        } else if yd >= 0.0 {
            core::f64::consts::PI - a
        } else {
            -core::f64::consts::PI + a
        };
        res as f32
    }
}

#[inline(always)]
pub fn fast_acos(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.acos()
    }
    #[cfg(not(feature = "std"))]
    {
        if x.is_nan() {
            return f32::NAN;
        }
        let clamped = x.clamp(-1.0, 1.0);
        let s = fast_sqrt(1.0 - clamped * clamped);
        fast_atan2(s, clamped)
    }
}
