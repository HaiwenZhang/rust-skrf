use num_complex::Complex64;

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

/// Trait for SIMD-accelerated complex number operations
pub trait ComplexSimd {
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
}

#[cfg(target_arch = "wasm32")]
#[derive(Copy, Clone)]
pub struct WasmComplex(v128);

#[cfg(target_arch = "wasm32")]
impl WasmComplex {
    #[inline]
    pub fn new(re: f64, im: f64) -> Self {
        Self(f64x2_replace_lane::<0>(
            f64x2_replace_lane::<1>(f64x2_splat(0.0), im),
            re,
        ))
    }

    #[inline]
    pub fn from_complex(c: Complex64) -> Self {
        Self::new(c.re, c.im)
    }

    #[inline]
    pub fn to_complex(self) -> Complex64 {
        Complex64::new(
            f64x2_extract_lane::<0>(self.0),
            f64x2_extract_lane::<1>(self.0),
        )
    }
}

#[cfg(target_arch = "wasm32")]
impl ComplexSimd for WasmComplex {
    #[inline]
    fn add(self, other: Self) -> Self {
        Self(f64x2_add(self.0, other.0))
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self(f64x2_sub(self.0, other.0))
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        // self = [a, b], other = [c, d]
        let a = f64x2_splat_lane::<0>(self.0); // [a, a]
        let b = f64x2_splat_lane::<1>(self.0); // [b, b]

        let cd = other.0; // [c, d]
        let dc = i64x2_shuffle::<1, 0>(cd, cd); // [d, c] (using i64x2 shuffle for bitwise swap)

        // Use a constant for [1.0, -1.0] if we wanted but let's just use negate logic
        // We need [ac - bd, ad + bc]
        // term1 = [a*c, a*d]
        // term2 = [b*d, b*c]
        // We need term1[0] - term2[0] and term1[1] + term2[1]

        let term1 = f64x2_mul(a, cd);
        let term2 = f64x2_mul(b, dc);

        // Negate d in term2: [b*d, b*c] -> [-b*d, b*c]
        let mask = f64x2_replace_lane::<1>(f64x2_replace_lane::<0>(f64x2_splat(0.0), -1.0), 1.0);
        let term2_adj = f64x2_mul(term2, mask);

        Self(f64x2_add(term1, term2_adj))
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub type SimdComplex = WasmComplex;

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub type SimdComplex = FallbackComplex;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Copy, Clone)]
pub struct FallbackComplex(pub Complex64);

#[cfg(all(target_arch = "wasm32", not(target_feature = "simd128")))]
#[derive(Copy, Clone)]
pub struct FallbackComplex(pub Complex64);

impl FallbackComplex {
    pub fn new(re: f64, im: f64) -> Self {
        Self(Complex64::new(re, im))
    }
    pub fn from_complex(c: Complex64) -> Self {
        Self(c)
    }
    pub fn to_complex(self) -> Complex64 {
        self.0
    }
}

impl ComplexSimd for FallbackComplex {
    #[inline]
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }
}
