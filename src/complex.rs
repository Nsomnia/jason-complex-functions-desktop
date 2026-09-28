//! Complex arithmetic in `f64`: the CPU reference type for the whole program.
//!
//! # Why `f64` and not `f32`
//!
//! The GPU draws the picture in `f32` because that is what the WGSL kernel gets
//! out of the uniform block (see [`crate::uniforms`]). This type is the *CPU
//! reference*: the values the app compares against, the values the unit tests
//! assert on, and the values a human reads when checking whether the plot is
//! mathematically right. Keeping one extra decimal digit of precision here
//! costs nothing and makes the tests able to distinguish "my formula is wrong"
//! from "the GPU lost a bit in the noise".
//!
//! Nothing in this module is GPGPU-specific. It is plain, careful textbook
//! complex arithmetic, written to be read.
//!
//! # What the WGSL kernel has to agree with
//!
//! The GPU has no complex number type, so every operation below is written
//! here in terms of the two real components, and `shaders/domain_coloring.wgsl`
//! transcribes them. Each method carries a note wherever the naive
//! transcription is *not* also the numerically best choice, so the shader author
//! knows exactly where a deliberate difference is allowed (and
//! [`Complex::norm_hypot`] versus [`Complex::norm`] is the one place where it
//! currently is).
//!
//! # Edge cases, once, for all
//!
//! IEEE-754 semantics are propagated rather than trapped, with a single
//! documented exception:
//!
//! * `x / 0` does **not** panic. It yields an infinite result (`0/0` yields
//!   `NaN`, which is the mathematically indeterminate form). See
//!   [`Div` for Complex`](std::ops::Div).
//! * `NaN` and infinity flow through the arithmetic operators, so a pixel whose
//!   orbit escapes to infinity is detectable with [`Complex::is_finite`].

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// The function dispatch table lives next door, and is compiled as a child
/// module of this one.
///
/// `src/main.rs` declares the other eight modules but not this one, and
/// `src/panel.rs` reaches the table as `crate::complex::functions`. So the
/// module is attached here with an explicit `#[path]`: that is what compiles
/// the table and what makes its test suite run at all. If `main.rs` ever gains
/// a top-level `mod functions;`, this line becomes a harmless second,
/// independently compiled copy of the same file and nothing breaks.
#[path = "functions.rs"]
pub mod functions;

/// A complex number `re + im * i`, stored as its two real components.
///
/// No invariant is maintained beyond "`re` and `im` are ordinary `f64`s" -
/// in particular `-0.0` and `NaN` are both legal, because the sign of a zero
/// carries real mathematical meaning here: it selects which side of the branch
/// cut of `log` and `sqrt` a point on the cut is taken from. See
/// [`Complex::arg`] and the principal square root in the function table.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex {
    /// Real part, the coefficient of `1`.
    pub re: f64,
    /// Imaginary part, the coefficient of `i`.
    pub im: f64,
}

impl Complex {
    /// The complex origin, `0 + 0i`.
    ///
    /// Also what [`Default`] produces.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    /// The multiplicative identity, `1 + 0i`.
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };

    /// The multiplicative and additive unit `i = sqrt(-1)`.
    ///
    /// Note `I * I == -ONE` rather than `ONE`: the GPU kernel stores the same
    /// pair `(0.0, 1.0)` in a `vec2<f32>` and gets the same answer.
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    /// The number `0 + 0i`, spelled the obvious way.
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Build a complex number from a real and an imaginary part, accepting
    /// anything that converts to `f64`.
    ///
    /// This is the lenient constructor, for bridging the `f32` world of
    /// [`crate::uniforms`] and the pixel-to-plane mapping into this `f64`
    /// type: `Complex::from_re_im(uniforms.center[0], uniforms.center[1])`
    /// compiles and does the widening for you. For `f64` arguments
    /// [`Complex::new`] is the more direct spelling and is exactly equivalent.
    pub fn from_re_im<A, B>(re: A, im: B) -> Self
    where
        A: Into<f64>,
        B: Into<f64>,
    {
        Self {
            re: re.into(),
            im: im.into(),
        }
    }

    /// The complex conjugate, `re - im * i`.
    ///
    /// This is a reflection in the real axis, which is why `f64::copysign`-free
    /// negation of `im` is the whole of it: `-0.0` maps to `0.0` and back.
    pub const fn conjugate(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Squared magnitude, `re * re + im * im`.
    ///
    /// Cheaper than [`Complex::norm`] (no square root) and is what the
    /// quadratic-form distance to the origin wants, but it can overflow to
    /// infinity for `|z| > ~1.34e154` and underflow to zero for very small
    /// `|z|`. When the magnitude itself matters rather than a ratio, use
    /// [`Complex::norm_hypot`].
    pub const fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Magnitude (absolute value), `|z| = sqrt(re^2 + im^2)`.
    ///
    /// The obvious implementation, and the one the WGSL kernel transcribes:
    /// it is [`Complex::norm_sqr`] with a square root on the end. It inherits
    /// that method's overflow and underflow limits - it returns infinity once
    /// `|z| > ~1.34e154` and zero once `|z| < ~1.6e-162` - so anywhere the
    /// magnitude itself has to be right rather than merely proportional, use
    /// [`Complex::norm_hypot`].
    pub fn norm(self) -> f64 {
        self.norm_sqr().sqrt()
    }

    /// Overflow- and underflow-safe magnitude, computed by `f64::hypot`.
    ///
    /// `hypot` scales its arguments internally, so it returns a correct
    /// magnitude for values spanning the whole `f64` range: `hypot(3e200,
    /// 4e200)` is `5e200` where [`Complex::norm`] returns infinity, and
    /// `hypot(3e-200, 4e-200)` is `5e-200` where [`Complex::norm`] returns
    /// zero. It costs a little more than a multiply-add, so this is for
    /// "compute the real magnitude" work - the `sinc` singularity guard, the
    /// principal `log` - and *not* for per-pixel inner loops.
    ///
    /// For every value where [`Complex::norm`] does not overflow or underflow,
    /// the two agree to within a couple of units in the last place, which is
    /// the difference between the two summation orders and nothing more.
    pub fn norm_hypot(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Principal argument in radians, in `(-pi, pi]`.
    ///
    /// Taken as `im.atan2(re)`, so it *is* `arg` for every quadrant and the
    /// range is half-open at the top: a point on the negative real axis with
    /// `im == +0.0` gets `+pi`, and the same point with `im == -0.0` gets
    /// `-pi`. That asymmetry is deliberate - it is what makes the branch cut of
    /// the principal `log` and `sqrt` single-valued on the cut instead of
    /// having to pick an arbitrary side. `arg()` of `0` is `0.0`.
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Whether both components are finite, i.e. whether this is an ordinary
    /// point of the complex plane rather than infinity or a hole.
    ///
    /// The plotter uses this to tell "orbit escaped, colour it as infinity"
    /// apart from "orbit hit a pole and the arithmetic broke", which are
    /// different pictures.
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

/// Componentwise addition, `(a.re + b.re) + (a.im + b.im) * i`.
///
/// Plain IEEE-754 addition, so `inf + -inf` is `NaN`. A constant-time add like
/// this is the one operation a WGSL kernel can write literally.
impl Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

/// Componentwise subtraction, `(a.re - b.re) + (a.im - b.im) * i`.
///
/// The naive form, valid whenever `|b|` is not so large that `b.norm_sqr()`
/// overflows; see the note on [`Div`] for why that is tolerated here.
impl Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

/// Complex multiplication, the "full" product
/// `(a.re * b.re - a.im * b.im) + (a.re * b.im + a.im * b.re) * i`.
///
/// This is the form the WGSL kernel uses, and it is exact for every pair of
/// binary floating-point numbers that does not overflow an intermediate (both
/// `a.re * b.re` terms are a single rounding each). Consequences worth knowing:
///
/// * `Complex::I * Complex::I == -Complex::ONE` exactly, the reason the
///   constants work out.
/// * An orbit that escapes does not stay at infinity. `inf * 0` is `NaN` in
///   IEEE-754, so the step after a component becomes infinite that component is
///   `NaN`, and the step after that both are. Callers must therefore detect an
///   escaped orbit with [`Complex::is_finite`] and never by testing for an
///   infinity, which is true for only one or two steps of the orbit.
/// * The one product whose two terms cancel, `a.re * b.re - a.im * b.im`, can
///   lose all significance to catastrophic cancellation even when the true
///   result is perfectly representable. Scaling first (Smith's algorithm) would
///   fix that, but it costs an extra square root and does not mirror the
///   shader, so it is not done. Nothing in the plotter's function table needs
///   it.
impl Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

/// Complex division by the naive formula
/// `(a.re * b.re + a.im * b.im) / d + (a.im * b.re - a.re * b.im) / d * i`,
/// where `d = b.re * b.re + b.im * b.im`.
///
/// # Division by zero does not panic
///
/// `d == 0` is detected rather than divided through, and the result is chosen
/// to say something useful to a domain plotter:
///
/// | numerator | denominator | result |
/// |---|---|---|
/// | zero | zero | `NaN + NaN * i` - the indeterminate form `0/0` |
/// | non-zero | zero | `inf + inf * i` - the pole, drawn as "at infinity" |
///
/// This mirrors how the extended complex plane actually behaves: `z -> 1/z`
/// sends a point to infinity exactly where it sends infinity to zero, and a
/// domain colouring of `1/z` is supposed to show a bright pole, not a crash.
/// The signs are not tracked, because the sign of an infinity is a property of
/// the limit taken and not of the point.
///
/// # Where the shader differs, and why it does not matter
///
/// The WGSL kernel's `c_div` uses Smith's algorithm
/// (`(a * conj(b)) / |b|^2`) with no guard at all, so its poles come out as
/// `NaN` rather than as an infinity. That is a genuine difference in the value
/// and a deliberate one here: a pole is not a `0/0`, and a plotter that paints
/// every non-finite value identically - which is what `shaders/domain_coloring`
/// does, black for both - shows the same pixel either way. Do not "fix" this
/// into agreement by making both of them `NaN`; the distinction between a pole
/// and a removable singularity is one of the things the picture is for.
///
/// # Overflow
///
/// `d` overflows to infinity when `|b| > ~1.34e154`, which makes the result a
/// signed zero rather than a very small number. Callers that care should scale
/// the divisor, or use [`Complex::norm_hypot`] to detect the situation first.
/// The shader is better conditioned here, because Smith's algorithm never forms
/// a sum that can cancel - but it is *d*, not a difference of products, that
/// overflows in this implementation, so the same threshold applies.
impl Div for Complex {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        let d = rhs.norm_sqr();
        if d == 0.0 {
            return if self == Self::ZERO {
                Self {
                    re: f64::NAN,
                    im: f64::NAN,
                }
            } else {
                Self {
                    re: f64::INFINITY,
                    im: f64::INFINITY,
                }
            };
        }
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / d,
            im: (self.im * rhs.re - self.re * rhs.im) / d,
        }
    }
}

/// Negation, `-re - im * i`.
///
/// Note that this maps `0.0` to `-0.0`, which is almost always what the
/// mathematician wants and occasionally surprising to the programmer.
impl Neg for Complex {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

/// Scaling by a real factor, `s * z`, on both sides of the operator.
///
/// `2.0 * z` and `z * 2.0` are the same value; a complex number times a real
/// one has no imaginary cross-term. This exists because writing
/// `z * Complex::new(2.0, 0.0)` to double a number is the sort of thing a
/// reader should not have to learn by hitting a type error, and the plotter
/// scales a lot of things by things.
impl Mul<f64> for Complex {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

/// [`Mul<f64>`](Mul) with the real factor on the left.
///
/// `f64 * Complex`, so that the two spellings of scaling are interchangeable.
impl Mul<Complex> for f64 {
    type Output = Complex;

    fn mul(self, rhs: Complex) -> Complex {
        Complex {
            re: self * rhs.re,
            im: self * rhs.im,
        }
    }
}

/// Division by a real factor, `z / s`.
///
/// `z / 2.0` is exact when `2.0` is a power of two and rounds to nearest
/// otherwise, exactly as `1.0 / 2.0` does. Division by `0.0` follows IEEE-754:
/// a zero numerator gives `NaN` and a non-zero one gives an infinity.
impl Div<f64> for Complex {
    type Output = Self;

    fn div(self, rhs: f64) -> Self {
        Self {
            re: self.re / rhs,
            im: self.im / rhs,
        }
    }
}

/// Division by a real factor with the factor on the left, `s / z`.
///
/// `(s * conjugate(z)) / |z|^2`, which is what makes this useful: `1.0 / z` is
/// the reciprocal without a subtraction and without a branch.
impl Div<Complex> for f64 {
    type Output = Complex;

    fn div(self, rhs: Complex) -> Complex {
        let d = rhs.norm_sqr();
        if d == 0.0 {
            return if self == 0.0 {
                Complex {
                    re: f64::NAN,
                    im: f64::NAN,
                }
            } else {
                Complex {
                    re: f64::INFINITY,
                    im: f64::INFINITY,
                }
            };
        }
        Complex {
            re: (self * rhs.re) / d,
            im: (-self * rhs.im) / d,
        }
    }
}

/// `a+bi`, in the spelling a maths textbook would use.
///
/// The rules, in the order they are applied:
///
/// * the origin prints as `0`;
/// * a missing part is omitted entirely, so a real number prints as `2.5`;
/// * a lone imaginary part prints as `5i`, or as `i` / `-i` when the
///   coefficient is one, and a leading minus is folded into the coefficient
///   rather than printed as a separate sign;
/// * otherwise the imaginary part is always signed, `-3-4i` for `-3 - 4i`.
///
/// Numbers are rendered with `f64`'s shortest round-tripping `Display`, so
/// `1.0` prints as `1`, `0.1` prints as `0.1`, and a value that has no short
/// decimal form prints in full. Non-finite components print as `NaN` or `inf`
/// and are not special-cased.
impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.re == 0.0 && self.im == 0.0 {
            return f.write_str("0");
        }
        if self.im == 0.0 {
            return write!(f, "{}", self.re);
        }
        if self.re == 0.0 {
            return match self.im {
                1.0 => f.write_str("i"),
                -1.0 => f.write_str("-i"),
                other => write!(f, "{}i", other),
            };
        }
        let sign = if self.im < 0.0 { "-" } else { "+" };
        write!(f, "{}{}{}i", self.re, sign, self.im.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use std::f64::EPSILON;

    /// `|a - b| <= tol * max(1, |a|, |b|)`, so the tolerance is relative where
    /// it can be and absolute near zero.
    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
    }

    fn close_complex(a: Complex, b: Complex, tol: f64) -> bool {
        close(a.re, b.re, tol) && close(a.im, b.im, tol)
    }

    #[test]
    fn new_stores_both_components() {
        let z = Complex::new(1.5, -2.5);
        assert_eq!(z.re, 1.5);
        assert_eq!(z.im, -2.5);
    }

    #[test]
    fn from_re_im_accepts_f32_and_f64_alike() {
        assert_eq!(
            Complex::from_re_im(1.5f32, -2.5f32),
            Complex::new(1.5, -2.5)
        );
        assert_eq!(
            Complex::from_re_im(1.5f64, -2.5f64),
            Complex::new(1.5, -2.5)
        );
        assert_eq!(
            Complex::from_re_im(1.5f32, -2.5f64),
            Complex::new(1.5, -2.5)
        );
    }

    #[test]
    fn constants_are_the_expected_points() {
        assert_eq!(Complex::ZERO, Complex::new(0.0, 0.0));
        assert_eq!(Complex::ONE, Complex::new(1.0, 0.0));
        assert_eq!(Complex::I, Complex::new(0.0, 1.0));
        assert_eq!(Complex::default(), Complex::ZERO);
    }

    #[test]
    fn conjugate_reflects_in_the_real_axis() {
        assert_eq!(Complex::new(3.0, 4.0).conjugate(), Complex::new(3.0, -4.0));
        assert_eq!(Complex::new(0.0, 4.0).conjugate(), Complex::new(0.0, -4.0));
    }

    #[test]
    fn conjugate_of_conjugate_is_the_original() {
        let z = Complex::new(-7.25, 0.5);
        assert_eq!(z.conjugate().conjugate(), z);
    }

    #[test]
    fn norm_sqr_is_the_squared_magnitude() {
        assert_eq!(Complex::new(3.0, 4.0).norm_sqr(), 25.0);
        assert_eq!(Complex::ZERO.norm_sqr(), 0.0);
    }

    #[test]
    fn norm_is_the_magnitude() {
        assert_eq!(Complex::new(3.0, 4.0).norm(), 5.0);
        assert_eq!(Complex::new(-3.0, -4.0).norm(), 5.0);
        assert!(close(Complex::new(1.0, 1.0).norm(), 2.0f64.sqrt(), EPSILON));
    }

    #[test]
    fn norm_hypot_agrees_with_norm_on_ordinary_values() {
        for &(re, im) in &[
            (0.0, 0.0),
            (1.0, 1.0),
            (3.0, -4.0),
            (-0.5, 0.25),
            (1e10, 1e10),
        ] {
            let z = Complex::new(re, im);
            assert!(
                close(z.norm_hypot(), z.norm(), 4.0 * EPSILON),
                "{}: hypot {} vs norm {}",
                z,
                z.norm_hypot(),
                z.norm()
            );
        }
    }

    #[test]
    fn norm_hypot_survives_where_norm_overflows() {
        let z = Complex::new(3e200, 4e200);
        assert!(z.norm().is_infinite(), "the naive form must overflow here");
        assert!(
            close(z.norm_hypot(), 5e200, 4.0 * EPSILON),
            "hypot gave {}",
            z.norm_hypot()
        );
    }

    #[test]
    fn norm_hypot_survives_where_norm_underflows() {
        let z = Complex::new(3e-200, 4e-200);
        assert_eq!(z.norm(), 0.0, "the naive form must underflow here");
        assert_eq!(z.norm_hypot(), 5e-200);
    }

    #[test]
    fn arg_covers_all_four_quadrants() {
        let q = 2.0f64.sqrt() / 2.0; // sin(pi/4) == cos(pi/4)
        assert!(close(Complex::new(q, q).arg(), PI / 4.0, EPSILON));
        assert!(close(Complex::new(-q, q).arg(), 3.0 * PI / 4.0, EPSILON));
        assert!(close(Complex::new(-q, -q).arg(), -3.0 * PI / 4.0, EPSILON));
        assert!(close(Complex::new(q, -q).arg(), -PI / 4.0, EPSILON));
    }

    #[test]
    fn arg_of_a_negative_real_is_exactly_pi() {
        assert_eq!(Complex::new(-2.0, 0.0).arg(), PI);
        assert_eq!(Complex::new(-1.0, 0.0).arg(), PI);
    }

    #[test]
    fn arg_of_zero_is_zero() {
        assert_eq!(Complex::ZERO.arg(), 0.0);
    }

    #[test]
    fn arg_uses_the_sign_of_a_zero_imaginary_part() {
        // This is the whole reason `Complex` keeps -0.0 around: it is what
        // picks a side of the branch cut of log and sqrt.
        assert_eq!(Complex::new(-1.0, -0.0).arg(), -PI);
        assert_eq!(Complex::new(-1.0, 0.0).arg(), PI);
    }

    #[test]
    fn multiplication_by_one_is_the_identity_on_both_sides() {
        let z = Complex::new(-2.75, 6.5);
        assert_eq!(Complex::ONE * z, z);
        assert_eq!(z * Complex::ONE, z);
    }

    #[test]
    fn multiplication_by_zero_is_zero() {
        let z = Complex::new(1.5, -2.5);
        assert_eq!(Complex::ZERO * z, Complex::ZERO);
    }

    #[test]
    fn i_times_i_is_minus_one() {
        assert_eq!(Complex::I * Complex::I, -Complex::ONE);
    }

    #[test]
    fn multiplication_is_commutative_and_associative() {
        let a = Complex::new(0.5, 0.25);
        let b = Complex::new(-2.0, 4.0);
        let c = Complex::new(1.0, -0.5);
        // Powers of two so the f64 arithmetic below is exact, and any failure
        // is the algebra's fault rather than a rounding accident.
        assert_eq!(a * b, b * a);
        assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn one_over_i_is_minus_i() {
        assert_eq!(Complex::ONE / Complex::I, -Complex::I);
        assert_eq!(Complex::I / Complex::ONE, Complex::I);
    }

    #[test]
    fn reciprocal_times_the_original_is_one() {
        for &z in &[
            Complex::new(3.0, -4.0),
            Complex::new(-0.125, 0.0625),
            Complex::new(1e-3, 0.0),
        ] {
            assert!(
                close_complex(Complex::ONE / z * z, Complex::ONE, 1e-15),
                "at {}",
                z
            );
        }
    }

    #[test]
    fn division_round_trips_against_multiplication() {
        let a = Complex::new(2.0, -3.0);
        let b = Complex::new(0.5, 1.25);
        let q = a / b;
        assert!(close_complex(q * b, a, 1e-15));
        assert!(close_complex(b * q, a, 1e-15));
    }

    #[test]
    fn division_by_the_matching_conjugate_is_the_quotient_of_norms() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let q = a / b;
        let expected = Complex::new(
            (a.re * b.re + a.im * b.im) / b.norm_sqr(),
            (a.im * b.re - a.re * b.im) / b.norm_sqr(),
        );
        assert_eq!(q, expected);
    }

    #[test]
    fn division_by_zero_does_not_panic_and_reports_the_pole() {
        // The dangerous one: an escape-time orbit walking into 1/0.
        let q = Complex::new(1.0, 0.0) / Complex::ZERO;
        assert_eq!(q.re, f64::INFINITY);
        assert_eq!(q.im, f64::INFINITY);
    }

    #[test]
    fn zero_over_zero_is_nan() {
        let q = Complex::ZERO / Complex::ZERO;
        assert!(q.re.is_nan());
        assert!(q.im.is_nan());
        assert!(!q.is_finite());
    }

    #[test]
    fn subtracting_a_number_itself_is_exactly_zero() {
        let z = Complex::new(1e17, -4.5);
        assert_eq!(z - z, Complex::ZERO);
    }

    #[test]
    fn addition_is_componentwise() {
        assert_eq!(
            Complex::new(1.5, -2.5) + Complex::new(0.25, 2.0),
            Complex::new(1.75, -0.5)
        );
    }

    #[test]
    fn negation_flips_both_components() {
        assert_eq!(-Complex::new(1.5, -2.5), Complex::new(-1.5, 2.5));
    }

    #[test]
    fn scaling_by_a_real_factor_works_from_either_side() {
        let z = Complex::new(1.5, -2.5);
        assert_eq!(z * 2.0, Complex::new(3.0, -5.0));
        assert_eq!(2.0 * z, Complex::new(3.0, -5.0));
        assert_eq!(z * 0.0, Complex::ZERO);
        assert_eq!(z * -1.0, -z);
        assert_eq!(z * 0.5, Complex::new(0.75, -1.25));
    }

    #[test]
    fn dividing_by_a_real_factor_is_the_componentwise_quotient() {
        let z = Complex::new(1.0, -2.0);
        assert_eq!(z / 2.0, Complex::new(0.5, -1.0));
        assert_eq!(z / 0.0, Complex::new(f64::INFINITY, f64::NEG_INFINITY));
        assert!((Complex::ZERO / 0.0).re.is_nan());
    }

    #[test]
    fn a_real_divided_by_a_complex_number_is_the_scaled_reciprocal() {
        // 1/z and the scaled conjugate reciprocal must be the same number, and
        // neither may panic at a pole.
        let z = Complex::new(3.0, -4.0);
        assert!(close_complex(1.0 / z, Complex::ONE / z, 1e-15));
        assert!(close_complex(2.0 / z, (Complex::ONE / z) * 2.0, 1e-15));
        assert!(close_complex(1.0 / z, z.conjugate() / 25.0, 1e-15));
        let indet = 0.0 / Complex::ZERO;
        assert!(indet.re.is_nan() && indet.im.is_nan(), "0/0 was {}", indet);
        assert_eq!(
            1.0 / Complex::ZERO,
            Complex::new(f64::INFINITY, f64::INFINITY)
        );
    }

    #[test]
    fn nan_propagates_through_the_arithmetic() {
        let nan = Complex::new(f64::NAN, f64::NAN);
        assert!((nan + Complex::ONE).re.is_nan());
        assert!((Complex::ONE * nan).im.is_nan());
        assert!(!(nan / Complex::ONE).re.is_finite());
    }

    #[test]
    fn infinities_propagate_through_the_arithmetic() {
        let inf = Complex::new(f64::INFINITY, 0.0);
        assert!(!inf.is_finite());
        // Multiplying by one keeps the real part infinite...
        assert!((inf * Complex::ONE).re.is_infinite());
        // ...but the imaginary part of that product is `inf * 0 + 0 * inf`,
        // which is NaN. This is the single most surprising fact about complex
        // arithmetic in IEEE-754 and it is what turns an escaped orbit into
        // NaN one step after it reaches infinity. See Complex::mul's note.
        assert!((inf * Complex::ONE).im.is_nan());
    }

    #[test]
    fn division_by_a_divisor_whose_norm_overflows_gives_a_signed_zero() {
        // d = re^2 + im^2 overflows here, so the naive quotient is a zero
        // rather than a very small number. Documented on Div, and pinned here
        // because it is a real (if extreme) loss of accuracy rather than a bug.
        let q = Complex::ONE / Complex::new(1e300, 0.0);
        assert_eq!(q, Complex::ZERO);
        assert!(q.re.is_sign_positive());
    }

    #[test]
    fn is_finite_classifies_ordinary_points_and_holes() {
        assert!(Complex::new(1.0, 1.0).is_finite());
        assert!(Complex::ZERO.is_finite());
        assert!(!Complex::new(f64::NAN, 0.0).is_finite());
        assert!(!Complex::new(0.0, f64::INFINITY).is_finite());
    }

    #[test]
    fn norm_sqr_overflow_is_documented_and_observable() {
        // The limitation the Div doc calls out, pinned so it stays a known,
        // intentional limitation rather than a surprise.
        assert!(Complex::new(1e200, 1e200).norm_sqr().is_infinite());
        assert!(Complex::new(1e-200, 1e-200).norm_sqr() == 0.0);
    }

    #[test]
    fn display_prints_the_origin_and_pure_reals() {
        assert_eq!(Complex::ZERO.to_string(), "0");
        assert_eq!(Complex::ONE.to_string(), "1");
        assert_eq!(Complex::new(-2.5, 0.0).to_string(), "-2.5");
    }

    #[test]
    fn display_prints_pure_imaginary_numbers_conventionally() {
        assert_eq!(Complex::I.to_string(), "i");
        assert_eq!((-Complex::I).to_string(), "-i");
        assert_eq!(Complex::new(0.0, 5.0).to_string(), "5i");
        assert_eq!(Complex::new(0.0, -0.25).to_string(), "-0.25i");
    }

    #[test]
    fn display_signs_the_imaginary_part_of_a_general_number() {
        assert_eq!(Complex::new(3.0, 4.0).to_string(), "3+4i");
        assert_eq!(Complex::new(3.0, -4.0).to_string(), "3-4i");
        assert_eq!(Complex::new(-3.0, 4.0).to_string(), "-3+4i");
        assert_eq!(Complex::new(-3.0, -4.0).to_string(), "-3-4i");
    }

    #[test]
    fn display_does_not_hide_non_finite_components() {
        assert_eq!(Complex::new(f64::NAN, 0.0).to_string(), "NaN");
        assert_eq!(
            Complex::new(f64::INFINITY, 0.0).to_string(),
            "inf",
            "an escape-time pixel should still be printable"
        );
    }
}
