//! The function dispatch table, and the CPU reference implementation of it.
//!
//! # This file is the reference. The shader is the copy.
//!
//! `shaders/domain_coloring.wgsl` is a transcription of this module, and the
//! two **must agree numerically** - the CPU is where the mathematics is
//! written down, checked by the tests below, and read by anyone debugging a
//! suspicious picture. The GPU is where it is evaluated a few million times a
//! second. If the two ever disagree about a function's value, the shader is
//! wrong, not this file.
//!
//! Three consequences shape everything below:
//!
//! 1. **The arithmetic is written out step by step.** Every function is the
//!    textbook formula spelled out in the two real components, in the order a
//!    WGSL author can transcribe line for line. No iterator, no shortcut, no
//!    "obviously this is the same as that". When the two files are compared,
//!    the diff should be about syntax, never about arithmetic.
//! 2. **The function ids are a hard ABI.** They go straight into
//!    [`crate::uniforms::Uniforms::func_id`] and are switched on by the shader.
//!    Renumbering anything here breaks the picture silently, not loudly. See
//!    [`FUNCTIONS`].
//! 3. **Nothing is smoothed over.** The branch cut of `log` and `sqrt`, the
//!    pole of `1/z`, the removable singularity of `sinc` - these are the
//!    features being plotted. A "helpful" continuous branch would erase the
//!    exact structure this program exists to show.
//!
//! # The one deliberate patch, and why it is allowed
//!
//! `sinc` is the only function that is not a literal transcription of its
//! formula: it returns `1` on a small disc around the origin instead of
//! `0/0`. See [`SINC_SINGULARITY_GUARD_RADIUS`]. Everywhere else, including the
//! infinities and NaNs the arithmetic produces, this module lets the answer
//! through untouched.
//!
//! # Function ids
//!
//! | id | name | formula | group |
//! |---:|---|---|---|
//! | 0 | identity | `z` | Polynomial |
//! | 1 | square | `z^2` | Polynomial |
//! | 2 | cube | `z^3` | Polynomial |
//! | 3 | reciprocal | `1/z` | Reciprocal |
//! | 4 | z2_minus_1 | `z^2 - 1` | Polynomial |
//! | 5 | z3_minus_1 | `z^3 - 1` | Polynomial |
//! | 6 | basilica | `z^3 - 2z` | Iterated |
//! | 7 | sin | `sin z` | Trigonometric |
//! | 8 | cos | `cos z` | Trigonometric |
//! | 9 | sinc | `sin z / z` | Trigonometric |
//! | 10 | sinh | `sinh z` | Hyperbolic |
//! | 11 | exp | `e^z` | Exponential |
//! | 12 | log | `ln z` | Transcendental |
//! | 13 | sqrt | `sqrt z` | Transcendental |
//! | 14 | mobius | `(z - 1) / (z+1)` | Mobius |
//! | 15 | julia | `z^2 + c` | Iterated |

//! # Why this module is partly `#[allow(dead_code)]`
//!
//! The *table* here is live: `src/panel.rs` derives its selector rows from
//! `FUNCTIONS`, and the plot's corner readout calls `label_for`, so the ids,
//! names, formulas and groups are what the application dispatches on and
//! displays. The *evaluator* — `eval`, `apply_once` and the constants only it
//! reads — is not, and is not supposed to be: nothing on the CPU evaluates these
//! functions, because the GPU is the thing that draws. The `f64` code exists to
//! be the oracle the WGSL is checked against, and the only caller of an oracle is
//! the test suite.
//!
//! So this is a deliberate, narrow suppression of one module rather than a
//! blanket one: the lint's complaint is that `rustc` cannot see the tests from a
//! binary crate, which is true and is not a defect in the code. What is **not**
//! covered by that sentence is anything new added to this file without a caller —
//! a new helper must be wired into the panel, used by `eval`, or deleted. Do not
//! add to this file on the strength of the attribute below.

#![allow(dead_code)]

use crate::complex::Complex;

/// How a function is presented in the control panel, and how the tests group
/// related behaviour.
///
/// The grouping is presentational only - nothing in the evaluator switches on
/// it - but it is chosen so that each variant has at least one member and the
/// iterated maps are set apart, since that is the distinction a user actually
/// makes when picking something to look at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionGroup {
    /// Algebraic maps: powers, shifts, and the identity.
    ///
    /// The identity map lives here because `z` is the degree-1 polynomial `z`,
    /// which is a fact rather than a convenience.
    Polynomial,
    /// Maps with a pole, `1/z` and friends.
    Reciprocal,
    /// `sin`, `cos`, and the normalised `sinc`.
    Trigonometric,
    /// `sinh`, and by extension `cosh`, `tanh` should they ever be added.
    Hyperbolic,
    /// `e^z` and `e^{iz}`.
    Exponential,
    /// Everything else transcendental: `log` and `sqrt`, which is where the
    /// interesting branch cuts live.
    Transcendental,
    /// Linear fractional (Möbius) transformations, which act on the Riemann
    /// sphere rather than the plane.
    Mobius,
    /// Quadratic polynomials shown for their Julia-set behaviour.
    ///
    /// These are polynomials too, and are implemented as polynomials; the
    /// group says "the point of this one is what happens when you iterate it".
    Iterated,
}

/// One row of the function table.
///
/// This is everything the UI needs to draw a menu entry and everything the
/// rest of the program needs to name a function. The `id` is the part that
/// crosses into the shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FunctionEntry {
    /// The dispatch id, written into
    /// [`crate::uniforms::Uniforms::func_id`] and switched on by the shader.
    pub id: u32,
    /// Short human-readable name, unique across the table, used in the UI and
    /// by [`label_for`].
    pub name: &'static str,
    /// The mathematical formula, as a human would write it, for display only.
    /// Never parsed.
    pub formula: &'static str,
    /// Which section of the menu this belongs in.
    pub group: FunctionGroup,
}

/// Upper bound on the number of applications performed by [`eval`] when
/// `iterate` is set.
///
/// A hostile or fat-fingered `max_iter` - `u32::MAX` from a slider that lost
/// its clamp, or a corrupted uniform - would otherwise turn a single frame into
/// a hang, so the count is clamped to this. It is high enough that no
/// interesting orbit in any of the 16 functions is cut short (the Julia-set
/// members need hundreds of iterations to resolve a boundary) and low enough
/// that a full-resolution frame finishes: 4096 steps is a few microseconds per
/// pixel in the best case and the worst case is a tight, branchy loop that the
/// GPU can absorb.
pub const ITERATION_CAP: u32 = 4096;

/// Radius inside which [`sinc`] returns `1` instead of dividing by zero.
///
/// The singularity of `sin(z)/z` at `z = 0` is *removable*: the limit is `1`.
/// Without a guard, the origin of the `sinc` picture is an exact `0/0` and
/// paints a black hole - a single pixel (or a disc, once iterated or
/// anti-aliased) of garbage in the middle of the smoothest function in the
/// table. `1e-12` is far below any visible scale at the default zoom and far
/// above the point where the true quotient has drifted appreciably from `1`, so
/// the patch is invisible and the singularity is gone.
pub const SINC_SINGULARITY_GUARD_RADIUS: f64 = 1e-12;

/// The iteration constant for the `julia` function.
///
/// The iteration constant for the `julia` function.
///
/// The sign convention matters: the parameter enters as `z^2 + c`, so `c`'s
/// real part is negative.
///
/// # Why this value, measured rather than chosen by eye
///
/// This was `-0.7269 + 0.1889i`, chosen for a dendrite's visible branching
/// structure. That was a mistake: a dendrite Julia set has **empty interior**,
/// so there is no filled set to draw and an iterated plot of it is almost
/// entirely escaping points. Measured over a 96x96 grid on `[-2,2]^2` after
/// 2000 iterations, counting points whose orbit is still finite:
///
/// | `c`                 | bounded fraction | filled set? |
/// |---------------------|------------------|-------------|
/// | `-0.7269 + 0.1889i` (old) | 0.00%      | no          |
/// | `-0.75`                    | 12.89%     | **yes**     |
/// | `-0.123 + 0.745i` (rabbit) | 8.01%      | **yes**     |
/// | `0.285 + 0.01i`             | 0.00%      | no          |
/// | `-0.8 + 0.156i`            | 0.00%      | no          |
/// | `-0.01 + 0.65i`            | 0.00%      | no          |
///
/// Note that `0.285 + 0.01i` was previously proposed as a "small, well
/// contained" candidate. It measures zero: it lies outside the Mandelbrot set,
/// so its Julia set is a Cantor dust with no interior. Do not reinstate it.
///
/// `-0.75` is the largest interior of the candidates, so it is the robust
/// choice: the filled set is unmistakable at any zoom and the plot still reads
/// correctly at a low `max_iter`. It is the root of the period-2 hyperbolic
/// component on the real axis, where the critical orbit converges to the
/// parabolic fixed point `-0.5`. Do **not** call it the basilica: the basilica
/// is `c = -1`. The distinction matters because the two look quite different,
/// and the period-2 root has a property the basilica does not - see the
/// saturation note on the `julia` arm of `apply_once`.
///
/// The rabbit (`-0.123 + 0.745i`) is the more intricate alternative and is a
/// reasonable substitute; it is a visual preference, not a correctness one.
///
/// Keep this constant in step with `JULIA_C` in `shaders/domain_coloring.wgsl`.
const JULIA_C: Complex = Complex { re: -0.75, im: 0.0 };

/// Function id: `z`, the identity.
const ID_IDENTITY: u32 = 0;
/// Function id: `z^2`.
const ID_SQUARE: u32 = 1;
/// Function id: `z^3`.
const ID_CUBE: u32 = 2;
/// Function id: `1/z`.
const ID_RECIPROCAL: u32 = 3;
/// Function id: `z^2 - 1`.
const ID_Z2_MINUS_1: u32 = 4;
/// Function id: `z^3 - 1`.
const ID_Z3_MINUS_1: u32 = 5;
/// Function id: `z^3 - 2z`.
const ID_BASILICA: u32 = 6;
/// Function id: `sin z`.
const ID_SIN: u32 = 7;
/// Function id: `cos z`.
const ID_COS: u32 = 8;
/// Function id: `sin(z)/z`.
const ID_SINC: u32 = 9;
/// Function id: `sinh z`.
const ID_SINH: u32 = 10;
/// Function id: `e^z`.
const ID_EXP: u32 = 11;
/// Function id: `ln z`, principal branch.
const ID_LOG: u32 = 12;
/// Function id: `sqrt(z)`, principal branch.
const ID_SQRT: u32 = 13;
/// Function id: `(z - 1)/(z + 1)`.
const ID_MOBIUS: u32 = 14;
/// Function id: `z^2 + c`, iterated.
const ID_JULIA: u32 = 15;

/// Number of entries in [`FUNCTIONS`]; also the exclusive upper bound on any
/// function id.
pub const FUNCTION_COUNT: u32 = 16;

/// The dispatch table, in id order.
///
/// **This array's order is the ABI.** Entry `k` has `id == k`, the kernel
/// switches on `Uniforms::func_id` with the numbers in the module
/// documentation, and neither side may renumber anything. Adding a function
/// means appending a row here and adding a case to `apply_once` *and* the
/// matching `case` in the WGSL, in the same change.
pub const FUNCTIONS: [FunctionEntry; 16] = [
    FunctionEntry {
        id: ID_IDENTITY,
        name: "identity",
        formula: "z",
        group: FunctionGroup::Polynomial,
    },
    FunctionEntry {
        id: ID_SQUARE,
        name: "square",
        formula: "z^2",
        group: FunctionGroup::Polynomial,
    },
    FunctionEntry {
        id: ID_CUBE,
        name: "cube",
        formula: "z^3",
        group: FunctionGroup::Polynomial,
    },
    FunctionEntry {
        id: ID_RECIPROCAL,
        name: "reciprocal",
        formula: "1/z",
        group: FunctionGroup::Reciprocal,
    },
    FunctionEntry {
        id: ID_Z2_MINUS_1,
        name: "z^2 - 1",
        formula: "z^2 - 1",
        group: FunctionGroup::Polynomial,
    },
    FunctionEntry {
        id: ID_Z3_MINUS_1,
        name: "z^3 - 1",
        formula: "z^3 - 1",
        group: FunctionGroup::Polynomial,
    },
    FunctionEntry {
        id: ID_BASILICA,
        name: "basilica",
        formula: "z^3 - 2z",
        group: FunctionGroup::Iterated,
    },
    FunctionEntry {
        id: ID_SIN,
        name: "sin",
        formula: "sin z",
        group: FunctionGroup::Trigonometric,
    },
    FunctionEntry {
        id: ID_COS,
        name: "cos",
        formula: "cos z",
        group: FunctionGroup::Trigonometric,
    },
    FunctionEntry {
        id: ID_SINC,
        name: "sinc",
        formula: "sin z / z",
        group: FunctionGroup::Trigonometric,
    },
    FunctionEntry {
        id: ID_SINH,
        name: "sinh",
        formula: "sinh z",
        group: FunctionGroup::Hyperbolic,
    },
    FunctionEntry {
        id: ID_EXP,
        name: "exp",
        formula: "e^z",
        group: FunctionGroup::Exponential,
    },
    FunctionEntry {
        id: ID_LOG,
        name: "log",
        formula: "ln z",
        group: FunctionGroup::Transcendental,
    },
    FunctionEntry {
        id: ID_SQRT,
        name: "sqrt",
        formula: "sqrt z",
        group: FunctionGroup::Transcendental,
    },
    FunctionEntry {
        id: ID_MOBIUS,
        name: "mobius",
        // The spacing here is `(z+1)` rather than `(z + 1)`. This is a display
        // string, never parsed, and the same string is what `src/panel.rs` and
        // the plot's corner readout display: the panel no longer keeps its own
        // copy, so there is nothing to keep in step with it any more.
        formula: "(z - 1) / (z+1)",
        group: FunctionGroup::Mobius,
    },
    FunctionEntry {
        id: ID_JULIA,
        name: "julia",
        formula: "z^2 + c",
        group: FunctionGroup::Iterated,
    },
];

/// Look up a function table row by id.
///
/// Returns `None` for any id outside `0..16`. The kernel has no such thing as
/// a partial match - its `switch` either hits a case or falls into a default -
/// so callers that must not fail should use [`label_for`], and callers that
/// are about to evaluate should check this first.
pub fn entry_for(func_id: u32) -> Option<&'static FunctionEntry> {
    if (func_id as usize) < FUNCTIONS.len() {
        Some(&FUNCTIONS[func_id as usize])
    } else {
        None
    }
}

/// The human-readable name of a function, never failing.
///
/// Out-of-range ids get `"unknown"` rather than a panic: this is the function
/// a status bar or a telemetry record calls with a value that came from
/// outside, and a label is never worth a crash.
pub fn label_for(func_id: u32) -> &'static str {
    match entry_for(func_id) {
        Some(entry) => entry.name,
        None => "unknown",
    }
}

/// Evaluate a function from the table.
///
/// With `iterate == false` the function is applied once, exactly, and
/// `max_iter` is ignored entirely - so `eval(id, z, 1, true)` and
/// `eval(id, z, 999_999, false)` are the same call.
///
/// With `iterate == true` the function is applied `max_iter` times, starting
/// from `z`: the returned value is `f(f(...f(z)))`. That is the control that
/// turns any function in the table into an escape-time picture - the Julia
/// members are designed for it, the polynomial members gain their classic
/// fractals, and the transcendental members grow their own orbit structures.
/// `max_iter == 0` returns `z` unchanged.
///
/// The count is clamped to [`ITERATION_CAP`]; see that constant for why. Note
/// that this loop does *not* stop early on a non-finite intermediate, unlike
/// the shader's. That is not a disagreement: in IEEE-754 arithmetic a
/// non-finite value can never become finite again, so every function in the
/// table is absorbing there and the two produce the same answer for the same
/// `max_iter`. The cap does the bounding on this side.
///
/// # Detecting an escaped orbit
///
/// An orbit that runs away does not settle at infinity. It is an infinity for
/// one or two steps - `|z|` passes `f64::MAX` and the next multiplication
/// overflows - and then `inf * 0` turns the imaginary component to `NaN`, and
/// the one after that has both components `NaN`. Callers must therefore test
/// [`Complex::is_finite`] and must never test for an infinity. The shader has
/// exactly the same behaviour, since it is the same IEEE-754 arithmetic, and
/// paints every non-finite value the same solid black, so the two cannot
/// disagree about which pixels are "at infinity".
///
/// An id outside `0..16` is treated as the identity, which is what the shader's
/// `default:` branch does - a stale UI selection renders something rather than
/// leaving the previous frame's texture on screen. The two must agree here;
/// neither is a panic, because the id arrives from a uniform buffer.
pub fn eval(func_id: u32, z: Complex, max_iter: u32, iterate: bool) -> Complex {
    if !iterate {
        return apply_once(func_id, z);
    }
    let steps = max_iter.min(ITERATION_CAP);
    let mut w = z;
    for _ in 0..steps {
        w = apply_once(func_id, w);
    }
    w
}

/// Apply a function from the table exactly once.
///
/// Private because [`eval`] is the whole public surface: a single application
/// is `eval(id, z, 1, true)`. The bodies below are the reference the WGSL
/// `switch` mirrors, so they are written to be read top to bottom against the
/// shader, not to be fast.
fn apply_once(func_id: u32, z: Complex) -> Complex {
    match func_id {
        // z
        ID_IDENTITY => z,

        // z^2
        ID_SQUARE => z * z,

        // z^3, formed as (z*z)*z so the GPU can reuse the square.
        ID_CUBE => z * z * z,

        // 1/z. The pole at 0 comes out as an infinity from Complex's Div, which
        // is exactly the feature being plotted.
        ID_RECIPROCAL => Complex::ONE / z,

        // z^2 - 1
        ID_Z2_MINUS_1 => z * z - Complex::ONE,

        // z^3 - 1
        ID_Z3_MINUS_1 => z * z * z - Complex::ONE,

        // z^3 - 2z. `z + z` is a doubling, which is exact in binary floating
        // point, so this is bit-for-bit "two z" and not an approximation of it.
        ID_BASILICA => z * z * z - (z + z),

        // sin z = sin(re) cosh(im) + i cos(re) sinh(im)
        // Separating even and odd parts like this is what stops the
        // transcendental sine from drowning: sin(re) is bounded by 1, and only
        // cosh(im) grows. The alternative, sin(re + i*im), would overflow for
        // |im| > ~710 and turn the whole vertical axis into a stripe of NaN.
        ID_SIN => {
            let s = z.re.sin();
            let c = z.re.cos();
            let sh = z.im.sinh();
            let ch = z.im.cosh();
            Complex::new(s * ch, c * sh)
        }

        // cos z = cos(re) cosh(im) - i sin(re) sinh(im)
        ID_COS => {
            let s = z.re.sin();
            let c = z.re.cos();
            let sh = z.im.sinh();
            let ch = z.im.cosh();
            Complex::new(c * ch, -(s * sh))
        }

        // sinc z = sin z / z, with the removable singularity at 0 patched out.
        // See SINC_SINGULARITY_GUARD_RADIUS. This is the only place in the
        // table where the arithmetic below is not a literal transcription of
        // the formula, and the shader must carry the same guard.
        ID_SINC => {
            if z.norm() < SINC_SINGULARITY_GUARD_RADIUS {
                return Complex::ONE;
            }
            let s = z.re.sin();
            let c = z.re.cos();
            let sh = z.im.sinh();
            let ch = z.im.cosh();
            let sin_z = Complex::new(s * ch, c * sh);
            sin_z / z
        }

        // sinh z = sinh(re) cos(im) + i cosh(re) sin(im)
        ID_SINH => {
            let sh = z.re.sinh();
            let ch = z.re.cosh();
            let s = z.im.sin();
            let c = z.im.cos();
            Complex::new(sh * c, ch * s)
        }

        // e^z = e^re (cos im + i sin im). The same separation as the
        // trigonometrics: the exponential growth lives in e^re alone, so the
        // imaginary part never overflows on its own.
        ID_EXP => {
            let e = z.re.exp();
            let s = z.im.sin();
            let c = z.im.cos();
            Complex::new(e * c, e * s)
        }

        // ln z = ln|z| + i arg(z), principal branch: arg in (-pi, pi], so the
        // branch cut is the non-positive real axis, where the imaginary part
        // jumps from +pi to -pi. That jump is the picture's most visible
        // feature; do not "fix" it by folding the argument into [0, 2pi).
        //
        // `norm_hypot` rather than `norm`: ln|z| must stay finite for |z| up to
        // ~1.8e308, and the naive form overflows the magnitude at ~1.3e154.
        // The shader uses its own length(), which is the naive form; the two
        // agree everywhere either is finite, and past that point the argument
        // is large enough that one ULP cannot be seen.
        ID_LOG => Complex::new(z.norm_hypot().ln(), z.arg()),

        // sqrt z, principal branch:
        //   Re = sqrt((|z| + re) / 2)
        //   Im = sign(im) sqrt((|z| - re) / 2)
        // Both square roots have non-negative radicands by construction, and
        // the second is the smaller one, so this is the square root with
        // argument in [-pi/2, pi/2] - the principal value, with its branch cut
        // on the negative real axis. Note `is_sign_negative` and not `< 0.0`:
        // it is true for -0.0, which is what makes sqrt(-1 - 0i) the conjugate
        // of sqrt(-1 + 0i) and agrees with arg()'s choice of -pi.
        ID_SQRT => {
            let r = z.norm_hypot();
            let re_part = ((r + z.re) * 0.5).sqrt();
            let im_part = ((r - z.re) * 0.5).sqrt();
            let im_sign = if z.im.is_sign_negative() { -1.0 } else { 1.0 };
            Complex::new(re_part, im_sign * im_part)
        }

        // (z - 1)/(z + 1), the linear fractional map that sends 0 to -1 and
        // fixes +-i. It maps the upper half plane to the unit disc. Its pole is
        // at -1 and its zero at 1, so the picture is a picture of the Riemann
        // sphere: both of those points are on the domain's edge in a default
        // view and both are where the interesting shading lives.
        //
        // Iterated, it does *not* return to where it started. It has order
        // four: f^2 is -1/z, f^3 is (1+z)/(1-z), and f^4 is the identity. The
        // tests pin all four, because "a Möbius map undoes itself" is the
        // natural wrong assumption here and the plotter draws the whole cycle.
        ID_MOBIUS => (z - Complex::ONE) / (z + Complex::ONE),

        // z^2 + c, the Julia map. c is JULIA_C; see that constant. Note the
        // dendrite: for this parameter the filled set has no interior, so an
        // iterated plot is a picture of escape times and nothing else.
        ID_JULIA => z * z + JULIA_C,

        // The shader's default branch, which is the identity. Keep these two in
        // step: an id outside the table is not a panic, because the id arrives
        // from a uniform buffer that a stale UI could have written.
        _ => z,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{PI, TAU};
    use std::f64::EPSILON;

    /// `|a - b| <= tol * max(1, |a|, |b|)`: relative where that means
    /// something, absolute near zero.
    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
    }

    fn close_complex(a: Complex, b: Complex, tol: f64) -> bool {
        close(a.re, b.re, tol) && close(a.im, b.im, tol)
    }

    #[test]
    fn every_id_resolves_to_its_own_entry() {
        for id in 0..FUNCTION_COUNT {
            let entry =
                entry_for(id).unwrap_or_else(|| panic!("id {} must resolve to an entry", id));
            assert_eq!(entry.id, id, "entry {} is filed under the wrong id", id);
        }
    }

    #[test]
    fn the_table_is_in_contract_order() {
        let expected: [(&str, &str); 16] = [
            ("identity", "z"),
            ("square", "z^2"),
            ("cube", "z^3"),
            ("reciprocal", "1/z"),
            ("z^2 - 1", "z^2 - 1"),
            ("z^3 - 1", "z^3 - 1"),
            ("basilica", "z^3 - 2z"),
            ("sin", "sin z"),
            ("cos", "cos z"),
            ("sinc", "sin z / z"),
            ("sinh", "sinh z"),
            ("exp", "e^z"),
            ("log", "ln z"),
            ("sqrt", "sqrt z"),
            ("mobius", "(z - 1) / (z+1)"),
            ("julia", "z^2 + c"),
        ];
        for (id, (name, formula)) in expected.iter().enumerate() {
            let entry = entry_for(id as u32).unwrap();
            assert_eq!(entry.name, *name, "id {} has the wrong name", id);
            assert_eq!(entry.formula, *formula, "id {} has the wrong formula", id);
        }
    }

    #[test]
    fn table_names_are_unique() {
        for (i, a) in FUNCTIONS.iter().enumerate() {
            for (j, b) in FUNCTIONS.iter().enumerate() {
                assert!(i == j || a.name != b.name, "duplicate name {}", a.name);
            }
        }
    }

    #[test]
    fn every_group_has_at_least_one_member() {
        let groups = [
            FunctionGroup::Polynomial,
            FunctionGroup::Reciprocal,
            FunctionGroup::Trigonometric,
            FunctionGroup::Hyperbolic,
            FunctionGroup::Exponential,
            FunctionGroup::Transcendental,
            FunctionGroup::Mobius,
            FunctionGroup::Iterated,
        ];
        for g in groups {
            assert!(
                FUNCTIONS.iter().any(|e| e.group == g),
                "{:?} has no members",
                g
            );
        }
    }

    #[test]
    fn the_julia_maps_are_the_iterated_group() {
        for id in [ID_BASILICA, ID_JULIA] {
            assert_eq!(entry_for(id).unwrap().group, FunctionGroup::Iterated);
        }
    }

    #[test]
    fn entry_for_rejects_ids_outside_the_table() {
        assert!(entry_for(FUNCTION_COUNT).is_none());
        assert!(entry_for(4096).is_none());
        assert!(entry_for(u32::MAX).is_none());
    }

    #[test]
    fn label_for_never_fails() {
        for id in 0..FUNCTION_COUNT {
            assert_eq!(label_for(id), FUNCTIONS[id as usize].name);
        }
        assert_eq!(label_for(FUNCTION_COUNT), "unknown");
        assert_eq!(label_for(u32::MAX), "unknown");
    }

    #[test]
    fn an_unknown_id_falls_back_to_the_identity_exactly_as_the_shader_does() {
        // The shader's `default:` branch returns the input, so this must too.
        // An id outside the table is not a panic: the value arrives from a
        // uniform buffer.
        for &id in &[FUNCTION_COUNT, 16, 4096, u32::MAX] {
            let z = Complex::new(1.0, -2.0);
            assert_eq!(eval(id, z, 1, false), z, "id {} did not fall back", id);
            assert_eq!(eval(id, z, 8, true), z, "id {} did not fall back", id);
        }
    }

    #[test]
    fn a_non_iterated_eval_ignores_max_iter() {
        let z = Complex::new(0.3, -0.7);
        for id in 0..FUNCTION_COUNT {
            let once = eval(id, z, 1, true);
            let many = eval(id, z, 999_999, false);
            assert_eq!(once, many, "id {} changed with max_iter", id);
        }
    }

    #[test]
    fn identity_and_square_match_their_formulas() {
        let z = Complex::new(1.5, -2.5);
        assert_eq!(eval(ID_IDENTITY, z, 1, false), z);
        assert_eq!(eval(ID_SQUARE, z, 1, false), z * z);
        assert_eq!(eval(ID_CUBE, z, 1, false), z * z * z);
    }

    #[test]
    fn the_cube_is_the_square_and_one_more_multiplication() {
        // Composition of the table's own entries, to check that `cube` is
        // z*z*z and not, say, z*z*z*z.
        let z = Complex::new(0.75, 1.25);
        let squared = eval(ID_SQUARE, z, 1, false);
        assert_eq!(eval(ID_CUBE, z, 1, false), squared * z);
        assert_eq!(eval(ID_CUBE, z, 1, false), z * squared);
    }

    #[test]
    fn i_cubed_is_minus_i() {
        // z*z*z with the multiplication above: a three-step check of the
        // sign convention, which is easy to get backwards and hard to see.
        assert_eq!(eval(ID_CUBE, Complex::I, 1, false), -Complex::I);
    }

    #[test]
    fn reciprocal_times_its_argument_is_one() {
        for &z in &[
            Complex::new(2.0, 3.0),
            Complex::new(-0.5, 0.125),
            Complex::new(0.0, 1.0),
        ] {
            let w = eval(ID_RECIPROCAL, z, 1, false);
            assert!(close_complex(w * z, Complex::ONE, 1e-15), "at {}", z);
        }
    }

    #[test]
    fn reciprocal_has_a_pole_at_the_origin() {
        let w = eval(ID_RECIPROCAL, Complex::ZERO, 1, false);
        assert!(!w.is_finite(), "1/0 must be a hole, not a crash");
    }

    #[test]
    fn the_shifted_polynomials_vanish_at_their_roots() {
        // z^2 - 1 at z = +-1
        assert!(close_complex(
            eval(ID_Z2_MINUS_1, Complex::new(1.0, 0.0), 1, false),
            Complex::ZERO,
            1e-15
        ));
        assert!(close_complex(
            eval(ID_Z2_MINUS_1, Complex::new(-1.0, 0.0), 1, false),
            Complex::ZERO,
            1e-15
        ));
    }

    #[test]
    fn z_cubed_minus_one_vanishes_at_all_three_roots() {
        let omega = Complex::new(-0.5, 3.0f64.sqrt() / 2.0);
        for root in [Complex::ONE, omega, omega.conjugate()] {
            let w = eval(ID_Z3_MINUS_1, root, 1, false);
            assert!(
                close_complex(w, Complex::ZERO, 1e-14),
                "z^3 - 1 at {} was {}",
                root,
                w
            );
        }
        // And is not accidentally zero somewhere else.
        assert!(!close_complex(
            eval(ID_Z3_MINUS_1, Complex::new(1.0, 1.0), 1, false),
            Complex::ZERO,
            1e-6
        ));
    }

    #[test]
    fn basilica_fixes_zero_and_the_real_sqrt_three() {
        for &x in &[0.0f64, 3.0f64.sqrt(), -3.0f64.sqrt()] {
            let z = Complex::new(x, 0.0);
            let w = eval(ID_BASILICA, z, 1, false);
            assert!(close_complex(w, z, 1e-15), "basilica moved {}", z);
        }
    }

    #[test]
    fn sin_and_cos_hit_their_exact_values() {
        assert_eq!(eval(ID_SIN, Complex::ZERO, 1, false), Complex::ZERO);
        assert_eq!(eval(ID_COS, Complex::ZERO, 1, false), Complex::ONE);
        assert!(close_complex(
            eval(ID_COS, Complex::new(PI / 2.0, 0.0), 1, false),
            Complex::ZERO,
            1e-15
        ));
        assert!(close_complex(
            eval(ID_SIN, Complex::new(PI / 2.0, 0.0), 1, false),
            Complex::ONE,
            1e-15
        ));
    }

    #[test]
    fn sin_and_cos_are_the_real_functions_on_the_real_axis() {
        for &x in &[0.0f64, 0.25, 1.0, 2.0, -3.5, 7.0] {
            let z = Complex::new(x, 0.0);
            assert!(close(eval(ID_SIN, z, 1, false).re, x.sin(), 1e-15));
            assert!(close(eval(ID_COS, z, 1, false).re, x.cos(), 1e-15));
            assert_eq!(eval(ID_SIN, z, 1, false).im, 0.0);
            assert_eq!(eval(ID_COS, z, 1, false).im, 0.0);
        }
    }

    #[test]
    fn sin_of_a_pure_imaginary_is_pure_imaginary() {
        // sin(i) = i sinh(1)
        let w = eval(ID_SIN, Complex::new(0.0, 1.0), 1, false);
        assert_eq!(w.re, 0.0);
        assert!(close(w.im, 1.0f64.sinh(), 1e-15));
    }

    #[test]
    fn sinh_of_a_pure_imaginary_is_pure_imaginary() {
        // sinh(i) = i sin(1)
        let w = eval(ID_SINH, Complex::new(0.0, 1.0), 1, false);
        assert_eq!(w.re, 0.0);
        assert!(close(w.im, 1.0f64.sin(), 1e-15));
    }

    #[test]
    fn sinh_and_exp_agree_through_their_standard_identities() {
        // sinh z = (e^z - e^-z) / 2, with e^-z computed as the reciprocal of
        // e^z, which is where the identity actually holds. (It does *not* hold
        // for -e^z, which is the mistake this test exists to catch.)
        for &z in &[
            Complex::new(0.75, 0.0),
            Complex::new(-0.75, 0.0),
            Complex::new(0.5, 0.5),
            Complex::new(0.0, 1.0),
        ] {
            let ez = eval(ID_EXP, z, 1, false);
            let e_minus_z = Complex::ONE / ez;
            let by_hand = (ez - e_minus_z) * Complex::new(0.5, 0.0);
            let actual = eval(ID_SINH, z, 1, false);
            assert!(
                close_complex(actual, by_hand, 1e-13),
                "sinh {} was {} but (e^z - e^-z)/2 was {}",
                z,
                actual,
                by_hand
            );
        }
    }

    #[test]
    fn exp_of_a_pure_imaginary_number_has_unit_modulus() {
        for &theta in &[0.0f64, 0.5, 1.0, 2.0, -1.25, 3.0] {
            let w = eval(ID_EXP, Complex::new(0.0, theta), 1, false);
            assert!(
                close(w.norm(), 1.0, 1e-15),
                "the modulus of exp(i * theta) at theta = {} was {}",
                theta,
                w.norm()
            );
        }
    }

    #[test]
    fn sinc_is_exactly_one_at_the_origin() {
        let w = eval(ID_SINC, Complex::ZERO, 1, false);
        assert_eq!(w, Complex::ONE, "the removable singularity must be patched");
    }

    #[test]
    fn sinc_is_one_throughout_the_guard_disc() {
        // Not just at the origin: the guard is a disc, and a point on its edge
        // must be handled the same way.
        for &r in &[0.0f64, 1e-20, 1e-13, SINC_SINGULARITY_GUARD_RADIUS / 2.0] {
            let w = eval(ID_SINC, Complex::new(r, 0.0), 1, false);
            assert_eq!(w, Complex::ONE, "at |z| = {}", r);
        }
    }

    #[test]
    fn sinc_outside_the_guard_is_sin_divided_by_z() {
        // sinc(i) = sin(i)/i = sinh(1)
        let w = eval(ID_SINC, Complex::new(0.0, 1.0), 1, false);
        assert!(close_complex(w, Complex::new(1.0f64.sinh(), 0.0), 1e-15));
        // And on the real axis it is the ordinary cardinal sine.
        let w = eval(ID_SINC, Complex::new(1.0, 0.0), 1, false);
        assert!(close(w.re, 1.0f64.sin(), 1e-15));
        assert_eq!(w.im, 0.0);
    }

    #[test]
    fn log_of_a_negative_real_number_has_argument_exactly_pi() {
        let w = eval(ID_LOG, Complex::new(-2.0, 0.0), 1, false);
        assert_eq!(w.im, PI, "the branch cut sits at +pi from above");
        assert!(close(w.re, 2.0f64.ln(), EPSILON));
    }

    #[test]
    fn log_of_the_branch_cut_from_below_is_minus_pi() {
        // The signed zero is the only difference between the two sides, and it
        // is the whole reason Complex keeps -0.0.
        let w = eval(ID_LOG, Complex::new(-2.0, -0.0), 1, false);
        assert_eq!(w.im, -PI);
    }

    #[test]
    fn log_of_one_is_zero_and_log_of_zero_is_minus_infinity() {
        assert!(close_complex(
            eval(ID_LOG, Complex::ONE, 1, false),
            Complex::ZERO,
            0.0
        ));
        let w = eval(ID_LOG, Complex::ZERO, 1, false);
        assert_eq!(w.re, f64::NEG_INFINITY);
        assert_eq!(w.im, 0.0);
    }

    #[test]
    fn log_inverts_exp() {
        for &z in &[
            Complex::new(0.75, 0.0),
            Complex::new(-1.5, 0.25),
            Complex::new(0.0, 2.0),
            Complex::new(3.0, -1.0),
        ] {
            let round_tripped = eval(ID_LOG, eval(ID_EXP, z, 1, false), 1, false);
            assert!(
                close_complex(round_tripped, z, 1e-12),
                "log(e^{}) was {}",
                z,
                round_tripped
            );
        }
    }

    #[test]
    fn log_uses_the_principal_branch_for_every_quadrant() {
        // arg in (-pi, pi], so the imaginary part is never below -pi.
        for &z in &[
            Complex::new(1.0, 1.0),
            Complex::new(-1.0, 1.0),
            Complex::new(-1.0, -1.0),
            Complex::new(1.0, -1.0),
        ] {
            let w = eval(ID_LOG, z, 1, false);
            assert!(w.im > -PI && w.im <= PI, "log({}) had argument {}", z, w.im);
            assert!(close(w.re, z.norm_hypot().ln(), 1e-15));
        }
    }

    #[test]
    fn sqrt_squares_back_to_its_input() {
        for &z in &[
            Complex::new(2.0, 0.0),
            Complex::new(-2.0, 0.0),
            Complex::new(-2.0, -0.0),
            Complex::new(-4.0, 0.0),
            Complex::new(1.0, 2.0),
            Complex::new(-1.0, 2.0),
            Complex::new(-1.0, -2.0),
            Complex::new(0.0, 0.0),
        ] {
            let w = eval(ID_SQRT, z, 1, false);
            assert!(
                close_complex(w * w, z, 1e-12),
                "(sqrt({}))^2 was {}",
                z,
                w * w
            );
        }
    }

    #[test]
    fn sqrt_of_a_negative_real_lands_on_the_branch_cut() {
        // Principal square root: argument in [-pi/2, pi/2], so the principal
        // value of -4 is +2i and the principal value of -4 with a negative
        // signed zero is -2i.
        assert!(close_complex(
            eval(ID_SQRT, Complex::new(-4.0, 0.0), 1, false),
            Complex::new(0.0, 2.0),
            EPSILON
        ));
        assert!(close_complex(
            eval(ID_SQRT, Complex::new(-4.0, -0.0), 1, false),
            Complex::new(0.0, -2.0),
            EPSILON
        ));
    }

    #[test]
    fn sqrt_agrees_with_the_real_sqrt_on_the_positive_real_axis() {
        for &x in &[0.0f64, 2.0, 3.0, 1e-8, 1e8] {
            let w = eval(ID_SQRT, Complex::new(x, 0.0), 1, false);
            assert!(close_complex(w, Complex::new(x.sqrt(), 0.0), 1e-15));
        }
    }

    #[test]
    fn mobius_sends_zero_to_minus_one() {
        assert!(close_complex(
            eval(ID_MOBIUS, Complex::ZERO, 1, false),
            -Complex::ONE,
            1e-15
        ));
    }

    #[test]
    fn mobius_is_not_an_involution_but_has_order_four() {
        // The tempting assumption is that a linear fractional map undoes
        // itself. This one does not. Composing it with itself gives -1/z, a
        // third distinct map; the fourth is the identity, so f generates a
        // cyclic group of order four and the plotter draws the whole cycle:
        //
        //   f^0 = z              f^2 = -1/z
        //   f^1 = (z-1)/(z+1)   f^3 = (1+z)/(1-z)
        //
        // (f^3 is *not* -z, despite the symmetry of the table above: it agrees
        // with -z only at the fixed points z = 1 +- sqrt(2).)
        let z = Complex::new(0.4, 0.9);
        let f = |w: Complex| eval(ID_MOBIUS, w, 1, false);

        let f2 = f(f(z));
        assert!(
            close_complex(f2, -Complex::ONE / z, 1e-12),
            "f^2 was {}",
            f2
        );

        let f3 = f(f2);
        let by_hand = (Complex::ONE + z) / (Complex::ONE - z);
        assert!(close_complex(f3, by_hand, 1e-12), "f^3 was {}", f3);
        assert!(
            !close_complex(f3, -z, 1e-3),
            "f^3 is not -z; the four maps are distinct"
        );

        assert!(close_complex(f(f3), z, 1e-12), "f^4 must be the identity");
    }

    #[test]
    fn mobius_fixes_plus_and_minus_i() {
        for &z in &[Complex::I, -Complex::I] {
            let w = eval(ID_MOBIUS, z, 1, false);
            assert!(close_complex(w, z, 1e-15), "mobius moved {}", z);
        }
    }

    #[test]
    fn iterating_from_a_fixed_point_returns_that_fixed_point() {
        // 0 is a fixed point of z^3 - 2z and stays there exactly, because
        // 0^3 - 2*0 is exactly 0 in binary floating point however many times
        // it is evaluated.
        assert_eq!(eval(ID_BASILICA, Complex::ZERO, 500, true), Complex::ZERO);
        // Same for z^2 at 0, where the orbit does not merely stay but is
        // attracting.
        assert_eq!(eval(ID_SQUARE, Complex::ZERO, 500, true), Complex::ZERO);
        // And for the identity, everywhere.
        let z = Complex::new(2.0, 3.0);
        assert_eq!(eval(ID_IDENTITY, z, 500, true), z);
    }

    #[test]
    fn mobius_iterated_from_its_fixed_point_stays_there() {
        // The arithmetic here is exact, not just close: (i-1)/(i+1) is
        // (-1+i)/(1+i) = i with every intermediate a small dyadic rational.
        for _ in 0..64 {
            assert_eq!(eval(ID_MOBIUS, Complex::I, 1, true), Complex::I);
        }
        assert_eq!(eval(ID_MOBIUS, Complex::I, 64, true), Complex::I);
    }

    #[test]
    fn iterating_the_square_on_the_unit_circle_stays_bounded() {
        for &z in &[
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(-1.0, 0.0),
        ] {
            let w = eval(ID_SQUARE, z, 128, true);
            assert!(w.is_finite(), "z^128 blew up for {}", z);
            assert!(close(w.norm(), 1.0, 1e-12), "|z^128| was {}", w.norm());
        }
    }

    #[test]
    fn the_julia_map_fixes_the_roots_of_its_own_quadratic() {
        // z^2 + c = z, i.e. the two roots of z^2 - z + c. Computed here with the
        // table's own sqrt so the test exercises nothing the shader lacks.
        // Uses `JULIA_C` rather than a literal, so this keeps testing the
        // shipped parameter if the constant ever changes.
        let c = JULIA_C;
        let discriminant = Complex::ONE - Complex::new(4.0, 0.0) * c;
        let root = eval(ID_SQRT, discriminant, 1, false);
        let expected = [(Complex::ONE + root) * 0.5, (Complex::ONE - root) * 0.5];
        for z in expected {
            let w = eval(ID_JULIA, z, 1, false);
            assert!(
                close_complex(w, z, 1e-12),
                "julia moved its fixed point {}",
                z
            );
        }
    }

    #[test]
    fn the_julia_parameter_has_a_filled_set_with_non_empty_interior() {
        // This is the property the parameter must have, and the reason the
        // previous one was replaced. A Julia set has non-empty interior
        // exactly when some open region of points stays bounded forever, so
        // counting survivors over a grid is a direct test of it.
        //
        // The old parameter, `-0.7269 + 0.1889i`, is a dendrite: its set is
        // infinitely branched but measure-zero thin, and it measures ZERO
        // survivors here. An iterated plot of it was ~92% black because there
        // was no filled set to draw, not because of f32 precision. The point of
        // this test is to make that regression impossible to reintroduce by
        // picking a prettier-looking constant.
        const PROBE: [Complex; 6] = [
            Complex::ZERO,
            Complex::new(0.3, 0.0),
            Complex::new(0.5, 0.0),
            Complex::new(-0.5, 0.0),
            Complex::new(0.0, 0.4),
            Complex::new(0.2, 0.3),
        ];
        let survivors = PROBE
            .iter()
            .filter(|&&z| eval(ID_JULIA, z, ITERATION_CAP, true).is_finite())
            .count();
        assert!(
            survivors >= 4,
            "expected the shipped julia parameter to have a filled set, but only \
             {survivors} of {} probe points stayed bounded. A parameter with no \
             bounded interior renders as a nearly black plot; see JULIA_C.",
            PROBE.len()
        );
    }

    #[test]
    fn the_replaced_dendrite_parameter_really_did_have_no_bounded_orbits() {
        // Pins the reason the parameter changed, so the comment above cannot
        // quietly become folklore. If someone reinstates this constant, this
        // test is the one that explains why that was a bad idea.
        for &z in &[
            Complex::ZERO,
            Complex::new(0.3, 0.0),
            Complex::new(-0.5, 0.2),
            Complex::new(0.1, 0.1),
        ] {
            let mut w = z;
            for _ in 0..ITERATION_CAP {
                w = w * w + Complex::new(-0.7269, 0.1889);
            }
            assert!(
                !w.is_finite(),
                "the dendrite orbit of {} unexpectedly stayed bounded at {}",
                z,
                w
            );
        }
    }

    #[test]
    fn iterating_the_square_outside_the_unit_circle_escapes() {
        // 2 -> 4 -> 16 -> 256 -> ... and the modulus passes f64::MAX after ten
        // steps. Note `!is_finite()` and not "is infinite": see the note on
        // escaped orbits in Complex's Mul.
        let w = eval(ID_SQUARE, Complex::new(2.0, 0.0), 200, true);
        assert!(!w.is_finite(), "2 should run away, got {}", w);
    }

    #[test]
    fn an_escaped_orbit_is_infinity_first_and_nan_afterwards() {
        // The consequence of `inf * 0 == NaN` that the renderer has to know
        // about: escape is visible as an infinity for a step or two and as a
        // NaN from then on. `is_finite()` is the only correct escape test.
        let mut w = Complex::new(2.0, 0.0);
        let mut first_escape = None;
        for step in 0..64 {
            w = eval(ID_SQUARE, w, 1, false);
            if !w.is_finite() {
                first_escape = Some((step + 1, w));
                break;
            }
        }
        let (step, escaped) = first_escape.expect("the orbit of 2 must escape");
        assert_eq!(escaped.re, f64::INFINITY, "escape is an infinity at first");
        assert_eq!(escaped.im, 0.0);

        // Two steps later the imaginary part is `inf * 0 + 0 * inf`.
        let once = eval(ID_SQUARE, escaped, 1, false);
        assert!(once.re.is_infinite() && once.im.is_nan(), "got {}", once);

        // And one step after that everything is NaN, for good.
        let twice = eval(ID_SQUARE, once, 1, false);
        assert!(twice.re.is_nan() && twice.im.is_nan(), "got {}", twice);
        assert!(!eval(ID_SQUARE, twice, 32, true).is_finite());
        assert_eq!(step, 10, "the escape step is a property of the arithmetic");
    }

    #[test]
    fn zero_iterations_returns_the_input_unchanged() {
        let z = Complex::new(0.9, 0.1);
        for id in 0..FUNCTION_COUNT {
            assert_eq!(eval(id, z, 0, true), z, "id {} moved z at 0 steps", id);
        }
    }

    #[test]
    fn the_iteration_count_is_capped() {
        // In `f64` the cap is a *termination* guarantee, not a numerical one.
        // The escape time of a double-precision point under any polynomial in
        // this table is bounded by the precision of the point - a z^2 orbit
        // escapes within about 60 steps however close to the unit circle it
        // starts - so no orbit can still be moving at step 4096 and a larger
        // count cannot change the answer. What the cap must guarantee is that
        // a corrupt `max_iter` of `u32::MAX` costs at most ITERATION_CAP
        // applications. Without the cap the first call below never returns, so
        // this test is the cap.
        for &id in &[
            ID_IDENTITY,
            ID_SQUARE,
            ID_JULIA,
            ID_RECIPROCAL,
            ID_MOBIUS,
            ID_SQRT,
        ] {
            for &z in &[
                Complex::new(0.31, 0.72),
                Complex::new(-0.4, 0.2),
                Complex::new(0.0, 0.0),
                Complex::new(-0.8333, -0.3333),
            ] {
                let capped = eval(id, z, ITERATION_CAP, true);
                let absurd = eval(id, z, u32::MAX, true);
                assert_eq!(
                    capped.re.to_bits(),
                    absurd.re.to_bits(),
                    "id {} at {} disagrees with its clamped result",
                    id,
                    z
                );
                assert_eq!(capped.im.to_bits(), absurd.im.to_bits());
            }
        }
    }

    #[test]
    fn no_meaningful_orbit_is_truncated_by_the_cap() {
        // The other half of the cap's justification: 4096 is above the slowest
        // escape anywhere in a dense sweep of the plane, so the cap never cuts
        // a picture short. The dendrite parameter is by far the worst case.
        let n = 48usize;
        let step = 4.0 / n as f64;
        for &id in &[
            ID_SQUARE,
            ID_Z2_MINUS_1,
            ID_Z3_MINUS_1,
            ID_BASILICA,
            ID_JULIA,
        ] {
            let mut slowest = 0u32;
            let mut slowest_at = Complex::ZERO;
            for i in 0..n {
                for j in 0..n {
                    let z = Complex::new(-2.0 + i as f64 * step, -2.0 + j as f64 * step);
                    let mut w = z;
                    for step_index in 1..=ITERATION_CAP {
                        w = eval(id, w, 1, false);
                        if !w.is_finite() {
                            if step_index > slowest {
                                slowest = step_index;
                                slowest_at = z;
                            }
                            break;
                        }
                    }
                }
            }
            assert!(
                slowest < ITERATION_CAP,
                "id {} needed {} steps to escape at {}, which the cap would cut off",
                id,
                slowest,
                slowest_at
            );
        }
    }

    #[test]
    fn an_absurd_max_iter_terminates() {
        // A corrupted uniform must cost at most ITERATION_CAP applications, not
        // u32::MAX of them. `0.5` is chosen because its z^2 orbit converges, so
        // the result stays a number and the comparison is meaningful rather
        // than infinity == infinity.
        let z = Complex::new(0.5, 0.0);
        let w = eval(ID_SQUARE, z, u32::MAX, true);
        assert!(w.is_finite());
        assert_eq!(w, eval(ID_SQUARE, z, ITERATION_CAP, true));
        assert_eq!(w, Complex::ZERO, "and the orbit really did converge");
    }

    #[test]
    fn every_function_survives_awkward_input_without_panicking() {
        // Poles, infinities, NaN, and arguments that overflow both sin and cosh.
        // Nothing here may panic or hang; a function that answers NaN at a NaN
        // is answering correctly.
        let probes = [
            Complex::ZERO,
            Complex::ONE,
            Complex::I,
            Complex::new(-1.0, 0.0),
            Complex::new(f64::NAN, 1.0),
            Complex::new(f64::INFINITY, f64::NEG_INFINITY),
            Complex::new(1e-300, 1e300),
            Complex::new(-1e300, -1e-300),
        ];
        for id in 0..FUNCTION_COUNT {
            for &z in &probes {
                for &iterate in &[false, true] {
                    let w = eval(id, z, 8, iterate);
                    let again = eval(id, z, 8, iterate);
                    // Determinism, compared bit for bit because a NaN result
                    // would otherwise fail to equal itself. The evaluator holds
                    // no state, so asking twice must give the identical bits.
                    assert_eq!(
                        (w.re.to_bits(), w.im.to_bits()),
                        (again.re.to_bits(), again.im.to_bits()),
                        "id {} is not deterministic at {}",
                        id,
                        z
                    );
                }
            }
        }
    }

    #[test]
    fn exp_is_periodic_in_the_imaginary_direction() {
        // e^{z + 2*pi*i} = e^z, which only holds if the argument of e^z is
        // reduced with the same principal `arg` the log uses.
        for &z in &[Complex::new(0.5, 0.0), Complex::new(-1.0, 0.25)] {
            let shifted = eval(ID_EXP, z, 1, false);
            let round_tripped = Complex::new(z.re, z.im + TAU);
            assert!(close_complex(
                eval(ID_EXP, round_tripped, 1, false),
                shifted,
                1e-12
            ));
        }
    }

    #[test]
    fn exp_and_log_stay_consistent_at_extreme_magnitudes() {
        // |z| far beyond where the naive norm overflows: log must still be
        // finite, which is why it uses norm_hypot.
        let z = Complex::new(1e300, 1e300);
        let w = eval(ID_LOG, z, 1, false);
        assert!(w.is_finite(), "log of a huge z was {}", w);
        assert!(close(w.re, z.norm_hypot().ln(), 1e-15));
    }

    #[test]
    fn a_full_frame_worth_of_pixels_evaluates_without_panicking() {
        // A cheap smoke test in the shape of the real workload: a grid of
        // points through every function, iterated. It may return anything the
        // arithmetic can produce - a number, an infinity, a NaN - and it must
        // do all of it without panicking, hanging, or looping forever.
        for id in 0..FUNCTION_COUNT {
            for i in 0..24 {
                for j in 0..24 {
                    let z = Complex::new(i as f64 * 0.2 - 2.4, j as f64 * 0.2 - 2.4);
                    let w = eval(id, z, 64, true);
                    // Whatever came back must round-trip through Display, which
                    // is what a tooltip or a telemetry record will do to it.
                    let _ = w.to_string();
                }
            }
        }
    }
}
