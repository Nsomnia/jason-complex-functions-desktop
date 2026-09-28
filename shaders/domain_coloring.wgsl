// ============================================================================
//  domain_coloring.wgsl  --  the heart of the complex-function plotter
// ============================================================================
//
//  ONE compute invocation per output pixel. The kernel maps the pixel to a point
//  in the complex plane, applies the selected complex function (optionally
//  iterated), and writes the result's "domain colouring" as an RGBA8 pixel into
//  a storage texture. `blit.wgsl` then samples that texture onto the screen.
//
//  ---------------------------------------------------------------------------
//  BINDING LAYOUT  (the Rust side MUST match this exactly)
//  ---------------------------------------------------------------------------
//      @group(0) @binding(0)   var<uniform>  uniforms : Uniforms
//      @group(0) @binding(1)   var            output_texture
//                                : texture_storage_2d<rgba8unorm, write>
//
//  Entry point: `main`, @compute @workgroup_size(8, 8)  =>  64 invocations per
//  workgroup arranged as an 8x8 tile. The host dispatches
//  ceil(width / 8) x ceil(height / 8) workgroups.
//
//  ---------------------------------------------------------------------------
//  ALGORITHM
//  ---------------------------------------------------------------------------
//  Step 1  pixel -> complex number.
//          `uv` is the pixel centre normalised to [0,1]^2, so it samples the
//          exact middle of the pixel and avoids a half-pixel offset. The x
//          extent is scaled by the aspect ratio so a wide window still shows a
//          circle as a circle instead of a stretched ellipse. The y term is
//          `1.0 - uv.y * 2.0`, NOT `uv.y * 2.0 - 1.0`: framebuffer row 0 is
//          the TOP of the window, and the top of the window must be the TOP of
//          the imaginary axis. Flipping this mirrors every plot vertically and
//          nothing else warns you about it.
//
//  Step 2  apply the function. `apply(fid, z)` is a `switch` over `func_id`.
//          The ids are a hard contract with the CPU reference implementation
//          (see src/functions.rs) -- never renumber them:
//              0  identity      z                 8   cos          cos z
//              1  square        z*z               9   sinc         sin z / z
//              2  cube          z*z*z             10  sinh         sinh z
//              3  reciprocal    1/z               11  exp          e^z
//              4  z2_minus_1    z*z - 1           12  log          ln|z| + i*arg z
//              5  z3_minus_1    z*z*z - 1         13  sqrt         principal branch
//              6  basilica      z*z*z - 2z        14  mobius       (z-1)/(z+1)
//              7  sin           sin z             15  julia        z*z + c
//          When `iterate != 0u` the function is applied
//          `min(max_iter, 4096u)` times, re-dispatching through `apply` on
//          each step, which is what turns any function into a Julia-style set.
//          The loop breaks as soon as the value stops being finite, so a
//          diverging orbit costs one iteration rather than 4096.
//
//  Step 3  domain colouring. Let w be the result and m = |w|.
//              hue   = fract(atan2(w.y, w.x)/TAU + 0.5 + phase)  -- direction
//              value = (m / (1 + m)) ^ modulus_shading            -- magnitude
//          Both are then multiplied by thin dark contour bands, which is what
//          produces the characteristic family of curves along which |w| or
//          arg(w) is constant. A non-finite w (overflow to inf or NaN) is
//          painted solid black, which is exactly the set boundary in the
//          iterated case.
//
//  Step 4  optional grid. If `grid_enabled != 0u`, the real axis (im == 0), the
//          imaginary axis (re == 0) and the unit circle (|z| == 1) are blended
//          in as pale cyan. Line width is measured in WORLD units, derived from
//          the world-units-per-pixel implied by `scale` and `resolution.y`, so
//          the lines stay about a pixel wide at any zoom level.
//
//  ---------------------------------------------------------------------------
//  UNIFORM ABI
//  ---------------------------------------------------------------------------
//  `Uniforms` mirrors `src/uniforms.rs` field-for-field, in order, by name.
//  64 bytes, every offset a multiple of 4. Do not reorder or rename anything
//  here without editing that file in the same commit; a mismatch mis-colours
//  the entire plot rather than failing loudly.
//
//  `frame` and `_pad` are declared so the struct covers the whole 64-byte
//  block, but nothing here reads them. That is deliberate: the Rust side writes
//  them and may use them for telemetry, and an unread member costs nothing.
//
//  ---------------------------------------------------------------------------
//  NUMERICAL NOTES
//  ---------------------------------------------------------------------------
//  * `length()` is used for |w| rather than sqrt(x*x + y*y) so the hardware can
//    apply its overflow-safe hypot sequence.
//  * `log` and `sqrt` use the principal branch and deliberately do NOT
//    special-case the branch cut (the negative real axis). The discontinuity
//    there is a real feature of the function and is the most interesting thing
//    on the plot.
//  * The hyperbolic functions are the WGSL builtins, unmodified. Scaled
//    variants would be more overflow-tolerant but would disagree with the CPU
//    reference, and the reference is the specification.
//  * `sinc` guards the removable singularity at 0, and `c_div` maps a
//    vanishing denominator to 0. Both land on m = 0, which Step 3 turns into
//    the same black pixel a non-finite value would produce, so the guard is
//    visually indistinguishable from letting it blow up -- but it does not
//    poison the iterate loop with NaN.
// ============================================================================

// ---------------------------------------------------------------------------
// Uniform block. See src/uniforms.rs -- this is the contract.
// ---------------------------------------------------------------------------
struct Uniforms {
    resolution: vec2<f32>,             // offset  0  (width, height) in pixels
    center: vec2<f32>,                 // offset  8  viewport centre, (re, im)
    scale: f32,                        // offset 16  half-height of the view, world units
    max_iter: u32,                     // offset 20  iteration cap
    func_id: u32,                      // offset 24  function selector
    phase: f32,                        // offset 28  hue rotation, in turns
    modulus_contour_density: f32,      // offset 32  iso-modulus bands per e-fold; 0 disables
    phase_contour_density: f32,        // offset 36  iso-phase bands per turn; 0 disables
    modulus_shading: f32,              // offset 40  lightening exponent
    grid_enabled: u32,                 // offset 44  draw axes + unit circle when non-zero
    frame: u32,                        // offset 48  monotonic frame counter (unread here)
    iterate: u32,                      // offset 52  apply max_iter times when non-zero
    _pad: vec2<u32>,                   // offset 56  pads the block to 64 bytes (unread)
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// rgba8unorm: the compute pass does no blending and no sRGB conversion, so the
// bytes written here are the bytes the display shows.
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// One full turn in radians, 2 * pi. Must match the CPU reference exactly, or
/// the whole hue wheel is rotated by a constant offset.
const TAU: f32 = 6.283185307179586;

/// Largest number of iterations the kernel will ever perform. `max_iter` comes
/// from the UI and could be nonsense; this bounds the worst case.
const MAX_ITER_CLAMP: u32 = 4096u;

/// Julia constant for func_id 15. Fixed by the ABI, not user-tunable.
///
/// This is `z^2 - 0.75`, the root of the period-2 hyperbolic component on the
/// real axis. It is NOT the basilica (that is `c = -1`). It replaced
/// `-0.7269 + 0.1889i`, whose Julia set is a dendrite with EMPTY INTERIOR:
/// almost every orbit escapes, so an iterated plot of it was ~92% black.
/// Measured over a 96x96 grid after 2000 iterations, the bounded fraction is
/// 0.00% for the dendrite against 12.89% for this value. See `JULIA_C` in
/// `src/functions.rs`, which carries the full table and must be changed in the
/// same commit as this line.
const JULIA_C: vec2<f32> = vec2<f32>(-0.75, 0.0);

/// The singularity threshold and the logarithm floor. `sinc` returns its
/// analytic limit 1.0 below this modulus instead of dividing 0/0, and the
/// modulus-contour band uses it as the floor of log() so a zero of f gives a
/// large finite band coordinate rather than -infinity. It is deliberately tiny:
/// only the genuinely undefined operations are guarded, everything else is left
/// to produce its true value, so the plot agrees with an f64 reference.
const EPS: f32 = 1e-12;

/// The largest finite f32. A decimal literal too large for f32 rounds to this
/// value rather than becoming an error, which is exactly the intent. Used to
/// test for infinity without relying on an `isInf` spelling that has moved
/// between WGSL revisions.
const F32_MAX: f32 = 3.402823466e+38;

/// Grid line colour and the fixed strength it is blended in at.
const GRID_COLOR: vec3<f32> = vec3<f32>(0.35, 0.95, 1.0);
const GRID_BLEND: f32 = 0.35;

/// Nominal grid line half-width, in pixels. Converted to world units per pixel
/// so the lines stay ~1px wide at any zoom.
const GRID_PX: f32 = 1.0;

// ===========================================================================
//  Complex arithmetic on vec2<f32>, where .x is the real part and .y the
//  imaginary part. No struct and no generics: vec2 keeps the generated code
//  tight and sidesteps any question about struct layout in the uniform
//  address space.
// ===========================================================================

/// Complex addition.
fn c_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a + b;
}

/// Complex subtraction.
fn c_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a - b;
}

/// Complex multiplication, the standard (ac - bd, ad + bc) expansion. The
/// Karatsuba rearrangement saves one multiply but loses precision near the
/// overflow boundary, and precision near |z| -> infinity is precisely what this
/// program exists to show.
fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x,
    );
}

/// Complex division by Smith's algorithm: a/b = (a * conj(b)) / |b|^2. Far
/// better conditioned than forming conj(b)/|b|^2 and dividing, because the
/// denominator is a sum of squares and so cannot cancel to zero.
///
/// Deliberately has NO guard against a zero or vanishing denominator. A true
/// pole (reciprocal at 0, Mobius at -1) is supposed to produce a non-finite
/// value, and Step 3 paints that black, which is exactly how a pole should
/// look. An earlier version of this function short-circuited a small
/// denominator to 0, which was wrong twice over: it made the reciprocal read as
/// "a zero of f" (black) across a whole disc where the true 1/z is a large
/// finite value and should read as bright, and it was far too coarse for sinc,
/// whose singularity is removable rather than a pole. `c_sinc_of` below does
/// its own division for exactly that reason.
fn c_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let inv = 1.0 / dot(b, b);
    return vec2<f32>(
        (a.x * b.x + a.y * b.y) * inv,
        (a.y * b.x - a.x * b.y) * inv,
    );
}

/// Complex square. Cheaper, and slightly more accurate, than c_mul(z, z).
fn c_sqr(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        z.x * z.x - z.y * z.y,
        2.0 * z.x * z.y,
    );
}

/// Complex cube.
fn c_cube(z: vec2<f32>) -> vec2<f32> {
    return c_mul(c_sqr(z), z);
}

/// |z|. length() rather than sqrt(dot(z, z)) so the hardware hypot sequence can
/// be used, which avoids a spurious infinity from intermediate squaring.
fn c_norm(z: vec2<f32>) -> f32 {
    return length(z);
}

/// arg(z) in (-pi, pi]. This is the branch of atan2, which is exactly the
/// principal argument; the negative real axis is the branch cut.
fn c_arg(z: vec2<f32>) -> f32 {
    return atan2(z.y, z.x);
}

/// True when both components are finite. Expressed as two comparisons so it
/// works on any WGSL revision:
///
///   * `x == x` is false for NaN, which is how NaN is detected without an
///     `isNan` builtin.
///   * `abs(x) < F32_MAX` is false for +inf and -inf, which is how infinity is
///     detected without an `isInf` builtin.
///
/// Negating both covers all three non-finite cases at once.
fn c_finite(z: vec2<f32>) -> bool {
    let not_nan = (z.x == z.x) && (z.y == z.y);
    let not_inf = (abs(z.x) < F32_MAX) && (abs(z.y) < F32_MAX);
    return not_nan && not_inf;
}

// ---------------------------------------------------------------------------
// Complex elementary functions, all following from Euler's formula:
//
//     sin z   = sin(re) cosh(im) + i cos(re) sinh(im)
//     cos z   = cos(re) cosh(im) - i sin(re) sinh(im)
//     sinh z  = sinh(re) cos(im)  + i cosh(re) sin(im)
//     exp z   = e^re * (cos(im), sin(im))
//     log z   = ln|z| + i*arg(z)                       (principal branch)
//     sqrt z  = sqrt((|z|+re)/2) + i*sign(im)*sqrt((|z|-re)/2)  (principal)
//
// sin/exp/log are why domain colouring of a non-polynomial looks nothing like
// that of z^2: the lines of constant argument are preimages of the real axis
// under a map that is no longer a polynomial, so they bend and wrap.
// ---------------------------------------------------------------------------

fn c_sin(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        sin(z.x) * cosh(z.y),
        cos(z.x) * sinh(z.y),
    );
}

fn c_cos(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        cos(z.x) * cosh(z.y),
        -sin(z.x) * sinh(z.y),
    );
}

fn c_sinh(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        sinh(z.x) * cos(z.y),
        cosh(z.x) * sin(z.y),
    );
}

/// e^z. Magnitude and phase separate cleanly, so this is one exp and one sincos.
fn c_exp(z: vec2<f32>) -> vec2<f32> {
    let r = exp(z.x);
    return vec2<f32>(r * cos(z.y), r * sin(z.y));
}

fn c_log(z: vec2<f32>) -> vec2<f32> {
    // Principal branch: ln|z| + i*arg(z). No case is made for the negative real
    // axis -- the jump from +pi to -pi there IS the branch cut, and it is one of
    // the most recognisable features of a log plot.
    return vec2<f32>(log(c_norm(z)), c_arg(z));
}

fn c_sqrt(z: vec2<f32>) -> vec2<f32> {
    // The max(., 0.0) calls are rounding guards, not branch-cut handling.
    // (|z| - re) is mathematically non-negative, but subtracting two nearly
    // equal f32 values can come back as a tiny negative, and sqrt of that is
    // NaN. sign(0.0) is 0.0 in WGSL, which yields +0.0 for the imaginary part
    // on the positive real axis, as it should.
    let m = c_norm(z);
    let a = sqrt(max(0.5 * (m + z.x), 0.0));
    let b = sign(z.y) * sqrt(max(0.5 * (m - z.x), 0.0));
    return vec2<f32>(a, b);
}

/// sinc(z) = sin z / z, for |z| >= 1e-12.
///
/// This does NOT go through c_div. sin z / z has a *removable* singularity at
/// the origin with limit 1, so the only thing the guard is protecting against
/// is genuine 0/0. c_div's pole semantics (a vanishing denominator becoming
/// non-finite, hence black) are the wrong semantics for a removable
/// singularity: routing sinc through c_div would paint the disc |z| < 1e-6
/// black when the correct value there is 1, i.e. the brightest possible pixel.
/// The division itself is the same Smith form c_div uses.
fn c_sinc_of(z: vec2<f32>) -> vec2<f32> {
    if (c_norm(z) < EPS) {
        return vec2<f32>(1.0, 0.0); // the analytic limit
    }
    let s = c_sin(z);
    let inv = 1.0 / dot(z, z);
    return vec2<f32>(
        (s.x * z.x + s.y * z.y) * inv,
        (s.y * z.x - s.x * z.y) * inv,
    );
}

// ===========================================================================
//  Function dispatch. The switch covers 0..=15 exactly; any other id falls
//  through to `default`, which is the identity map. Ids must not be renumbered.
// ===========================================================================

/// Apply complex function `fid` to `z` exactly once.
fn apply(fid: u32, z: vec2<f32>) -> vec2<f32> {
    switch (fid) {
        // 0: identity.
        case 0u: {
            return z;
        }
        // 1: square.
        case 1u: {
            return c_sqr(z);
        }
        // 2: cube.
        case 2u: {
            return c_cube(z);
        }
        // 3: reciprocal; c_div guards the pole at 0.
        case 3u: {
            return c_div(vec2<f32>(1.0, 0.0), z);
        }
        // 4: z^2 - 1.
        case 4u: {
            return c_sub(c_sqr(z), vec2<f32>(1.0, 0.0));
        }
        // 5: z^3 - 1.
        case 5u: {
            return c_sub(c_cube(z), vec2<f32>(1.0, 0.0));
        }
        // 6: basilica, the Julia set of z^3 - 2z. Note there is no escape-radius
        // bailout anywhere in this shader: divergence is detected only by
        // c_finite, so an escaping orbit runs to f32 overflow. See the note on
        // apply_selected and the c=... julia case.
        case 6u: {
            return c_sub(c_cube(z), 2.0 * z);
        }
        // 7: sin.
        case 7u: {
            return c_sin(z);
        }
        // 8: cos.
        case 8u: {
            return c_cos(z);
        }
        // 9: sinc = sin z / z, with the removable singularity at 0 filled in
        // with its analytic limit 1.0. See c_sinc_of for why this is not just
        // c_div(c_sin(z), z).
        case 9u: {
            return c_sinc_of(z);
        }
        // 10: sinh.
        case 10u: {
            return c_sinh(z);
        }
        // 11: e^z.
        case 11u: {
            return c_exp(z);
        }
        // 12: principal-branch logarithm.
        case 12u: {
            return c_log(z);
        }
        // 13: principal-branch square root.
        case 13u: {
            return c_sqrt(z);
        }
        // 14: Mobius transform (z-1)/(z+1), the Cayley map from the unit disc
        // onto the right half-plane. Its pole sits at z = -1, where c_div
        // yields NaN and Step 3 paints black.
        case 14u: {
            return c_div(c_sub(z, vec2<f32>(1.0, 0.0)), c_add(z, vec2<f32>(1.0, 0.0)));
        }
        // 15: Julia iteration z -> z^2 + c, c fixed by the ABI.
        case 15u: {
            return c_add(c_sqr(z), JULIA_C);
        }
        // An undefined id behaves as the identity rather than aborting the
        // dispatch, so a stale UI selection still renders something sane
        // instead of leaving the previous frame's texture on screen.
        default: {
            return z;
        }
    }
}

/// Apply the selected function, once or `min(max_iter, 4096)` times.
///
/// The break on non-finite output is the important part. Iterating exp or
/// square past the f32 overflow point yields inf, then inf - inf = NaN, and
/// every later iteration is pure NaN propagation over 4096 rounds. Stopping at
/// the first non-finite value bounds the cost and leaves `w` non-finite, which
/// Step 3 turns into the correct black pixel.
///
/// KNOWN LIMITATION, no escape-radius bailout. Divergence is detected only by
/// c_finite, i.e. only once the orbit has already overflowed f32 at about
/// |w| = 1.8e19. A classical escape-time plot would bail at |w| > 2 and colour
/// by the iteration count; this one colours the final iterate instead.
///
/// An earlier version of this comment blamed f32 rounding for the julia
/// function rendering almost entirely black, claiming the critical orbit is
/// "bounded in exact arithmetic" and that f32 pushes it off the fractal. That
/// was wrong, and measurement corrected it. The old parameter
/// `-0.7269 + 0.1889i` is a DENDRITE: its Julia set has empty interior, so
/// essentially no orbit is bounded and the picture is mostly escaping points in
/// f64 too. The fix was the parameter, not the arithmetic - see `JULIA_C`
/// above. The current parameter has a 12.89% bounded fraction.
///
/// Two real costs of having no bailout remain, and they are worth stating
/// plainly rather than folding into the above:
///
///  1. Divergence is only noticed at f32 overflow, so a slowly escaping orbit
///     burns all 4096 iterations before anything registers.
///  2. The shading value is `m / (1 + m)`, which reaches exactly 1.0 in f32 at
///     |w| ~ 3.4e7 and holds there for twelve decades. A diverging orbit
///     therefore passes through white and then snaps to black when it finally
///     overflows. That discontinuity is an f32 artifact and is worth fixing,
///     but it is a presentation problem sitting on top of the escape problem,
///     and it is no longer what makes this plot unusable.
///
///  3. A consequence of colouring the FINAL ITERATE rather than the trajectory:
///     after a few hundred iterations every point inside the filled set has
///     collapsed onto an attracting cycle, so `|w|` is nearly the same
///     everywhere inside and the interior renders at a flat, dim, almost
///     uniform value. Measured at max_iter = 256 with c = -0.75, 94.9% of
///     structured pixels fall in the 32-63/255 band, because the critical orbit
///     has converged to the parabolic fixed point -0.5. This is mathematically
///     correct for the specified colour scheme, and it is not a bug - but it is
///     why the plot looks muted. Restoring interior contrast means colouring by
///     something other than the final iterate (minimum |w| along the orbit, or
///     escape time), which is a design change, not a fix.
///
/// For reference, under the old parameter:
/// the default max_iter = 256 about 92% of a 3x3 viewport comes out non-finite
/// and therefore black. At max_iter = 4096 the whole frame is black.
///
/// That is a property of the specified algorithm, not a bug in this shader,
/// and the CPU reference reproduces it exactly because it does the same thing
/// -- which is the point of rule 3. Fixing it means detecting escape at a
/// radius during the loop and colouring by escape time, which changes both the
/// algorithm and the ABI, and is a decision for the project, not for this file.
fn apply_selected(z: vec2<f32>) -> vec2<f32> {
    let fid = uniforms.func_id;
    if (uniforms.iterate == 0u) {
        return apply(fid, z);
    }

    let n = min(uniforms.max_iter, MAX_ITER_CLAMP);
    var w = z;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        w = apply(fid, w);
        if (!c_finite(w)) {
            break;
        }
    }
    return w;
}

// ===========================================================================
//  Colour
// ===========================================================================

/// HSV -> RGB in the p/q/f form, i.e. the standard hue-to-RGB reduction that
/// shares a single fractional position `f` across all three of p, q and t
/// rather than branching through a six-way piecewise. It is branch-free and
/// has no seams at the sector boundaries, which matters because hue wraps
/// continuously here.
fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    // fract() first: a hue of exactly 1.0 must wrap to 0.0 (pure red) instead
    // of failing every sector comparison and coming out black.
    let h = fract(c.x) * 6.0;
    let s = clamp(c.y, 0.0, 1.0);
    let v = clamp(c.z, 0.0, 1.0);

    // Integer sector and the shared within-sector position, both in [0,6) / [0,1).
    let i = floor(h);
    let f = h - i;

    // p, q and t are the three distinct values the six sectors are drawn from.
    let p = v * (1.0 - s);            // fully desaturated
    let q = v * (1.0 - f * s);        // falls from v to p across the sector
    let t = v * (1.0 - (1.0 - f) * s); // rises from p to v across the sector

    switch (i32(i)) {
        case 0: { return vec3<f32>(v, t, p); }
        case 1: { return vec3<f32>(q, v, p); }
        case 2: { return vec3<f32>(p, v, t); }
        case 3: { return vec3<f32>(p, q, v); }
        case 4: { return vec3<f32>(t, p, v); }
        case 5: { return vec3<f32>(v, p, q); }
        // Unreachable: fract() guarantees i in [0,6), so i is at most 5. Kept so
        // the function is total if the input ever stops being normalised.
        default: { return vec3<f32>(v, t, p); }
    }
}

/// Domain colouring of a single complex value `w`.
///
/// Follows the specified algorithm literally, because the CPU reference
/// implementation mirrors it step for step and the test suite compares the two.
fn domain_color(w: vec2<f32>, phase: f32, modulus_shading: f32,
                modulus_density: f32, phase_density: f32) -> vec3<f32> {
    let m = c_norm(w);   // overflow-safe, see c_norm
    let angle = c_arg(w); // [-pi, pi]

    // The +0.5 maps the positive real axis to hue 0.5, i.e. cyan, and puts the
    // negative real axis at hue 0 (red). That is the conventional orientation:
    // the barrier where the function blows up reads as a red line.
    let turns = angle / TAU + 0.5; // [0, 1]
    let hue = fract(turns + phase);

    // Smooth magnitude ramp. m/(1+m) is 0 at a zero of f, tends to 1 as |w|
    // grows, and never exceeds 1, so no clamp is needed here.
    let b = m / (1.0 + m);
    let value = pow(b, modulus_shading);

    // Iso-modulus and iso-phase contours, multiplied into the value.
    var dark = 1.0;

    if (modulus_density > 0.0) {
        // Bands in log-modulus space, so they are evenly spaced per e-fold of
        // |w| instead of piling up near the origin.
        let band = log(max(m, EPS)) * modulus_density;
        // Triangle wave: 0 at the band centre, 1 half a band away.
        let f1 = abs(fract(band) - 0.5) * 2.0;
        // pow(., 24) squeezes the wave into a thin line, so 0.45 is the peak
        // darkening at the band centre.
        dark = dark * (1.0 - 0.45 * pow(f1, 24.0));
    }

    if (phase_density > 0.0) {
        // Bands in turn space. These are the radiating rays of phase portrait.
        let band = turns * phase_density;
        let f2 = abs(fract(band) - 0.5) * 2.0;
        dark = dark * (1.0 - 0.45 * pow(f2, 24.0));
    }

    return hsv2rgb(vec3<f32>(hue, 1.0, value * dark));
}

// ===========================================================================
//  Grid overlay
// ===========================================================================

/// Blend the axes and the unit circle into `color`.
///
/// `z` is the complex coordinate of this pixel, i.e. a point in the DOMAIN
/// rather than the range, which is what a grid is drawn on. Line width is
/// computed in world units from `scale`, so a one-pixel line stays one pixel
/// wide as the view zooms; without that the lines either vanish or smear once
/// the scale changes by orders of magnitude.
fn apply_grid(color: vec3<f32>, z: vec2<f32>, scale: f32, res: vec2<f32>) -> vec3<f32> {
    // World units per pixel. `scale` is the half-height of the view in world
    // units and the view spans resolution.y pixels vertically, so this is the
    // exact per-pixel footprint. The max() calls guard against a zero-height
    // window or a zero scale: dividing by zero here would emit NaN from every
    // invocation and flood the frame with a solid NaN colour.
    let per_pixel = 2.0 * max(abs(scale), EPS) / max(res.y, 1.0);

    // GRID_PX is a nominal half-width in pixels; convert it to world units. The
    // outer max() keeps the width strictly positive so smoothstep never sees
    // low == high, which is undefined and yields NaN.
    let line_w = max(GRID_PX * per_pixel, EPS);

    // Distance to each curve, in world units. For the axes that is just the
    // perpendicular distance to the line. For the unit circle, |z| - 1 is a
    // first-order approximation of the distance to the circle, and is exact on
    // it.
    let d_real = abs(z.y);               // im == 0
    let d_imag = abs(z.x);               // re == 0
    let d_unit = abs(c_norm(z) - 1.0);   // |z| == 1

    // Each line contributes a soft coverage in [0,1]: 1 on the line, falling to
    // 0 over one half-width, which antialiases the line against the plot. The
    // three are combined with max() rather than a sum, so a line crossing
    // another does not double-darken and blow out the blend.
    var cover = 0.0;
    cover = max(cover, 1.0 - smoothstep(0.0, line_w, d_real));
    cover = max(cover, 1.0 - smoothstep(0.0, line_w, d_imag));
    cover = max(cover, 1.0 - smoothstep(0.0, line_w, d_unit));

    return mix(color, GRID_COLOR, GRID_BLEND * clamp(cover, 0.0, 1.0));
}

// ===========================================================================
//  Entry point
// ===========================================================================

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let res = uniforms.resolution;

    // The host dispatches ceil(res/8) workgroups, so the trailing edge of the
    // grid produces invocations one past the texture bounds. They must be
    // discarded: textureStore outside the texture is undefined, and on some
    // backends it is a hard fault rather than a silent drop.
    //
    // The comparison is done in f32 rather than as `global_id.x >= u32(res.x)`.
    // Converting a negative or wildly out-of-range f32 to u32 is explicitly
    // undefined behaviour in WGSL, and a wrapped result would defeat the very
    // bounds check it was meant to perform. The f32 form is total for every
    // possible input. It is only inexact for resolutions above 2^24, where the
    // uv mapping below is already meaningless.
    if (f32(global_id.x) >= res.x || f32(global_id.y) >= res.y) {
        return;
    }

    // Integer pair for textureStore, float pair for the arithmetic below.
    let pixel = vec2<u32>(global_id.x, global_id.y);
    let pixf = vec2<f32>(f32(global_id.x), f32(global_id.y));

    // ---- Step 1: pixel -> complex plane ---------------------------------
    // +0.5 samples the pixel centre.
    let uv = (pixf + vec2<f32>(0.5, 0.5)) / res;

    // Widen x by the aspect ratio so the plane is not stretched.
    let aspect = res.x / max(res.y, 1.0);

    // NOTE the y flip: `1.0 - uv.y * 2.0`. Row 0 is the top of the window and
    // must be the top of the imaginary axis, so uv.y = 0 maps to +scale, not
    // -scale. Writing `uv.y * 2.0 - 1.0` here silently mirrors every plot.
    let re = uniforms.center.x + (uv.x * 2.0 - 1.0) * aspect * uniforms.scale;
    let im = uniforms.center.y + (1.0 - uv.y * 2.0) * uniforms.scale;
    let z = vec2<f32>(re, im);

    // ---- Step 2: apply the function -------------------------------------
    let w = apply_selected(z);

    // ---- Step 3: domain colouring ---------------------------------------
    // An overflow to inf or a NaN means the orbit escaped. Paint it solid
    // black: there is no meaningful argument to a non-finite number. Note that
    // in the iterated case it is the ESCAPING points, i.e. the complement of
    // the filled set, that go black -- the filled set itself stays finite and
    // keeps its domain colouring. This inverts the classical convention, where
    // the set is the black part; see the note on apply_selected.
    if (!(c_finite(w))) {
        textureStore(output_texture, pixel, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }


    var color = domain_color(
        w,
        uniforms.phase,
        uniforms.modulus_shading,
        uniforms.modulus_contour_density,
        uniforms.phase_contour_density,
    );

    // ---- Step 4: optional grid ------------------------------------------
    if (uniforms.grid_enabled != 0u) {
        color = apply_grid(color, z, uniforms.scale, res);
    }

    textureStore(output_texture, pixel, vec4<f32>(color, 1.0));
}
