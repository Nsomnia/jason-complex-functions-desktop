//! Viewport state: pan, zoom, and the pixel <-> complex-plane mapping.
//!
//! # The one thing this module must get right
//!
//! The CPU and the GPU must agree, to the bit, on which complex number a given
//! pixel shows. If they disagree the picture drifts away from the cursor while
//! you pan, "zoom toward the mouse" zooms toward the wrong point, and no test
//! anywhere else in the crate will catch it. So the mapping is written here in
//! the exact shape the WGSL kernel uses it, with the kernel's own lines quoted
//! beside it, and the two are meant to be diffed by eye whenever either changes.
//!
//! # Deliberate decoupling from `complex::Complex`
//!
//! The viewport centre is stored as two bare `f64`s (`center_re`, `center_im`)
//! and all the arithmetic is done on raw `f64`. This is intentional: this module
//! must not depend on `crate::complex`, which is owned and written in parallel,
//! so the two can be developed against the same shared ABI without a cyclic
//! build-order dependency. The cost is a handful of field accesses instead of
//! operator sugar; the benefit is that `camera` is a leaf module that compiles
//! on its own. If `complex::Complex` lands with a `+`, `-`, and a
//! `From<(f64, f64)>`, the conversion at the boundary in `app.rs` is
//! mechanical and not this module's problem.
//!
//! # Coordinate conventions
//!
//! * Pixels are `(px, py)` with `px` growing **right** and `py` growing
//!   **down**, matching image space, wgpu's `pixel_index`, and egui's
//!   `PointerState::pos` (after flipping it, which `app.rs` owns).
//! * The mapping is evaluated at the pixel **centre**, i.e. `px + 0.5`. The
//!   kernel does the same, so a `f32` pixel index and an integer pixel column
//!   select the same point.
//! * `py = 0` is the **top** row, which carries the **largest** imaginary part.
//!   The imaginary axis therefore points up the screen, and `im` *decreases*
//!   as `py` increases. This is the standard computer-graphics orientation and
//!   it is the single most common place for a plot to come out mirrored.

/// Half-height of the viewport in complex units at startup.
///
/// Matches the `Default` view of [`crate::uniforms::Uniforms`], so the window
/// opens showing exactly the view the uniform block describes before the first
/// frame.
pub const DEFAULT_SCALE: f64 = 1.5;

/// Smallest half-height the camera will permit, in complex units.
///
/// At `1e-9` the visible width of a 4:3 viewport is `~5e-9` complex units,
/// which still resolves to ~2000 significant decimal digits of detail, so this
/// is far past anything legible. The point of the floor is numerical: below it
/// the product `uv * aspect * scale` starts losing bits to the subnormals, and
/// `scale / scale` in the inverse starts returning garbage.
pub const MIN_SCALE: f64 = 1e-9;

/// Largest half-height the camera will permit, in complex units.
///
/// The mirror image of [`MIN_SCALE`]. At `1e9` the viewport spans `~6.7e9`
/// complex units, far outside the interesting region, and it keeps `scale`
/// small enough that `2 * aspect * scale` cannot overflow an `f32` uniform.
pub const MAX_SCALE: f64 = 1e9;

/// Exponential smoothing rate for [`Camera::update`], in units of `1/s`.
///
/// A rate of 18 corresponds to a 55 ms time constant: fast enough that the
/// image feels attached to the cursor, slow enough that a scripted transition
/// reads as motion rather than a jump cut. At 60 Hz the per-frame blend factor
/// is `1 - exp(-18/60) = 0.26`, so a step is covered in about eight frames.
const SMOOTHING_RATE: f64 = 18.0;

/// Relative distance below which [`Camera::is_settled`] reports "at rest".
///
/// The centre is compared in units of `scale` and the scale itself in relative
/// terms, so a single epsilon works from `1e-9` to `1e9` without the test
/// being vacuous at one end and unreachable at the other.
const SETTLE_EPSILON: f64 = 1e-9;

/// Decimal places used by the viewport readout in [`fmt_component`], and
/// mantissa places (one fewer) in [`fmt_exponential`].
const READOUT_DECIMALS: usize = 3;

/// The view onto the complex plane: where we are, and where we are going.
///
/// `center_*` and `scale` are the *current* state and are what
/// [`Camera::complex_at_pixel`] and the uniform bridge read. The `target_*`
/// fields are where the camera is heading; [`Camera::update`] eases the current
/// values toward them.
///
/// Direct manipulation ([`Camera::pan`], [`Camera::zoom_at`]) moves the current
/// state *and* its target together, so pointer input is never lagged by the
/// smoother. Programmatic moves that should be animated should change only the
/// target, via [`Camera::set_target`].
///
/// All public fields are `f64` rather than `f32` even though the uniform block
/// is `f32`: repeated multiply/divide in the pan and zoom paths compounds
/// rounding error, and doing the arithmetic in double leaves well over a decade
/// of headroom before the `f32` narrowing at [`Camera::as_uniform_center`]
/// becomes visible. That narrowing happens exactly once, at the boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct Camera {
    /// Real part of the viewport centre, in complex units.
    pub center_re: f64,
    /// Imaginary part of the viewport centre, in complex units.
    pub center_im: f64,
    /// Half-height of the viewport in complex units.
    ///
    /// This is the `scale` field of the uniform block: the visible rectangle is
    /// `2 * scale` tall and `2 * aspect * scale` wide, centred on
    /// `(center_re, center_im)`.
    pub scale: f64,
    /// Real part the camera is easing toward.
    pub target_center_re: f64,
    /// Imaginary part the camera is easing toward.
    pub target_center_im: f64,
    /// Half-height the camera is easing toward, in complex units.
    pub target_scale: f64,
}

impl Default for Camera {
    /// The view the app opens in: centred on the origin at
    /// [`DEFAULT_SCALE`], fully settled so no opening animation plays.
    fn default() -> Self {
        Self {
            center_re: 0.0,
            center_im: 0.0,
            scale: DEFAULT_SCALE,
            target_center_re: 0.0,
            target_center_im: 0.0,
            target_scale: DEFAULT_SCALE,
        }
    }
}

impl Camera {
    /// Creates a camera centred on the origin at [`DEFAULT_SCALE`].
    ///
    /// Identical to [`Camera::default`]; both exist because callers reach for
    /// one or the other by habit.
    pub fn new() -> Self {
        Self::default()
    }

    /// Maps a pixel to the complex number under its centre, as `(re, im)`.
    ///
    /// # Agreement with the kernel
    ///
    /// The WGSL side of `shaders/domain_coloring.wgsl` computes, per pixel:
    ///
    /// ```wgsl
    /// let uv     = (vec2<f32>(pixel) + 0.5) / resolution;
    /// let aspect = resolution.x / resolution.y;
    /// let re     = center.x + (uv.x * 2.0 - 1.0) * aspect * scale;
    /// let im     = center.y + (1.0 - uv.y * 2.0) * scale;
    /// ```
    ///
    /// The four lines below are those four lines, in the same order and with the
    /// same operations, widened to `f64`. If the shader changes, change this and
    /// re-run `centre_pixel_maps_to_the_viewport_centre` and
    /// `row_zero_is_the_top_of_the_viewport`; those two tests are the tripwire.
    ///
    /// Note the `1.0 - uv.y * 2.0` on the imaginary line. That is the y-flip:
    /// `py = 0` is the top of the image and therefore the *largest* `im`, while
    /// the bottom of the image is the *smallest* `im`.
    ///
    /// # Degenerate viewports
    ///
    /// A zero, negative, or non-finite width or height (a minimised window, a
    /// collapsed dock) yields a `1.0 x 1.0` viewport rather than `NaN`. A `NaN`
    /// here would poison the camera permanently, because every pan and zoom
    /// computes a difference of two calls to this function.
    pub fn complex_at_pixel(&self, px: f64, py: f64, width: f32, height: f32) -> (f64, f64) {
        let (w, h, aspect) = normalized_viewport(width, height);
        let uv_x = (px + 0.5) / w; // uv = (pixel + 0.5) / resolution;
        let uv_y = (py + 0.5) / h;
        let re = self.center_re + (uv_x * 2.0 - 1.0) * aspect * self.scale;
        let im = self.center_im + (1.0 - uv_y * 2.0) * self.scale;
        (re, im)
    }

    /// Maps a complex number back to the pixel that shows it, as `(px, py)`.
    ///
    /// The exact algebraic inverse of [`Camera::complex_at_pixel`], and its
    /// round trip is tested to `1e-9`. Used for "what am I hovering over?",
    /// double-click-to-centre, and drawing a cursor readout in world units.
    ///
    /// The returned coordinates address pixel centres, so an exact hit on the
    /// centre of pixel `n` returns `n + 0.5`.
    ///
    /// `#[allow(dead_code)]`: no runtime caller, and there is nothing honest to
    /// invent one for. Every feature that exists today reads the *forward*
    /// direction — pan, zoom-to-cursor and the viewport readout all start from a
    /// pixel — so the inverse is a one-way street that only a future
    /// world-space marker or a "fly to this point" command would walk. It is
    /// kept, rather than deleted, because it is half of the crate's main defence
    /// against rule 3: `round_trip_is_accurate_within_one_e_minus_nine` and
    /// `round_trip_survives_panning_and_zooming` are the tests that fail if this
    /// drifts away from the shader's arithmetic. Deleting the function would
    /// delete the ability to check that.
    #[allow(dead_code)]
    pub fn pixel_at_complex(&self, re: f64, im: f64, width: f32, height: f32) -> (f64, f64) {
        let (w, h, aspect) = normalized_viewport(width, height);
        // Invert re = center_re + (uv_x * 2 - 1) * aspect * scale  for uv_x...
        let nx = (re - self.center_re) / (aspect * self.scale);
        // ...and im = center_im + (1 - uv_y * 2) * scale  for uv_y.
        let ny = (im - self.center_im) / self.scale;
        // Then undo the +0.5 pixel-centre offset and the y-flip.
        let px = (nx + 1.0) * 0.5 * w - 0.5;
        let py = (1.0 - ny) * 0.5 * h - 0.5;
        (px, py)
    }

    /// Translates the view by a pixel-space drag, moving the image with the
    /// cursor.
    ///
    /// `dx_px` and `dy_px` are the drag deltas in the caller's coordinate
    /// system: positive `dy_px` is downward on screen. The content follows the
    /// cursor, so the world point that was under the cursor at the start of the
    /// drag is still under it at the end.
    ///
    /// # Why there is no drift
    ///
    /// The obvious implementation is a closed form such as
    /// `center_re -= 2.0 * dx_px * aspect * scale / width`, which is correct
    /// but repeats the viewport algebra in a second place, and the two copies
    /// eventually disagree - that is how zoom-dependent pan drift gets in.
    /// Instead this samples the mapping twice, at the origin and at the drag
    /// vector, and applies the negated difference. Because both samples share
    /// the same centre and scale, those cancel exactly and the difference is
    /// precisely the world vector the drag spans, with no dependence on where
    /// the viewport is or how far it is zoomed in.
    ///
    /// A non-finite drag is ignored rather than propagated.
    pub fn pan(&mut self, dx_px: f64, dy_px: f64, width: f32, height: f32) {
        if !dx_px.is_finite() || !dy_px.is_finite() {
            return;
        }
        let (re0, im0) = self.complex_at_pixel(0.0, 0.0, width, height);
        let (re1, im1) = self.complex_at_pixel(dx_px, dy_px, width, height);
        self.center_re += re0 - re1;
        self.center_im += im0 - im1;
        // A drag is a direct manipulation: move the target too, so the smoother
        // never fights the user's hand.
        self.target_center_re = self.center_re;
        self.target_center_im = self.center_im;
        self.clamp_scale();
    }

    /// Multiplies the scale by `factor` while holding the world point under
    /// `cursor_px` fixed on screen. The classic "zoom toward the mouse".
    ///
    /// `factor > 1.0` zooms in, `factor < 1.0` zooms out; egui's
    /// `ScrollDelta::LineDelta` maps to `factor.powf(lines)`.
    ///
    /// The implementation is three steps and the order matters: sample the world
    /// point under the cursor *before* changing the scale, apply the new scale,
    /// then shift the centre by however much that point moved. Doing it in any
    /// other order - for instance scaling the cursor offset itself - reintroduces
    /// a term that is only correct in the limit of an infinitesimal zoom, which
    /// shows up as the view creeping away from the cursor on every wheel notch.
    ///
    /// A non-positive or non-finite `factor`, or a non-finite cursor position,
    /// is ignored: a flipped or `NaN` scale cannot be recovered from by
    /// [`Camera::clamp_scale`], so the input has to be rejected at the door.
    pub fn zoom_at(&mut self, factor: f64, cursor_px: (f64, f64), width: f32, height: f32) {
        if !factor.is_finite() || factor <= 0.0 {
            return;
        }
        if !cursor_px.0.is_finite() || !cursor_px.1.is_finite() {
            return;
        }
        let (re_before, im_before) = self.complex_at_pixel(cursor_px.0, cursor_px.1, width, height);
        self.scale *= factor;
        self.clamp_scale();
        let (re_after, im_after) = self.complex_at_pixel(cursor_px.0, cursor_px.1, width, height);
        self.center_re += re_before - re_after;
        self.center_im += im_before - im_after;
        self.target_scale = self.scale;
        self.target_center_re = self.center_re;
        self.target_center_im = self.center_im;
    }

    /// Forces the scale, and its target, back inside
    /// `[MIN_SCALE, MAX_SCALE]`, replacing a non-finite value with
    /// [`DEFAULT_SCALE`].
    ///
    /// Called by every mutator, so out-of-range scale cannot enter the camera
    /// from the app layer. The target is clamped alongside the current value on
    /// purpose: leaving an out-of-range target would make [`Camera::update`]
    /// asymptote towards a value [`Camera::clamp_scale`] forbids, so the camera
    /// would never quite settle and `is_settled` would report motion forever.
    pub fn clamp_scale(&mut self) {
        self.scale = sanitize_scale(self.scale);
        self.target_scale = sanitize_scale(self.target_scale);
    }

    /// Snaps the current and target state back to the origin at
    /// [`DEFAULT_SCALE`].
    ///
    /// Instant rather than animated. The app layer animates home by pointing
    /// the target at the default view with [`Camera::set_target`] and polling
    /// [`Camera::is_settled`].
    pub fn reset(&mut self) {
        self.center_re = 0.0;
        self.center_im = 0.0;
        self.scale = DEFAULT_SCALE;
        self.target_center_re = 0.0;
        self.target_center_im = 0.0;
        self.target_scale = DEFAULT_SCALE;
    }

    /// Points the camera at a new view without moving it there yet.
    ///
    /// The camera then travels to `(re, im, scale)` over subsequent
    /// [`Camera::update`] calls, easing out of wherever it currently is. The
    /// scale argument is clamped as usual. This is the entry point for animated
    /// transitions - "home", "zoom to fit", "fly to a clicked point" - and the
    /// one that makes the `target_*` fields and [`Camera::is_settled`] mean
    /// something.
    pub fn set_target(&mut self, re: f64, im: f64, scale: f64) {
        self.target_center_re = if re.is_finite() { re } else { self.center_re };
        self.target_center_im = if im.is_finite() { im } else { self.center_im };
        self.target_scale = if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            self.target_scale
        };
        self.clamp_scale();
    }

    /// Advances the eased motion by `dt` seconds.
    ///
    /// Uses the frame-rate-independent blend factor `1 - exp(-k * dt)` rather
    /// than the naive `k * dt`. The naive form is a first-order Taylor expansion
    /// of the correct one, and it is only valid while `k * dt` is small: its
    /// step size is fixed per *frame*, so a 30 Hz machine eases at half the
    /// rate of a 60 Hz one, and once `k * dt > 1` the factor exceeds 1 and the
    /// camera overshoots. The exponential form is exact at any `dt` - one step of
    /// 0.1 s lands in the same place as two steps of 0.05 s, to rounding - and
    /// it is stable for arbitrarily long frames.
    ///
    /// The scale is eased in *log* space, i.e. geometrically rather than
    /// linearly, because scale is a ratio: interpolating `1.5 -> 0.01` linearly
    /// spends almost the entire animation in the top decade and then snaps. In
    /// log space the zoom reads as a constant-rate approach, which is what the
    /// eye expects. Both eases have the same `1 - exp(-k * dt)` factor, so they
    /// stay in step with each other and remain frame-rate independent.
    ///
    /// A non-finite or non-positive `dt` is ignored, so a first frame stamped
    /// with no elapsed time cannot divide by zero.
    pub fn update(&mut self, dt: f64) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let t = 1.0 - (-SMOOTHING_RATE * dt).exp();
        self.center_re += (self.target_center_re - self.center_re) * t;
        self.center_im += (self.target_center_im - self.center_im) * t;
        if self.scale > 0.0 && self.target_scale > 0.0 {
            // Geometric interpolation: scale *= (target / scale) ^ t.
            let ratio = self.target_scale / self.scale;
            self.scale *= ratio.powf(t);
        }
        self.clamp_scale();
    }

    /// Reports whether the eased motion has converged on its target.
    ///
    /// True immediately after a direct manipulation, since those move the
    /// current state and the target together. The app layer uses this to decide
    /// when a transition animation may be dropped, and to avoid re-issuing a
    /// `set_target` for a view it has already reached.
    pub fn is_settled(&self) -> bool {
        // Express the centre tolerance in world units scaled by the current
        // zoom, so "settled" means the same thing at every magnification.
        let center_tolerance = SETTLE_EPSILON * self.scale.abs().max(MIN_SCALE);
        let scale_tolerance = SETTLE_EPSILON * self.scale.abs().max(self.target_scale.abs());
        (self.target_center_re - self.center_re).abs() <= center_tolerance
            && (self.target_center_im - self.center_im).abs() <= center_tolerance
            && (self.target_scale - self.scale).abs() <= scale_tolerance
    }

    /// The viewport centre as the uniform block's `center: [f32; 2]`.
    ///
    /// This is the `f64` to `f32` narrowing mentioned on [`Camera`]: the only
    /// place it happens. Call it once per frame, immediately before
    /// `queue.write_buffer`.
    pub fn as_uniform_center(&self) -> [f32; 2] {
        [self.center_re as f32, self.center_im as f32]
    }

    /// The half-height as the uniform block's `scale: f32`.
    ///
    /// The only `f64` to `f32` narrowing on the scale, for the same reason as
    /// [`Camera::as_uniform_center`].
    pub fn as_uniform_scale(&self) -> f32 {
        self.scale as f32
    }

    /// A one-line human-readable description of the viewport, for the status
    /// bar. For example: `center = -0.162+0.020i, height = 3.00e-02`.
    ///
    /// Components print in fixed point while they are in a comfortable range
    /// and switch to a two-digit-exponent scientific form once they are not, so
    /// the line neither loses a zoomed-out centre to `0.000` nor fills the bar
    /// with `0.00` at extreme zoom. The exponent is zero-padded to match what a
    /// calculator shows. A signed zero is normalised to `0.000` so the readout
    /// cannot flicker between `0.000` and `-0.000` as the camera drifts.
    ///
    /// Reports the *current* values, not the targets, because this is what the
    /// user is looking at while it eases.
    pub fn viewport_description(&self) -> String {
        format!(
            "center = {}{}{}i, height = {}",
            fmt_component(self.center_re),
            if self.center_im < 0.0 { "-" } else { "+" },
            fmt_component(self.center_im.abs()),
            fmt_exponential(self.scale),
        )
    }
}

/// Returns `(width, height, aspect)` as finite, positive `f64`s.
///
/// A viewport can legitimately be zero-sized for a frame or two (minimised
/// window, collapsed dock). Dividing by that would make the mapping `NaN`, and
/// because pan and zoom are built from *differences* of mapping results, one
/// `NaN` frame would poison the camera for the rest of the session. Substituting
/// a `1 x 1` viewport keeps the arithmetic finite and makes the frame a no-op.
fn normalized_viewport(width: f32, height: f32) -> (f64, f64, f64) {
    let w = if width.is_finite() && width > 0.0 {
        f64::from(width)
    } else {
        1.0
    };
    let h = if height.is_finite() && height > 0.0 {
        f64::from(height)
    } else {
        1.0
    };
    (w, h, w / h)
}

/// Clamps one scale into `[MIN_SCALE, MAX_SCALE]`, mapping a non-finite value
/// to [`DEFAULT_SCALE`].
///
/// `f64::clamp` would happily return `NaN` for a `NaN` input, and `NaN` scale
/// survives every later comparison silently, so it is caught here.
fn sanitize_scale(scale: f64) -> f64 {
    if !scale.is_finite() {
        return DEFAULT_SCALE;
    }
    scale.clamp(MIN_SCALE, MAX_SCALE)
}

/// Formats a real or imaginary component of the viewport readout.
///
/// Fixed point inside `[1e-4, 1e5)`, scientific outside it, and always
/// `READOUT_DECIMALS` places within the fixed-point band so the text does not
/// jitter in width as the value changes.
fn fmt_component(value: f64) -> String {
    if !value.is_finite() {
        return "n/a".to_string();
    }
    if value == 0.0 {
        // Catches -0.0 as well, which would otherwise print as "-0.000".
        return format!("{:.*}", READOUT_DECIMALS, 0.0);
    }
    let magnitude = value.abs();
    if !(1e-4..1e5).contains(&magnitude) {
        return fmt_exponential(value);
    }
    let fixed = format!("{:.*}", READOUT_DECIMALS, value);
    if fixed == format!("-{:.*}", READOUT_DECIMALS, 0.0) {
        // Rounded down to zero from a small negative: show it as zero rather
        // than as a sign that means nothing at this resolution.
        return format!("{:.*}", READOUT_DECIMALS, 0.0);
    }
    fixed
}

/// Formats a value in scientific notation with a zero-padded two-digit
/// exponent, e.g. `3.00e-02` rather than Rust's default `3.00e-2`.
///
/// Purely a string fix-up: the mantissa is already produced by `{:.*e}` and the
/// exponent is only re-padded when it is a single digit, so there is nothing
/// here that can fail to parse.
fn fmt_exponential(value: f64) -> String {
    let raw = format!("{:.*e}", READOUT_DECIMALS - 1, value);
    let Some((mantissa, exponent)) = raw.split_once('e') else {
        // Unreachable: `{:e}` always emits an 'e'.
        return raw;
    };
    let (sign, digits) = match exponent.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("+", exponent),
    };
    if digits.len() < 2 {
        format!("{mantissa}e{sign}0{digits}")
    } else {
        raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wide viewport, the common case for this app.
    const W: f32 = 1600.0;
    /// Tall counterpart of [`W`]; aspect 2:1.
    const H: f32 = 800.0;

    /// Pixel coordinate of the exact centre of the [`W`]-wide viewport.
    ///
    /// The `- 0.5` is the half-pixel again: pixel `n` is sampled at `n + 0.5`,
    /// so the geometric centre of a 1600-wide viewport falls at 799.5, the
    /// *centre* of pixel 799, and not at the index 800. Tests that mean "the
    /// middle of the screen" have to say so explicitly.
    fn centre_px() -> f64 {
        f64::from(W) / 2.0 - 0.5
    }

    /// Pixel coordinate of the exact centre of the [`H`]-tall viewport.
    fn centre_py() -> f64 {
        f64::from(H) / 2.0 - 0.5
    }

    /// Asserts two `f64`s agree to within `tolerance`, with a message that
    /// names both values so a failure is readable without a debugger.
    #[track_caller]
    fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{what}: {actual} != {expected} (tolerance {tolerance})"
        );
    }

    #[test]
    fn new_starts_centred_on_the_origin_at_the_default_scale() {
        let cam = Camera::new();
        assert_eq!(cam.center_re, 0.0);
        assert_eq!(cam.center_im, 0.0);
        assert_eq!(cam.scale, DEFAULT_SCALE);
        assert_eq!(cam.scale, 1.5);
    }

    #[test]
    fn new_and_default_agree() {
        assert_eq!(Camera::new(), Camera::default());
    }

    #[test]
    fn default_matches_the_uniform_default_view() {
        // `Uniforms::default()` is the state the first frame uploads, so the
        // camera that produces it has to agree or the picture jumps on frame 1.
        let cam = Camera::default();
        assert_eq!(cam.as_uniform_center(), [0.0, 0.0]);
        assert_eq!(cam.as_uniform_scale(), 1.5);
    }

    #[test]
    fn centre_pixel_maps_to_the_viewport_centre() {
        let cam = Camera::new();
        let (re, im) = cam.complex_at_pixel(centre_px(), centre_py(), W, H);
        assert_close(re, 0.0, 1e-12, "centre re");
        assert_close(im, 0.0, 1e-12, "centre im");

        // And the centre of the view really is 799.5 / 399.5: pixel 800 sits a
        // half pixel right of and a half pixel below the middle, so it samples
        // a point slightly right of and slightly below the origin.
        let (re, im) = cam.complex_at_pixel(800.0, 400.0, W, H);
        assert_close(re, 0.001_875, 1e-12, "half-pixel offset in re");
        assert_close(im, -0.001_875, 1e-12, "half-pixel offset in im");
    }

    #[test]
    fn mapping_matches_the_wgsl_kernel_line_for_line() {
        // Hard-coded literals, transcribed by hand from the kernel's four
        // statements rather than recomputed in Rust. Recomputing them would
        // only restate the implementation; literals pin the values, and a
        // change to either side of the contract has to show up here.
        //
        //   uv = (pixel + 0.5) / resolution   aspect = width / height
        //   re = center.x + (uv.x * 2 - 1) * aspect * scale
        //   im = center.y + (1 - uv.y * 2) * scale
        //
        // with center = (0, 0), scale = 1.5, resolution = (1600, 800), so
        // aspect = 2 and one world unit per pixel is 2*2*1.5/1600 in re and
        // 2*1.5/800 in im.
        let cam = Camera::new();

        // Pixel (0, 0): top-left. uv = (0.0003125, 0.000625).
        let (re, im) = cam.complex_at_pixel(0.0, 0.0, W, H);
        assert_close(re, -2.998_125, 1e-12, "top-left re");
        assert_close(im, 1.498_125, 1e-12, "top-left im");

        // Pixel (1599, 799): bottom-right. uv = (0.9996875, 0.999375).
        let (re, im) = cam.complex_at_pixel(1599.0, 799.0, W, H);
        assert_close(re, 2.998_125, 1e-12, "bottom-right re");
        assert_close(im, -1.498_125, 1e-12, "bottom-right im");

        // Pixel (800, 400): the middle. uv = (0.5003125, 0.500625), so the
        // half-pixel offset shows up here as the quarter-unit world offset.
        let (re, im) = cam.complex_at_pixel(800.0, 400.0, W, H);
        assert_close(re, 0.001_875, 1e-12, "centre-ish re");
        assert_close(im, -0.001_875, 1e-12, "centre-ish im");
    }

    #[test]
    fn row_zero_is_the_top_of_the_viewport() {
        // The y-flip. Image row 0 carries the largest imaginary part; this is
        // the convention a mirrored plot gets wrong, and it cannot be seen from
        // the round-trip test because the inverse is wrong in the same way.
        let cam = Camera::new();
        let (_, top) = cam.complex_at_pixel(0.0, 0.0, W, H);
        let (_, bottom) = cam.complex_at_pixel(0.0, f64::from(H) - 1.0, W, H);
        assert!(
            top > bottom,
            "top row {top} must have a larger imaginary part than bottom row {bottom}"
        );
        // Sampling at the edges of the outermost rows (-0.5 and H-0.5) hits the
        // viewport boundary exactly rather than half a pixel inside it.
        let (_, exact_top) = cam.complex_at_pixel(0.0, -0.5, W, H);
        let (_, exact_bottom) = cam.complex_at_pixel(0.0, f64::from(H) - 0.5, W, H);
        assert_close(exact_top, cam.scale, 1e-12, "top edge is +scale");
        assert_close(exact_bottom, -cam.scale, 1e-12, "bottom edge is -scale");
    }

    #[test]
    fn horizontal_extent_is_aspect_scaled() {
        let cam = Camera::new();
        let (left, _) = cam.complex_at_pixel(-0.5, 0.0, W, H);
        let (right, _) = cam.complex_at_pixel(f64::from(W) - 0.5, 0.0, W, H);
        let half_width = f64::from(W) / f64::from(H) * cam.scale;
        assert_close(left, -half_width, 1e-12, "left edge");
        assert_close(right, half_width, 1e-12, "right edge");
        // Aspect 2:1, so the viewport is twice as wide as it is tall.
        assert_close(
            right - left,
            2.0 * half_width,
            1e-12,
            "visible width is 2 * aspect * scale",
        );
    }

    #[test]
    fn round_trip_is_accurate_within_one_e_minus_nine() {
        // The most valuable test in this file: forward then inverse must be a
        // no-op to nine decimal places, at the corners, the centre, and the
        // edge midpoints, across several magnifications.
        for scale in [1.5_f64, 0.05, 40.0, 3.0e-3, 250.0] {
            let mut cam = Camera::new();
            cam.scale = scale;
            cam.center_re = -0.162;
            cam.center_im = 0.020;
            cam.clamp_scale();

            let w = f64::from(W);
            let h = f64::from(H);
            let points = [
                (0.0, 0.0),
                (w - 1.0, 0.0),
                (0.0, h - 1.0),
                (w - 1.0, h - 1.0),
                (w / 2.0, h / 2.0),
                (w / 4.0, h / 4.0),
                (0.5, 0.5),
                (w - 0.5, h - 0.5),
            ];

            for (px, py) in points {
                let (re, im) = cam.complex_at_pixel(px, py, W, H);
                let (rx, ry) = cam.pixel_at_complex(re, im, W, H);
                assert_close(rx, px, 1e-9, "round-trip px");
                assert_close(ry, py, 1e-9, "round-trip py");
            }
        }
    }

    #[test]
    fn round_trip_survives_panning_and_zooming() {
        // The round trip has to hold after the camera has been moved, not just
        // in the default view - a pan that only shifts one of the two axes
        // would pass the default-view test.
        let mut cam = Camera::new();
        for step in 0..20 {
            cam.pan(37.5, -11.25, W, H);
            cam.zoom_at(1.07, (213.0, 655.0), W, H);
            cam.clamp_scale();
            let (re, im) = cam.complex_at_pixel(1001.5, 22.5, W, H);
            let (rx, ry) = cam.pixel_at_complex(re, im, W, H);
            assert_close(rx, 1001.5, 1e-9, "round-trip px after motion");
            assert_close(ry, 22.5, 1e-9, "round-trip py after motion");
            assert!(cam.scale > 0.0, "scale went non-positive at step {step}");
        }
    }

    #[test]
    fn inverse_is_exact_for_the_viewport_corners() {
        // A hand-checkable case that pins the pixel-centre offset and the flip.
        let mut cam = Camera::new();
        cam.scale = 2.0;
        cam.center_re = 0.5;
        cam.center_im = -0.25;
        let aspect = f64::from(W) / f64::from(H);

        let (px, py) = cam.pixel_at_complex(cam.center_re, cam.center_im, W, H);
        assert_close(px, centre_px(), 1e-9, "centre px");
        assert_close(py, centre_py(), 1e-9, "centre py");

        let (top_left_re, top_left_im) = cam.complex_at_pixel(-0.5, -0.5, W, H);
        let (rx, ry) = cam.pixel_at_complex(top_left_re, top_left_im, W, H);
        assert_close(rx, -0.5, 1e-9, "top-left px");
        assert_close(ry, -0.5, 1e-9, "top-left py");
        assert!(top_left_im > cam.center_im, "top-left is above the centre");
        assert!(
            aspect > 1.0 && top_left_re < cam.center_re,
            "wide viewport shows more real units"
        );
    }

    #[test]
    fn zoom_holds_the_world_point_under_the_cursor() {
        // Zoom toward the mouse. The point under the cursor must not move, for
        // any zoom factor and any cursor position, including the extreme ones.
        let cam = Camera::new();
        for &(px, py) in &[(0.0, 0.0), (800.0, 400.0), (1599.0, 799.0), (1.0, 799.0)] {
            for &factor in &[0.5, 0.9, 1.01, 1.1, 2.0, 10.0, 0.001] {
                let mut zoomed = cam.clone();
                let (before_re, before_im) = zoomed.complex_at_pixel(px, py, W, H);
                zoomed.zoom_at(factor, (px, py), W, H);
                let (after_re, after_im) = zoomed.complex_at_pixel(px, py, W, H);
                assert_close(after_re, before_re, 1e-12, "re under cursor");
                assert_close(after_im, before_im, 1e-12, "im under cursor");
            }
        }
    }

    #[test]
    fn zoom_holds_the_cursor_point_from_an_offset_view() {
        // Same invariant, but starting off-axis and zoomed in, which is where a
        // sign error in the centre correction would survive the previous test.
        let mut cam = Camera::new();
        cam.center_re = 0.37;
        cam.center_im = -1.9;
        cam.scale = 0.004;
        let (px, py) = (1234.5, 77.5);
        let (before_re, before_im) = cam.complex_at_pixel(px, py, W, H);
        cam.zoom_at(3.0, (px, py), W, H);
        let (after_re, after_im) = cam.complex_at_pixel(px, py, W, H);
        assert_close(after_re, before_re, 1e-13, "re under cursor");
        assert_close(after_im, before_im, 1e-13, "im under cursor");
    }

    #[test]
    fn zoom_scales_the_visible_height_by_exactly_the_factor() {
        let mut cam = Camera::new();
        for &factor in &[0.25, 0.5, 2.0, 4.0] {
            let before = cam.scale;
            cam.zoom_at(factor, (800.0, 400.0), W, H);
            assert_close(cam.scale, before * factor, 1e-12, "scale after zoom");
        }
    }

    #[test]
    fn zoom_at_the_viewport_centre_only_changes_the_scale() {
        let mut cam = Camera::new();
        cam.zoom_at(2.5, (centre_px(), centre_py()), W, H);
        assert_close(cam.center_re, 0.0, 1e-12, "re unchanged");
        assert_close(cam.center_im, 0.0, 1e-12, "im unchanged");
    }

    #[test]
    fn zoom_rejects_non_positive_and_non_finite_factors() {
        let mut cam = Camera::new();
        for &bad in &[0.0, -2.0, f64::NAN, f64::INFINITY] {
            cam.zoom_at(bad, (800.0, 400.0), W, H);
            assert_eq!(cam.scale, DEFAULT_SCALE, "factor {bad} changed the scale");
            assert_eq!(cam.center_re, 0.0, "factor {bad} moved the centre");
        }
    }

    #[test]
    fn pan_translates_the_image_by_exactly_the_drag() {
        // The world point under a fixed pixel must move by the world vector the
        // drag spans, and by the same amount at any zoom level.
        let drag = (240.0_f64, -90.0_f64);
        for &scale in &[1.5_f64, 0.02, 100.0] {
            let mut cam = Camera::new();
            cam.scale = scale;
            let probe = (640.0_f64, 400.0_f64);
            let (re0, im0) = cam.complex_at_pixel(probe.0, probe.1, W, H);
            cam.pan(drag.0, drag.1, W, H);
            let (re1, im1) = cam.complex_at_pixel(probe.0, probe.1, W, H);

            // Independent closed form for the drag's world vector. The signs
            // differ between the axes purely because screen `y` grows downward
            // while world `im` grows upward: dragging right lowers the centre's
            // real part, and dragging up lowers its imaginary part, because in
            // both cases the content is being pulled after the cursor.
            let w_per_px = 2.0 * (f64::from(W) / f64::from(H)) * scale / f64::from(W);
            let h_per_px = 2.0 * scale / f64::from(H);
            assert_close(re0 - re1, drag.0 * w_per_px, 1e-12, "re moved by drag");
            assert_close(im0 - im1, -drag.1 * h_per_px, 1e-12, "im moved by drag");
        }
    }

    #[test]
    fn pan_moves_the_image_under_the_cursor_exactly() {
        // The definition of dragging: the world point that was under the cursor
        // before the drag is under it after, and it is now `d` pixels along.
        let mut cam = Camera::new();
        cam.center_re = -0.4;
        cam.center_im = 0.9;
        let cursor = (321.5_f64, 654.5_f64);
        let drag = (-77.0_f64, 123.0_f64);

        let (before_re, before_im) = cam.complex_at_pixel(cursor.0, cursor.1, W, H);
        cam.pan(drag.0, drag.1, W, H);
        let (after_re, after_im) = cam.complex_at_pixel(cursor.0 + drag.0, cursor.1 + drag.1, W, H);
        assert_close(after_re, before_re, 1e-12, "re stayed under the cursor");
        assert_close(after_im, before_im, 1e-12, "im stayed under the cursor");

        // Which is the same statement as: the world point under the cursor is
        // now `drag` pixels away, via the inverse.
        let (px, py) = cam.pixel_at_complex(before_re, before_im, W, H);
        assert_close(px, cursor.0 + drag.0, 1e-9, "px follows the drag");
        assert_close(py, cursor.1 + drag.1, 1e-9, "py follows the drag");
    }

    #[test]
    fn pan_does_not_change_the_scale() {
        let mut cam = Camera::new();
        cam.zoom_at(7.0, (100.0, 100.0), W, H);
        let scale = cam.scale;
        cam.pan(13.0, -77.0, W, H);
        assert_eq!(cam.scale, scale, "panning must not zoom");
    }

    #[test]
    fn pan_by_the_viewport_width_is_one_whole_period() {
        // A drag of exactly the viewport width translates the mapping by one
        // full period: the same region is on screen, anchored one width to the
        // left, and panning back returns the mapping *exactly* where it started.
        // That last part is the drift test - a pan that quietly loses or gains
        // bits fails here at any offset and any zoom.
        let w = f64::from(W);
        let h = f64::from(H);
        for &scale in &[1.5_f64, 0.03, 12.0] {
            let mut cam = Camera::new();
            cam.center_re = -0.25;
            cam.center_im = 0.75;
            cam.scale = scale;
            cam.clamp_scale();
            let probe = (123.0_f64, 456.0_f64);
            let before = cam.complex_at_pixel(probe.0, probe.1, W, H);

            // The centre moves by exactly the full visible extent.
            let half_width = w / h * scale;
            cam.pan(w, 0.0, W, H);
            assert_close(cam.center_re, -0.25 - 2.0 * half_width, 1e-12, "centre re");
            assert_close(cam.center_im, 0.75, 1e-12, "centre im untouched");

            // The mapping is periodic in x with period W, and untouched in y.
            let shifted = cam.complex_at_pixel(probe.0 + w, probe.1, W, H);
            assert_close(shifted.0, before.0, 1e-12, "re is periodic in x");
            assert_close(shifted.1, before.1, 1e-12, "im is periodic in x");

            // Every world point has moved a full width to the right on screen.
            let (px, py) = cam.pixel_at_complex(before.0, before.1, W, H);
            assert_close(px, probe.0 + w, 1e-9, "px moved one width");
            assert_close(py, probe.1, 1e-9, "py unmoved by an x pan");

            // And back again: exact, with no accumulated drift.
            cam.pan(-w, 0.0, W, H);
            let after = cam.complex_at_pixel(probe.0, probe.1, W, H);
            assert_close(after.0, before.0, 1e-12, "re after pan there and back");
            assert_close(after.1, before.1, 1e-12, "im after pan there and back");
        }
    }

    #[test]
    fn pan_by_the_viewport_height_is_one_whole_period() {
        let h = f64::from(H);
        for &scale in &[1.5_f64, 0.03, 12.0] {
            let mut cam = Camera::new();
            cam.center_re = -0.25;
            cam.center_im = 0.75;
            cam.scale = scale;
            cam.clamp_scale();
            let probe = (123.0_f64, 456.0_f64);
            let before = cam.complex_at_pixel(probe.0, probe.1, W, H);

            cam.pan(0.0, h, W, H);
            assert_close(cam.center_im, 0.75 + 2.0 * scale, 1e-12, "centre im");
            assert_close(cam.center_re, -0.25, 1e-12, "centre re untouched");

            let shifted = cam.complex_at_pixel(probe.0, probe.1 + h, W, H);
            assert_close(shifted.0, before.0, 1e-12, "re is periodic in y");
            assert_close(shifted.1, before.1, 1e-12, "im is periodic in y");

            cam.pan(0.0, -h, W, H);
            let after = cam.complex_at_pixel(probe.0, probe.1, W, H);
            assert_close(after.0, before.0, 1e-12, "re after pan there and back");
            assert_close(after.1, before.1, 1e-12, "im after pan there and back");
        }
    }

    #[test]
    fn pan_rejects_non_finite_deltas() {
        let mut cam = Camera::new();
        cam.pan(f64::NAN, 0.0, W, H);
        cam.pan(0.0, f64::INFINITY, W, H);
        assert_eq!(cam, Camera::default(), "a bad drag must be a no-op");
    }

    #[test]
    fn degenerate_viewport_does_not_produce_nan() {
        let mut cam = Camera::new();
        for &(w, h) in &[(0.0_f32, 0.0_f32), (0.0, 800.0), (1600.0, 0.0), (-5.0, 5.0)] {
            let (re, im) = cam.complex_at_pixel(10.0, 20.0, w, h);
            assert!(re.is_finite() && im.is_finite(), "mapping went non-finite");
            cam.pan(5.0, 5.0, w, h);
            assert!(cam.center_re.is_finite(), "pan poisoned the centre");
            cam.zoom_at(2.0, (1.0, 1.0), w, h);
            assert!(cam.scale.is_finite(), "zoom poisoned the scale");
        }
    }

    #[test]
    fn clamp_scale_clamps_both_ends() {
        let mut cam = Camera::new();

        cam.scale = 0.0;
        cam.target_scale = 0.0;
        cam.clamp_scale();
        assert_eq!(cam.scale, MIN_SCALE);
        assert_eq!(
            cam.target_scale, MIN_SCALE,
            "the target must be clamped too"
        );

        cam.scale = 1e300;
        cam.target_scale = 1e300;
        cam.clamp_scale();
        assert_eq!(cam.scale, MAX_SCALE);
        assert_eq!(
            cam.target_scale, MAX_SCALE,
            "the target must be clamped too"
        );

        cam.scale = 0.5;
        cam.target_scale = 0.5;
        cam.clamp_scale();
        assert_eq!(cam.scale, 0.5, "an in-range scale is untouched");
    }

    #[test]
    fn clamp_scale_recovers_from_a_non_finite_scale() {
        let mut cam = Camera::new();
        cam.scale = f64::NAN;
        cam.clamp_scale();
        assert_eq!(cam.scale, DEFAULT_SCALE);
        assert!(cam.scale.is_finite());
    }

    #[test]
    fn zoom_cannot_escape_the_scale_limits() {
        let mut cam = Camera::new();
        for _ in 0..64 {
            cam.zoom_at(4.0, (800.0, 400.0), W, H);
        }
        assert!(
            cam.scale <= MAX_SCALE,
            "scale {} exceeded the cap",
            cam.scale
        );
        for _ in 0..256 {
            cam.zoom_at(0.25, (800.0, 400.0), W, H);
        }
        assert!(
            cam.scale >= MIN_SCALE,
            "scale {} fell below the floor",
            cam.scale
        );
    }

    #[test]
    fn direct_manipulation_settles_immediately() {
        let mut cam = Camera::new();
        cam.pan(10.0, 10.0, W, H);
        assert!(cam.is_settled(), "a drag must not leave easing pending");
        cam.zoom_at(1.5, (400.0, 200.0), W, H);
        assert!(cam.is_settled(), "a zoom must not leave easing pending");
    }

    #[test]
    fn reset_restores_the_default_view() {
        let mut cam = Camera::new();
        cam.pan(500.0, -500.0, W, H);
        cam.zoom_at(0.01, (10.0, 10.0), W, H);
        cam.set_target(3.0, 4.0, 9.0);
        assert!(!cam.is_settled());

        cam.reset();
        assert_eq!(cam, Camera::default());
        assert!(cam.is_settled(), "reset must leave no pending motion");
    }

    #[test]
    fn set_target_defers_the_move() {
        let mut cam = Camera::new();
        cam.set_target(2.0, -3.0, 0.5);
        assert_eq!(cam.center_re, 0.0, "set_target must not teleport");
        assert_eq!(cam.target_center_re, 2.0);
        assert!(!cam.is_settled());

        for _ in 0..600 {
            cam.update(1.0 / 60.0);
        }
        assert!(cam.is_settled(), "easing never converged");
        assert_close(cam.center_re, 2.0, 1e-6, "converged re");
        assert_close(cam.center_im, -3.0, 1e-6, "converged im");
        assert_close(cam.scale, 0.5, 1e-9, "converged scale");
    }

    #[test]
    fn set_target_clamps_a_wild_scale() {
        let mut cam = Camera::new();
        cam.set_target(0.0, 0.0, 1e200);
        assert_eq!(cam.target_scale, MAX_SCALE);
        cam.set_target(0.0, 0.0, -1.0);
        assert_eq!(cam.target_scale, MAX_SCALE, "a negative scale is rejected");
        assert!(cam.target_scale > 0.0);
    }

    #[test]
    fn update_is_frame_rate_independent() {
        // Two 50 ms steps must land in the same place as one 100 ms step. With
        // the naive `k * dt` factor these differ, and the gap grows with dt -
        // this is the regression that the exponential factor exists to prevent.
        let target = (7.0_f64, -4.0_f64, 0.125_f64);

        let mut coarse = Camera::new();
        coarse.set_target(target.0, target.1, target.2);
        coarse.update(0.1);

        let mut fine = Camera::new();
        fine.set_target(target.0, target.1, target.2);
        fine.update(0.05);
        fine.update(0.05);

        assert_close(fine.center_re, coarse.center_re, 1e-12, "re");
        assert_close(fine.center_im, coarse.center_im, 1e-12, "im");
        assert_close(fine.scale, coarse.scale, 1e-12, "scale");
    }

    #[test]
    fn update_never_overshoots_and_converges_monotonically() {
        let mut cam = Camera::new();
        cam.set_target(1.0, 0.0, 4.0);
        let mut last = 0.0_f64;
        for step in 0..300 {
            cam.update(1.0 / 60.0);
            assert!(cam.scale >= last, "scale moved backwards at step {step}");
            assert!(cam.scale <= 4.0, "scale overshot its target at step {step}");
            last = cam.scale;
        }
        assert!(cam.is_settled());
    }

    #[test]
    fn update_ignores_non_positive_and_non_finite_dt() {
        let mut cam = Camera::new();
        cam.set_target(5.0, 5.0, 5.0);
        for &bad in &[0.0, -0.016, f64::NAN, f64::INFINITY] {
            cam.update(bad);
        }
        assert_eq!(cam.center_re, 0.0, "a bad dt must not move the camera");
        assert!(cam.center_re.is_finite() && cam.scale.is_finite());
    }

    #[test]
    fn scale_easing_is_geometric() {
        // Log-space easing: the midpoint of a 1 -> 4 zoom is sqrt(4) = 2, not
        // 2.5. This is what makes a scripted zoom read as a steady approach.
        let mut cam = Camera::new();
        cam.set_target(0.0, 0.0, 4.0);
        let mut previous = cam.scale;
        for _ in 0..400 {
            cam.update(1.0 / 60.0);
            if cam.scale > 2.0 {
                // First frame past the halfway point: must still be close to
                // the geometric mean, not the arithmetic one.
                assert!(
                    cam.scale < 2.6,
                    "scale {} passed the halfway mark too abruptly",
                    cam.scale
                );
                assert!(cam.scale > previous, "scale must keep rising");
                return;
            }
            previous = cam.scale;
        }
        panic!("scale never passed the halfway mark");
    }

    #[test]
    fn is_settled_is_scale_relative() {
        // A residual centre error of 1e-6 world units is a thousand viewports
        // of drift at a 1e-9 half-height and utterly invisible at a 1e9 one.
        // The tolerance has to scale with the zoom in order to say that.
        let mut tight = Camera::new();
        tight.scale = 1e-9;
        tight.target_scale = 1e-9;
        tight.center_re = 1e-6;
        tight.target_center_re = 0.0;
        assert!(
            !tight.is_settled(),
            "1e-6 is a thousand viewports of drift at this magnification"
        );

        let mut wide = Camera::new();
        wide.scale = 1e9;
        wide.target_scale = 1e9;
        wide.center_re = 1e-6;
        wide.target_center_re = 0.0;
        assert!(wide.is_settled(), "1e-6 is invisible at this magnification");
    }

    #[test]
    fn uniform_bridge_reports_the_current_state() {
        let mut cam = Camera::new();
        cam.set_target(1.25, -0.5, 0.125);
        for _ in 0..600 {
            cam.update(1.0 / 60.0);
        }
        assert_eq!(cam.as_uniform_center(), [1.25, -0.5]);
        assert_eq!(cam.as_uniform_scale(), 0.125);
    }

    #[test]
    fn viewport_description_reads_well() {
        let mut cam = Camera::new();
        cam.center_re = -0.162;
        cam.center_im = 0.020;
        cam.scale = 0.03;
        assert_eq!(
            cam.viewport_description(),
            "center = -0.162+0.020i, height = 3.00e-02"
        );
    }

    #[test]
    fn viewport_description_signs_and_pads_the_exponent() {
        let mut cam = Camera::new();
        cam.center_re = 0.0;
        cam.center_im = -1.5;
        cam.scale = 1.5;
        assert_eq!(
            cam.viewport_description(),
            "center = 0.000-1.500i, height = 1.50e+00"
        );

        cam.center_re = 123_456.0;
        cam.center_im = 0.0;
        assert_eq!(
            cam.viewport_description(),
            "center = 1.23e+05+0.000i, height = 1.50e+00"
        );
    }

    #[test]
    fn viewport_description_normalises_negative_zero() {
        let mut cam = Camera::new();
        cam.center_re = -0.0;
        cam.center_im = -1e-9;
        let text = cam.viewport_description();
        assert!(!text.contains("-0.000"), "signed zero leaked into {text}");
        assert_eq!(text, "center = 0.000-1.00e-09i, height = 1.50e+00");
    }
}
