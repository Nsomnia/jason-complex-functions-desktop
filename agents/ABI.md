# ABI

The contract between the CPU and the GPU. **Read this before editing
`src/uniforms.rs` or either shader.**

`src/uniforms.rs` is the single source of truth. The WGSL struct in
`shaders/domain_coloring.wgsl` is a transcription of it, and no compiler in
this toolchain cross-checks the two. That is why the rules in this document
exist and why they are worded the way they are.

## Rule 1: the layout

`Uniforms` is 64 bytes, `#[repr(C)]`, and derives `bytemuck::Pod` and
`bytemuck::Zeroable`. It is bound as a `var<uniform>` in WGSL and uploaded whole.

| Offset | Size | Rust field | WGSL type | Meaning |
|---:|---:|---|---|---|
| 0 | 8 | `resolution: [f32; 2]` | `vec2<f32>` | render-target size in **physical** pixels, `(width, height)` |
| 8 | 8 | `center: [f32; 2]` | `vec2<f32>` | centre of the viewport in the complex plane, `(re, im)` |
| 16 | 4 | `scale: f32` | `f32` | **half-height** of the view, in complex units. Visible height is `2 * scale`, visible width is `2 * aspect * scale`. |
| 20 | 4 | `max_iter: u32` | `u32` | iteration cap for escape-time and series-based functions |
| 24 | 4 | `func_id: u32` | `u32` | function selector, 0–15. See the table below. |
| 28 | 4 | `phase: f32` | `f32` | hue rotation, in turns. Purely cosmetic. |
| 32 | 4 | `modulus_contour_density: f32` | `f32` | iso-modulus bands per e-fold of `|w|`. `0.0` disables. |
| 36 | 4 | `phase_contour_density: f32` | `f32` | iso-phase bands per turn. `0.0` disables. |
| 40 | 4 | `modulus_shading: f32` | `f32` | exponent on the modulus term. `< 1` exaggerates, `0` flattens to a pure hue map. |
| 44 | 4 | `grid_enabled: u32` | `u32` | non-zero draws the axes and the unit circle in **world** space |
| 48 | 4 | `frame: u32` | `u32` | monotonic frame counter. Reserved; the kernel does not read it yet. |
| 52 | 4 | `iterate: u32` | `u32` | non-zero applies the function `max_iter` times instead of once |
| 56 | 8 | `_pad: [u32; 2]` | `vec2<u32>` | explicit padding to 64. Never read. |

Verified on 2026-09-28 by compiling the struct standalone and printing
`std::mem::size_of` and each member's offset, and by the in-crate test
`field_offsets_match_the_documented_wgsl_layout`, which asserts all thirteen
offsets individually. `size_of == 64`, `align_of == 4`.

### Why 64, and why the padding is explicit

WGSL rounds a struct's alignment in the uniform address space up to 16. A Rust
struct of nothing but 32-bit scalars aligns to 4. Those are different numbers
and that is fine, because a standalone uniform binding starts at offset 0 in
both worlds. The one requirement is that the **total size** be a multiple of 16,
or wgpu rejects the buffer when the bind group layout is created — which is
reported as a layout validation error with no mention of the real cause, at a
moment that has nothing to do with the mistake. Two `const _: () = assert!`
guards in `src/uniforms.rs` catch it at compile time instead:

```rust
const _: () = assert!(std::mem::size_of::<Uniforms>() % 16 == 0);
const _: () = assert!(std::mem::size_of::<Uniforms>() == 64);
```

If you add a field and the second assertion fires, take 4 or 8 bytes out of
`_pad` and update `field_offsets_match_the_documented_wgsl_layout` **and** the
WGSL struct in the same edit. If the assertion is firing, do not "fix" it by
changing 64 to 65; the 64 is a constraint, not an observation.

> Stale comment warning: the module doc-comment at the top of
> `src/uniforms.rs` says "`Uniforms` is 80 bytes". It is 64. The asserts, the
> tests, and the WGSL all agree on 64. Fix the comment, not the layout.

## Rule 2: the WGSL side must change in the same commit

The WGSL declaration, verbatim, with the offsets its author documented:

```wgsl
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
```

The offset comments in the WGSL are not decoration. They are how the next person
checks the transcription without running anything. Keep them.

**The procedure, and it is not negotiable:**

1. Edit `src/uniforms.rs`: add, remove, reorder, or retype a field. Adjust
   `_pad`. Update both `const _: () = assert!` guards if the size moved.
2. Edit the WGSL struct to match, field for field, in the same order, with the
   offset comments updated.
3. Update `field_offsets_match_the_documented_wgsl_layout` with the new offsets.
4. Update the table at the top of this file.
5. `cargo test uniforms`.
6. Commit **all** of it as one commit.

Splitting steps 1 and 2 across two commits is the specific failure this document
exists to prevent, because a half-done ABI change produces a plot that is
wrong rather than a build that fails.

### The shader paths are duplicated too

`src/renderer.rs` holds both a path constant and an `include_str!` literal for
each shader, and they have to be kept in step **by hand** because `include_str!`
requires a literal and will not accept the constant:

```rust
const COMPUTE_SHADER_PATH: &str = "shaders/domain_coloring.wgsl";
const BLIT_SHADER_PATH: &str = "shaders/blit.wgsl";
const COMPUTE_SHADER_SRC: &str = include_str!("../shaders/domain_coloring.wgsl");
const BLIT_SHADER_SRC: &str = include_str!("../shaders/blit.wgsl");
```

If you move a shader, change all four lines, not two. And note the consequence:
because the source is `include_str!`-embedded rather than read from disk at
runtime, **editing a `.wgsl` file requires a cargo rebuild to take effect**.
There is no hot reload.

## Rule 3: the function id table

Ids are **append-only**. Never renumber, never reuse, never leave a gap that
something else later fills with a different function. `src/functions.rs` is the
source of truth; the `switch` in `shaders/domain_coloring.wgsl` (`fn apply`) is
the transcription.

| id | constant | formula | name | group |
|---:|---|---|---|---|
| 0 | `ID_IDENTITY` | `z` | identity | Polynomial |
| 1 | `ID_SQUARE` | `z^2` | square | Polynomial |
| 2 | `ID_CUBE` | `z^3` | cube | Polynomial |
| 3 | `ID_RECIPROCAL` | `1/z` | reciprocal | Reciprocal |
| 4 | `ID_Z2_MINUS_1` | `z^2 - 1` | `z^2 - 1` | Polynomial |
| 5 | `ID_Z3_MINUS_1` | `z^3 - 1` | `z^3 - 1` | Polynomial |
| 6 | `ID_BASILICA` | `z^3 - 2z` | basilica | Iterated |
| 7 | `ID_SIN` | `sin z` | sin | Trigonometric |
| 8 | `ID_COS` | `cos z` | cos | Trigonometric |
| 9 | `ID_SINC` | `sin z / z` | sinc | Trigonometric |
| 10 | `ID_SINH` | `sinh z` | sinh | Hyperbolic |
| 11 | `ID_EXP` | `e^z` | exp | Exponential |
| 12 | `ID_LOG` | `ln z`, principal branch | log | Transcendental |
| 13 | `ID_SQRT` | `sqrt z`, principal branch | sqrt | Transcendental |
| 14 | `ID_MOBIUS` | `(z - 1) / (z + 1)` | mobius | Mobius |
| 15 | `ID_JULIA` | `z^2 + c`, `c = -0.7269 + 0.1889i` | julia | Iterated |

`FUNCTION_COUNT = 16` and the array `FUNCTIONS: [FunctionEntry; 16]` are in id
order, entry `k` having `id == k`. That is asserted, not assumed.

**Why the ordering matters so much.** A `func_id` is not an internal detail. It
is written into the uniform block, switched on by the shader, and it is what any
saved view, config file, screenshot caption, or bug report will name. If
"sin" moves from 7 to 16, a user's stored view of the sine plot silently becomes
a view of whatever now occupies 7, and nothing anywhere reports an error. Hence:

- **Append only.** A new function takes 16, then 17.
- **Never reuse an id**, including one from a function that was removed. It may
  have been written to somebody's disk.
- **Never reorder** `FUNCTIONS`, even to sort it by name. The array order *is*
  the id order.
- When the table grows, `FUNCTION_COUNT` and the array length grow together, in
  the same edit as the constant.

## Rule 4: constants that exist on both sides

These pairs must be equal. Neither side may drift, and neither is in `Uniforms`.

| Rust | WGSL | Value | Why |
|---|---|---|---|
| `ITERATION_CAP` | `MAX_ITER_CLAMP` | `4096` | bounds the `iterate` loop so a fat-fingered `max_iter` cannot hang a frame |
| `SINC_SINGULARITY_GUARD_RADIUS` | `EPS` | `1e-12` | the removable singularity of `sin(z)/z` at 0, patched to its limit `1.0` |
| `JULIA_C` | `JULIA_C` | `-0.7269 + 0.1889i` | the Julia parameter for id 15 |
| `Complex::norm_hypot` | `c_norm` | — | see below |
| — | `TAU` | `6.283185307179586` | one turn, in radians |

`Complex::norm` is the naive `sqrt(re*re + im*im)`. `c_norm` in the shader is
the overflow-safe form, because a `f32` intermediate of `re*re` overflows at
about `1.8e19` and the naive form returns `inf` where the true modulus is
merely large. The two disagree on purpose, and both source files say so. The
shader author should use `c_norm`; the CPU tests may use either.

## Rule 5: the pixel-to-complex mapping

Both implementations, side by side. The WGSL is `main`, step 1; the Rust is
`Camera::complex_at_pixel`. They must stay identical, including the order of
the operations, because the Rust one is what the tests assert against.

```wgsl
let uv     = (pixf + vec2<f32>(0.5, 0.5)) / res;   // +0.5 samples the pixel centre
let aspect = res.x / max(res.y, 1.0);
let re     = uniforms.center.x + (uv.x * 2.0 - 1.0) * aspect * uniforms.scale;
let im     = uniforms.center.y + (1.0 - uv.y * 2.0) * uniforms.scale;
```

```rust
let uv_x = (px + 0.5) / w;
let uv_y = (py + 0.5) / h;
let re = self.center_re + (uv_x * 2.0 - 1.0) * aspect * self.scale;
let im = self.center_im + (1.0 - uv_y * 2.0) * self.scale;
```

**The y-axis is flipped, and this is the single easiest bug in this project to
ship.** Row 0 of the texture is the top of the window and is the **largest**
imaginary part. The bottom of the window is the **smallest** imaginary part.
That is what `1.0 - uv.y * 2.0` says. Writing `uv.y * 2.0 - 1.0` instead mirrors
every plot vertically, and a mirrored plot is a perfectly plausible-looking
picture, so it will not be caught by "does it look like a plot".

The tripwire tests in `src/camera.rs` are named `row_zero_is_the_top_of_the_viewport`
and `centre_pixel_maps_to_the_viewport_centre`. Run them. When you first get a
frame on screen, check the orientation **before** you go looking for anything
else, because every other observation you make will be conditioned on it.

There is a third test worth knowing about, and it is the best defence this
contract has: `mapping_matches_the_wgsl_kernel_line_for_line` asserts the
mapping against **hard-coded literals** transcribed from the kernel, rather than
recomputing it in Rust. Recomputing would only restate the implementation;
literals pin the values, so a change on either side of the contract shows up
there. If you touch the mapping, that test is the one that should fail, and
failing is the point.

`scale` is the **half-height**. The view spans `2 * scale` vertically and
`2 * aspect * scale` horizontally. The exact inverse, for hover readouts and
double-click-to-centre, is `Camera::pixel_at_complex`, and it addresses pixel
*centres*: an exact hit on the middle of pixel `n` returns `n + 0.5`.

`resolution` is the size in **physical** pixels. egui input is in logical
points. On a Retina display those differ by the scale factor, and multiplying a
drag delta by the wrong one is a bug that will not appear on a 1.0 display.
Unresolved; see [TODO.md](../TODO.md).

## Rule 6: the domain-colouring algorithm

Normative. The Rust reference and the WGSL must produce the same colour for the
same `w` and the same uniform values, to within `f32` rounding. At present the
WGSL in `shaders/domain_coloring.wgsl` (`fn domain_color`) is the only
implementation and there is no Rust counterpart — **that is a real gap**, and
rule 3 in [../AGENTS.md](../AGENTS.md) says a divergence here would be invisible.
Writing the Rust `domain_color` with unit tests that assert these exact values
is worth doing; until then, treat the pseudocode below as the specification that
the shader was written from.

`TAU = 2*pi` is a named constant in the shader. The contour strength `0.45` and
the line sharpness `24.0` are **inline literals in `domain_color`, not named
constants and not in `Uniforms`**. They are the only tunables in the colour
scheme that the user cannot reach, which is a known gap — see the open questions
in [TODO.md](../TODO.md). Below they are spelled `CONTOUR_DARK` and
`CONTOUR_SHARPNESS` so the pseudocode reads clearly; the *values* are what is
normative.

```
domain_color(w, phase, modulus_shading, modulus_density, phase_density) -> rgb:

    m     = |w|                 # overflow-safe: c_norm on the GPU
    angle = atan2(w.im, w.re)   # in [-pi, pi]

    # --- hue: the argument of the IMAGE, offset so negative reals read red ---
    turns = angle / TAU + 0.5          # -> [0, 1]
    hue   = fract(turns + phase)

    # --- brightness: the modulus of the IMAGE ---
    b     = m / (1.0 + m)              # 0 at the zero of f, -> 1, never exceeds 1
    value = pow(b, modulus_shading)    # shading = 0 gives value = 1: a flat hue map

    # --- contour lines: thin dark bands multiplied into the brightness ---
    dark = 1.0

    if modulus_density > 0.0:                      # iso-modulus
        band = log(max(m, EPS)) * modulus_density # even per e-fold, not near 0
        f    = abs(fract(band) - 0.5) * 2.0       # triangle wave, 0 at centre, 1 at edge
        dark = dark * (1.0 - CONTOUR_DARK * pow(f, CONTOUR_SHARPNESS))

    if phase_density > 0.0:                        # iso-phase; these are the rays
        band = turns * phase_density
        f    = abs(fract(band) - 0.5) * 2.0
        dark = dark * (1.0 - CONTOUR_DARK * pow(f, CONTOUR_SHARPNESS))

    return hsv2rgb(hue, saturation = 1.0, value * dark)
```

Points that matter and are easy to get wrong:

- **Hue comes from `w`, brightness comes from `w`, contours come from `w`.** All
  three are properties of the *image*, not the domain. The only thing drawn in
  domain coordinates is the grid, in `apply_grid`. Mixing the two up produces a
  plot that looks like a plot and means nothing.
- **The `+ 0.5` on `turns`** puts the positive real axis at hue 0.5 (cyan) and
  makes the negative real axis red. This is a convention, not a physical fact,
  and it is currently hardcoded in the shader and absent from `Uniforms`. That
  is an implicit ABI, which is exactly what rule 1 exists to prevent. Raised in
  [TODO.md](../TODO.md); not resolved.
- **`max(m, EPS)` before `log`.** `|w| = 0` is common (`z^2` at the origin,
  `sin` at every multiple of `pi`) and `log(0)` is `-inf`, and `fract(-inf)` is
  undefined. Without the guard a single pixel goes black for the wrong reason.
  This is the same constant as the `sinc` singularity guard.
- **`fract` of a negative.** `turns` is in `[0,1]` so the phase band is safe, but
  the triangle wave `abs(fract(x) - 0.5) * 2.0` relies on `fract` being
  floor-based, not C's truncating `fmod`. WGSL's `fract` is floor-based. On the
  CPU, `x - x.floor()` matches; `x % 1.0` does not.
- **`hsv2rgb` must `fract` its hue before the six-way sector switch.** A hue of
  exactly `1.0` has to wrap to sector 0 (red), not fall through and come out
  black. The shader's `hsv2rgb` is branch-free apart from the final switch,
  using the `p`/`q`/`t`/`f` form so there are no seams between sectors.
- **Non-finite `w` is handled before this function is called.** `main` checks
  `c_finite(w)` and writes solid black, because a non-finite number has no
  meaningful argument. That black *is* the escape-time set boundary in the
  iterated case.

### The grid overlay

`apply_grid` is a separate step, applied to the finished colour, and it is the
one thing that works in **domain** (`z`) coordinates:

```
per_pixel = 2 * max(abs(scale), EPS) / max(res.y, 1.0)   # world units per pixel
line_w    = GRID_PX * per_pixel                          # world-space line width

d_real = abs(z.im)                 # the real axis,  im == 0
d_imag = abs(z.re)                 # the imaginary axis, re == 0
d_unit = abs(|z| - 1)              # the unit circle

cover = max(1 - smoothstep(0, line_w, d_real),
            1 - smoothstep(0, line_w, d_imag),
            1 - smoothstep(0, line_w, d_unit))

color = mix(color, GRID_COLOR, GRID_BLEND * cover)
```

The width is computed in **world** units from `scale`, so a one-pixel line stays
one pixel wide at any zoom. Computing it in screen space instead makes the lines
either vanish or smear once the scale changes by orders of magnitude. `max()`
rather than a sum, so a crossing does not double-darken.

## Rule 7: how to add a 17th function

Work through this in order. The shader edit and the table edit are the same
commit, always.

1. **`src/functions.rs`** — add a constant:
   ```rust
   /// Function id: `cosh z`.
   const ID_COSH: u32 = 16;
   ```
   Next free id. Do not reuse one and do not shift the others.

2. **`src/functions.rs`** — append a row to `FUNCTIONS`. It must be the last
   element, and its `id` must equal its index. Give it a `name` that is unique
   across the table, a `formula` written as a human would, and a `group`.

3. **`src/functions.rs`** — bump `FUNCTION_COUNT` to 17 and the array type to
   `[FunctionEntry; 17]`, in the same edit.

4. **`src/functions.rs`** — add the arm to `apply_once` (or whatever the
   single-application function is called) and a matching arm to `eval`. A
   function that works in one and not the other is worse than one that works in
   neither, because the CPU tests will pass and the plot will be wrong.

5. **`src/complex.rs`** — add any new complex-arithmetic helper the function
   needs, in `f64`, with the same IEEE edge-case policy: `x / 0` yields an
   infinity, `NaN` flows through, no panics. Add tests.

6. **`shaders/domain_coloring.wgsl`** — add `case 16u:` to the switch in `fn
   apply`, dispatching to a `c_*` helper transcribed from the Rust. Any new
   complex helper goes alongside the existing `c_*` family. **Same commit as
   steps 1–5.**

7. **`shaders/domain_coloring.wgsl`** — if the new function needs a parameter
   (a Julia constant, an exponent) that is not already in `Uniforms`, **stop**.
   That is an ABI change: go back to [Rule 2](#rule-2-the-wgsl-side-must-change-in-the-same-commit)
   and grow the uniform block properly. Do not hardcode a parameter that the
   user will want to change, and do not quietly reuse an unrelated field.

8. **`src/panel.rs`** — add the entry to whatever list or menu the panel builds
   from `FUNCTIONS`. If it iterates `FUNCTIONS`, there is nothing to do; check
   before editing.

9. **`agents/ABI.md`** — add a row to the table in
   [Rule 3](#rule-3-the-function-id-table).

10. **`agents/TODO.md`** and **`agents/SESSION-LOG.md`** — note it. The session
    log entry is not optional; it is how the next session knows the id exists.

11. **`cargo test`** — the id-table tests and the `f64` function tests must
    pass. Then confirm the new id is actually reachable from the UI.

12. Commit. Include the `[ABI.md]` table row in the same commit. Rule 2 in
    [../AGENTS.md](../AGENTS.md) covers the uniform block; this rule covers the
    id table, and it is the same discipline.

## Related

- [ARCHITECTURE.md](ARCHITECTURE.md) — why this pipeline is shaped this way
- [../TODO.md](../TODO.md) — the open ABI questions: the `+0.5` hue offset, the
  contour constants, `frame`, the missing escape radius
- [../AGENTS.md](../AGENTS.md) — the rules this document enforces
