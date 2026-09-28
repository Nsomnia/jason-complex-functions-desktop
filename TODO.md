# TODO

Legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked

Status is a snapshot. Re-derive it with `cargo check --release` before
believing it — several lanes write these files concurrently.

**As of 2026-09-28 the crate is one error away from building**, and that error
is the missing `app::run()`. Everything else compiles.

---

## Milestone 1 — Working renderer core

**Goal:** a window opens, the complex plane is drawn with a function on it, and
the mouse pans and wheel-zooms it. This milestone is about proving the pipeline
works end to end, not about feature breadth.

### Done

- [x] **Uniform ABI.** `src/uniforms.rs`: the 64-byte `#[repr(C)]` block, its
      `bytemuck::Pod` derive, the compile-time size guards, the `Default`
      initial view, and five tests including
      `field_offsets_match_the_documented_wgsl_layout` which pins all thirteen
      field offsets. First commit `14a08d1`.
- [x] **`f64` complex arithmetic.** `src/complex.rs`: `Complex { re, im }`,
      the operator impls, `norm` / `norm_hypot` / `arg` / `conjugate` /
      `is_finite`, IEEE edge-case policy (no panic on `x / 0`; `NaN` and
      infinity propagate), and tests.
- [x] **Function table.** `src/functions.rs`: 16 entries with fixed ids 0–15,
      `FunctionEntry` / `FunctionGroup`, `eval` with iteration, `entry_for`,
      `label_for`, `ITERATION_CAP = 4096`, `SINC_SINGULARITY_GUARD_RADIUS`.
      Attached as `crate::complex::functions` through an explicit `#[path]`,
      because `src/main.rs` has no top-level `mod functions;`.
- [x] **WGSL compute kernel.** `shaders/domain_coloring.wgsl`: the `Uniforms`
      struct mirroring the Rust layout with offset comments, the `c_*` complex
      helpers, `apply` dispatching on `func_id`, `apply_selected` implementing
      the iteration loop, `c_finite`, `hsv2rgb`, `domain_color`, `apply_grid`,
      and `main` at `@workgroup_size(8, 8)`.
- [x] **Camera.** `src/camera.rs`: `f64` viewport state with `target_*` easing,
      `complex_at_pixel` / `pixel_at_complex` as exact inverses, drag and zoom
      expressed as sampling the mapping twice rather than re-deriving it,
      `DEFAULT_SCALE = 1.5`, `MIN_SCALE = 1e-9`, `MAX_SCALE = 1e9`, and the
      y-flip tripwire tests.
- [x] **Telemetry.** `src/telemetry.rs`: 128-entry `FrameHistory` ring buffer,
      `Telemetry` with `begin_frame` / `end_frame` / `record_cpu_time` /
      `record_gpu_time`, `TelemetrySnapshot`, instant and average FPS.
- [x] **Theme.** `src/theme.rs`: the `VOID` / `PANEL` / `SUNKEN` / `RAISED` /
      `HAIRLINE` palette, `TEXT` ramp, `CYAN` accent, the type scale, and the
      FPS thresholds (`FPS_GOOD` 55, `FPS_WARN` 30, `FPS_BAD` 24,
      `FRAME_BUDGET_MS` 16.67).
- [x] **`shaders/blit.wgsl`.** Full-screen triangle from `vertex_index` alone,
      no vertex and no index buffer; `input_texture` at
      `@group(0) @binding(0)`, `linear_sampler` at binding 1, and `fs_main`
      returns a single `textureSampleLevel` with no gamma and no tone curve. It
      makes no colour decisions at all, which is the point of the file.
- [x] **`src/renderer.rs`.** 1,328 lines. Compiles clean against wgpu 30: the
      adapter and device, the surface, the compute and blit pipelines and their
      bind groups, resize handling, the frame loop, and the timestamp-query
      readback.
- [x] **`src/panel.rs`.** 1,857 lines. Compiles clean against egui 0.36.

### In progress

Nothing. Every lane that had a file in flight has landed it.

### Todo

- [ ] **`src/app.rs` — the integration step, and the last thing standing.**
      It is the **only** remaining compile error in the crate: `src/main.rs:30`
      calls `app::run()` and `src/app.rs` is a one-line placeholder. Write it
      and the crate builds. Must provide `pub fn run() -> eframe::Result<()>`:
      - `eframe::NativeOptions` with the theme applied via
        `eframe::egui::Style` / the theme module.
      - An `eframe::App` struct owning `Camera`, `Telemetry`, the `egui_dock`
        `DockState`, the function selection, the panel parameter set, and an
        `Option<Renderer>` (the adapter may fail; the window should still open
        and say so).
      - `App::update`, calling `renderer.render(...)` when the view is dirty.
      - egui viewport: central panel draws nothing (wgpu presents through the
        window surface), left or bottom panel calls into `panel.rs`.
      - Input: drag to pan, wheel to zoom, routed into `Camera`'s
      `target_*` fields rather than the live ones, so easing works.

- [ ] **First successful frame.** The milestone's definition of done. A window
      opens, `main` in the kernel runs, and the identity map fills the viewport
      with a recognisable hue wheel that is *not* upside down. Take a
      screenshot to `target/` and confirm the orientation before celebrating —
      an unflipped y is the single easiest bug in this project to ship.

      **Scope: 15 of the 16 functions render correctly with `iterate` off**,
      which is the path the first frame should be judged on. The iterated path
      is the only one with a known content problem, and only for `julia`
      (id 15), whose orbit escapes in `f32`. Do not let that block the first
      frame: un-iterated `julia` draws a perfectly good phase portrait, because
      with `iterate = 0` `apply_selected` is a single `apply` and nothing
      overflows.

- [ ] **Verify the two cross-language tripwires by hand.** Once the kernel runs,
      sample a known point: `z = 1.0` with `func_id = 0` (identity) and
      `max_iter = 0` must produce `w = 1 + 0i`, so hue is fixed by
      `angle = 0` and value is `pow(0.5, modulus_shading)`. `z = i` with
      identity must be the same colour as `z = 1` rotated a quarter turn, not
      the same colour. If these do not hold, the uniform offsets are wrong.

- [ ] **Confirm the blit is not doing colour work.** Disable
      `domain_color` (return a constant) and check the blit still shows
      *something* the right shape. This keeps the "presentation is
      presentation" rule honest.

- [ ] **Escape handling for iterated plots.** The iterated path currently
      produces an all-black frame for `julia` (id 15). This is a design question
      about the specified algorithm, **not** a shader bug — the CPU reference
      reproduces it exactly, which is the point of rule 3. Full analysis, the
      measured numbers, and the candidate fix are in
      [Known open questions](#escape-handling-for-iterated-plots-the-biggest-open-question-in-milestone-1)
      and in [agents/ROADMAP.md](agents/ROADMAP.md) phase 1. Do not paper over
      it with a clamp; the fix is a real decision with a visual-tuning
      dependency, and it needs a rendered frame to choose the constants. Not
      blocking the first successful frame — see the note above.

- [ ] **Cross-compile / CI notes.** Record, in
      [agents/ENVIRONMENT.md](agents/ENVIRONMENT.md):
      - The exact command that builds the binary on macOS, from clean.
      - Whether `cargo test --release` passes with the window absent (it should
        — no test may require a GPU).
      - That cross-compilation to Windows was not attempted, and which of the two
        viable routes (`*-pc-windows-msvc` needs a Windows host; `*-gnu` needs
        mingw) is the intended one.
      - Note explicitly that there is **no CI configured**. Nothing validates
        this crate anywhere but this machine.

- [ ] **`git commit` the completed milestone 1 in pieces, not one commit.** The
      ABI, the maths, the shader, the renderer, the panel, and the app are
      independently reviewable. Rule 2 requires the ABI and shader edits to
      share a commit with each other; nothing else has to.

---

## Milestone 2 — Riemann surfaces and manifold exploration

This is the "radical expansion" the original brief asked for and the reason the
project is worth building. It is **not** solved. The technical content and the
candidate designs are in [agents/ROADMAP.md](agents/ROADMAP.md) phase 2; read
that before planning anything here.

- [ ] **Decide the sheet model before writing any code.** One mesh with vertex
      duplication at branch cuts, or N stacked mesh layers with a per-sheet
      index? The trade-offs are written up in the roadmap. The decision
      determines everything downstream, including whether the project needs a
      3D camera at all.
- [ ] **Decide whether this is a mesh project or a compute project.** The
      existing compute-then-blit pipeline can draw n sheets in one pass into an
      n-layer storage texture, reusing everything in milestone 1. It is much
      less capable than real geometry. Pick deliberately.
- [ ] **Extend the ABI for branches.** A branch selector `k` is not in
      `Uniforms`, and a vertex shader cannot read the compute kernel's bind group
      without either a second binding or a separate bind group layout. This is
      an ABI change and therefore needs the full treatment in
      [agents/ABI.md](agents/ABI.md).
- [ ] **Work out how a branch point is hit at all.** A uniform grid over the
      base domain essentially never places a vertex exactly on a branch point.
      Either an explicit cut, or an epsilon snap, or the branch points are
      simply absent from the mesh and you live with it.
- [ ] **Ship it, and keep the 2D plotter working.** The plain domain-colouring
      plot is the primary artefact. Riemann surfaces are an additional view, not
      a replacement.

---

## Milestone 3 — Windows packaging, and the performance work

- [ ] **A reproducible benchmark.** One command that reports ms/frame and the
      adapter name for each of the 16 functions at a few viewports. Adding
      `criterion` for this needs a rule-7 justification, or the harness has to
      be hand-rolled and then the risk is that it is wrong.
- [ ] **Dirty-rectangle rendering.** A static plot should cost zero GPU work.
      This is the single largest win available and it is boring; do it before
      anything clever.
- [ ] **Tile decomposition for escape-time regions.** Skip tiles that have
      already converged. Bigger win, much more work.
- [ ] **Supersampling / distance estimator** to kill the aliasing moiré near
      Julia-set boundaries. Undecided whether this is milestone 2 or 3.
- [ ] **GPU timestamp queries on Metal.** `renderer.rs` has the scaffolding;
      verify they actually resolve on Intel, and that they degrade gracefully
      on Apple Silicon where availability differs.
- [ ] **A Windows installer** that runs on a clean Windows 10 22H2 and a clean
      Windows 11. Nothing has been built or tested for Windows. Cross-compiling
      from macOS was never attempted.
- [ ] **A macOS `.app` bundle** that runs from the same artifact on Intel and
      Apple Silicon (universal binary, or two thin wrappers around one
      `wgpu`-backed binary).

---

## Stretch

- [ ] **Continuous zoom and pan.** Easing targets exist in `Camera` but nothing
      drives them yet. Exponential smoothing in log-scale feels dramatically
      better than linear and is a few lines.
- [ ] **Saved views and config persistence.** Serialise `Uniforms` to JSON on
      disk. Only safe once rule 4 (append-only ids) is genuinely respected,
      since a stored `func_id` is a permanent commitment to the numbering.
- [ ] **Function animation.** A uniform that shifts with time, e.g. the
      transcendental argument, to make phase contours sweep. Needs an ABI
      field and a decision about whether the compute pass runs every frame.
- [ ] **A CPU/GPU numerical cross-check harness.** Render the same scene with
      the `f64` reference and the `f32` kernel, compare, and report the max
      deviation. This is the single highest-value test in the project, because
      it is the only thing that would actually catch a rule-3 violation.
- [ ] **More functions.** The table is 16 by convention, not by mathematics.
      `cosh`, `tan`, `tanh`, `z^(1/3)`, `atan`, Möbius families are all easy
      additions and each one is a good exercise of the 17th-function checklist.

---

## Known open questions

Genuine uncertainties. Do not treat any of these as settled; if you settle one,
move it into a commit and delete it here.

**Milestone 1**

- Is `iterate` supposed to be a boolean-ish flag or a count? The uniform is
  `u32` and is currently treated as "zero or not zero, iterate `max_iter`
  times", which makes intermediate values unreachable from the UI. A count
  would let you show, say, `f^5(z)` for a real rational map. Undecided.
- `phase` rotates the hue. The WGSL adds a fixed `+0.5` so negative reals come
  out red. Is that offset a permanent property of the colour scheme, or should
  it be a field? It is currently hardcoded in the shader and absent from
  `Uniforms`, which is exactly the kind of implicit ABI that rule 1 exists to
  prevent.
- Contour darkness (0.45) and the `pow(f, 24)` line profile are magic constants
  baked into `domain_color`. They are duplicated nowhere, but they are also not
  user-tunable, and they should probably be fields.
- `frame` is in the uniform for "cheap deterministic dithering" but the kernel
  does not use it. Either the plan was dropped or the implementation is
  missing.
- Should the blit do gamma correction? The compute pass writes linear-ish values
  to an `rgba8unorm` (not `-srgb`) texture and the comment in the shader claims
  the bytes are what the display shows. On most surfaces that is wrong, and on
  a Retina display it is visibly wrong. Undecided and worth an experiment.
- Multi-monitor and DPI: nothing has been tested with a non-1.0 scale factor.
  The `resolution` the kernel sees is the surface size in physical pixels, but
  egui input is in logical points, so the pan/zoom deltas need scaling by the
  ratio. This is very likely a bug that has not been found yet.

### Escape handling for iterated plots (the biggest open question in milestone 1)

**This is not a bug. Do not "fix" it with a clamp, and do not describe it as one.**
It is a property of the algorithm as specified, the CPU reference reproduces it
exactly, and the fix is a design decision with a visual-tuning dependency. It
gets its own section because it is the one question in milestone 1 that cannot
be settled without rendering something and looking at it.

**The symptom.** With `iterate` on and `func_id = 15` (`julia`,
`c = -0.7269 + 0.1889i`), the frame is black. Measured on this machine: the
critical orbit of 0 escapes at iteration 120; at the default `max_iter = 256`
about **92% of a 3x3 viewport** comes out non-finite and therefore black; at
`max_iter = 4096` the frame is entirely black.

**Why.** The `julia` function's critical orbit is chaotic but bounded in exact
arithmetic — that is what makes the Julia set interesting. `f32` rounding does
not respect the bound, and pushes the orbit off the fractal. There is also a
second, slower-growing reason: the loop has no bailout, so a *slowly* escaping
orbit burns all 4096 iterations before anything notices.

**The real half of the problem, which is easy to miss.** The obvious story is
"overflow goes non-finite, non-finite draws black". True, and it is the smaller
half. The colour scheme does something worse first:

```
value = pow(m / (1 + m), modulus_shading)      // modulus_shading = 1.0 by default
```

At `|w| = 1e4` the value is already `0.9999` — visually white. From about
`|w| = 3.4e7` it is **exactly** `1.0` in `f32`, i.e. pure white, and it stays
there for the next twelve orders of magnitude. Only at `|w| ~ 1.8e19`, where
`f32` overflows and `c_finite` finally goes false, does the pixel snap to black.

So a diverging orbit passes through **white and then snaps to black**. That
white-to-black discontinuity is an `f32` artifact, not the intended design. The
intended reading of the scheme is that a large modulus means bright, and
"bright" saturating into "black" contradicts it. A plot that went smoothly to
black at the escape boundary would be a defensible design choice; a plot that
goes white, sits in white for twelve decades, and then snaps to black is
neither. **Judge the fix on the second half of this, not the first.**

**The candidate fix — renormalisation, undecided.** Standard practice, and cheap:
in the iteration loop, once `|w|` exceeds a threshold `T` (`1e4` is reasonable),
divide `w` by `T` and accumulate `ln(T)` into a running log-modulus. Two things
fall out of this:

- `|w|` stays bounded inside `f32`, so it **can never overflow**, and the
  `c_finite` black never fires.
- Because the divisor is a *positive real*, the argument of `w` — and therefore
  the hue — is untouched. Only the modulus is rescaled, and the accumulated
  log-modulus recovers exactly the information the rescaling destroyed.

The accumulated log-modulus is monotonically related to escape time for an
exponentially diverging orbit, which makes it a usable stand-in for one. Feed
it into the shading as an extra attenuation term:

```
value *= 1 / (1 + log_modulus * escape_scale)
```

so the filled set stays bright and the complement recedes smoothly to black
instead of through white. Note that this is a **deliberate inversion of the
classical Julia convention**, where the complement is bright and the set is
black. It is chosen to match this app's existing behaviour at the zeros of a
function, where the origin is already drawn black. That is a judgement call and
a user will disagree with it eventually; it should probably be a toggle, and
making it a toggle means an ABI field.

**It needs no ABI change.** The accumulator is a kernel-local variable, not a
uniform. `Uniforms` stays at 64 bytes. This is worth stating explicitly because
the shader's own `KNOWN LIMITATION` comment observes that colouring by escape
time "changes both the algorithm and the ABI" — that is true of the *classical*
escape-time fix, which needs a uniform to carry the escape threshold and the
count. Renormalisation needs neither. Do not conflate the two fixes when
deciding.

**What it needs that is not available yet: a rendered frame.** The exact look is
a tuning problem. `T` and especially `escape_scale` are visual judgements —
`escape_scale` too small and the whole viewport is black; too large and the
escape boundary is invisible and everything is white. **A future session must
not pick these blind.** Render, look, adjust. This is the explicit dependency
that makes the item not-blocked-by-accident: it is checkable in one minute of
work once the first frame is up, and it is not checkable at all until then.

**Scope.** Not blocking the first successful frame. 15 of the 16 functions are
correct with `iterate` off, and un-iterated `julia` is a perfectly good phase
portrait, because with `iterate = 0` `apply_selected` is a single `apply` and
nothing iterates and nothing overflows. Land the first frame first.

**Not a rule-3 violation.** The CPU reference in `src/functions.rs` does the
same thing, which is exactly the agreement rule 3 asks for. The two sides agree
on a bad answer, which is a different problem from the two sides disagreeing.

**One structural asymmetry to watch while fixing it.** The two iteration loops
are not literally the same code. The shader's `apply_selected` breaks out as
soon as `c_finite(w)` goes false; `functions::eval` runs all `steps`
iterations with no early exit. For a polynomial the two still agree on the
observable outcome — once a value is non-finite, applying a polynomial to it
leaves it non-finite, so both end black — but for any function that can map a
non-finite input back to a finite one, `1/z` being the obvious candidate, they
can part company. The two loops are already close enough to being the same
transcription that this is easy to miss and hard to debug, so if you are in this
area anyway, make them match deliberately rather than by accident.

**Milestone 2, the hard part**

- How do you tessellate a Riemann surface in wgpu? A surface is not a function
  graph; it is a 2-manifold. There is no general answer that avoids explicit
  mesh construction.
- Do the sheets go in one mesh with duplicated vertices at the branch cut, or in
  N layers? See the roadmap; both are viable and they are not equivalent in
  capability.
- Multiple render passes per sheet, or a single instanced draw with one
  instance per sheet? Instancing is the obvious default because it needs no
  per-sheet index buffers, but it constrains what a per-sheet uniform can be.
- How do you sample a multi-valued function for surface plotting? You need a
  branch index `k` at the vertex, and the answer has to be a root-finder
  (Newton on `w^n - z = 0`), not the principal branch. The Newton iteration
  itself needs a bailout and a "which root did I converge to" tie-break, and
  that tie-break is where the branch points hide.
- Where does the third dimension come from? A sheet is a mesh over a 2D base
  domain, so the surface is 2D embedded in something. Options: extrude the
  sheets vertically in screen space, or build a real 3D scene. These are
  different products.
- Is a branch cut a rendering artefact or part of the model? For `sqrt` the
  cut along the negative real axis is conventional, but it is a choice, and
  the surface does not actually have an edge there. Rotating the cut by hand to
  watch the picture change would be a genuinely nice way to teach the point.
- How do you light a surface? There is no lighting, no depth buffer, and no
  normal computation anywhere in the project today. A mesh is useless without
  at least per-vertex normals, and normals across a branch point are the
  singular case.

**Cross-cutting**

- Is a `f32` kernel numerically adequate? At `max_iter = 4096` on Apple
  Silicon, `f32` reassociation rules are stricter than on the Intel GPU used
  for development, so a plot that looks right on the dev machine may show
  banding on a Mac. Untested and untestable without the hardware.
- Once the escape handling above is decided, does `f32` renormalisation also buy
  a much higher usable `max_iter`? It is the same technique, but that is a
  secondary benefit and the constants will already have been tuned for the
  first one. Fold it into the same decision rather than reopening it.
