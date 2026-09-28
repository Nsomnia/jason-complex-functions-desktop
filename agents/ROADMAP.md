# ROADMAP

The multi-phase plan for STEEL-PULSE v2.0-TURBO. Current position: **end of
phase 1**. For what is being worked on right now see [../TODO.md](../TODO.md);
for what has happened see [SESSION-LOG.md](SESSION-LOG.md).

Each phase below has a goal, an approach, the open problems, and a definition of
done that is specific enough to be checked.

---

## Phase 1 — Renderer core (current)

### Goal

A window opens, the complex plane is drawn with a function on it, and the mouse
pans and wheel-zooms it. Proving the pipeline works end to end, not feature
breadth. Sixteen functions that already exist, a compute kernel that already
exists, and an application shell that does not.

### Approach

Compute-then-blit. See [ARCHITECTURE.md](ARCHITECTURE.md) for the shape and the
reasoning; the short version is that a compute shader gives per-pixel work
without quad-wide agreement, and its output is an ordinary texture that later
phases can accumulate into rather than throw away.

- `Uniforms`, the 64-byte block, is done and tested. See [ABI.md](ABI.md).
- The `f64` reference (`Complex`, the 16-function table) is done and tested.
- The WGSL transcription of both is done.
- The camera, telemetry, and theme modules are done.
- `shaders/blit.wgsl`, `src/renderer.rs` and `src/panel.rs` are done and
  compile.
- Remaining: `src/app.rs`, and the first frame. That is all.

### Open problems

- The crate does not compile. As of the latest check there is **exactly one**
  error, and it is the whole of what remains in phase 1: `src/main.rs:30` calls
  `app::run()` and `src/app.rs` does not define it. The renderer, the panel,
  and both shaders all compile.
- Nothing has ever been run. There is no evidence yet that the y-flip is right,
  that the uniform offsets survive the round trip, or that the blit is
  colour-neutral. All three are cheap to check and all three are the kind of
  thing that is much harder to check once there is more on screen.
- A failed adapter request has no defined behaviour yet. It should not be a
  panic: the window should open and say so.
- **The iterated path draws nothing useful for `julia`.** Not a defect to be
  patched and not blocking the first frame; it is a design question that cannot
  be settled without a rendered picture to look at. See
  [escape handling for iterated plots](#escape-handling-for-iterated-plots) below.

### Escape handling for iterated plots

The one phase-1 item that is a genuine decision rather than a task, so it is
written out here rather than left as a line in the task list.

**The situation.** There is no escape-radius bailout anywhere in the renderer.
The iteration loop runs `max_iter` times and detects divergence only afterwards,
with the `c_finite` check in `main`, which does not fire until `f32` has
overflowed at about `|w| = 1.8e19`. A classical escape-time plot would bail at
`|w| > 2` and colour by iteration count; this one colours the final iterate.

For most functions that is fine. For `julia` (id 15, `c = -0.7269 + 0.1889i`) it
is not: the critical orbit is chaotic but bounded in exact arithmetic — that is
what makes the Julia set interesting — and `f32` rounding does not respect the
bound. Measured on this machine: the orbit of 0 escapes at iteration 120; at the
default `max_iter = 256` about 92% of a 3x3 viewport comes out non-finite and
therefore black; at `max_iter = 4096` the whole frame is black.

**The subtler half, which is the one that matters.** The above is the obvious
story and it is the smaller half. The colour scheme does something worse first.
With `value = pow(m / (1 + m), modulus_shading)` and the default
`modulus_shading = 1.0`:

| `\|w\|` | value | appearance |
|---:|---:|---|
| 1 | 0.500 | mid brightness |
| 1e2 | 0.990 | bright |
| 1e4 | 0.9999 | visually white |
| ~3.4e7 | **exactly 1.0 in `f32`** | pure white |
| 1.8e19 | `f32` overflows, `c_finite` false | snaps to black |

So a diverging orbit spends twelve orders of magnitude in **pure white** and
then snaps to **black**. That discontinuity is an `f32` artifact, not the
intended design: the scheme's own convention is that a large modulus means
bright, and bright saturating into black contradicts it. A plot that receded
smoothly to black at the escape boundary would be defensible. A plot that goes
white, sits in white for twelve decades, then snaps to black, is neither. **Any
proposed fix should be judged on the white-to-black half, not the overflow
half.**

**This is not a bug, and specifically not a rule-3 violation.** The CPU
reference in `src/functions.rs` does the same thing, which is precisely what the
CPU/GPU agreement rule asks for. Two sides agreeing on a bad answer is a
different problem from two sides disagreeing, and it is the far more tractable
one. Do not clamp it, do not raise it as a shader defect, and do not "fix" one
side to make the plot prettier.

**Two candidate fixes, and they are not equivalent.**

*The classical one — colour by escape time.* Bail at `|w| > T`, accumulate the
iteration count, shade by the count. This is what most fractal software does. It
needs `T` and the count in the uniform block, so it is a full ABI change with
the whole procedure in [ABI.md](ABI.md), and it inverts this app's current
convention (classical Julia rendering is bright-outside, black-inside; this app
draws the origin black and brightens with modulus).

*Renormalisation, undecided.* Once `|w|` exceeds `T` (`1e4` is reasonable),
divide `w` by `T` and accumulate `ln(T)` into a running log-modulus. Three
consequences, and they are all good ones:

1. `|w|` stays bounded, so it can never overflow and the `c_finite` black never
   fires.
2. Because the divisor is a **positive real**, the argument of `w` — and
   therefore the hue — is untouched. Only the modulus is rescaled.
3. The accumulated log-modulus recovers exactly the information the rescaling
   destroyed, and it is monotonically related to escape time for an
   exponentially diverging orbit, so it is a usable stand-in for one.

Feed it into the shading as an extra attenuation, `value *= 1 / (1 +
log_modulus * escape_scale)`, and the filled set stays bright while the
complement recedes smoothly to black instead of through white.

**Renormalisation needs no ABI change at all.** The accumulator is a kernel-local
variable, not a uniform; `Uniforms` stays 64 bytes. This is worth stating
because the shader's own `KNOWN LIMITATION` comment concludes that fixing this
"changes both the algorithm and the ABI" — that is true of the *classical*
escape-time fix, which must carry the threshold and the count in a uniform. Do
not conflate the two when deciding.

**What is missing, and it is not a small thing: a rendered frame.** `T` and above
all `escape_scale` are visual judgements, not derivable constants. Too small an
`escape_scale` and the entire viewport is black; too large and the escape
boundary is invisible and everything is white. **A future session must not pick
these blind**, because either wrong value ships a plot that looks broken for a
reason unrelated to the code. This is the explicit dependency that keeps the
item honestly not-blocked-by-accident: it is checkable in one minute once the
first frame is up, and it is not checkable at all until then.

**Scope: not blocking the first successful frame.** 15 of the 16 functions are
correct with `iterate` off, and un-iterated `julia` draws a perfectly good
phase portrait, because with `iterate = 0` `apply_selected` is a single `apply`
— nothing iterates and nothing overflows. Land the first frame first; settle
this second, with a picture on screen.

### Done means

- `cargo build --release` succeeds and `cargo run --release` opens a window.
- The identity map fills the viewport with a recognisable hue wheel.
- **The picture has been screenshotted and the orientation confirmed by eye.**
  An unflipped y is the one bug that produces a plausible-looking wrong answer.
- `z = 1.0` with `func_id = 0` and `iterate = 0` gives `w = 1 + 0i`, so hue is
  fixed and value is `pow(0.5, modulus_shading)`. If not, the offsets are wrong.
- `z = i` under identity is `z = 1` rotated a quarter turn, not the same colour.
- Disabling `domain_color` still shows the right *shape* through the blit. This
  is what keeps "presentation is presentation" honest.
- Drag pans, wheel zooms, and both go to the `target_*` fields so the easing is
  visible.
- 15 of the 16 functions render correctly with `iterate` off. That is the bar
  for "working renderer core"; escape handling for the iterated path is a
  separate decision, tracked in
  [../TODO.md](../TODO.md) and in
  [escape handling above](#escape-handling-for-iterated-plots).
- `cargo test` passes with no GPU and no window.
- Milestone 1 is committed in reviewable pieces, per the list in
  [../TODO.md](../TODO.md).

---

## Phase 2 — Riemann surfaces

This is the "radical expansion" the original brief asked for. It is the part of
the project that is not solved, and the part that makes it worth doing. Nothing
below is decided.

### The problem, stated properly

A single-valued holomorphic function `f: C -> C` is something you can draw: pick
a point in the domain, evaluate, colour. The plotter does exactly that, and it
is the whole of phase 1.

A **multi-valued** function does not work that way. `sqrt(z)` has two values at
almost every point of the domain. `z^(1/3)` has three. `log z` has infinitely
many, differing by multiples of `2*pi*i`. There is no single-valued function
here to plot, so a plot of `sqrt` must be a plot of *something else* — and the
mathematically correct something is the graph of the function, which turns out
not to be a subset of `C x C` at all.

The graph of a multi-valued holomorphic function is a **Riemann surface**: a
two-dimensional manifold that locally looks like the complex plane everywhere
except at **branch points**, and folds over the plane as you travel. It is
locally a disk *including* at the branch points — that is the theorem — but
`w -> w^n` for `n > 1` is the local model there, so `n` sheets come together
and meet. The surface is a perfectly respectable object and has an edge
nowhere; it merely cannot be drawn without introducing one, because a flat
picture has edges.

**The branch cut is the fake edge.** You draw a curve from each branch point to
infinity, forbid the surface from crossing it, and the function becomes
single-valued on the cut domain. For `sqrt` the conventional cut is the negative
real axis. It is a *choice*: the surface has no edge there, and rotating the cut
by hand to watch the picture change is a genuinely good way to teach the point.
Which is an argument for making the cut a user-visible control, and an argument
that a design which hardcodes one cut is a design that will need rewriting.

Three things follow, and they are the actual engineering content of this phase:

1. Each **sheet** is a mesh over a two-dimensional base domain, so the surface
   is 2D embedded in something and you have to decide what.
2. The sheets have to be **joined** at the branch points, and a naive grid never
   lands a vertex on a branch point.
3. Evaluating branch `k` of an `n`-valued function at a point is a **root
   finding** problem, not a library call. `sqrt` has a closed form per branch,
   but the general answer is Newton iteration on `w^n - z = 0`, and Newton
   needs a bailout and a tie-break for "which root did I converge to". The
   tie-break is where the branch points hide.

### The constraint nobody has costed yet

There is no 3D in this project. No vertex buffers, no index buffers, no depth
buffer, no matrices, no perspective camera, no normal computation, no lighting.
Everything written so far is a two-dimensional pixel-to-complex mapping. Phase 2
as a mesh project adds all of that at once, on top of the hardest problem in
complex analysis. That is the main reason the "cheapest credible path" below is
worth writing down.

### Candidate approach A — N separate mesh layers, per-sheet index

Tessellate the base domain once per sheet: a `G x G` grid over the visible
viewport, duplicated `n` times, once for each branch `k`. For each vertex of
layer `k`, evaluate branch `k` of the function. Draw the `n` layers — as `n`
draw calls, or as one instanced draw with `n` instances, one per sheet.

- **Cost.** `n` times the vertex work, which is trivial. One instanced draw, so
  one pipeline, one bind group per sheet, no per-sheet index buffers. Fully
  parallel; no cross-sheet coherence is required, and no ordering.
- **Benefit.** Cut-agnostic. The same code draws `sqrt`, `z^(1/3)`, and `log`
  (modulo an `n` that is not finite) without knowing anything about cuts. Each
  sheet is an independent grid you can colour, clip, or discard triangles from
  independently — so you can start at one sheet and add more.
- **Cost of the cost.** You get *stacked* sheets, not a *joined* surface. Branch
  points are only implied by sheets converging on a common point, which they do
  not actually share, so they will be slightly wrong and there is no way to shade
  continuously across one. Sheets that nearly touch will z-fight, and z-fighting
  needs a depth buffer and a sorting policy, which is exactly the machinery this
  approach was chosen to avoid needing. If the third dimension is a screen-space
  vertical offset, you get the "pages of a flipbook" picture, which is legible
  and honest and is not a Riemann surface.

### Candidate approach B — one mesh, vertex duplication at the branch cut

Cut the base domain along a branch cut, split the grid into the pieces either
side of it, and tessellate the cut domain for all `n` sheets. Along the cut,
weld the boundary: the lower edge of sheet `k` on one side is the same surface
point as the upper edge of sheet `k` on the other, and the two must share
vertices. The result is a single connected mesh whose only seam is the cut.

- **Benefit.** Correct connectivity. The surface is one manifold, it can be
  shaded smoothly across the seam, normals can be averaged across it, and a
  real lighting model becomes meaningful.
- **Cost.** A real index-buffer builder that tracks which boundary vertices are
  glued. This is where the bugs live: a weld tolerance that is too loose welds
  distinct vertices and creates a non-manifold, too tight leaves a visible
  crack, and welding a vertex to itself produces a mesh that cannot be
  triangulated. It is a real piece of work with a real bug surface and no
  tests to lean on, and it is the part of this phase most likely to consume a
  week.
- **Cost of the cost.** Cut-specific. A new multi-valued function means a new
  cut, a new decomposition of the base domain, and a new weld. Approach A is
  cut-agnostic and this is not. The cut also becomes a piece of state the user
  can see and would reasonably want to move.

### Candidate approach C — n sheets in one compute pass, n-layer storage texture

Do it in the compute kernel. One invocation per pixel per sheet, writing to layer
`k` of an `n`-layer `rgba8unorm` storage texture; the blit resolves. This
reuses **everything** from phase 1: the same pipeline, the same blit, the same
storage texture, only a different format and a dispatch over `n` times the
pixels.

- **Benefit.** By far the cheapest of the three, and it is the only one that
  ships without a 3D pipeline. It is also the only one that keeps the
  compute-then-blit architecture intact rather than adding a second, parallel one.
- **Cost.** It is not geometry. There is no rotation except by moving the camera
  plane, no lighting, no depth reasoning beyond whatever you hand-roll in the
  blit, and the sheets are `n` images rather than `n` pieces of a surface. But it
  will show the user what a branch cut is and what a branch point does, which
  is most of what the phase is for.

### How to choose

Not on capability — on capability, B wins and it is not close. On what can be
finished and verified. C is the right first step because it is a few hundred
lines on top of code that already works, it can be verified by looking at it, and
it will tell you whether the interesting part of the feature is the surface
topology or the branch selection. If C turns out to be uninformative — if
stacked sheets do not teach anything that the plain plot does not — then the
value was in the topology after all, and that is the argument for paying for B.
That is a cheap experiment that resolves a question worth weeks.

### Open problems

- **Sheet model, or mesh, or image.** Undecided. See above.
- **Multiple render passes per sheet, or one instanced draw with one instance
  per sheet?** Instancing is the obvious default: it needs no per-sheet index
  buffers and one pipeline. What it constrains is the per-sheet data — an
  instance index is the only per-sheet input, and it has to reach a *vertex*
  shader, which is a different pipeline from the compute kernel. Undecided.
- **Extending the ABI for the branch index `k`.** It is not in `Uniforms`, and
  it will need to be, because a vertex shader cannot read the compute kernel's
  bind group without either a second binding or a separate bind group layout.
  That is an ABI change, and it gets the full treatment in [ABI.md](ABI.md):
  both sides, the offset test, the size guards, the same commit. Do not
  special-case it.
- **How a branch point gets hit at all.** A uniform grid almost never places a
  vertex exactly on a branch point. Either an explicit cut (approach B), or an
  epsilon snap, or the branch points are simply absent from the mesh and the
  surface has small holes exactly at the most interesting points. Undecided.
- **Root finding per vertex.** Newton on `w^n - z = 0` needs a bailout and a
  "which root did I converge to" tie-break, and `f^n` has bad basins.
  Undecided whether the tie-break is a per-vertex choice (fragile, slow) or a
  propagated one (cheap, has seams).
- **The third dimension.** Where does it come from, and is it a screen-space
  offset or a real 3D scene? These are different products and choosing wrongly
  is expensive. Undecided, and it is the decision that most constrains the rest.
- **Lighting.** No normals exist anywhere in the project. A mesh without
  per-vertex normals cannot be lit, and normals *across a branch point* are the
  singular case — the surface is locally `w -> w^n` there, so the normal
  direction is not well defined by averaging faces. Undecided.
- **Is a branch cut a rendering artefact or part of the model?** For `sqrt` the
  negative real axis is a convention, not a fact. A feature that lets the user
  rotate the cut and watch the picture change would be a good one. Undecided
  whether that is scope for this phase or the next.

### Done means

- A single multi-valued function — `sqrt` — is navigable across all of its
  sheets, and the branch point is visible and explicable.
- The user can move the branch cut and see the picture change in a way they can
  explain. If this is not achievable, the feature has taught nothing.
- The plain 2D plotter is unchanged and still works. It is the primary
  artefact; Riemann surfaces are an additional view, not a replacement.
- The ABI changes went through [ABI.md](ABI.md) with both sides in one commit
  and the offset tests updated.
- A CPU reference for the branch evaluation exists in `f64` and is tested, so
  the vertex shader has something to be transcribed from.

### Explicitly not in this phase

Anti-aliasing. Escape-time plots alias badly near the set boundary — the
boundary is a fractal curve and a one-sample-per-pixel grid cannot resolve it —
and the fix (supersampling, or a distance estimator) is a well-understood
independent piece of work that shares nothing with surface topology. It is
milestone 3. Whether it is milestone 2 or 3 is undecided and it does not
matter, because it can be done either way.

---

## Phase 3 — Performance, and shipping it

### Goal

Hold 60 fps at Retina resolution on integrated graphics, and have the thing
installable on a machine that has never seen it. The plotter has to be
responsive to a drag to feel alive; a correct answer that arrives a second late
is a worse product than a slightly wrong one that keeps up.

### Approach

**Performance, in descending order of value per unit of difficulty:**

1. **Dirty-rectangle rendering.** A static plot should cost zero GPU work. The
   plot only changes when the camera or a uniform changes, and both are known on
   the CPU. This is by far the largest win available and it is completely
   boring; do it before anything clever, because it may make the clever things
   unnecessary.
2. **A reproducible benchmark.** One command that reports ms/frame and the
   adapter name for each of the 16 functions at a few viewport sizes. Adding
   `criterion` needs a justification under rule 7 in [../AGENTS.md](../AGENTS.md)
   and a hand-rolled alternative has its own correctness risk; this has to be
   decided deliberately rather than drifted into.
3. **Tile decomposition for escape-time regions.** Skip a tile whose orbit has
   already converged. Bigger win, much more work, and it interacts with (1): a
   tile cache is just a persistent dirty-rect scheme.
4. **Supersampling or a distance estimator** for the Julia-set boundary. Well
   understood, unexciting, and it is the difference between a plot that looks
   professional and one that looks noisy.
5. **GPU timestamp queries on Metal.** `src/telemetry.rs` already has the CPU
   ring buffer and `src/renderer.rs` has scaffolding for query sets. Verify they
   resolve on Intel and degrade gracefully on Apple Silicon, where availability
   differs and reassociation is stricter.

**Distribution:**

- A **macOS `.app` bundle** that runs from the same artifact on Intel and Apple
  Silicon. A universal binary if that works; otherwise two thin wrappers around
  one `wgpu`-backed binary.
- A **Windows installer** that runs on a clean Windows 10 22H2 and a clean
  Windows 11.

### Open problems

- **There is no CI.** Nothing in this repository is validated anywhere but one
  macOS x86_64 machine. The ABI offset tests, the `f64` reference tests, and
  `cargo fmt` are all candidates for a cheap CI job that would catch the exact
  class of bug rule 1 in [../AGENTS.md](../AGENTS.md) exists to prevent. The
  obstacle is the 21-minute cold dependency build, not the difficulty.
- **Windows packaging is unresolved and cross-compilation was not attempted.**
  Two routes exist. `x86_64-pc-windows-msvc` needs a Windows host, because the
  MSVC ABI cannot be cross-linked from macOS. `x86_64-pc-windows-gnu` can
  cross-compile from macOS with mingw-w64, and wgpu will build, but nothing
  about the result has been tested and a `wgpu`+Metal-porting layer on a
  cross-compiled toolchain is a real risk. Which route is intended is undecided
  and should be, before the first line of installer work.
- **`f32` adequacy on Apple Silicon.** Untestable from here. Apple GPUs are
  stricter about reassociation than the Intel one this was developed on, so a
  plot that looks right here may band there. Needs someone with the hardware.
- **f32 renormalisation** would allow far higher `max_iter`, at the cost of
  changing the colour scheme, because the accumulated modulus carries the
  information the shading depends on. This is the same technique as
  [escape handling for iterated plots](#escape-handling-for-iterated-plots) and
  it should be the *same decision*: settle that one, and this falls out as a
  secondary benefit rather than a reopening.

### Done means

- One documented command produces a benchmark table: adapter name, ms/frame,
  across all 16 functions and at least three viewport sizes, committed to
  `target/` and summarised in this file.
- A static plot with no input and no parameter change issues no GPU work.
  Measurable, not asserted.
- A `.app` bundle launches on Intel and on Apple Silicon.
- A Windows installer runs on a clean Windows 10 22H2 and a clean Windows 11,
  from a build produced by a command written down in
  [ENVIRONMENT.md](ENVIRONMENT.md).
- A CI job exists and is green, running the tests that do not need a GPU.
