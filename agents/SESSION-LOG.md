# SESSION LOG

Newest first. One entry per working session, appended, never rewritten. This
file is the project's memory: the original mission brief was truncated and lost,
so what a session decided and why is recoverable only from here and from the
commits.

Rules for an entry:

- One entry per session. The heading is `## <n>. <short title> — YYYY-MM-DD`.
- Under the heading, in this order and with these headings: **Situation**,
  **Attempted**, **Worked**, **Did not work**, **Decisions**, **Next**.
- Record *why*, not just *what*. A future session reading a list of changes can
  get those from the git log; it cannot get the reasoning.
- Record the state of the build honestly. "Does not compile, 11 errors, listed
  below" is far more useful than "in progress".
- If a decision was made and later reversed, the reversal gets its own entry
  and the old one stays. Do not tidy history here.
- One line per fact. This is a log, not a document.

Template at the bottom of this file.

---

## 1. Milestone 1 kickoff — 2026-09-28

### Situation

The working directory `/Users/derekvanee/Documents/jason-complex-functions-desktop`
was empty apart from `.git`. The mission brief for the project was truncated and
lost before it was read, so the intent had to be reconstructed from the project
name and the domain: STEEL-PULSE v2.0-TURBO, a hardware-accelerated native
desktop recreation and expansion of Samuel J. Li's WebGL Complex Function Plotter
(<https://samuelj.li/complex-function-plotter/>), with Riemann-surface
exploration as the intended expansion. Rust 2024 (not 2021, as originally
assumed) on wgpu 30, egui/eframe 0.36, egui_dock 0.21.

Machine: x86_64 macOS 14.8.8, Intel UHD Graphics 617, Metal 3, Retina 2560x1600
at scale factor 2.0. Development machine only; Apple Silicon is a target.

Work was parallelised across several lanes, each owning one file. This creates
the standing hazard that `git status` is permanently dirty and that
`cargo check` failures belong to a lane that is still writing.

### Attempted

Build the project skeleton and get a first frame out.

### Worked

- **`Cargo.toml` and `Cargo.lock`.** Five direct dependencies, 423 crates locked
  in total. `Cargo.lock` is committed deliberately: the dependency set is part
  of the reproducibility story and the resolution was a non-trivial decision.
- **Toolchain: found and worked around the Homebrew breakage.** The first
  `rustc` on the default `PATH` is a Homebrew Rust at `/usr/local/bin/rustc`
  that fails to load:

  ```
  dyld: Symbol not found: __ZN4llvm17OptimizationLevel2O0E
    Referenced from: /usr/local/Cellar/rust/1.97.1/lib/librustc_driver-*.dylib
    Expected in:     /usr/local/Cellar/llvm/23.1.2/lib/libLLVM.23.1.dylib
  ```

  A dylib linkage failure between `librustc_driver` and Homebrew's LLVM, not a
  project problem. It could not be fixed: a `brew.rb install rustup` process was
  found stuck in `S` state, sleeping, holding nothing and accomplishing nothing,
  running since 00:30. **Rule established: never run `brew` on this machine, and
  never run `/usr/local/bin/cargo`.** The working toolchain is rustup's stable
  at `~/.rustup/toolchains/stable-x86_64-apple-darwin/`, already symlinked into
  `~/.cargo/bin/`, so every command carries `PATH="$HOME/.cargo/bin:$PATH"`.
  Full details in [ENVIRONMENT.md](ENVIRONMENT.md).
- **`src/main.rs`** — the module list, the module-map doc comment, and
  `fn main() -> eframe::Result<()> { app::run() }`. It establishes the shape
  early: `app::run()` does not exist yet and its absence is a hard compile
  error, which is deliberate. Nothing works until the integration step lands.
- **`src/uniforms.rs` — the ABI, complete and tested.** 64 bytes, `#[repr(C)]`,
  `bytemuck::Pod` and `Zeroable`, thirteen fields, 64-byte `UNIFORM_BUFFER_SIZE`.
  The important part is not the struct, it is the defence: five tests, of which
  `field_offsets_match_the_documented_wgsl_layout` asserts **each of the
  thirteen field offsets individually**, plus two `const _: () = assert!` guards
  on `size_of` (a multiple of 16, and exactly 64). The reason for such a
  defensive posture is written up in [ABI.md](ABI.md) and abbreviated to a rule
  in [../AGENTS.md](../AGENTS.md): the WGSL struct is a separate text artifact
  that no compiler cross-checks, so a one-field shift does not fail, it silently
  mis-colours the entire plot.
- **The documentation set.** `AGENTS.md`, `TODO.md`, and this file plus
  [ARCHITECTURE.md](ARCHITECTURE.md), [ABI.md](ABI.md), [ROADMAP.md](ROADMAP.md),
  [GLOSSARY.md](GLOSSARY.md), [ENVIRONMENT.md](ENVIRONMENT.md). Written before
  the code was finished, on the theory that a future session with no memory of
  the intent is better served by a decision record than by a changelog.

First commit, `14a08d1` "Scaffold: uniform ABI contract, module map, dependency
lock" — `Cargo.toml`, `Cargo.lock`, `.gitignore`, `src/main.rs`,
`src/uniforms.rs`, and one-line placeholders for the other eight modules.

### Did not work

The crate does not compile. At the last check of this session,
`cargo check --release` reported 31 errors, dropping to 11 on an immediate
re-run as the owning lanes landed fixes. The causes, in order of severity:

- `shaders/blit.wgsl` does not exist, and `src/renderer.rs` does
  `include_str!("../shaders/blit.wgsl")`. This is a hard error with no
  workaround, and it is the critical path: without a blit there is no way to see
  the compute kernel's output at all. **This is the single highest-priority
  missing file in the project.**
- `src/app.rs` has no `run()`. `src/main.rs` calls it. Expected; the integration
  step has not started.
- `src/renderer.rs` against wgpu 30: `Instance::new` takes `InstanceDescriptor`
  by value and it has no `Default`; `create_storage_texture` is called and not
  defined; `TextureView::default()` and `BindGroup::default()` do not exist (a
  half-built renderer cannot be constructed, so the textures and bind groups
  have to be built in `new` or the type has to become `Option<Renderer>`); and
  `TimestampState` is not yet a `Mutex`, so four `self.timestamp_state.lock()`
  sites do not compile.
- `src/panel.rs` against egui 0.36: `RichText`'s `text` field is private (use
  the method); `layout_no_wrap` returns `Arc<Galley>` with no `.galley` field
  (five sites); `FontSelection::Monospace` and `TextEdit::clip` are gone;
  `egui::Id::new` is not `const`, so the two `pub const` state ids cannot be
  const; `egui::PathStroke` is not exported; `Theme::hairline_strong()` returns
  a `Stroke` where a `Color32` is wanted; and `wrapping_mul` is being called on
  a reference to a `u64`.

Nothing has been run. There is no screenshot, no frame, no evidence the y-flip
is right. Three cheap and important checks are queued in
[TODO.md](../TODO.md): the orientation of the first frame, the two
cross-language tripwires at `z = 1` and `z = i`, and that the blit does no
colour work.

### Decisions

- **The uniform block and the function id table live in `src/uniforms.rs` and
  nowhere else.** Anything that crosses into a shader is a contract, and one
  file is the only way to make it reviewable. A consequence, recorded because it
  will look like an inconsistency: `src/functions.rs` owns the *shape* of the
  table (ids, names, groups, order) while `src/complex.rs` owns the *semantics*.
- **Function ids are append-only, 0–15, forever.** Not because a reorder is hard
  but because a reorder is silent. A saved view, a config file, or a bug report
  that names id 7 will keep naming id 7, and id 7 has to keep meaning `sin`.
- **The camera is raw `f64` and does not depend on `Complex`.** The camera is
  updated every frame and pan/zoom accumulate over thousands of frames; putting
  a display type in that hot path buys nothing. It also keeps `camera.rs`
  compilable and testable with no dependency on the rest of the crate.
- **`f32` on the GPU, `f64` on the CPU, deliberately.** The kernel gets `f32`
  because that is what the uniform block carries; `Complex` is `f64` because it
  is what the tests assert on and what a human reads to check the mathematics.
- **No dependencies without a stated reason.** 423 crates and a 21-minute cold
  build is already the cost floor.
- **Compute-then-blit, not a fragment shader.** Per-pixel work without
  quad-wide agreement, a reusable output texture, and evaluation decoupled from
  presentation. Argued in full in [ARCHITECTURE.md](ARCHITECTURE.md).
- **Phase 2 (Riemann surfaces) is not solved and is not pretended to be.** Two
  mesh approaches and one compute approach are written up with their trade-offs
  in [ROADMAP.md](ROADMAP.md), and the recommendation is to build the cheap
  compute one first, as an experiment that answers whether the interesting part
  is the topology or the branch selection.

### Next

1. Write `shaders/blit.wgsl`. Three vertices, no vertex buffer, a fragment
   entry point that samples the `rgba8unorm` storage texture and returns it
   unchanged. Nothing else in the project can be verified until this exists.
2. Finish `src/renderer.rs` against the wgpu 30 API, including the frame loop,
   resize handling, and the timestamp-query readback.
3. Fix `src/panel.rs` against the egui 0.36 API.
4. Write `src/app.rs`. Nothing works without it.
5. First frame. Screenshot it. **Check the orientation before anything else** —
   an unflipped y produces a perfectly plausible-looking wrong picture, and
   every subsequent observation is conditioned on it.
6. Commit milestone 1 in reviewable pieces rather than one commit: the ABI, the
   maths, the shader, the renderer, the panel, the app. The ABI and shader edits
   must share a commit; nothing else has to.

---

## 2. Escape-handling analysis, and the state correction — 2026-09-28

### Situation

Same day, later. The build lanes that entry 1 reported as in flight had all
landed by the time this entry was written: `shaders/blit.wgsl` (6.9 KB),
`src/renderer.rs` (1,328 lines) and `src/panel.rs` (1,857 lines) all exist and
compile. `cargo check --release` now reports **exactly one error**, down from
31:

```
error[E0425]: cannot find function `run` in module `app`
  --> src/main.rs:30:10
   |
30 |     app::run()
   |          ^^^ not found in `app`
```

Every API-drift error listed in entry 1's **Did not work** is resolved. The
wgpu 30 `InstanceDescriptor` by-value fix, the missing
`create_storage_texture`, the `TextureView::default()` / `BindGroup::default()`
problem, the `TimestampState` mutex, and all of the egui 0.36 breakage are
gone. `src/app.rs` is still the one-line placeholder and is now the entire
remaining scope of milestone 1.

### Attempted

Turn the WGSL lane's finding about iterated plots into documentation, and
re-derive the build state so entry 1's error list did not go stale.

### Worked

- **The escape-handling question is now written down properly** — in
  [../TODO.md](../TODO.md) as its own subsection of Known open questions, and
  in [ROADMAP.md](ROADMAP.md) phase 1 as its own subsection. Both record the
  measurement (critical orbit of 0 escapes at iteration 120; ~92% of a 3x3
  viewport non-finite at `max_iter = 256`; whole frame black at 4096), the
  *second* and more important half of the problem, and the tuning dependency
  that blocks any attempt to settle it before there is a rendered frame.
- **Verified the white-saturates-before-overflow claim numerically** rather than
  taking it on trust. With `value = m/(1+m)` and `modulus_shading = 1.0`:
  `|w| = 1e4` gives `0.9999`, and the value is **exactly `1.0` in `f32` from
  about `|w| = 3.4e7`**, staying there until `f32` overflows at `1.8e19` and
  `c_finite` finally goes false. So the white plateau is roughly twelve orders
  of magnitude wide, then a hard snap to black. The claim holds.
- **Found the one real bug in the neighbourhood**, which is *not* the one
  reported. The shader's `apply_selected` breaks out of the loop as soon as
  `c_finite(w)` goes false; `functions::eval` in `src/functions.rs` has no such
  break and runs all `steps` iterations. For a polynomial the two converge to
  the same observable answer (a non-finite value stays non-finite, so both end
  black), but the loops are not the same code and the difference is invisible
  until a function maps a non-finite input back to a finite one. Recorded in
  [../TODO.md](../TODO.md).

### Did not work

Nothing attempted here failed. Two things worth recording as *not* done:

- **The rendered frame still does not exist**, so the escape-handling fix is
  documented and nothing more. `T` and `escape_scale` are visual judgements and
  picking them blind would ship either an all-black or an all-white plot. The
  dependency is now written into both documents explicitly so that a future
  session does not treat the item as ready to implement.
- **The 21-minute cold build was still not re-timed.** Deleting a 598 MB
  `target/` to measure it is not a decision this session should make unilaterally
  while other lanes are compiling against it.

### Decisions

- **The iterated plot is documented as an open design question, not a defect,
  and the renormalisation candidate is documented without being implemented.**
  It is not this lane's to implement. The two candidate fixes are recorded as
  deliberately *not* equivalent: classical escape-time colouring needs a uniform
  field and so is a full ABI change, whereas renormalisation is a kernel-local
  variable and leaves `Uniforms` at 64 bytes. The shader's own `KNOWN
  LIMITATION` comment concludes that fixing this "changes both the algorithm and
  the ABI"; that is true only of the classical fix, and the documents say so
  rather than letting the two get conflated.
- **15 of the 16 functions are correct with `iterate` off**, so escape handling
  does not gate the first successful frame. Stated in the first-frame item, in
  the phase-1 done-means list, and in the scope paragraph of the escape section.
  A future session should not stall the whole milestone behind a question that
  only affects one of sixteen plots on one code path.

### Next

1. Write `src/app.rs`. One function, `pub fn run() -> eframe::Result<()>`, and
   the crate builds. This is the whole of what stands between the project and a
   first frame.
2. First frame: screenshot it, **check the orientation before anything else**.
3. Then settle escape handling, with a picture on screen, against the analysis
   already written. Start from the renormalisation candidate; do not re-derive
   the analysis.

---

## Template

Copy this block, put it directly above the `---` that precedes the template, and
number the entry. Delete the parenthetical reminders as you fill it in.

```markdown
## <n>. <short title> — YYYY-MM-DD

### Situation

What the repo looked like when this session started: which files existed, which
compiled, what `git log --oneline -5` said, what the last entry left open.
One paragraph. If nothing changed, say so and say what you did instead.

### Attempted

What this session set out to do. Not a task list — the thing. If the session
turned into something else, that goes in **Did not work**.

### Worked

What is now different, and why it is right. One line per fact, with the
reasoning. Include anything that looks like a mistake but is deliberate, and
say so explicitly, because a future session will otherwise "fix" it.

### Did not work

What was tried and did not land, and the actual error text where it is
load-bearing. The build state at the end, honestly: how many errors, which
files, which lanes own them. A failed experiment with its diagnosis is worth
more here than a success with no caveat.

### Decisions

Anything decided that a future session would otherwise re-litigate. What was
chosen, what was rejected, and why. If a decision reverses an earlier one, cite
the entry number. If something is still undecided, say it is undecided rather
than implying a conclusion was reached.

### Next

Concrete, in order, each a thing a person can start on. Not aspirations.
```

---

## Index of decisions

Kept here so a reader can find the reasoning without reading every entry. Each
line points at the entry that made the call.

| Decision | Entry |
|---|---|
| Uniform block and function ids owned by `src/uniforms.rs` alone | 1 |
| Function ids append-only, 0–15 | 1 |
| Camera is raw `f64`, decoupled from `Complex` | 1 |
| `f32` on GPU, `f64` on CPU, deliberately | 1 |
| No dependency without a stated reason | 1 |
| Compute-then-blit rather than a fragment shader | 1 |
| Never run `brew`; always `PATH="$HOME/.cargo/bin:$PATH"` | 1 |
| Escape handling is a design question, not a defect; do not clamp it | 2 |
| Renormalisation needs no ABI change; classical escape-time does | 2 |
| Escape handling does not gate the first frame (15/16 un-iterated are fine) | 2 |
