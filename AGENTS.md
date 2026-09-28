# AGENTS.md

Read this file first. It is the entry point for anyone picking this repository
up cold.

- [TODO.md](TODO.md) — the live task list, by milestone
- [agents/ARCHITECTURE.md](agents/ARCHITECTURE.md) — how the system fits together
- [agents/ABI.md](agents/ABI.md) — the CPU/GPU contract. Read before touching `src/uniforms.rs` or the shaders
- [agents/ROADMAP.md](agents/ROADMAP.md) — the multi-phase plan
- [agents/GLOSSARY.md](agents/GLOSSARY.md) — vocabulary
- [agents/ENVIRONMENT.md](agents/ENVIRONMENT.md) — toolchain, platform, build times
- [agents/SESSION-LOG.md](agents/SESSION-LOG.md) — what past sessions did, and why

## What this is

STEEL-PULSE v2.0-TURBO is a hardware-accelerated native desktop recreation and
expansion of Samuel J. Li's WebGL Complex Function Plotter
(<https://samuelj.li/complex-function-plotter/>), written in Rust on top of
`wgpu` and `egui`. It renders complex functions over the complex plane using the
standard domain-colouring scheme — hue encodes the argument of the image,
brightness encodes its modulus, with iso-modulus and iso-phase contour lines
drawn as thin dark bands — by evaluating the function per-pixel in a WGSL
compute kernel that writes into an `rgba8unorm` storage texture which is then
blitted to the screen. The intended expansion beyond the original plotter is
Riemann-surface exploration, which is the hard part and is **not** solved yet
(see [agents/ROADMAP.md](agents/ROADMAP.md)).

Target platforms are macOS (Apple Silicon and Intel) and Windows 10/11.

## Current state

Milestone 1, "working renderer core", is in progress. As of 2026-09-28:

| Component | State |
|---|---|
| `src/uniforms.rs` | Complete. 64-byte uniform block, the CPU/GPU ABI. Tests pass. |
| `src/complex.rs` | Complete. `f64` complex arithmetic, the CPU reference. |
| `src/functions.rs` | Complete. 16-function dispatch table with fixed ids. Compiled as `crate::complex::functions` via an explicit `#[path]`. |
| `shaders/domain_coloring.wgsl` | Complete. Compute kernel, complex helpers, `domain_color`, grid overlay. |
| `shaders/blit.wgsl` | Complete. Full-screen triangle, texture at binding 0, sampler at binding 1, single `textureSampleLevel` in `fs_main`. No colour work. |
| `src/camera.rs` | Complete. `f64` viewport state, easing targets, pixel/complex round trip. |
| `src/telemetry.rs` | Complete. 128-entry frame ring buffer, CPU/GPU timing. |
| `src/theme.rs` | Complete. Neon "3AM tweaker" palette and typography scale. |
| `src/renderer.rs` | Complete. Compiles clean against wgpu 30. |
| `src/panel.rs` | Complete. Compiles clean against egui 0.36. |
| `src/app.rs` | **Not started.** One line: `// placeholder - owned by a build lane`. |
| First frame on screen | Not achieved. |

**The crate has exactly one compile error, and it is the whole of the remaining
milestone 1:** `src/main.rs:30` calls `app::run()` and `src/app.rs` does not
define it. Every other module — 8,278 lines of Rust across nine files, plus
two WGSL shaders — compiles clean. Write `app.rs` and the crate builds.

Re-verify with `cargo check --release` rather than trusting this table: several
lanes write these files concurrently and the table will go stale again.


## Non-negotiable rules

**1. `src/uniforms.rs` is the single source of truth for the GPU buffer layout
and the function id table.**
Nothing else in the crate may invent a field, reorder one, or renumber a
function. *Reason:* the WGSL struct in `shaders/domain_coloring.wgsl` is a
separate text artifact that no compiler cross-checks. A one-field shift does not
fail — it silently mis-colours the entire plot, or reads `center` as `scale`.
The struct is `#[repr(C)]` and `bytemuck::Pod`, so the layout is pinned, and the
`field_offsets_match_the_documented_wgsl_layout` test asserts all thirteen
offsets. That test is the only automated defence; do not weaken it.

**2. Any change to the uniform struct or to a function id must land in the same
commit as the matching shader edit.**
Same reason as rule 1. [agents/ABI.md](agents/ABI.md) holds the full procedure
and the "how to add a 17th function" checklist.

**3. CPU and WGSL implementations of each function must be kept numerically in
agreement.**
`src/complex.rs` is the oracle; the WGSL is a transcription of it. *Reason:* a
divergence produces a wrong picture, not an error, and there is currently no
cross-language test harness that would catch it. Where the transcription is
deliberately different for numerical reasons — `Complex::norm_hypot` versus
`Complex::norm`, for instance — the divergence is deliberate and is documented
in the source at both ends. Only those documented exceptions are permitted.
`src/functions.rs` also defines `ITERATION_CAP` and
`SINC_SINGULARITY_GUARD_RADIUS`; `shaders/domain_coloring.wgsl` defines
`MAX_ITER_CLAMP` and `EPS`. These pairs must stay equal.

**4. Function ids are append-only. Never reorder, never reuse.**
Ids 0–15 are fixed forever. A new function takes id 16, then 17. *Reason:* ids
are what the shader switches on and what any saved view or config will store. A
reorder silently changes which function a stored id selects, and nothing fails.

**5. `Uniforms` must stay a multiple of 16 bytes, and `_pad` absorbs the
difference.**
*Reason:* WGSL rounds a uniform struct's alignment up to 16. A mis-sized
binding is rejected by wgpu at buffer-creation time rather than at first draw,
which is a miserable way to find out. The `const _: () = assert!` guards in
`src/uniforms.rs` enforce this at compile time; if you add a field, update the
size assertion and the offset test in the same edit.

**6. The camera works in raw `f64` and must not depend on `complex::Complex`.**
`Camera` holds `center_re`, `center_im`, `scale` and their `target_*` easing
counterparts as plain `f64`. *Reason:* the camera is updated every frame from
egui input, and pan and zoom accumulate over thousands of frames; that arithmetic
needs more precision than the display has, and routing it through a display type
puts a semantic type in the hot path for no gain. It also keeps `camera.rs`
compilable and testable with no dependency on the rest of the crate. The
f32/f64 boundary is crossed exactly once, when the camera is narrowed into
`Uniforms` for upload.

**7. Do not add dependencies without a stated reason.**
*Reason:* `wgpu` and `eframe` already pull in a very large tree — 423 crates in
`Cargo.lock` — and this machine takes about 21 minutes to build them cold. Every
new dependency is build time, binary size, and a licence to track. A dependency
is acceptable when it removes a hand-rolled implementation that would otherwise
be a correctness liability (a criterion benchmark harness is the likely first
candidate), and the reason must be written into the commit message and a line in
[agents/ENVIRONMENT.md](agents/ENVIRONMENT.md).

**8. `f32` on the GPU, `f64` on the CPU, and that is deliberate.**
`Complex` is `f64` because it is the reference the tests assert on and the thing
a human reads to check the mathematics. The kernel is `f32` because that is what
the uniform block carries. Do not "fix" one to match the other.

## Build, test, run

**Every command must be prefixed with the rustup toolchain path.** The Homebrew
`rustc` at `/usr/local/bin/rustc` is broken on this machine — its
`librustc_driver` cannot resolve LLVM symbols
(`Symbol not found: __ZN4llvm17OptimizationLevel2O0E`) — and `brew` itself
hangs, so it cannot be used to repair anything. The working toolchain is
rustup's, already symlinked into `~/.cargo/bin/`.

```sh
cd /Users/derekvanee/Documents/jason-complex-functions-desktop
export PATH="$HOME/.cargo/bin:$PATH"
```

Verify before anything else:

```sh
PATH="$HOME/.cargo/bin:$PATH" cargo --version    # cargo 1.97.1
PATH="$HOME/.cargo/bin:$PATH" rustc --version    # rustc 1.97.1
```

Then:

```sh
export PATH="$HOME/.cargo/bin:$PATH"

cargo check --release          # fastest way to see if the crate builds
cargo build --release          # produces target/release/steel-pulse
cargo test                     # unit tests; the ABI tests live in src/uniforms.rs
cargo test --release uniforms  # just the ABI layout tests
cargo run --release            # launches the app; needs a window and a GPU
cargo fmt
cargo clippy --release -- -D warnings
```

Notes that will save you time:

- `cargo check --release` of this crate alone, with all 423 dependencies already
  built, takes about **35 seconds**. A cold build of the dependency tree takes
  about **21 minutes**. Do not conclude the build is hung before 25 minutes on a
  cold run.
- If you see `Blocking waiting for file lock on build directory`, another cargo
  process owns `target/`. That is not a hang. Check with `ps aux | grep cargo`.
- Do **not** run `/usr/local/bin/cargo`. It is the broken Homebrew build.
- Do **not** run `brew` for any reason. It hangs indefinitely; there is a stuck
  `brew.rb install rustup` process on this machine that has been running for
  hours.
- `cargo fmt` and `cargo clippy` are available from the same toolchain
  (rustfmt 1.9.0-stable).

## Platform notes

- The development machine is **x86_64** macOS 14.8.8, Intel UHD Graphics 617,
  Metal 3. `uname -m` returns `x86_64`. The rustup toolchain is
  `stable-x86_64-apple-darwin`.
- The eventual product must also run on **Apple Silicon**. Nothing in the
  codebase should assume x86: there is no hand-written assembly, no
  architecture-specific SIMD, and no x86-only linkage. The one thing to watch
  is `f32` behaviour near the edges, where Apple GPUs are stricter about
  reassociation than the Intel one.
- wgpu 30 on macOS talks to Metal. There is no Vulkan or GL path here.
- Windows 10/11 is a target but nothing has been built or tested for it, and
  cross-compiling from macOS to Windows has not been attempted. See
  [agents/ROADMAP.md](agents/ROADMAP.md) phase 3.

## Where to look

| If you need to change... | Go to |
|---|---|
| The uniform buffer layout, field offsets, or a function id | [`src/uniforms.rs`](src/uniforms.rs), then [agents/ABI.md](agents/ABI.md) |
| The maths of a function, or add a function to the table | [`src/functions.rs`](src/functions.rs) (`FUNCTIONS`, `eval`, `apply_once`) |
| `f64` complex arithmetic, the reference semantics | [`src/complex.rs`](src/complex.rs) |
| The matching GPU implementation of a function | [`shaders/domain_coloring.wgsl`](shaders/domain_coloring.wgsl) (`apply`, the complex helpers) |
| The colour scheme, contours, or the grid overlay | [`shaders/domain_coloring.wgsl`](shaders/domain_coloring.wgsl) (`domain_color`, `apply_grid`) |
| Pixel-to-complex mapping, pan, zoom, easing | [`src/camera.rs`](src/camera.rs) |
| wgpu device, pipelines, the frame loop, timing queries | [`src/renderer.rs`](src/renderer.rs) |
| Presenting the storage texture to the screen | [`shaders/blit.wgsl`](shaders/blit.wgsl) (`vs_main`, `fs_main`) |
| Frame timing, FPS, GPU/CPU stats | [`src/telemetry.rs`](src/telemetry.rs) |
| Colours, fonts, spacing, the neon look | [`src/theme.rs`](src/theme.rs) |
| The docked control panel and its widgets | [`src/panel.rs`](src/panel.rs) |
| Wiring everything into an `eframe::App` | [`src/app.rs`](src/app.rs) — **does not exist yet** |
| What still needs doing | [TODO.md](TODO.md) |
| What the plan is and why | [agents/ROADMAP.md](agents/ROADMAP.md) |
| What a word means | [agents/GLOSSARY.md](agents/GLOSSARY.md) |

## Git

`git log` works and the repository has history. The first commit is

```
14a08d1  Scaffold: uniform ABI contract, module map, dependency lock
```

which established `Cargo.toml`, `Cargo.lock`, `src/main.rs`, `src/uniforms.rs`
and the placeholder module files. Milestone 1 is that first commit plus
everything after it. `src/*.rs` and `shaders/*.wgsl` are under active concurrent
development, so `git status` will usually be dirty. `.gitignore` contains
`/target` (twice) — `target/` is ~600 MB and must not be committed.
