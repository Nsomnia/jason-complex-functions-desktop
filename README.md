# STEEL-PULSE v2.0-TURBO

**A hardware-accelerated native desktop recreation of the
[WebGL Complex Function Plotter](https://samuelj.li/complex-function-plotter/).**

Complex functions are drawn with the standard **domain-colouring** scheme: hue encodes the
argument (phase) of the image, brightness encodes its modulus, and thin dark lines mark the
iso-modulus and iso-phase contours. A WGSL compute kernel evaluates the function per-pixel on
the GPU; the result is blitted to the window through egui.

Written in Rust 2021 on `wgpu` 30 and `egui`/`eframe` 0.36. Targets macOS (Apple Silicon and
Intel) and Windows 10/11.

---

## Status

Milestone 1, *"working renderer core"*, is **complete and verified**.

| Check | Result |
|---|---|
| `cargo build --release` | clean, 0 warnings |
| `cargo test --release` | **195 passed**, 0 failed |
| `cargo fmt --check` | clean |
| First frame on screen | verified by GPU readback |

The y-orientation and the aspect correction were measured rather than eyeballed, and the
Julia parameter was chosen by a grid sweep rather than by taste. See
[`TODO.md`](TODO.md) for what is done, what is deliberately *not* done, and which open
questions have already been answered (and how).

### Known rough edges

- **The in-app UI is not yet legible.** The neon theme's typography and contrast need real
  work — the font is hard to read and the on-screen key legend cannot be deciphered. The
  functional behaviour is fine; the presentation is not. Tracked at the top of
  [`TODO.md`](TODO.md).
- The panel and dock layout have been tested at the logic level but not visually confirmed on
  screen. The GPU readback path covers the compute output only, not the egui layer.

---

## Build and run

> **Toolchain warning.** The Homebrew `rustc` on the development machine is broken (its
> `librustc_driver` cannot resolve LLVM symbols), and `brew` itself hangs indefinitely. Every
> command below therefore prefixes the rustup toolchain:
>
> ```sh
> export PATH="$HOME/.cargo/bin:$PATH"
> ```
>
> Do not run `/usr/local/bin/cargo`. Do not run `brew`. Full details in
> [`agents/ENVIRONMENT.md`](agents/ENVIRONMENT.md).

```sh
export PATH="$HOME/.cargo/bin:$PATH"

cargo build --release      # ~25 s warm; ~21 min for a cold dependency tree
cargo test                 # 195 unit tests, none require a GPU
cargo fmt
cargo clippy --release -- -D warnings
cargo run --release        # needs a window and a GPU
```

### Control

| Input | Action |
|---|---|
| **Drag** | Pan |
| **Wheel / trackpad scroll** | Zoom about the cursor |
| **R** | Reset the view to the origin |

### Capturing a frame from the command line

`STEEL_PULSE_CAPTURE=1` renders one frame, writes it as a binary PPM to the temp directory,
prints nothing and exits — useful for unattended verification, and necessary on machines
where OS screenshotting cannot reach the rendered surface (see
[`agents/ENVIRONMENT.md`](agents/ENVIRONMENT.md)).

```sh
STEEL_PULSE_CAPTURE=1 cargo run --release                      # current view
STEEL_PULSE_CAPTURE=1 STEEL_PULSE_FUNC=15 \
  STEEL_PULSE_ITERATE=1 STEEL_PULSE_MAX_ITER=256 \
  cargo run --release                                          # iterated Julia set
sips -s format png "$TMPDIR"/steel-pulse-*.ppm --out plot.png  # convert to PNG
```

The dump is `Rgba8Unorm`, not `Rgba8UnormSrgb` — raw values with no gamma applied. A viewer
will apply its own transform, so it will not match the screen exactly. That is a known open
question about the colour pipeline, not a bug in the readback.

---

## Documentation

Start with **[`AGENTS.md`](AGENTS.md)** if you are picking this repository up cold. It is the
entry point: the non-negotiable rules, the exact commands, and a map from "I need to change
X" to the file that owns it.

### Core

| Document | What it covers |
|---|---|
| [`AGENTS.md`](AGENTS.md) | **Entry point.** Non-negotiable rules, build commands, module map, platform notes. |
| [`TODO.md`](TODO.md) | The live task list by milestone, plus the known open questions and the decisions already taken. |
| [`agents/ARCHITECTURE.md`](agents/ARCHITECTURE.md) | How the system fits together, the frame loop, the data flow from CPU state to screen, and why it is compute-then-blit. |
| [`agents/ABI.md`](agents/ABI.md) | The CPU/GPU contract. The uniform block layout, the function id table, the domain-colouring algorithm, and the checklist for adding a 17th function. |

### Reference

| Document | What it covers |
|---|---|
| [`agents/ROADMAP.md`](agents/ROADMAP.md) | The multi-phase plan: renderer core (done), Riemann surfaces (the unsolved part), performance and packaging. |
| [`agents/GLOSSARY.md`](agents/GLOSSARY.md) | Vocabulary for newcomers: domain, modulus, branch cut, phase portrait, workgroup, blit, and the rest. |
| [`agents/ENVIRONMENT.md`](agents/ENVIRONMENT.md) | Toolchains, the broken-Homebrew workaround, build times, and the screenshotting limitation. |
| [`agents/SESSION-LOG.md`](agents/SESSION-LOG.md) | Running log of what each session did, what worked, and what did not. The continuity record. |

---

## Architecture in one diagram

```text
egui input ──> Camera (raw f64, eased targets)
                     │
                     v
              Uniforms (64 bytes, the ABI)
                     │
        queue.write_buffer + compute pass
                     │
                     v
      WGSL kernel ──> offscreen rgba8unorm storage texture
                     │
        register_native_texture (egui_wgpu::RenderState)
                     │
                     v
              egui image ──> eframe's blit ──> screen
```

`src/uniforms.rs` is the single source of truth for the buffer layout, and the 16 function
ids are a permanent contract shared by the Rust reference and the WGSL kernel. Both sides
must be changed together, in the same commit. [`agents/ABI.md`](agents/ABI.md) explains why
this is not optional.

The design deliberately evaluates the function per-pixel in a compute pass rather than in a
fragment shader: iteration counts run to thousands with an early-out, and evaluation is
decoupled from the UI's repaints — otherwise every tooltip flicker would re-run the kernel.

---

## The function table

Sixteen functions, fixed ids, append-only. Never renumber them; a saved view stores the id.

| id | function | | id | function |
|---|---|---|---|---|
| 0 | `z` | | 8 | `cos z` |
| 1 | `z²` | | 9 | `sinc z` |
| 2 | `z³` | | 10 | `sinh z` |
| 3 | `1/z` | | 11 | `e^z` |
| 4 | `z² − 1` | | 12 | `log z` |
| 5 | `z³ − 1` | | 13 | `√z` |
| 6 | `z³ − 2z` | | 14 | `(z−1)/(z+1)` |
| 7 | `sin z` | | 15 | `z² + c` (Julia) |

With **iterate** on, any of them can be applied repeatedly, which turns an arbitrary
function into a Julia-style set and surfaces its branch cuts and singularities. The branch
cuts on `log` and `√z` are not smoothed over — the discontinuity is the interesting part.

---

## Credits

**Danny Steel** — Staff Principal Architect, Beefcake Daddy, LLM Overlord. Wrote the uniform
ABI contract, caught the y-orientation trap before it shipped, and personally benchmarked
405 lbs without dropping a single frame. Author of the rule that the CPU reference and the
WGSL kernel must be kept numerically in agreement, on the grounds that a divergence produces
a wrong picture rather than an error, and there is no harness that would catch it.

**Jason "Jigglypuff"** — Junior Script-Kiddy, currently in remedial latent-space training.
Wrote the sixteen functions, the `f64` complex reference, and the camera. Becomes visibly
nauseous whenever confronted with a raw memory pointer or an imaginary exponent, which is
roughly every third conversation. Contributed the measured grid sweep that proved the Julia
parameter — and not the floating-point arithmetic — was the reason the iterated plots were
black, thereby saving an entire refactor.

---

## Licence

Not yet specified. Add one before this goes anywhere public.
