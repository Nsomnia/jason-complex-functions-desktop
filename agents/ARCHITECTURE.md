# ARCHITECTURE

How STEEL-PULSE v2.0-TURBO fits together. For the buffer layout and function
numbering, read [ABI.md](ABI.md); for the plan, [ROADMAP.md](ROADMAP.md).

## The one-paragraph version

Every frame, the CPU narrows its `f64` view state into a 64-byte uniform block
and uploads it. A compute shader reads that block once per invocation,
evaluates the selected complex function at the point under that pixel, maps the
result to a colour by the domain-colouring scheme, and writes a byte into an
`rgba8unorm` storage texture. A second, trivial render pass samples that texture
and writes it to the window surface. The compute pass does all the work; the
blit does none. The egui panel runs in the same window and never touches either
pass — it only writes the uniform block.

## The pipeline

```
   CPU (main thread, f64)                         GPU
   ─────────────────────                           ───

   Camera { center_re: f64,                       ┌──────────────────────────┐
             center_im: f64,                      │  COMMAND ENCODER         │
             scale:       f64 }                    │  (one encoder per frame) │
        │                                          └──────────────────────────┘
        │ narrow to f32                                  │
        ▼                                                 │
   Uniforms (64 B, #[repr(C)]) ──── write_buffer ──────────┤
        │            (offset 0, whole struct)              │
        │                                                  ▼
        │            ┌─────────────────────────────────────────────────┐
        │            │  PASS 1: COMPUTE                                │
        │            │  domain_coloring.wgsl  @workgroup_size(8, 8)     │
        │            │                                                 │
        │            │  @group(0) @binding(0) var<uniform>  uniforms    │──┐
        │            │  @group(0) @binding(1) var storage_2d<rgba8unorm,│  │
        │            │                                             write>│  │
        │            │                                                 │  │
        │            │  per invocation:                                │  │
        │            │    1. pixel -> z          (complex_at_pixel)    │  │
        │            │    2. w = f(z), or f^n(z) if iterate != 0      │  │
        │            │    3. w non-finite?  -> black, done            │  │
        │            │    4. w -> RGB        (domain_color)            │  │
        │            │    5. optional axes + unit circle on z          │  │
        │            │    6. textureStore(output_texture, pixel, rgba) │  │
        │            └──────────────────────────┬──────────────────────┘  │
        │                                       │ writes                   │
        │                                       ▼                         │
        │            ┌─────────────────────────────────────────────────┐  │
        │            │  STORAGE TEXTURE                                │◀─┘
        │            │  rgba8unorm, usage STORAGE_BINDING |            │
        │            │              TEXTURE_BINDING | COPY_DST          │
        │            │  size = surface size in PHYSICAL pixels         │
        │            └──────────────────────────┬──────────────────────┘
        │                                       │ textureSampleLevel
        │            ┌──────────────────────────▼──────────────────────┐
        │            │  PASS 2: RENDER                                 │
        │            │  blit.wgsl  (3-vertex full-screen triangle)     │
        │            │                                                 │
        │            │  @group(0) @binding(0) var input_texture         │  │
        │            │                    : texture_2d<f32>             │  │
        │            │  @group(0) @binding(1) var linear_sampler        │  │
        │            │                                                 │
        │            │  vs_main: 3 verts, no vertex buffer,            │
        │            │            clip position from vertex_index      │
        │            │  fs_main: uv from @builtin(position) / size,   │
        │            │            output textureSampleLevel(input_texture,
        │            │                     linear_sampler, uv, 0.0)     │
        │            │            ALPHA = 1, no gamma, no tone curve   │
        │            └──────────────────────────┬──────────────────────┘
        │                                       │ render pass
        ▼                                       ▼
   Telemetry { cpu_ms, gpu_ms,          ──► WINDOW SURFACE (swapchain)
              fps_instant, ... }              (bgra8unorm on macOS/Metal)
```

The dotted arrow from the CPU to PASS 2 is deliberate: the blit does not read
`Uniforms` at all in the current design. If it turns out it needs to, that is a
layout change and it goes through [ABI.md](ABI.md) first.

## Data flow, step by step

1. **Input.** egui hands `App::update` a `RawInput`. Drag deltas and wheel
   deltas are converted to camera moves and written to the `target_*` fields of
   `Camera`, never to the live fields. This is what makes the easing in
   `camera.rs` work; writing straight to the live fields would make every pan
   and zoom instantaneous and the easing dead code.

2. **Narrowing.** Immediately before the frame, the live `f64` camera fields are
   narrowed to `f32` and written into a `Uniforms` value. This is the single
   place the f64/f32 boundary is crossed. `Uniforms` is `#[repr(C)]` and
   `bytemuck::Pod`, so it uploads with
   `queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[uniforms]))` — no
   staging, no `Vec`, no per-field marshalling.

3. **Compute.** The kernel is dispatched over `ceil(w/8) x ceil(h/8)` workgroups.
   Invocations past the right and bottom edges return early, because
   `textureStore` with an out-of-range coordinate is undefined in WGSL and
   validation layers will complain. That guard is in `main` and must stay.

4. **Blit.** One `RenderPass` draws three vertices. The vertex stage emits a
   triangle that covers the clip volume with no vertex buffer at all, deriving
   positions from `vertex_index` alone. The fragment stage samples the storage
   texture and returns it unchanged.

5. **Present.** `queue.submit`. The telemetry ring buffer gets a CPU duration
   from `begin_frame`/`end_frame` and, once the timestamp-query readback
   plumbing lands, a GPU duration from a pair of query sets.

## Why compute-then-blit, and not a fragment shader

A plain fragment shader that evaluates the complex function per pixel is the
obvious design and it is wrong for this program, for three reasons.

**Arbitrary per-pixel work.** Every pixel wants a different amount of work. Near
the boundary of a Julia set, one pixel needs 4000 iterations to decide and its
neighbour needs 12. A fragment shader gives every pixel in a 2x2 quad the same
shader invocation and therefore, on real hardware, very nearly the same
execution time — the whole quad waits for the slowest lane. The standard
mitigation is dynamic loop length with an `any` flag across the quad, which is
fiddly, still imbalanced, and on some drivers still costs the full iteration
count for the quad. A compute shader has one invocation per pixel with no
quad-wide agreement requirement, so a cheap pixel simply finishes and retires.
On the Intel UHD 617 this app is developed on, that is the difference between a
plot that redraws at 60 fps and one that does not. The kernel already does this:
`apply_selected` breaks out of the loop the moment `c_finite` goes false, and
`MAX_ITER_CLAMP` bounds the worst case at 4096.

**The storage texture is reusable.** The compute pass writes to a plain
`rgba8unorm` storage texture that nothing else in the pipeline is required to
treat as an image. That means the same buffer can be:

- the input to the blit, which is the normal path;
- the accumulator for a multi-pass algorithm (milestone 3's dirty-rectangle
  scheme, or a multi-sample supersampling scheme);
- an n-layer image, one layer per Riemann sheet (milestone 2's cheapest
  candidate design);
- read back to the CPU for a screenshot, with a plain `COPY_SRC` usage flag.

Every one of those is a no-change-or-one-flag extension to a fragment shader
design, and an architectural change to a compute design.

**Evaluation is decoupled from presentation.** The kernel's only job is "given
`z`, return a colour". It knows nothing about the window, the display, sRGB,
rounding, or the aspect ratio beyond the single `resolution` field it needs for
the pixel-to-plane mapping. The blit knows nothing about complex numbers. This
is the single most useful property in the codebase: it means the interesting
maths can be unit-tested in `f64` on the CPU and transposed to WGSL, with no
interference from presentation, and it means the blit can be replaced
(a gamma-correct blit, a scaling blit, a magnifier) without touching a line of
the maths.

The cost of this choice is one extra full-surface read and write per frame. At
2560x1600 that is 16 MB of traffic for the blit, which is nothing next to the
function evaluation, and it is the price of the four properties above.

## Module ownership

| Module | Owns | Must not |
|---|---|---|
| `src/main.rs` | the module list, `fn main` | contain logic |
| `src/uniforms.rs` | the 64-byte block, its field order, the `UNIFORM_BUFFER_SIZE` constant, the ABI offset tests | be edited without the matching WGSL edit |
| `src/complex.rs` | `f64` complex arithmetic; the reference semantics every other side is checked against | know about wgpu, textures, or the UI |
| `src/functions.rs` | the 16-entry table, the ids, `eval`, iteration | know about WGSL |
| `shaders/domain_coloring.wgsl` | the GPU transcription of the above, plus the colour scheme and the grid | disagree numerically with `src/functions.rs` |
| `src/camera.rs` | viewport state, the pixel-to-complex mapping, easing targets | depend on `complex::Complex`; see rule 6 in [../AGENTS.md](../AGENTS.md) |
| `src/renderer.rs` | the wgpu device, surface, pipelines, bind groups, the frame loop, timestamp queries | decide what a pixel looks like |
| `src/telemetry.rs` | the frame ring buffer, FPS, CPU/GPU durations | touch the uniform block |
| `src/theme.rs` | the neon palette, the type scale, the FPS thresholds | contain layout or state |
| `src/panel.rs` | the docked egui controls | evaluate a function, or hold the only copy of any state |
| `src/app.rs` | the `eframe::App`, the `DockState`, the wiring | contain rendering or maths |

`src/complex.rs` owns the function table's *semantics*; `src/functions.rs` owns
its *shape* (ids, names, groups, order). The module is attached with an explicit
`#[path = "functions.rs"]` because `src/main.rs` has no top-level
`mod functions;` — see the comment at the top of `src/complex.rs`. If a
`mod functions;` line is ever added to `main.rs`, that `#[path]` must be
removed, or the table gets compiled twice and the two copies drift.

## Threading model

There is effectively one thread that matters, and it is worth being precise
about where the boundaries are.

- **The main thread** owns the wgpu `Device`, the `Queue`, the `Surface`, all
  pipelines, and every resource. wgpu is not internally synchronised for you
  in the way you want, and a single-owner device is the model that does not
  surprise anybody. The main thread runs `eframe::run_native`, which drives
  `App::update`, and `App::update` is where the frame is encoded and submitted.

- **The GPU** is asynchronous. `queue.submit` returns immediately; the work is
  still in flight. The `Index` and the surface's present path handle
  synchronisation between frames. Consequence: never read a mapped buffer or a
  `QuerySet` result from the same frame you submitted it in. The timestamp
  readback must use a staging buffer written on a *later* frame, which is why
  the timestamp state in `renderer.rs` keeps an `in_flight` set and a `latest`
  result.

- **wgpu's own worker threads** exist inside the library — the `wgpu-core`
  device poll, the readback staging, and on Metal the driver's own queues.
  They are not yours to reason about. What you must not do is hand a `wgpu`
  resource to another thread without going through an `Arc` and a channel.

- **The `Uniforms` value is `Copy` and is written on the main thread** into a
  buffer that wgpu then copies into a staging allocation. There is no shared
  mutable CPU state anywhere. `Camera` and `Telemetry` are plain `Send` structs
  owned by the `App`; if a future feature needs a worker pool (a benchmark
  harness, say), the value to send is a *copy* of `Uniforms`, never a reference
  into `Camera`.

- **`eframe` and egui are single-threaded by design.** The `App` is not `Sync`
  and does not need to be. Do not add an `Arc<Mutex<App>>` anywhere.

The practical summary: there is one thread, one device, and the only concurrency
worth designing for is the GPU's. If a future change wants a second thread, the
right shape is "clone the `Uniforms`, send it over a channel, get a
`Duration` back", not "lock the App".

## What is deliberately not here yet

- **No depth buffer, no 3D matrices, no vertex buffers.** Everything is 2D.
  Milestone 2 adds all three at once; see [ROADMAP.md](ROADMAP.md).
- **No dirty-rectangle or partial redraw.** Every frame recomputes every
  pixel. Milestone 3.
- **No multi-sample or distance-estimator anti-aliasing.** Escape-time plots
  alias badly near the set boundary. Milestone 3, and it is a genuine open
  question whether it belongs in 2 or 3.
- **No sRGB conversion anywhere.** The kernel writes to `rgba8unorm`, not
  `rgba8unorm-srgb`, and the blit copies bytes through. Whether that is correct
  on a Retina display is an open question in [TODO.md](../TODO.md), not a
  decision.
- **No log or error channel.** There is no logging crate in the dependency set
  and rule 7 in [../AGENTS.md](../AGENTS.md) makes adding one a deliberate act.
  Diagnostics go to stderr for now.
