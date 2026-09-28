# GLOSSARY

The vocabulary of this project, for someone who is a competent programmer and
new to complex analysis or to this particular stack. One line each. Where a word
means something narrower here than in general, that is said.

## Mathematics

**Complex plane** — the 2D number line, written `C`, where a point is a pair
`(re, im)` meaning `re + im*i`. Drawn as a normal x/y plane with the real axis
horizontal and the imaginary axis vertical. `i^2 = -1`.

**Domain** — the set of inputs a function accepts. Here, almost always the
complex plane. The plane you look at is the domain.

**Image** — the set of outputs. For one point, the image is the *value*
`w = f(z)`, and both the hue and the brightness of that pixel are read from `w`,
not from `z`. Getting this backwards produces a plot that looks like a plot and
means nothing.

**Conjugate** — `conj(z) = re - im*i`, the reflection of `z` across the real
axis. `|conj(z)| = |z|`, and `arg(conj(z)) = -arg(z)`.

**Modulus** — the distance from the origin, `|z| = sqrt(re^2 + im^2)`. Also
called magnitude or absolute value. Here, brightness.

**Argument (phase)** — the angle from the positive real axis,
`arg(z) = atan2(im, re)`, in `[-pi, pi]` or `(-pi, pi]`. Here, hue. It is only
defined for `z != 0`, which is one reason the zero of a function is a special
colour case.

**Holomorphy** — being complex-differentiable everywhere in a region.
Equivalent to being expressible as a power series there, which is why a
holomorphic function is locally a function *graph* and can be drawn with one
value per point. The moment a function is multi-valued, that stops being true.

**Branch cut** — a curve in the domain along which a multi-valued function is
declared not to switch sheets, which makes it single-valued there. For `sqrt`
the conventional cut is the negative real axis. **The cut is a rendering
artefact, not a property of the function** — the surface underneath has no edge
there. See [ROADMAP.md](ROADMAP.md) phase 2.

**Branch point** — a point in the domain where a multi-valued function's sheets
come together, locally behaving like `w -> w^n`. For `sqrt` it is `z = 0`; for
`z^(1/3)` it is also `0`. Infinitely many branch points, for `log`. A plot of a
multi-valued function has to say what it does at these; see
[ROADMAP.md](ROADMAP.md).

**Riemann surface** — the actual home of a multi-valued function. A
two-dimensional manifold that locally looks like the complex plane, including at
branch points, but which covers the complex plane several times over. The
`n`-th root function is the standard example: a surface with `n` sheets that
join at `z = 0`. Not yet implemented here; this is milestone 2.

**Sheet** — one local copy of the complex plane in a Riemann surface, i.e. one
branch of a multi-valued function over a region with no branch points in it. A
whole Riemann surface is a set of sheets glued along cuts.

**Julia set** — the boundary of the set of points whose orbits under iteration
of a function stay bounded. For a quadratic polynomial it is the whole
interesting picture, and its fine structure is fractal, which is why plot 15
exists and why the plotter aliases near it.

**Escape-time iteration** — `w = z; repeat w = f(w)` until `|w|` exceeds some
threshold, then report how long it took, or that it never did. A cheap way to
draw the basin structure of an iterated function, and the reason `max_iter` and
`iterate` are the two most important controls in the UI. Note that this codebase
has no escape *radius* — divergence is detected only by a non-finite check.
See [TODO.md](../TODO.md).

**Newton-Raphson** — the iteration `w <- w - g(w)/g'(w)` for finding a zero of
`g`. Not used in the 2D plotter at all. It is on the critical path for
milestone 2, because evaluating branch `k` of an `n`-valued function means
finding one root of `w^n - z = 0` and a closed form does not generally exist.

**Iso-modulus contour** — the locus of points where the image's modulus is
constant. Here: the thin dark bands spaced evenly in `log|w|`, so they are
`e`-fold apart rather than crowded near the origin. A band is drawn wherever
`log|w| * density` is an integer.

**Iso-phase contour** — the locus of points where the image's argument is
constant. In a domain-coloured plot these are the rays emanating from the zero
of the function, and they are what makes the structure of the function legible.

**Domain colouring** — the standard way of drawing a complex function. Evaluate
at every point, map the *image's* argument to hue and its modulus to
brightness, and optionally draw contours. The whole scheme is defined precisely
in [ABI.md](ABI.md#rule-6-the-domain-colouring-algorithm).

**Phase portrait** — a plot showing *where a function's values go* rather than
their magnitude, usually by drawing arrows or a phase contour. A domain-coloured
plot is a phase portrait that also encodes modulus, which is more informative.

**Winding number / turn** — how many full turns the argument has made. Here the
phrase "in turns" in a parameter means this unit, not radians: a hue rotation
of `0.25` is a quarter turn, i.e. 90 degrees, not 0.25 radians.

## The stack

**Rust** — the language. This crate is edition 2024 and builds with rustc 1.97.1
on macOS x86_64. See [ENVIRONMENT.md](ENVIRONMENT.md).

**wgpu** — a portable, safe abstraction over Vulkan, Metal and Direct3D 12.
The rendering and compute API used here. On macOS it talks to Metal. Version 30.

**WGSL** — the WebGPU Shading Language, the one used for wgpu's compute and
render shaders. It has no complex number type, so a complex number is a
`vec2<f32>` and every operation is spelled out by hand. The shaders here are
`shaders/domain_coloring.wgsl` and `shaders/blit.wgsl`.

**naga** — the library that validates and compiles WGSL. wgpu uses it
internally, and its validation errors are what you get when a shader is wrong.
"the naga error" is a common and sufficiently alarming phrase.

**Compute shader** — a shader that runs a grid of invocations over a set of
pixels or elements with no geometry and no framebuffer, and writes to buffers or
storage textures. This project does all the maths in one; see
[ARCHITECTURE.md](ARCHITECTURE.md).

**Uniform buffer** — a small block of GPU-visible memory written by the CPU once
per frame and read by a shader. `Uniforms` in this project is exactly 64 bytes
and is the entire CPU-to-GPU interface. See [ABI.md](ABI.md).

**Storage texture** — a texture a shader can write to directly, as opposed to
one attached to a render pass. This project writes `rgba8unorm` pixels into one
in the compute pass and then samples it in the blit. The reason for the extra
step is in [ARCHITECTURE.md](ARCHITECTURE.md).

**Bind group** — the set of resource bindings a shader sees, declared as a
layout. `@group(0) @binding(0)` in the WGSL is the group and slot. The compute
kernel uses binding 0 for the uniform block and binding 1 for the output
storage texture.

**Workgroup** — a batch of compute invocations that run together and can share
memory. `@workgroup_size(8, 8)` means 64 invocations per workgroup, arranged 8
by 8 to match the shape of a pixel grid, so memory access is coalesced.

**Blit** — a pass whose only job is to copy one texture to the screen. It is
verb by extension of the graphics term for a block transfer. This project's
blit is a three-vertex full-screen triangle and a texture sample, and it makes
no colour decisions at all; see [ABI.md](ABI.md) and the "presentation is
presentation" rule.

**egui** — an immediate-mode GUI library. You call `ui.button(...)` inside a
closure and the widget's state is whatever your variables hold. There is no
widget tree, no retained hierarchy, and no layout engine beyond the immediate
one. Convenient; repaints everything every frame.

**eframe** — egui's application shell. Creates a window, owns the event loop,
runs the render pass for the egui layer, and hands your `App` implementation a
`&mut Context` and a `&RawInput` once per frame.

**egui_dock** — a docking-tabbed container for egui panels, which is what gives
the app its "left sidebar, movable, resizable" shape.

**raw-window-handle** — the cross-platform "here is a window" trait that
`winit` implements and that `wgpu` accepts when creating a surface. It is the
seam between the window and the GPU. eframe brings it in transitively.

**winit** — the windowing and input library. eframe is built on it; this project
does not depend on it directly.

**sRGB** — a colour space, and in wgpu also a texture format suffix
(`rgba8unorm-srgb`) that makes the hardware do the linear-to-display conversion
on write. **This project writes `rgba8unorm`, not `-srgb`, and does no gamma
conversion anywhere.** Whether that is right on a Retina display is an open
question in [TODO.md](../TODO.md), not a decision.

**Metal** — Apple's graphics API. wgpu 30 talks to it on macOS and there is no
Vulkan or GL path on this platform.

**MSL** — Metal Shading Language, naga's compile target when the backend is
Metal. A WGSL error can be an naga front-end error or an MSL back-end error, and
they are reported differently.

**bytemuck** — the crate that turns a plain `#[repr(C)]` struct into something
that can be uploaded to the GPU without a copy, via the `Pod` trait. The catch
is that `Pod` is an unsafe promise: if the struct has a padding hole, bytemuck
will happily read indeterminate bytes. This project handles that with an
explicit `_pad` field rather than relying on the struct having none.

**DMA** — Direct Memory Access: a device reading memory without the CPU being
involved. Relevant here because the CPU-to-GPU upload of `Uniforms` is a DMA
transfer that happens on the queue, and because wgpu's readback path (for
timestamp queries and screenshots) is the same mechanism in reverse.

## Project-specific usages

These are the places where this project means something narrower than the general
word.

**"iterate"** — the `Uniforms::iterate` flag, and the thing it is *not* easy to
describe accurately. Non-zero, it applies the selected function `max_iter`
times instead of once: `z -> f(z) -> f(f(z)) -> ...`. So "iterated view of
`sin`" is `sin(sin(sin(...(z))))` with the count set by `max_iter`, not
`sin^n(z)` in the trigonometric sense. It is the same machinery that draws a
Julia set, which is why the control is so important. Whether it should be a
count rather than a flag — so you could show `f^5(z)` for its own sake — is
undecided; see [TODO.md](../TODO.md).

**"storage texture"** — always singular in this project, always one
`rgba8unorm` texture the size of the window surface, with usages
`STORAGE_BINDING | TEXTURE_BINDING | COPY_DST`. It is recreated on resize.

**"the plane"** — the complex plane as currently framed by the camera. `scale`
is its half-height, `center` is its middle.

**"hue is red at the negative real axis"** — a convention, not a fact. The
shader adds a fixed `+0.5` turn to the argument before taking the hue, which
puts the positive real axis at cyan and the negative at red. Red reads as "the
barrier". It is hardcoded in the shader and is not a `Uniforms` field, which is
an implicit ABI and is flagged as such in [TODO.md](../TODO.md).

## Related

- [ABI.md](ABI.md) — the colour scheme and the buffer, written out exactly
- [ARCHITECTURE.md](ARCHITECTURE.md) — the pipeline, for the stack half
- [../AGENTS.md](../AGENTS.md) — the rules
