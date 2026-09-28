# ENVIRONMENT

Everything about the machine, the toolchain, and the build that a future
session needs before it touches anything. Measured 2026-09-28 on this
repository; re-measure rather than trusting a number that has quietly gone stale.

## The one thing that will waste your time if you do not read this

**There are two Rust installations on this machine and the one that is first on
the default `PATH` is broken.**

`/usr/local/bin/rustc` is a Homebrew Rust, 1.97.1, symlinked to
`/usr/local/Cellar/rust/1.97.1/bin/rustc`. Running it fails immediately:

```
$ /usr/local/bin/rustc --version
dyld[65715]: Symbol not found: __ZN4llvm17OptimizationLevel2O0E
  Referenced from: <...> /usr/local/Cellar/rust/1.97.1/lib/librustc_driver-fcfdf5fdc8f046f0.dylib
  Expected in:     <...> /usr/local/Cellar/llvm/23.1.2/lib/libLLVM.23.1.dylib
```

It is a dylib linkage failure between `librustc_driver` and Homebrew's LLVM, not
anything about this project. **You cannot fix it**, because:

- **`brew` hangs on this machine.** There is a `brew.rb install rustup` process
  that has been stuck for hours:

  ```
  $ ps aux | grep brew
  derekvanee  42524  0.0  0.1 34686732  6360  ??  S  12:30am  0:11.84  .../ruby -W1 --disable=gems,rubyopt /usr/local/Homebrew/Library/Homebrew/brew.rb install rustup
  ```

  It is sleeping, not working. Do not wait on it and do not run `brew` for any
  reason, including to "just check the version". It has been observed to sit
  there for the better part of a day.

### The working toolchain

A rustup-managed stable toolchain exists at
`~/.rustup/toolchains/stable-x86_64-apple-darwin/`, and its binaries are already
symlinked into `~/.cargo/bin/`:

```
~/.cargo/bin/rustc   -> ~/.rustup/toolchains/stable-x86_64-apple-darwin/bin/rustc
~/.cargo/bin/cargo   -> ~/.rustup/toolchains/stable-x86_64-apple-darwin/bin/cargo
~/.cargo/bin/rustfmt -> ...
~/.cargo/bin/clippy-driver, cargo-fmt, cargo-clippy, rustdoc
```

These are direct symlinks to the toolchain binaries, **not** rustup's
multiplexing shims, and the `rustup` binary itself is not on the `PATH` at all.
There is therefore no rustup proxy layer to go wrong, and no way to switch
toolchains with `rustup toolchain`. If a second toolchain is ever needed, it has
to be added by hand.

**Every command must therefore carry the `PATH` prefix.** Either per command:

```sh
PATH="$HOME/.cargo/bin:$PATH" cargo check --release
```

or once per shell, which is what you want:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
```

Confirm before doing anything else:

```
$ PATH="$HOME/.cargo/bin:$PATH" cargo --version
cargo 1.97.1 (c980f4866 2026-06-30)

$ PATH="$HOME/.cargo/bin:$PATH" rustc --version
rustc 1.97.1 (8bab26f4f 2026-07-14)
```

If `cargo --version` prints the Homebrew one, or errors with the `librustc_driver`
symbol, the prefix is missing. Do not work around it by invoking the full path;
set the `PATH`, because every subcommand needs it too.

## Machine

| | |
|---|---|
| Architecture | `x86_64` (`uname -m`) — **Intel, not Apple Silicon** |
| OS | macOS 14.8.8, build 23J620 (Sonoma) |
| GPU | Intel UHD Graphics 617, integrated, Metal 3 support, 1536 MB shared VRAM |
| Display | Built-in Retina LCD, 2560 x 1600, scale factor 2.0 |
| Toolchain triple | `stable-x86_64-apple-darwin` |
| Shell | zsh |

The Retina display matters more than it looks. A 1.0-scale-factor assumption
about `resolution` versus egui's logical-point input space is a real, currently
unresolved bug class; see the DPI note in [TODO.md](../TODO.md).

The GPU is the **Intel** one. Apple Silicon GPUs are stricter about `f32`
reassociation, so numerical results that look right here may band there. Nothing
in the codebase may assume x86: no hand-written assembly, no architecture-specific
SIMD, no x86-only linkage, and the binary must eventually be universal.

## Dependencies

From `Cargo.toml`:

| Crate | Version | Why it is here |
|---|---|---|
| `bytemuck` | 1.25.2 | `Pod` derive on `Uniforms`, so the 64-byte block uploads with no copy |
| `eframe` | 0.36.2 | the window, the event loop, the render surface |
| `egui` | 0.36.2 | immediate-mode GUI |
| `egui_dock` | 0.21.1 | docking tab container for the control panel |
| `wgpu` | 30.0.1 | the GPU API: device, compute pipeline, render pipeline, buffers |

Pinned transitive versions worth knowing, because they are where API surprises
come from:

| Crate | Version | Note |
|---|---|---|
| `naga` | 30.0.1 | WGSL front end. The source of shader validation errors. |
| `winit` | 0.30.13 | the windowing layer under eframe |
| `raw-window-handle` | via wgpu 30 | the window/surface seam |
| `image` | 0.25.10 | pulled in by egui for image loading |
| `arboard` | 3.6.1 | pulled in by egui for clipboard access |

`Cargo.lock` pins **423 crates** in total. Edition is **2024**, rustc 1.97.1.

### Adding a dependency

Do not, without a reason written in the commit message and added to this file.
The reasoning is in rule 7 of [../AGENTS.md](../AGENTS.md): 423 crates already,
a 21-minute cold build, and every addition is build time, binary size, and a
licence to track. The realistic candidate is a benchmark harness (`criterion`),
because the alternative is a hand-rolled one that might be wrong.

## Build times

Measurements from this repository, 2026-09-28:

| Operation | Time |
|---|---|
| `cargo check --release`, this crate only, 423 deps already built | **~35 s** (measured) |
| Cold build of the full dependency tree, from an empty `target/` | **~21 min** (reported by the milestone-1 build lane; not re-timed, because re-timing it means deleting a 600 MB `target/`) |
| `cargo --version`, `rustc --version` | instant |

`target/` is currently **598 MB** and holds 515 built artifacts. It is
gitignored — in fact `.gitignore` contains the line `/target` twice — and must
never be committed.

**Do not conclude a build is hung before 25 minutes on a cold run.** On a warm
run, 35 seconds of a `cargo check` with no output is normal; cargo buffers its
progress through a pipe, so with output redirected you will see nothing at all
until it finishes. Pipe to `tail` and the first thing you see is the error.

### Concurrent builds

Several lanes work on this repository at once and all of them run cargo. If you
see

```
Blocking waiting for file lock on build directory
Blocking waiting for file lock on package cache
```

another cargo process owns `target/`. **That is not a hang.** Check with
`ps aux | grep cargo` and either wait or work on something that does not need
the compiler. Killing someone else's build to get your own going is worse than
waiting, and cargo's own queue is fair.

Note also that a concurrent `cargo build` (debug) and `cargo check --release`
will repeatedly evict each other's fingerprints, so a concurrent session can
make your build look far slower than it is. Re-measure alone before drawing
conclusions.

## Getting a GPU adapter on macOS

wgpu 30 on macOS talks to Metal and only Metal. There is no Vulkan and no GL
fallback to configure. The sequence a renderer needs, in the shape wgpu 30
wants it (note: `Instance::new` takes its descriptor **by value** in wgpu 30,
and `InstanceDescriptor` does **not** implement `Default` — this is a live
compile error in `src/renderer.rs` at the moment):

```rust
let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
    backends: wgpu::Backends::all(),
    flags: wgpu::InstanceFlags::default(),
    dx12_shader_compiler: wgpu::Dx12Compiler::default(),
    gles_minor_version: wgpu::Gles3MinorVersion::default(),
});

let adapter = instance
    .request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: Some(&surface),
    })
    .await
    .map_err(|e| format!("no Metal adapter: {e}"))?;

let (device, queue) = adapter
    .request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
    })
    .await
    .map_err(|e| format!("no device: {e}"))?;
```

Notes that will save a future session a confusing afternoon:

- **`required_limits: adapter.limits()` is the safe default.** The Intel UHD 617
  reports a maximum texture dimension of 8192. A 5K display at 2x would exceed
  the maximum 2D texture dimension at *some* resolutions, and a Retina 2560x1600
  window is 2560x1600 physical pixels, which is fine. The number to check if
  textures start failing to create is `max_texture_dimension_2d`.
- **A failed adapter request is not a programming error.** The window should
  still open and say that no adapter was available; it should not panic and take
  the process down before the user has read anything.
- **A headless smoke test is possible** on this machine by dropping
  `compatible_surface` and `force_fallback_adapter: true`. macOS has one Metal
  driver, so the fallback adapter is the same hardware; this gives you a device
  without a window, which is enough to validate shader compilation. A
  `wgslcheck` scratch crate for exactly this purpose already exists in the
  opencode temp directory.
- **naga compiles WGSL to MSL for this backend.** A shader can therefore fail at
  the MSL stage with a message that mentions neither WGSL nor your code. When
  validating a shader by hand, expect Metal-flavoured errors.

## The commands

```sh
cd /Users/derekvanee/Documents/jason-complex-functions-desktop
export PATH="$HOME/.cargo/bin:$PATH"

cargo check --release            # fastest signal; ~35 s warm
cargo build --release            # -> target/release/steel-pulse
cargo test                       # unit tests, no GPU, no window required
cargo test --release uniforms    # just the ABI layout tests
cargo run --release              # needs a window and a GPU
cargo fmt                        # rustfmt 1.9.0-stable
cargo clippy --release -- -D warnings
```

`cargo fmt` and `cargo clippy` come from the same toolchain and are already
symlinked into `~/.cargo/bin/`.

The tests must pass **with no window and no GPU**. The ABI layout tests are pure
arithmetic and nothing in the test suite may require a display adapter; that is
what makes the suite usable in CI, which does not exist yet.

## Windows

**Unresolved. Nothing has been built or tested for Windows, and cross-compiling
from macOS to Windows has not been attempted.** No target is installed, no
`cargo` target has been added, and there is no code in the crate that has been
run on anything but Metal.

The two viable routes:

| Route | Requirement | Assessment |
|---|---|---|
| `x86_64-pc-windows-msvc` | a **Windows host** | The MSVC ABI cannot be cross-linked from macOS. This is the route a Windows CI runner or a Windows build machine would use. Most likely to produce a binary that actually works. |
| `x86_64-pc-windows-gnu` | mingw-w64 on this machine | Can cross-compile from macOS, and `wgpu` will build against `vulkan`/`dx12`, but a `wgpu` + winit + mingw cross-link is a genuine risk and the result would be entirely untested. |

Which one is intended is **undecided** and should be decided before any
installer work starts. See [ROADMAP.md](ROADMAP.md) phase 3.

macOS distribution is equally unresolved: there is no `.app` bundle, no code
signing, and no universal binary. A release build is a bare executable in
`target/release/`.

## CI

**There is none.** No `.github/`, no config file, no badge. Nothing in this
repository is validated anywhere but this one macOS x86_64 machine. The tests
that would be worth putting in CI — the `Uniforms` offset assertions in
[ABI.md](ABI.md) and the `f64` reference tests — are pure computation and need
no GPU, so a CI job is well within reach. The obstacle is the 21-minute cold
dependency build, not the work.

## Related

- [../AGENTS.md](../AGENTS.md) — the rules
- [TODO.md](../TODO.md) — the outstanding cross-compile and CI notes
- [SESSION-LOG.md](SESSION-LOG.md) — the toolchain breakage as it was found
