//! GPU renderer: the wgpu device, the per-pixel complex-function compute
//! pipeline, and the fullscreen blit that presents the result to the window.
//!
//! # The two-pass shape, and why
//!
//! Nothing in this program draws geometry in the usual sense. Every pixel is
//! the independent evaluation of a complex function, so the natural mapping is
//! *one invocation per pixel* — which is what a compute shader is for, and what
//! a fragment shader is bad at (fragment shaders are latency-bound on
//! divergence; a million escape-time iterations per pixel is precisely the
//! workload that makes every lane in a SIMD quad wait for the slowest one).
//! So the frame is:
//!
//! 1. a **compute pass** evaluates `f(z)` for every texel of an off-screen
//!    `rgba8unorm` storage texture;
//! 2. a **render pass** samples that texture through a fullscreen triangle and
//!    writes it into the swap-chain image.
//!
//! The split exists for one concrete reason: `texture_storage_2d<..., write>`
//! is **write-only** by the WebGPU spec, so the compute pass physically cannot
//! read the image back, and a plain `texture_2d<f32>` binding — which is what a
//! fragment shader can sample — is not legal in a compute shader at all in the
//! core spec. Two passes, two binding tables, one texel-for-texel blit.
//!
//! # The WGSL contract (READ THIS BEFORE EDITING THE SHADERS)
//!
//! The Rust side declares explicit [`wgpu::BindGroupLayout`]s and hands the
//! shader compiler an explicit `PipelineLayout`, so a mismatch with the WGSL is
//! a **hard validation error at pipeline-creation time**, not a silently wrong
//! picture. The layout the shaders must match exactly is:
//!
//! | Pass | `@group` | `@binding` | WGSL declaration |
//! |---|---|---|---|
//! | compute | `0` | `0` | `var<uniform> uniforms: Uniforms;` |
//! | compute | `0` | `1` | `var output_texture: texture_storage_2d<rgba8unorm, write>;` |
//! | blit | `0` | `0` | `var input_texture: texture_2d<f32>;` |
//! | blit | `0` | `1` | `var linear_sampler: sampler;` |
//!
//! The blit pass deliberately binds **no** uniform buffer. The fullscreen
//! triangle covers the viewport, so the shader needs nothing but its own
//! position, which the rasteriser supplies. The two passes therefore have
//! different binding tables at group 0, which is one more reason they live in
//! separate shader modules: a pipeline layout is shared by every entry point in
//! one module, so combining them would force a dummy binding into one of them.
//!
//! The same applies to the entry-point names in the constants below. If a
//! shader author renames `main` to something else, update the constant in this
//! file, or pipeline creation will fail loudly rather than silently.
//!
//! The Y-axis is flipped in *both* shaders, and the two flips cancel — do not
//! "fix" either one. The kernel maps texture row 0 to `+scale` on the imaginary
//! axis (`im = center.y + (1.0 - uv.y * 2.0) * scale`), and the blit maps
//! texture `v = 0` to the top of the screen (`uv.y = (1.0 - clip.y) * 0.5`).
//! Remove either flip and every plot in the program is mirrored.
//!
//! # Async without a runtime
//!
//! [`Renderer::new`] is `async` because wgpu's adapter and device requests are
//! futures. It deliberately does **not** block on them: `pollster` and
//! `futures-lite` appear in `Cargo.lock` only as transitive dependencies of
//! `eframe`, and Rust does not let you `use` a crate that is not in your own
//! `[dependencies]`. Adding one would mean editing `Cargo.toml`, which this
//! module does not own. The caller is expected to have an executor available;
//! that keeps the renderer free of any opinion about *how* the frame loop is
//! driven.

use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::uniforms::{Uniforms, UNIFORM_BUFFER_SIZE};

// ---------------------------------------------------------------------------
// Shader sources and the names inside them.
//
// These constants are the *only* places in the program that know where the
// shaders live or what they are called. Both WGSL files are embedded with
// `include_str!` rather than read at runtime: a missing or renamed shader then
// fails the build with a file-not-found error instead of producing a blank
// window that is painful to diagnose.
//
// NOTE FOR INTEGRATORS: the WGSL files are authored in a separate lane. If
// either is renamed, change `COMPUTE_SHADER_PATH` / `BLIT_SHADER_PATH` *and*
// the `include_str!` literals below. The macro requires a string literal, so
// the constant and the literal have to be kept in step by hand; the constant
// exists to record the path in one obvious, greppable place.
// ---------------------------------------------------------------------------

/// Where the domain-colouring compute shader lives, relative to `Cargo.toml`.
const COMPUTE_SHADER_PATH: &str = "shaders/domain_coloring.wgsl";

/// Where the fullscreen blit shader lives, relative to `Cargo.toml`.
const BLIT_SHADER_PATH: &str = "shaders/blit.wgsl";

/// Source of the compute kernel, embedded at compile time.
const COMPUTE_SHADER_SRC: &str = include_str!("../shaders/domain_coloring.wgsl");

/// Source of the blit vertex/fragment pair, embedded at compile time.
const BLIT_SHADER_SRC: &str = include_str!("../shaders/blit.wgsl");

/// `@compute` entry point in [`COMPUTE_SHADER_SRC`].
///
/// This is passed explicitly rather than left to wgpu's "exactly one compute
/// stage in the module" inference, so that adding a second experimental kernel
/// to the same file later cannot silently change which one we run.
const COMPUTE_ENTRY_POINT: &str = "main";

/// `@vertex` entry point in [`BLIT_SHADER_SRC`].
const BLIT_VERTEX_ENTRY_POINT: &str = "vs_main";

/// `@fragment` entry point in [`BLIT_SHADER_SRC`].
const BLIT_FRAGMENT_ENTRY_POINT: &str = "fs_main";

/// Edge length of the compute workgroup square, in invocations.
///
/// **This number is a contract, not a tuning knob.** The workgroup count
/// dispatched in [`Renderer::render`] is `ceil(size / WORKGROUP_SIZE)` in each
/// axis, and the kernel guards every invocation with a bounds check, so a
/// partial final workgroup writes only the pixels that exist. It must match
/// `@workgroup_size(x, y)` in [`COMPUTE_SHADER_SRC`] — currently `(8, 8)` — and
/// wgpu cannot check that for us. A mismatch here is a disagreement about how
/// many invocations a workgroup holds, which is an out-of-bounds write, not a
/// compile error.
const WORKGROUP_SIZE: u32 = 8;

/// Format of the intermediate image the compute pass writes.
///
/// `Rgba8Unorm` is one of the formats the WebGPU spec *guarantees* supports
/// `STORAGE_BINDING` with write-only access (the `Bgra8Unorm` case is the one
/// that needs a feature flag; `Rgba8Unorm` does not), so this needs no optional
/// device feature and cannot fail on a conformant backend. The WGSL side must
/// spell it `texture_storage_2d<rgba8unorm, write>`.
const STORAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Version string shown in the UI's backend line.
const WGPU_VERSION_LABEL: &str = "30.0.1";

/// Vertex count of the fullscreen triangle.
///
/// Three, not four: a single triangle that overshoots the viewport covers every
/// pixel with no diagonal seam and no index buffer, and the GPU clips the
/// overhang for free. The shader derives clip-space positions from
/// `vertex_index` alone, so there is no vertex buffer at all.
const FULLSCREEN_TRIANGLE_VERTICES: u32 = 3;

/// Frames allowed between acquiring a swap-chain image and presenting it.
///
/// Capped at 2 rather than 1. With 1 the CPU and GPU are forced into lockstep —
/// the CPU cannot record frame N+1 until the GPU has finished N, which halves
/// throughput and makes pan/zoom feel mushy. With 3 or more, extra frames of
/// input latency become perceptible. 2 is the balance point, and it is also what
/// wgpu's own `get_default_config` picks. On Metal this becomes
/// `CAMetalLayer.maximumDrawableCount = 3`.
const DESIRED_MAX_FRAME_LATENCY: u32 = 2;

/// Bytes occupied by a single resolved timestamp query.
///
/// `wgpu::QUERY_SIZE`; re-declared through a named constant so the readback
/// buffer size below reads as arithmetic rather than as a magic number.
const TIMESTAMP_QUERY_BYTES: u64 = wgpu::QUERY_SIZE as u64;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Everything that can go wrong while bringing the renderer up.
///
/// Each variant carries a human-readable explanation rather than a bare wgpu
/// enum variant. A user who sees this in a window title needs to learn *what to
/// do* — "your GPU is too old" is actionable, `RequestAdapterError::NotFound`
/// is not.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RendererError {
    /// No adapter satisfied the request. By far the most common failure on a
    /// developer machine, and almost never a bug in this program.
    NoAdapter {
        /// Human-readable cause, naming the likely fix.
        cause: String,
    },
    /// The surface cannot be used with the adapter we picked, or reports no
    /// usable configuration. Usually means the surface was created from a
    /// window handle the chosen backend does not support.
    UnsupportedSurface {
        /// Human-readable cause.
        cause: String,
    },
    /// The adapter existed but the device request was refused, typically
    /// because a requested limit or feature is out of range.
    DeviceRequest {
        /// Human-readable cause, including wgpu's own explanation.
        cause: String,
    },
}

impl fmt::Display for RendererError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter { cause } => write!(
                f,
                "could not find a usable GPU adapter ({cause}). \
                 This usually means the machine has no GPU that wgpu can drive, \
                 that a driver is too old for wgpu {WGPU_VERSION_LABEL}, or that \
                 the process is sandboxed without graphics access. \
                 Try updating the graphics driver, or run on a machine with a \
                 discrete or integrated GPU exposed to the window server."
            ),
            Self::UnsupportedSurface { cause } => write!(
                f,
                "the window surface is not usable with the selected GPU adapter \
                 ({cause}). The surface was probably created from a window \
                 handle the active wgpu backend does not support; recreating the \
                 surface after the device is known usually fixes it."
            ),
            Self::DeviceRequest { cause } => write!(
                f,
                "the GPU adapter was found but refused to create a device \
                 ({cause}). This usually means the adapter does not support the \
                 requested limits or optional features."
            ),
        }
    }
}

impl std::error::Error for RendererError {}

// ---------------------------------------------------------------------------
// Optional GPU timing
// ---------------------------------------------------------------------------

/// Mutable state shared between the frame loop and the `map_async` callback
/// wgpu invokes once the GPU has finished copying query results.
///
/// It lives behind an [`Arc`] + [`Mutex`] because the mapping callback is
/// `FnOnce(..) + Send + 'static` — it cannot borrow from `self`. The [`Mutex`]
/// is never held across a GPU call, so contention is not a concern; it is only
/// ever locked for a handful of field assignments.
#[derive(Debug, Default)]
struct TimestampState {
    /// `true` while a staging-buffer mapping is outstanding. Prevents a second
    /// `map_async` on a buffer that is already mapped, which wgpu rejects.
    in_flight: bool,
    /// Most recent successfully decoded compute-pass duration.
    latest: Option<Duration>,
    /// How many readbacks failed or produced nonsense. Surfaced so that a
    /// silently-frozen telemetry readout is distinguishable from a GPU that is
    /// genuinely never finishing.
    dropped: u64,
}

/// Locks the timestamp state, recovering from poisoning instead of panicking.
///
/// A panic inside the mapping callback would poison this lock. Everything behind
/// it is a `bool`, an `Option<Duration>` and a `u64` — all independently valid
/// after a panic — so it is always safe to keep using. Propagating the poison
/// would turn a telemetry glitch into a dead render loop, which is a
/// disproportionate outcome for a value nothing blocks on.
fn lock_timestamps(state: &Mutex<TimestampState>) -> std::sync::MutexGuard<'_, TimestampState> {
    state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The intermediate image and the two bind groups that reference it.
///
/// Bundled into one value because the objects are only ever valid together:
/// recreating the texture without recreating the bind groups leaves the pipelines
/// bound to a destroyed view.
struct StorageTarget {
    /// The `rgba8unorm` image the compute pass writes.
    texture: wgpu::Texture,
    /// View of [`StorageTarget::texture`] narrowed to `STORAGE_BINDING`, for the
    /// compute pass's write-only image.
    storage_view: wgpu::TextureView,
    /// View of [`StorageTarget::texture`] narrowed to `TEXTURE_BINDING`, for the
    /// blit pass's `texture_2d<f32>`.
    ///
    /// Two views of one image rather than one shared view. They must exist
    /// separately because a `texture_2d<f32>` and a
    /// `texture_storage_2d<..., write>` are different binding types, and
    /// narrowing each view to the single usage it is bound with makes that
    /// distinction impossible to get wrong by accident.
    sampled_view: wgpu::TextureView,
    /// Uniform + storage view, for the compute pass.
    compute_bind_group: wgpu::BindGroup,
    /// Sampled view + sampler, for the blit pass.
    blit_bind_group: wgpu::BindGroup,
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

/// Owns the wgpu device, the two pipelines that make up a frame, and every
/// resource they bind.
///
/// # Ownership and lifetime
///
/// The `Surface` is taken as an `Arc` and held for as long as the renderer
/// lives. wgpu's `Surface<'window>` borrows the platform window handle, and the
/// handle must outlive every acquire; keeping the `Arc` in a `'static` renderer
/// makes that the caller's problem exactly once, at surface creation, instead of
/// a constraint on every subsequent frame.
///
/// # Threading
///
/// `Renderer` is `Send + Sync` — asserted at compile time at the bottom of this
/// file, not merely claimed — but it is *not* internally synchronised. Treat it
/// as owned by whichever thread drives the frame loop, which in this program is
/// the egui main thread. The only interior mutability is the timestamp state,
/// which is genuinely shared with a wgpu callback and is why it is the one
/// place a `Mutex` appears.
pub struct Renderer {
    /// The logical device. Every pipeline, buffer and texture hangs off this.
    device: wgpu::Device,
    /// The queue all uploads and submissions go through.
    queue: wgpu::Queue,
    /// What the adapter told us about itself; the source of the UI's backend
    /// line. Captured at construction because the `Adapter` itself is not kept.
    adapter_info: wgpu::AdapterInfo,
    /// The window surface we present into. Shared with the caller, which may
    /// need it to reconfigure on its own schedule.
    surface: Arc<wgpu::Surface<'static>>,
    /// Live swap-chain configuration. Mutated by [`Renderer::resize`] and
    /// re-applied to the surface whenever the size actually changes.
    config: wgpu::SurfaceConfiguration,
    /// Size of the intermediate image, `(width, height)`, in physical pixels.
    ///
    /// Clamped to at least `1x1`; see [`Renderer::resize`] for why a zero is
    /// fatal rather than merely useless.
    size: (u32, u32),

    /// Evaluates the complex function, one invocation per texel.
    compute_pipeline: wgpu::ComputePipeline,
    /// Explicit layout for [`Renderer::compute_pipeline`]; kept alive because a
    /// `BindGroup` may not outlive the layout it was created against.
    compute_layout: wgpu::BindGroupLayout,
    /// Samples the intermediate image onto the swap-chain image.
    blit_pipeline: wgpu::RenderPipeline,
    /// Explicit layout for [`Renderer::blit_pipeline`].
    blit_layout: wgpu::BindGroupLayout,

    /// 64 bytes of [`Uniforms`], re-uploaded once per frame.
    uniform_buffer: wgpu::Buffer,

    /// The `rgba8unorm` image the compute pass writes. Recreated on resize.
    storage_texture: wgpu::Texture,
    /// Storage-narrowed view, for the compute pass's write-only image.
    storage_view: wgpu::TextureView,
    /// Sampled-narrowed view, for the blit pass's `texture_2d<f32>`.
    sampled_view: wgpu::TextureView,
    /// Storage view + uniform, for the compute pass.
    compute_bind_group: wgpu::BindGroup,
    /// Sampled view + sampler, for the blit pass.
    blit_bind_group: wgpu::BindGroup,
    /// Linear, clamped sampler. Linear filtering is what makes the blit
    /// resample gracefully if the swap-chain size and the intermediate size ever
    /// disagree by a pixel; clamping stops the fullscreen triangle's overshoot
    /// from wrapping to the opposite edge.
    sampler: wgpu::Sampler,

    /// Two timestamp slots, present only after [`Renderer::enable_timestamps`]
    /// succeeds. Slot 0 is the start of the compute pass, slot 1 the end.
    timestamp_query_set: Option<wgpu::QuerySet>,
    /// Destination for `resolve_query_set` and the CPU-side map of those
    /// results. Created together with the query set.
    timestamp_readback: Option<wgpu::Buffer>,
    /// Shared with the `map_async` callback; see [`TimestampState`].
    timestamp_state: Arc<Mutex<TimestampState>>,
}

impl Renderer {
    /// Requests an adapter and device, configures the surface, and builds both
    /// pipelines.
    ///
    /// `window` is the [`egui::ViewportBuilder`] the host window was created
    /// from; it is consulted only for the initial logical size, which gives the
    /// swap chain a sane geometry before the first resize event arrives. The
    /// `surface` is expected to have been created from the same window.
    ///
    /// # Errors
    ///
    /// Returns [`RendererError::NoAdapter`] if no GPU is usable,
    /// [`RendererError::UnsupportedSurface`] if the surface and adapter are
    /// incompatible, and [`RendererError::DeviceRequest`] if the adapter
    /// refuses the device. No `unwrap` is reached: every fallible step below is
    /// either a `Result` or a documented non-fallible wgpu operation.
    pub async fn new(
        window: &egui::ViewportBuilder,
        surface: Arc<wgpu::Surface<'static>>,
    ) -> Result<Self, RendererError> {
        // `Backends::all()` rather than naming Metal/Vulkan/DX12: wgpu already
        // filters to backends that were compiled in for this target, so this
        // stays correct on a machine we did not anticipate.
        //
        // `display: None` is right for us even though winit normally supplies
        // one. The only consumer of a display handle is GLES/Wayland surface
        // creation, and eframe hands us an already-created `wgpu::Surface`
        // instead of asking us to build one. On Metal, Vulkan and DX12 the field
        // is documented as unused.
        let descriptor = wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        };
        // `with_env` lets `WGPU_BACKEND` / `WGPU_ADAPTER_NAME` override the
        // defaults, so a bug report about the wrong backend can be reproduced
        // on a different one without a rebuild.
        let instance = wgpu::Instance::new(descriptor.with_env());

        // `compatible_surface` is what lets wgpu reject adapters that cannot
        // present to *this* window. Passing it is the difference between a
        // clean "no adapter" error and a crash later inside the swap chain.
        // The surface is only borrowed for the duration of the request, so the
        // `Arc` is not captured.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&*surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| RendererError::NoAdapter {
                cause: e.to_string(),
            })?;

        let adapter_info = adapter.get_info();

        // Ask for `TIMESTAMP_QUERY` *only* if this adapter actually has it.
        // Device features are immutable after `request_device`, so this is the
        // one and only chance to opt in; conversely, requesting a feature the
        // adapter lacks makes the whole device request fail, which would take
        // the renderer down on a machine that could otherwise run fine. Gate on
        // the capability and the request stays safe everywhere.
        let wants_timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let required_features = if wants_timestamps {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };

        // `request_device` hands back `(Device, Queue)` in wgpu 30; there is no
        // separate `adapter.get_queue()` any more.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("steel-pulse:device"),
                required_features,
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                // Tracing writes every wgpu call to a directory; off unless
                // someone is actively debugging the driver interface.
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| RendererError::DeviceRequest {
                cause: e.to_string(),
            })?;

        // Install an uncaptured-error handler. wgpu's default already panics, but
        // a panic from deep inside a render pass takes the window down with it
        // and prints a backtrace nobody reads. Routing validation and
        // out-of-memory errors to stderr keeps the app alive and makes the
        // message findable in a log.
        device.on_uncaptured_error(std::sync::Arc::new(|error| {
            eprintln!("steel-pulse: wgpu error: {error}");
        }));

        let capabilities = surface.get_capabilities(&adapter);
        let surface_format =
            *capabilities
                .formats
                .first()
                .ok_or_else(|| RendererError::UnsupportedSurface {
                    cause: "the surface reports no presentable texture formats".to_owned(),
                })?;

        // The requested logical size is only a starting guess: a window that has
        // not been shown yet, or one that is minimised, can report zero. Clamp
        // before it reaches `Surface::configure`, which documents a zero
        // dimension as a panic.
        let initial = window
            .inner_size
            .unwrap_or(egui::emath::Vec2::new(1280.0, 720.0));
        // `as u32` on a negative or NaN float saturates to 0 in Rust, so the
        // `max(1)` is doing real work for a malformed builder, not just for a
        // minimised window.
        let width = (initial.x as u32).max(1);
        let height = (initial.y as u32).max(1);

        let config = wgpu::SurfaceConfiguration {
            // The blit pass renders into the swap-chain image, and nothing
            // else. Requesting COPY_SRC/COPY_DST here would make wgpu build a
            // slower, copy-capable swap chain for no benefit.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width,
            height,
            // `AutoVsync` rather than `Fifo`: wgpu documents `Immediate` and
            // `Mailbox` as *crashing* when unsupported, and `AutoVsync`
            // degrades gracefully to `Fifo` instead. For an interactive plotter
            // we want vsync (no tearing, no spinning the CPU flat out when the
            // plot is static), and we want the degrade path rather than a hard
            // failure on an exotic compositor.
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: DESIRED_MAX_FRAME_LATENCY,
            // `Auto` lets the OS pick: opaque on a normal desktop window, and
            // premultiplied where the compositor wants that. The blit writes
            // alpha 1.0 in the shader, so the window is fully opaque either
            // way and the choice costs nothing.
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            // The wgpu 30 alias is `SurfaceConfiguration<Vec<TextureFormat>>`;
            // an owned empty vec means "no alternate srgb views".
            view_formats: Vec::new(),
        };

        surface.configure(&device, &config);

        // -------------------------------------------------------------------
        // Bind group layouts. Explicit, because an explicit layout turns a
        // shader-side mistake into a pipeline-creation error instead of a
        // picture that is subtly wrong.
        // -------------------------------------------------------------------

        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("steel-pulse:compute:bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    // COMPUTE, not VERTEX_FRAGMENT: the uniform is read by the
                    // kernel and by nothing else.
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        // Pinning the size lets wgpu validate against the
                        // WGSL struct's minimum binding size at layout time.
                        min_binding_size: wgpu::BufferSize::new(UNIFORM_BUFFER_SIZE),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: STORAGE_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("steel-pulse:blit:bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        // `filterable: true` pairs with the linear sampler
                        // below and with a plain unorm format, which is always
                        // filterable. Getting this wrong while using a
                        // `Filtering` sampler is a validation error.
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // -------------------------------------------------------------------
        // Pipelines.
        //
        // The shader modules are created with `create_shader_module`, which in
        // wgpu 30 runs Naga's frontend and reports WGSL syntax errors through
        // the uncaptured-error handler installed above. Note also that in this
        // version `compilation_options` lives on the per-stage state, not on the
        // pipeline descriptor, and the `cache` field is `None` (no on-disk
        // pipeline cache: it would need a Cargo feature we do not enable).
        // -------------------------------------------------------------------

        let compute_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("steel-pulse:domain_coloring"),
            source: wgpu::ShaderSource::Wgsl(COMPUTE_SHADER_SRC.into()),
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("steel-pulse:compute:pl"),
                bind_group_layouts: &[Some(&compute_layout)],
                // No push-constant-style immediate data; the uniforms go
                // through the uniform buffer so they survive across passes.
                immediate_size: 0,
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("steel-pulse:domain_coloring:compute"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_module,
            entry_point: Some(COMPUTE_ENTRY_POINT),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let blit_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("steel-pulse:blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER_SRC.into()),
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("steel-pulse:blit:pl"),
            bind_group_layouts: &[Some(&blit_layout)],
            immediate_size: 0,
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("steel-pulse:blit:render"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_module,
                entry_point: Some(BLIT_VERTEX_ENTRY_POINT),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                // No vertex buffers: positions come from `vertex_index`.
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // The triangle is wound counter-clockwise in clip space but the
                // projection flips Y, so it arrives clockwise. Culling is off
                // anyway, so winding is irrelevant — but being explicit means
                // the back-face cull setting cannot drift.
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            // No depth buffer: this is a single full-screen draw and there is
            // nothing to occlude.
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &blit_module,
                entry_point: Some(BLIT_FRAGMENT_ENTRY_POINT),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    // Opaque output. Alpha is meaningless for a plot, and
                    // `CompositeAlphaMode::Auto` will composite the window as
                    // opaque regardless, so blending would only cost bandwidth.
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        // -------------------------------------------------------------------
        // Per-frame resources.
        // -------------------------------------------------------------------

        // `mapped_at_creation` would let us seed it without a queue write, but
        // an all-zero uniform is a perfectly good starting state and the first
        // frame overwrites it anyway. COPY_DST is the only usage strictly
        // required, and the narrowest one that validates.
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("steel-pulse:uniforms"),
            size: UNIFORM_BUFFER_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("steel-pulse:blit:sampler"),
            // Clamp on every axis: the fullscreen triangle samples slightly
            // outside `[0, 1]` at its third vertex, and a repeat or mirror
            // address mode would drag the far edge of the plot into view.
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let target = build_storage_target(
            &device,
            &compute_layout,
            &blit_layout,
            &uniform_buffer,
            &sampler,
            width,
            height,
        );

        Ok(Self {
            device,
            queue,
            adapter_info,
            surface,
            config,
            size: (width, height),
            compute_pipeline,
            compute_layout,
            blit_pipeline,
            blit_layout,
            uniform_buffer,
            storage_texture: target.texture,
            storage_view: target.storage_view,
            sampled_view: target.sampled_view,
            compute_bind_group: target.compute_bind_group,
            blit_bind_group: target.blit_bind_group,
            sampler,
            timestamp_query_set: None,
            timestamp_readback: None,
            timestamp_state: Arc::new(Mutex::new(TimestampState::default())),
        })
    }

    /// Notifies the renderer that the drawable area changed size.
    ///
    /// Reconfigures the surface and, when the size genuinely changed,
    /// **recreates the storage texture and both bind groups**. That
    /// recreation is not optional: a bind group captures the view it was built
    /// from, so once the texture is replaced the old bind groups reference a
    /// texture that no longer exists. Recreating them is the only correct
    /// response, and it is cheap — three small objects, once per resize.
    ///
    /// Sizes are clamped to `1x1`. A minimised window legitimately reports
    /// `0x0`, and both `create_texture` and `Surface::configure` treat a zero
    /// dimension as a validation error (and, for `configure`, a documented
    /// panic) rather than as something to skip. Clamping keeps the renderer
    /// alive across a minimise/restore cycle without a special case at every
    /// call site.
    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);

        // Nothing to do if the size is unchanged: egui emits a resize event
        // for sub-pixel and HiDPI changes that do not alter the pixel count,
        // and reallocating the image on those would be pure waste.
        if (width, height) == self.size {
            return;
        }

        self.size = (width, height);
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.recreate_storage_target();
    }

    /// Renders one frame into the swap chain.
    ///
    /// `target` is the [`egui::TextureId`] the caller intends to display. egui
    /// hands the app a `TextureId::Managed(_)` for anything it owns — the font
    /// atlas above all — and such a frame is not addressed at a real GPU
    /// surface, so all GPU work is skipped and the call returns immediately.
    /// Only [`egui::TextureId::User`], which is what a `TextureOptions`/
    /// `egui::TextureManager` registration for the plot produces, actually
    /// drives a frame.
    ///
    /// This is a best-effort operation and reports nothing: a dropped frame is
    /// a normal outcome, not an error. The surface legitimately refuses to hand
    /// out an image when the window is occluded, minimised, or mid-resize, and
    /// the correct response to all of those is "skip this frame and try again",
    /// which is what the `None` arms below do.
    /// # The `uniforms.resolution` contract
    ///
    /// The kernel bounds-checks every invocation against
    /// `uniforms.resolution`, but `textureStore`s at its own integer pixel
    /// coordinate in an image of `self.size()`. Those two numbers **must** be
    /// equal. If the uniform is larger, the shader's own bounds check admits
    /// invocations that store past the end of the image — an out-of-bounds
    /// write, which is undefined behaviour rather than a dropped pixel. If it
    /// is smaller, the right and bottom edges of the plot are simply never
    /// written and keep stale contents.
    ///
    /// So the caller should set `uniforms.resolution` from
    /// [`Renderer::size`], in **physical pixels** — not from egui's logical
    /// points, and not from the central panel's `available_size`, which is a
    /// different rectangle from the window surface. On a Retina display those
    /// differ by the pixel scale; the window surface is the authority.
    ///
    /// This function deliberately does *not* overwrite the field to "fix" a
    /// mismatch. Silently substituting a different resolution would shift every
    /// plot's mapping and hide the disagreement that caused it, which is a far
    /// worse failure than an obvious one.
    pub fn render(&mut self, uniforms: &Uniforms, target: &egui::TextureId) {
        // egui repaint semantics: a headless or hidden frame has no drawable
        // to present to. Checking this *before* touching the uniform buffer
        // also means a hidden window does not keep the queue busy with uploads.
        let egui::TextureId::User(_) = target else {
            return;
        };

        // `bytemuck::cast_slice` on a one-element array borrows a temporary
        // array, which is fine: `write_buffer` copies immediately and the
        // borrow does not outlive the statement.
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[*uniforms]));

        let surface_texture = match self.acquire_surface_texture() {
            Some(texture) => texture,
            None => return,
        };

        let target_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("steel-pulse:surface:view"),
                format: None,
                dimension: None,
                usage: None,
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: 0,
                array_layer_count: None,
            });

        let (width, height) = self.size;

        // Workgroup counts are rounded *up*: the shader guards each invocation
        // with a bounds check, so the overhang of the last workgroup in each
        // axis is discarded on the GPU rather than clamped on the CPU. Rounding
        // down instead would leave the right and bottom edges of the plot
        // unrendered after any non-multiple-of-8 resize.
        let groups_x = width.div_ceil(WORKGROUP_SIZE);
        let groups_y = height.div_ceil(WORKGROUP_SIZE);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("steel-pulse:frame"),
            });

        // -- Pass 1: evaluate the function, one invocation per texel. --------
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("steel-pulse:domain_coloring:pass"),
                // Only written when timing is on; `None` otherwise so the
                // pass costs exactly what it costs without instrumentation.
                timestamp_writes: self.timestamp_query_set.as_ref().map(|query_set| {
                    wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }
                }),
            });

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
            compute_pass.dispatch_workgroups(groups_x, groups_y, 1);
        }

        // -- Pass 2: blit the result onto the swap-chain image. ---------------
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("steel-pulse:blit:pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    // A 2D view, so no depth slice.
                    depth_slice: None,
                    // No MSAA, so nothing to resolve into.
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // The blit covers every pixel of the triangle, so
                        // clearing is pure cost. `LoadOp::Clear` is kept anyway
                        // as a cheap guard against a driver that leaves
                        // undefined content at the triangle's clipped
                        // overhang on some platforms.
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(&self.blit_pipeline);
            render_pass.set_bind_group(0, &self.blit_bind_group, &[]);
            render_pass.draw(0..FULLSCREEN_TRIANGLE_VERTICES, 0..1);
        }

        // In wgpu 30 timestamps live on the pass descriptors, so the resolve has
        // to happen in a *later* submission than the pass that wrote them. It
        // is encoded here rather than in `resolve_timestamps` so the copy is
        // queued in the same breath as the frame that produced it.
        if let (Some(query_set), Some(readback)) = (
            self.timestamp_query_set.as_ref(),
            self.timestamp_readback.as_ref(),
        ) {
            encoder.resolve_query_set(query_set, 0..2, readback, 0);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        // wgpu 30 moved presentation off `SurfaceTexture::present` and onto the
        // queue, because the queue is what owns the submission the present must
        // be ordered against. Forgetting this makes the image appear one frame
        // late and then stall.
        self.queue.present(surface_texture);
    }

    /// A one-line description of the GPU stack, for the UI's status bar.
    ///
    /// Built from the real [`wgpu::AdapterInfo`] rather than from a compile-time
    /// constant, so it reflects the adapter wgpu actually handed us — which is
    /// the only thing a bug report actually needs. `Backend` and `DeviceType`
    /// implement neither `Display` nor a lowercase name, so they are formatted
    /// explicitly.
    pub fn backend_description(&self) -> String {
        format!(
            "wgpu {WGPU_VERSION_LABEL} / {} / {}",
            backend_label(self.adapter_info.backend),
            device_type_label(self.adapter_info.device_type),
        )
    }

    /// The uniform buffer, for tests and for debug tooling that wants to read
    /// back what the last frame uploaded.
    pub fn uniform_buffer(&self) -> &wgpu::Buffer {
        &self.uniform_buffer
    }

    /// Current intermediate-image size, `(width, height)`, in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Tries to turn on GPU timing for the compute pass.
    ///
    /// Returns `true` if timing is now available. This is a *real* capability
    /// check, not a stub: [`Renderer::new`] already requested
    /// [`wgpu::Features::TIMESTAMP_QUERY`] if and only if the adapter advertised
    /// it, and creating a two-slot `QueryType::Timestamp` set is unconditionally
    /// valid for a device holding that feature, so no speculative creation (and
    /// no error scope) is needed here.
    ///
    /// Calling this on a device without the feature is not an error; it just
    /// returns `false` and [`Renderer::resolve_timestamps`] keeps returning
    /// `None`. `Features::TIMESTAMP_QUERY` is an ordinary device feature
    /// available in a default wgpu build — it needs no Cargo feature flag, which
    /// is why this is a runtime check rather than a `compile_error!`.
    pub fn enable_timestamps(&mut self) -> bool {
        if self.timestamp_query_set.is_some() {
            return true;
        }
        if !self
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return false;
        }

        self.timestamp_query_set = Some(self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("steel-pulse:timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        }));

        // `QUERY_RESOLVE` and `MAP_READ` on one buffer lets the resolve target be
        // mapped directly, with no staging copy. Resolution writes 8 bytes per
        // query with an offset aligned to `QUERY_RESOLVE_BUFFER_ALIGNMENT`; the
        // buffer is sized and aligned to match.
        self.timestamp_readback = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("steel-pulse:timestamps:readback"),
            size: 2 * TIMESTAMP_QUERY_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));

        {
            let mut state = lock_timestamps(&self.timestamp_state);
            state.in_flight = false;
            state.latest = None;
            state.dropped = 0;
        }
        true
    }

    /// Returns the most recent GPU duration measured for the compute pass.
    ///
    /// Deliberately **non-blocking**. The readback is a map of a buffer the GPU
    /// wrote a frame or two ago; waiting on it with `PollType::Wait` would
    /// serialise CPU and GPU and throw away the double buffering the frame
    /// latency of 2 exists to provide. So this kicks off at most one readback at
    /// a time, gives the callback a single non-blocking poll to run, and
    /// returns whatever the previous readback produced — which is a frame stale
    /// and therefore useless for synchronisation, but exactly right for
    /// telemetry.
    ///
    /// Returns `None` when timing was never enabled, when the device lacks the
    /// feature, or when no readback has completed yet.
    pub fn resolve_timestamps(&mut self) -> Option<Duration> {
        let readback = self.timestamp_readback.clone()?;

        {
            let mut state = lock_timestamps(&self.timestamp_state);
            if state.in_flight {
                // A readback is outstanding; report the last decoded value
                // rather than trying to map an already-mapped buffer.
                return state.latest;
            }
            state.in_flight = true;
        }

        // `get_timestamp_period` converts raw query ticks to nanoseconds. It is
        // 1.0 on the web and roughly 1.0 on most native backends, but reading
        // it is the only correct way to do the arithmetic.
        let period = f64::from(self.queue.get_timestamp_period());

        let readback_for_callback = readback.clone();
        let state = Arc::clone(&self.timestamp_state);
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let mut state = lock_timestamps(&state);
                // Always clear the in-flight flag, even on failure, or the
                // renderer would never attempt another readback.
                state.in_flight = false;

                if result.is_err() {
                    state.dropped += 1;
                    return;
                }

                let mapped = readback_for_callback.slice(..).get_mapped_range();
                let duration = match mapped {
                    Ok(view) => {
                        // Two little-endian u64s, in submission order. The
                        // buffer is exactly two queries wide, so `chunks_exact`
                        // yields exactly two words and anything malformed shows
                        // up as a short iterator rather than as a panic.
                        let mut words = view.chunks_exact(8).map(|w| {
                            u64::from_le_bytes([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]])
                        });
                        let start = words.next();
                        let end = words.next();
                        drop(view);
                        // Unmap before returning: the next `map_async` is only
                        // legal on an unmapped buffer.
                        readback_for_callback.unmap();

                        match (start, end) {
                            (Some(start), Some(end)) => {
                                end.checked_sub(start).map(|ticks| ticks as f64 * period)
                            }
                            _ => None,
                        }
                    }
                    Err(_) => None,
                };

                match duration {
                    // A zero-length interval means the query set was never
                    // written this frame (for instance a frame that was skipped
                    // because the window was occluded), not a zero-cost pass.
                    Some(seconds) if seconds > 0.0 && seconds.is_finite() && seconds < 1.0 => {
                        state.latest = Some(Duration::from_secs_f64(seconds));
                    }
                    _ => state.dropped += 1,
                }
            });

        // One non-blocking poll so a readback that already completed is decoded
        // on this call rather than the next. `Poll` never waits, so this cannot
        // stall the frame loop; a device error here is not actionable and is
        // reported through the uncaptured-error handler anyway.
        let _ = self.device.poll(wgpu::PollType::Poll);

        lock_timestamps(&self.timestamp_state).latest
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    /// Allocates a fresh intermediate image plus the two bind groups that read
    /// and write it.
    ///
    /// Called from [`Renderer::resize`]. Not `pub`: both bind groups bake in a
    /// `TextureView`, so this must never be reachable without also refreshing
    /// `self.size` and the surface configuration.
    fn recreate_storage_target(&mut self) {
        let (width, height) = self.size;
        let target = build_storage_target(
            &self.device,
            &self.compute_layout,
            &self.blit_layout,
            &self.uniform_buffer,
            &self.sampler,
            width,
            height,
        );
        // Assigning the texture last is deliberate: the new bind groups are
        // already valid by the time this runs, so the renderer is never
        // momentarily missing a binding.
        self.compute_bind_group = target.compute_bind_group;
        self.blit_bind_group = target.blit_bind_group;
        self.storage_view = target.storage_view;
        self.sampled_view = target.sampled_view;
        self.storage_texture = target.texture;
    }

    /// Acquires the next swap-chain image, reconfiguring if the surface has gone
    /// stale.
    ///
    /// wgpu 30 replaced the old `Result<SurfaceTexture, SurfaceError>` with a
    /// six-way enum, and four of the six arms are "do not draw this frame"
    /// rather than "something went wrong": the window can be minimised
    /// (`Occluded`), mid-move (`Outdated`), timed out, or the device can have
    /// reported a validation error. Only `Success` and `Suboptimal` carry a
    /// drawable.
    fn acquire_surface_texture(&self) -> Option<wgpu::SurfaceTexture> {
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => Some(texture),
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
                // The image no longer matches the window (a resize raced the
                // frame). Reconfigure now so the *next* frame is clean; this
                // one is still drawable, so use it rather than dropping a frame.
                self.surface.configure(&self.device, &self.config);
                Some(texture)
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Outdated
            | wgpu::CurrentSurfaceTexture::Lost
            | wgpu::CurrentSurfaceTexture::Validation => None,
        }
    }
}

/// Allocates a `width` x `height` `rgba8unorm` image and the two bind groups
/// that reference it.
///
/// Free function rather than a method so that the constructor can build the
/// target from local handles and `resize` can build it from `self`'s, without
/// either path needing placeholder values to satisfy struct initialisation.
///
/// `width` and `height` are clamped rather than trusted. A `0x0`
/// `create_texture` is a validation error that poisons the device, and a
/// minimised window really does report zero, so the invariant is enforced at the
/// one place that allocates rather than trusted from every call site.
fn build_storage_target(
    device: &wgpu::Device,
    compute_layout: &wgpu::BindGroupLayout,
    blit_layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    sampler: &wgpu::Sampler,
    width: u32,
    height: u32,
) -> StorageTarget {
    let (width, height) = (width.max(1), height.max(1));

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("steel-pulse:plot:rgba8unorm"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: STORAGE_FORMAT,
        // `STORAGE_BINDING` for the compute pass's write-only image,
        // `TEXTURE_BINDING` for the blit's `texture_2d<f32>`.
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    // Two views of one image: the compute pass needs a write-only storage view,
    // the blit a filterable sampled view. They are the same pixels, so nothing
    // has to be copied between them, and because they live in different passes
    // of one submission wgpu inserts the storage-write to texture-read barrier
    // on its own.
    let storage_view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("steel-pulse:plot:storage_view"),
        format: None,
        dimension: None,
        usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: None,
        base_array_layer: 0,
        array_layer_count: None,
    });

    let sampled_view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("steel-pulse:plot:sampled_view"),
        format: None,
        dimension: None,
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: None,
        base_array_layer: 0,
        array_layer_count: None,
    });

    let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("steel-pulse:compute:bg"),
        layout: compute_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: uniform_buffer,
                    offset: 0,
                    // `None` means "the rest of the buffer", which is exactly
                    // `UNIFORM_BUFFER_SIZE` and stays correct if the struct
                    // ever grows.
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&storage_view),
            },
        ],
    });

    let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("steel-pulse:blit:bg"),
        layout: blit_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&sampled_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });

    StorageTarget {
        texture,
        storage_view,
        sampled_view,
        compute_bind_group,
        blit_bind_group,
    }
}

/// Human-readable name for a wgpu backend.
///
/// `Backend` implements `Debug` but not `Display`, and the `Debug` rendering
/// happens to be the exact spelling we want ("Metal", "Vulkan"), so the only
/// thing to add is the `BrowserWebGpu` spelling, which reads as
/// `BrowserWebGpu` and is worth humanising.
fn backend_label(backend: wgpu::Backend) -> &'static str {
    match backend {
        wgpu::Backend::Noop => "Noop",
        wgpu::Backend::Vulkan => "Vulkan",
        wgpu::Backend::Metal => "Metal",
        wgpu::Backend::Dx12 => "Direct3D 12",
        wgpu::Backend::Gl => "OpenGL",
        wgpu::Backend::BrowserWebGpu => "WebGPU (browser)",
    }
}

/// Human-readable name for a physical device class.
///
/// Worth spelling out because the distinction is the whole point of asking for
/// a `HighPerformance` adapter: "DiscreteGpu" tells a user nothing, "Discrete
/// GPU" tells them why their laptop fan is spinning.
fn device_type_label(device_type: wgpu::DeviceType) -> &'static str {
    match device_type {
        wgpu::DeviceType::Other => "Unclassified device",
        wgpu::DeviceType::IntegratedGpu => "Integrated GPU",
        wgpu::DeviceType::DiscreteGpu => "Discrete GPU",
        wgpu::DeviceType::VirtualGpu => "Virtual GPU",
        wgpu::DeviceType::Cpu => "Software (CPU)",
    }
}

// Compile-time guards for the invariants this module documents but cannot
// enforce at a call site. A failure here names the exact assumption that broke,
// which is worth more than the first validation error it would otherwise cause.
const _: () = {
    assert!(
        WORKGROUP_SIZE > 0,
        "a zero workgroup size dispatches nothing"
    );
    assert!(DESIRED_MAX_FRAME_LATENCY >= 1);
    assert!(FULLSCREEN_TRIANGLE_VERTICES == 3);
    // 64 bytes today; the point is that the buffer is allocated from the shared
    // constant rather than a literal that could drift from `Uniforms`.
    assert!(UNIFORM_BUFFER_SIZE == 64);
    // The dispatch grid is derived from the image size divided by this, and the
    // kernel's bounds check is written against the same number on its side.
    // Keeping them numerically equal here makes the *next* reader's job obvious.
    assert!(WORKGROUP_SIZE == 8, "kernel declares @workgroup_size(8, 8)");
};

/// Backs the `Send + Sync` claim in [`Renderer`]'s documentation without adding
/// a dependency. Naming the generic function at its concrete type makes the
/// compiler resolve the bound here and now; the value is discarded.
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    let _checked: fn() = assert_send_sync::<Renderer>;
};
