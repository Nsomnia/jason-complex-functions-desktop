//! GPU renderer: the per-pixel complex-function compute kernel, and the
//! hand-off of its output to egui for display.
//!
//! # Where the GPU work actually happens
//!
//! This module owns **no surface, no swap chain and no window**. eframe 0.36
//! creates the `wgpu::Instance`, the adapter, the device, the queue, the surface
//! and the swap chain before `App::new` is ever called, and there is no public
//! hook to replace any of it — `eframe::native::wgpu_integration` is entirely
//! private. So this renderer *shares* eframe's device rather than competing with
//! it. The device and queue arrive by cloning the handles out of
//! [`egui_wgpu::RenderState`], which is a few `Arc` bumps rather than a copy.
//!
//! The frame is then one pass:
//!
//! 1. a **compute pass** evaluates `f(z)` for every texel of an off-screen
//!    `rgba8unorm` storage texture, one invocation per texel;
//! 2. that texture is **registered with egui's own renderer**, which blits it to
//!    the window as an ordinary egui image in a later part of the same frame.
//!
//! Nothing here does colour work, gamma correction or tone mapping, and nothing
//! here touches the swap chain. The compute pass exists because every pixel is
//! an independent function evaluation, which is what compute shaders are for and
//! what fragment shaders are bad at: a fragment shader is latency-bound on
//! divergence, and thousands of escape-time iterations per pixel is precisely
//! the workload that makes every lane of a SIMD quad wait for the slowest one.
//!
//! # The WGSL contract (READ THIS BEFORE EDITING THE SHADER)
//!
//! The Rust side declares an explicit [`wgpu::BindGroupLayout`] and hands the
//! compiler an explicit `PipelineLayout`, so a mismatch with the WGSL is a
//! **hard validation error at pipeline-creation time**, not a silently wrong
//! picture. `shaders/domain_coloring.wgsl` must match exactly:
//!
//! | `@group` | `@binding` | WGSL declaration | Rust [`wgpu::BindingType`] |
//! |---|---|---|---|
//! | `0` | `0` | `var<uniform> uniforms: Uniforms;` | `Buffer { ty: Uniform }` |
//! | `0` | `1` | `var output_texture: texture_storage_2d<rgba8unorm, write>;` | `StorageTexture { access: WriteOnly, format: Rgba8Unorm }` |
//!
//! The `@compute` entry point is `main` and the workgroup is `(8, 8)`; both are
//! passed explicitly to wgpu rather than inferred, so renaming either in the
//! shader without updating the constants below fails loudly at startup instead
//! of silently running a different kernel.
//!
//! **The kernel flips Y and nothing else must.** It maps texture row 0 to
//! `+scale` on the imaginary axis (`im = center.y + (1.0 - uv.y * 2.0) * scale`).
//! That is correct *because* egui samples the texture with `v = 0` at the top of
//! the screen. There is no second flip anywhere in this file, so the flip looks
//! redundant until you remember it is what makes row 0 the top. Do not "fix" it.
//!
//! # The `uniforms.resolution` contract
//!
//! `uniforms.resolution` **must** equal [`Renderer::size`] exactly, in physical
//! pixels. The kernel bounds-checks its dispatch against the uniform but
//! `textureStore`s into an image of `size()`.
//!
//! * Uniform **larger** than the image: the shader's own bounds check admits
//!   invocations that store past the end of the texture. That is undefined
//!   behaviour, not a dropped pixel.
//! * Uniform **smaller**: the right and bottom edges are never written and keep
//!   stale contents from the previous frame, which during a resize is a smear of
//!   the old plot.
//!
//! [`Renderer::render`] does **not** overwrite the field to paper over a
//! mismatch. Silently substituting a different resolution would shift every
//! plot's mapping and hide the disagreement that caused it, which is a far worse
//! failure than an obvious one. The caller must set the field from
//! [`Renderer::size`] — see [`Renderer::resize`] for which pixel count that is.

use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;

// The wgpu renderer is reached through eframe's re-export rather than by adding
// a direct `egui-wgpu` dependency. eframe re-exports `egui_wgpu` under its
// `wgpu_no_default_features` feature, which its own default `wgpu` feature
// enables, so this path is live in a default build. It also guarantees a single
// wgpu crate in the type namespace: a direct `egui-wgpu` dependency that
// resolved to a different feature set would produce a *second*, nominally
// identical `wgpu::Device` type, and nothing would compile. `Cargo.lock` pins one
// wgpu 30.0.1 for both, so `state.device` is the same type as the `wgpu::Device`
// imported below.
use eframe::egui;
use eframe::egui_wgpu;

use crate::uniforms::{Uniforms, UNIFORM_BUFFER_SIZE};

// ---------------------------------------------------------------------------
// Shader source and the names inside it.
//
// `include_str!` rather than a runtime read: a missing or renamed shader then
// fails the build with a file-not-found error instead of producing a permanently
// black plot that is miserable to diagnose.
//
// NOTE FOR INTEGRATORS: `shaders/domain_coloring.wgsl` is authored in a
// separate lane. If it is renamed, change `COMPUTE_SHADER_PATH` *and* the
// `include_str!` literal below. The macro needs a string literal, so the
// constant and the literal must be kept in step by hand; the constant exists to
// record the path in one obvious, greppable place.
// ---------------------------------------------------------------------------

/// Where the domain-colouring compute shader lives, relative to `Cargo.toml`.
const COMPUTE_SHADER_PATH: &str = "shaders/domain_coloring.wgsl";

/// Source of the compute kernel, embedded at compile time.
const COMPUTE_SHADER_SRC: &str = include_str!("../shaders/domain_coloring.wgsl");

/// `@compute` entry point in [`COMPUTE_SHADER_SRC`].
///
/// Passed explicitly rather than left to wgpu's "exactly one compute stage in
/// the module" inference, so that adding a second experimental kernel to the
/// same file later cannot silently change which one runs.
const COMPUTE_ENTRY_POINT: &str = "main";

/// Edge length of the compute workgroup square, in invocations.
///
/// **A contract, not a tuning knob.** [`Renderer::render`] dispatches
/// `ceil(size / WORKGROUP_SIZE)` workgroups in each axis and the kernel guards
/// every invocation with a bounds check, so a partial final workgroup writes
/// only the pixels that exist. It must match `@workgroup_size(x, y)` in
/// [`COMPUTE_SHADER_SRC`] — currently `(8, 8)`. wgpu cannot check that for us, and
/// a mismatch is a disagreement about how many invocations a workgroup holds,
/// which is an out-of-bounds write rather than a compile error.
const WORKGROUP_SIZE: u32 = 8;

/// Format of the image the compute pass writes and egui samples.
///
/// `Rgba8Unorm` is required, not chosen: `egui_wgpu::Renderer::register_native_texture`
/// documents that a native texture "must have the format
/// [`wgpu::TextureFormat::Rgba8Unorm`]", and the WGSL side must spell it
/// `texture_storage_2d<rgba8unorm, write>`.
///
/// It is also one of the formats the WebGPU spec *guarantees* supports
/// `STORAGE_BINDING` with write-only access — the `Bgra8Unorm` case is the one
/// that needs a feature flag, this does not — so no optional device feature is
/// required and the compute pass cannot fail on a conformant backend.
const STORAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// How egui should sample the plot when it draws it as an image.
///
/// This is a description, not a `wgpu::Sampler`. egui builds the actual sampler
/// and the bind group inside its own render pass, so there is no binding left
/// here for a sampler object to occupy; keeping the *configuration* as a named
/// constant is what preserves the intent.
///
/// `Linear` is **mandatory**, not a preference. egui-wgpu's own texture bind
/// group layout declares binding 1 as `SamplerBindingType::Filtering` alongside
/// `TextureSampleType::Float { filterable: true }`, so registering with
/// `FilterMode::Nearest` is a validation error at registration time.
///
/// `ClampToEdge` on every axis matters because egui images are drawn with uv
/// coordinates that can reach slightly outside `0..1`; a repeat or mirror mode
/// would drag the opposite edge of the plot into view.
///
/// `lod_max_clamp: 0.0` pins sampling to mip level 0. The plot is a single mip
/// level, so any nonzero maximum is meaningless, and pinning it makes an
/// accidental future mip chain impossible to sample by accident.
const PLOT_SAMPLER: wgpu::SamplerDescriptor<'static> = wgpu::SamplerDescriptor {
    label: Some("steel-pulse:plot:sampler"),
    address_mode_u: wgpu::AddressMode::ClampToEdge,
    address_mode_v: wgpu::AddressMode::ClampToEdge,
    address_mode_w: wgpu::AddressMode::ClampToEdge,
    mag_filter: wgpu::FilterMode::Linear,
    min_filter: wgpu::FilterMode::Linear,
    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
    lod_min_clamp: 0.0,
    lod_max_clamp: 0.0,
    // egui's own bind group supplies its own sampler; a comparison function
    // here would be ignored at best and misleading at worst.
    compare: None,
    // Must be at least 1. Anything higher would force every filter mode to be
    // linear, including the mip filter, which is not what `Nearest` above says.
    anisotropy_clamp: 1,
    // Only consulted by `AddressMode::ClampToBorder`, which the plot never uses.
    border_color: None,
};

/// Version string shown in the UI's backend line.
const WGPU_VERSION_LABEL: &str = "30.0.1";

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
/// enum variant. A user who sees this in a window needs to learn *what to do* —
/// "this build of eframe was compiled without the wgpu renderer" is actionable,
/// `Renderer::Wgpu` is not.
///
/// There is no `NoAdapter` or `DeviceRequest` variant any more: eframe already
/// requested the adapter and the device before this module is ever reached, and
/// a failure there is an eframe startup failure with its own error.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RendererError {
    /// The `AdapterInfo` could not be read, or the supplied state was
    /// inconsistent. In practice this means the render state did not come from a
    /// live wgpu adapter.
    UnusableRenderState {
        /// Human-readable cause.
        cause: String,
    },
    /// The compute pipeline could not be built. Almost always a WGSL error in
    /// [`COMPUTE_SHADER_SRC`] or a binding layout that no longer matches the
    /// table at the top of this file. The underlying wgpu diagnostic is in
    /// `cause`.
    PipelineCreation {
        /// Human-readable cause.
        cause: String,
    },
}

impl fmt::Display for RendererError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnusableRenderState { cause } => write!(
                f,
                "the eframe wgpu render state is unusable ({cause}). \
                 This usually means eframe was built with the `glow` renderer \
                 instead of `wgpu`. Check that `eframe`'s default features are \
                 enabled and that `Renderer::Wgpu` is passed in the native \
                 options."
            ),
            Self::PipelineCreation { cause } => write!(
                f,
                "the domain-colouring compute pipeline could not be created \
                 ({cause}). Check {COMPUTE_SHADER_PATH} for a WGSL compile \
                 error, and check that its `@group(0) @binding(0)` uniform and \
                 `@binding(1)` rgba8unorm storage texture still match the \
                 bind group layout in src/renderer.rs."
            ),
        }
    }
}

impl std::error::Error for RendererError {}

// ---------------------------------------------------------------------------
// What one frame produced
// ---------------------------------------------------------------------------

/// The result of one [`Renderer::render`] call: what to draw, at what size, and
/// how long the GPU took.
///
/// Returned by value rather than pulled off the renderer with accessors so that
/// a frame is self-describing. The app layer caches the three fields in its own
/// state and only has to look at [`Renderer`] again when it resizes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderOutcome {
    /// The egui texture handle to draw with [`egui::Image::from_texture`], or
    /// `None` if the GPU path is unavailable.
    ///
    /// **Always `Some` in the current implementation.** egui's renderer lock is
    /// `epaint::mutex::RwLock`, which wraps `parking_lot` and therefore has no
    /// poisoning state, and registration cannot otherwise fail. The `Option` is
    /// kept anyway, for two reasons: it forces callers to handle the
    /// unavailable case rather than `unwrap`, and it gives a future failure mode
    /// — a lost device, a renderer dropped by eframe — somewhere honest to
    /// report. Handling it costs one `if let`.
    ///
    /// If it is ever `None`, the compute pass still ran; the result is sitting
    /// in the storage texture, it simply cannot be referenced from the UI. The
    /// app should surface an error rather than silently draw nothing.
    ///
    /// The handle is stable across frames and is only replaced after a
    /// [`Renderer::resize`], so it is safe to cache. Always take the value from
    /// the current frame's outcome rather than holding an old one: after a
    /// resize the previous handle has been freed and refers to a destroyed view.
    pub texture: Option<egui::TextureId>,
    /// Size of the storage texture, `(width, height)`, in **physical pixels**.
    ///
    /// This is the value that must be written into `uniforms.resolution`; see
    /// the module documentation. Always the renderer's current size, never a
    /// stale one, so it doubles as the trigger for a resize comparison.
    pub size: (u32, u32),
    /// GPU time for the compute pass, in milliseconds, or `None` if it is not
    /// known yet.
    ///
    /// `None` is the honest answer in three distinct cases, and it is never a
    /// fabricated `0.0`:
    ///
    /// * timestamps were never enabled (the default — see
    ///   [`Renderer::enable_timestamps`]);
    /// * the device was created without `wgpu::Features::TIMESTAMP_QUERY`;
    /// * the readback for this frame has not completed yet. The value is a frame
    ///   or two behind by construction, because reading it is non-blocking.
    pub gpu_ms: Option<f64>,
}

// ---------------------------------------------------------------------------
// Optional GPU timing
// ---------------------------------------------------------------------------

/// Mutable state shared between the frame loop and the `map_async` callback wgpu
/// invokes once the GPU has finished copying query results.
///
/// It lives behind an [`Arc`] + [`Mutex`] because the mapping callback is
/// `FnOnce(..) + Send + 'static` — it cannot borrow from `self`. The [`Mutex`] is
/// never held across a GPU call, so contention is not a concern; it is only ever
/// locked for a handful of field assignments.
#[derive(Debug, Default)]
struct TimestampState {
    /// `true` while a staging-buffer mapping is outstanding. Prevents a second
    /// `map_async` on a buffer that is already mapped, which wgpu rejects.
    in_flight: bool,
    /// Most recent successfully decoded compute-pass duration.
    latest: Option<Duration>,
    /// How many readbacks failed or produced nonsense. Surfaced so that a
    /// silently frozen telemetry readout is distinguishable from a GPU that is
    /// genuinely never finishing.
    dropped: u64,
}

/// Locks the timestamp state, recovering from poisoning instead of panicking.
///
/// A panic inside the mapping callback would poison this lock. Everything behind
/// it is a `bool`, an `Option<Duration>` and a `u64` — all independently valid
/// after a panic — so it is always safe to keep using. Propagating the poison
/// would turn a telemetry glitch into a dead render loop, which is wildly
/// disproportionate for a value nothing blocks on.
fn lock_timestamps(state: &Mutex<TimestampState>) -> std::sync::MutexGuard<'_, TimestampState> {
    state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

// ---------------------------------------------------------------------------
// The storage target
// ---------------------------------------------------------------------------

/// The intermediate image and the bind group that writes it.
///
/// Bundled because the objects are only ever valid together: recreating the
/// texture without recreating the bind group leaves the pipeline bound to a
/// destroyed view.
struct StorageTarget {
    /// The `rgba8unorm` image the compute pass writes and egui samples.
    texture: wgpu::Texture,
    /// View narrowed to `STORAGE_BINDING`, for the compute pass's write-only
    /// image.
    storage_view: wgpu::TextureView,
    /// View narrowed to `TEXTURE_BINDING`, for egui's `texture_2d<f32>`.
    ///
    /// Two views of one image rather than one shared view: a
    /// `texture_2d<f32>` and a `texture_storage_2d<..., write>` are different
    /// binding types, and narrowing each view to the single usage it is bound
    /// with makes that distinction impossible to get wrong by accident.
    sampled_view: wgpu::TextureView,
    /// Uniform + storage view, for the compute pass.
    compute_bind_group: wgpu::BindGroup,
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

/// Owns the compute pipeline that evaluates the complex function per pixel, and
/// the off-screen image it writes, and keeps that image registered with egui.
///
/// # Sharing eframe's device
///
/// The `Device` and `Queue` are clones of eframe's, obtained from
/// [`egui_wgpu::RenderState`]. Both are `Arc`-backed handles inside wgpu, so
/// cloning is a refcount bump and the renderer is not creating a second logical
/// device. That is the point: a second device would mean a second set of
/// resources with no shared memory, and there would be no way to get our
/// texture onto eframe's swap chain anyway.
///
/// # Threading
///
/// `Renderer` is `Send + Sync` — asserted at compile time at the bottom of this
/// file, not merely claimed — but it is *not* internally synchronised. Treat it
/// as owned by whichever thread drives the frame loop, which in this program is
/// the egui main thread. The only shared mutable state is
/// [`TimestampState`], which is genuinely shared with a wgpu callback.
pub struct Renderer {
    /// Clone of eframe's logical device. Every resource here hangs off it.
    device: wgpu::Device,
    /// Clone of eframe's queue. All uploads and submissions go through it.
    queue: wgpu::Queue,
    /// egui's renderer, shared with eframe. Held so the compute output can be
    /// registered and unregistered without the app layer handling the lock.
    ///
    /// `epaint::mutex::RwLock`, not `std::sync::RwLock`: that is the type
    /// `RenderState::renderer` actually is, and naming anything else here would
    /// be a distinct type that the compiler refuses to assign. epaint's wrapper
    /// is `parking_lot` underneath, so it has no poisoning state to recover from
    /// and `write()` hands back a guard directly rather than a `Result`.
    egui_renderer: Arc<egui::epaint::mutex::RwLock<egui_wgpu::Renderer>>,
    /// What the adapter reported about itself; the source of the UI's backend
    /// line. Captured once because the adapter itself is not kept.
    adapter_info: wgpu::AdapterInfo,
    /// Rendered once at construction: the list of adapters wgpu could see.
    ///
    /// A `String` rather than the `Vec<wgpu::Adapter>` itself because
    /// `RenderState::available_adapters` is `#[cfg]`-gated to non-wasm targets
    /// and holding adapter handles open for the life of the app serves no
    /// purpose. Reading it at construction also keeps that platform assumption
    /// in exactly one place.
    adapter_list: String,

    /// Evaluates the complex function, one invocation per texel.
    compute_pipeline: wgpu::ComputePipeline,
    /// Explicit layout for [`Renderer::compute_pipeline`]; kept alive because a
    /// `BindGroup` may not outlive the layout it was created against.
    compute_layout: wgpu::BindGroupLayout,
    /// 64 bytes of [`Uniforms`], re-uploaded once per frame.
    uniform_buffer: wgpu::Buffer,

    /// The image the compute pass writes. Recreated on resize.
    storage_texture: wgpu::Texture,
    /// Storage-narrowed view, for the compute pass.
    storage_view: wgpu::TextureView,
    /// Sampled-narrowed view, handed to egui.
    sampled_view: wgpu::TextureView,
    /// Uniform + storage view, for the compute pass.
    compute_bind_group: wgpu::BindGroup,

    /// Size of the image, `(width, height)`, in physical pixels, clamped to at
    /// least `1x1`.
    size: (u32, u32),
    /// egui's handle for [`Renderer::sampled_view`].
    ///
    /// `None` until the first [`Renderer::render`], and reset to `None` by
    /// [`Renderer::resize`] so the next frame re-registers. Caching matters:
    /// `register_native_texture` allocates a fresh `TextureId` and a fresh
    /// bind group on every call, so calling it per frame would leak one of each
    /// per frame.
    registered: Option<egui::TextureId>,

    /// Two timestamp slots, present only after [`Renderer::enable_timestamps`]
    /// succeeds. Slot 0 is the start of the compute pass, slot 1 the end.
    timestamp_query_set: Option<wgpu::QuerySet>,
    /// Resolve target for `resolve_query_set`, and the CPU-side map of those
    /// results. Created together with the query set.
    timestamp_readback: Option<wgpu::Buffer>,
    /// Shared with the `map_async` callback; see [`TimestampState`].
    timestamp_state: Arc<Mutex<TimestampState>>,
}

impl Renderer {
    /// Builds the compute pipeline against **eframe's** device and queue.
    ///
    /// `state` is what [`eframe::CreationContext::wgpu_render_state`] hands back.
    /// Nothing is requested, awaited or configured here: eframe has already
    /// chosen an adapter, created the device and queue, and configured its
    /// surface. In particular this function is **not** `async` — there is no
    /// adapter request to await, so there is no executor requirement and no
    /// reason for the app layer to block on anything.
    ///
    /// # The initial image is 1x1
    ///
    /// There is no window here to measure, so the image starts at `1x1`, which
    /// is a valid texture and a valid one-pixel plot. **The app layer must call
    /// [`Renderer::resize`] before the first [`Renderer::render]`** — in practice
    /// it does this on the very first frame, because the central panel's
    /// `available_rect` is the natural source for the pixel count. Until it
    /// does, the plot is a single pixel.
    ///
    /// # Errors
    ///
    /// Returns [`RendererError::UnusableRenderState`] if the state does not
    /// describe a live adapter, and [`RendererError::PipelineCreation`] if the
    /// shader and the bind group layout disagree.
    ///
    /// wgpu's own validation runs asynchronously and is not visible in either
    /// variant: a WGSL compile error surfaces through the uncaptured-error
    /// handler installed below, which prints to stderr rather than panicking. A
    /// panic from inside pipeline creation would take the whole window down and
    /// print a backtrace nobody reads.
    pub fn new(state: &egui_wgpu::RenderState) -> Result<Self, RendererError> {
        // Cloning `Device`/`Queue` bumps a refcount; there is no second
        // physical device and no separate resource namespace.
        let device = state.device.clone();
        let queue = state.queue.clone();
        let egui_renderer = Arc::clone(&state.renderer);

        let adapter_info = state.adapter.get_info();
        let adapter_list = describe_adapters(state);

        // Validation and out-of-memory errors are reported rather than fatal, so
        // a mis-sized texture or a binding mismatch degrades to a black plot
        // with a readable message instead of a dead window.
        device.on_uncaptured_error(Arc::new(|error| {
            eprintln!("steel-pulse: wgpu error: {error}");
        }));

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
                        // Pinning the size lets wgpu validate the binding against
                        // the WGSL struct's minimum binding size at layout time,
                        // which is how a field-count mismatch between
                        // `src/uniforms.rs` and the shader gets caught at
                        // startup rather than as a mis-coloured plot.
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

        // `create_shader_module` in wgpu 30 runs Naga's WGSL front-end; syntax
        // errors are reported through the uncaptured-error handler above rather
        // than returned here, which is why `PipelineCreation` mostly surfaces
        // layout and entry-point mismatches.
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("steel-pulse:domain_coloring"),
            source: wgpu::ShaderSource::Wgsl(COMPUTE_SHADER_SRC.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("steel-pulse:compute:pl"),
            bind_group_layouts: &[Some(&compute_layout)],
            // No push-constant-style immediate data. The uniforms go through the
            // uniform buffer so they are addressable by the shader rather than
            // baked into the pipeline, which is what makes them changeable per
            // frame at all.
            immediate_size: 0,
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("steel-pulse:domain_coloring:compute"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(COMPUTE_ENTRY_POINT),
            // Pass-through is right here: naga's own front-end already runs on
            // this machine, and a second front-end would add a compile-time
            // feature dependency for no benefit.
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            // No on-disk pipeline cache: that needs a Cargo feature this crate
            // does not enable.
            cache: None,
        });

        // `mapped_at_creation` would let us seed this without a queue write, but
        // an all-zero uniform is a perfectly good starting state and the first
        // frame overwrites it anyway. `COPY_DST` is the only other usage
        // `write_buffer` strictly requires, and the narrowest one that validates.
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("steel-pulse:uniforms"),
            size: UNIFORM_BUFFER_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 1x1 until the app layer reports the real size. Valid, and cheap.
        let size = (1u32, 1u32);
        let target =
            build_storage_target(&device, &compute_layout, &uniform_buffer, size.0, size.1);

        Ok(Self {
            device,
            queue,
            egui_renderer,
            adapter_info,
            adapter_list,
            compute_pipeline,
            compute_layout,
            uniform_buffer,
            storage_texture: target.texture,
            storage_view: target.storage_view,
            sampled_view: target.sampled_view,
            compute_bind_group: target.compute_bind_group,
            size,
            registered: None,
            timestamp_query_set: None,
            timestamp_readback: None,
            timestamp_state: Arc::new(Mutex::new(TimestampState::default())),
        })
    }

    /// Notifies the renderer that its drawable area changed size.
    ///
    /// Rebuilds the storage texture, both views, and the compute bind group, and
    /// **invalidates the cached egui registration** so the next
    /// [`Renderer::render`] re-registers. That last part is not optional: egui's
    /// bind group holds a reference to the old view, and once this function
    /// replaces the texture that view is destroyed. Leaving the old
    /// [`RenderOutcome::texture`] in place would leave the app drawing through a
    /// bind group that points at freed memory.
    ///
    /// Sizes are clamped to `1x1`. A minimised window legitimately reports
    /// `0x0`, and a `0x0` `create_texture` is a validation error that poisons the
    /// device rather than something to skip, so clamping keeps the renderer
    /// alive across a minimise/restore cycle without a special case at every
    /// call site.
    ///
    /// Returns early when the size is unchanged. egui emits resize events for
    /// sub-pixel and HiDPI changes that do not alter the pixel count, and
    /// reallocating the image on those would be pure waste.
    ///
    /// # Which pixel count to pass
    ///
    /// The plot's physical size, and **only** the plot's physical size. The
    /// image is sampled by egui into a UI rectangle, so the right argument is
    /// that rectangle's size in physical pixels — the central panel's
    /// `available_rect` times the viewport's `pixels_per_point` — not the window
    /// size and not the value in logical points. Getting this wrong stretches or
    /// crops the plot even though nothing errors.
    ///
    /// eframe owns the surface, so unlike a self-presenting renderer this
    /// function reconfigures nothing: no swap chain, no `SurfaceConfiguration`,
    /// no `present`. It only has to agree with eframe about how big the plot is.
    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);

        if (width, height) == self.size {
            return;
        }

        // Release egui's handle on the view we are about to destroy. Without
        // this, every resize would leave a dead bind group in egui's texture
        // map, and the map would grow without bound over a session with many
        // window drags.
        self.unregister();

        self.size = (width, height);
        let target = build_storage_target(
            &self.device,
            &self.compute_layout,
            &self.uniform_buffer,
            width,
            height,
        );
        // Assigning the texture last is deliberate: the new bind group is
        // already valid by the time this runs, so the renderer is never
        // momentarily missing a binding.
        self.compute_bind_group = target.compute_bind_group;
        self.storage_view = target.storage_view;
        self.sampled_view = target.sampled_view;
        self.storage_texture = target.texture;
    }

    /// Uploads the uniforms, runs the compute pass, and makes the result
    /// available to egui.
    ///
    /// This is the whole frame. It does not draw anything itself: the compute
    /// pass fills an off-screen image, and egui's own renderer blits that image
    /// to the window later in the same frame, as part of the ordinary UI paint.
    /// That is what makes this renderer work inside eframe at all — eframe owns
    /// the swap chain and will present whatever its renderer produced.
    ///
    /// # The `resolution` contract
    ///
    /// `uniforms.resolution` **must** equal [`RenderOutcome::size`] exactly, in
    /// physical pixels. The kernel bounds-checks its dispatch against the
    /// uniform but `textureStore`s into an image of `size()`:
    ///
    /// * uniform larger than the image — the shader's own bounds check admits
    ///   invocations that store past the end of the texture, which is undefined
    ///   behaviour rather than a dropped pixel;
    /// * uniform smaller — the right and bottom edges are never written and keep
    ///   stale contents, which during a resize is a visible smear of the old
    ///   plot.
    ///
    /// This function deliberately does **not** overwrite the field to "fix" a
    /// mismatch. Silently substituting a different resolution would shift every
    /// plot's mapping and hide the disagreement that caused it, which is much
    /// worse than an obvious failure. Set the field from [`Renderer::size`].
    ///
    /// # Errors
    ///
    /// None. A dropped or unusable frame is a normal outcome here — egui's
    /// renderer lock may be poisoned — and is reported through
    /// [`RenderOutcome::texture`] rather than as an error, because there is
    /// nothing the caller could do about it mid-frame.
    pub fn render(&mut self, uniforms: &Uniforms) -> RenderOutcome {
        // `bytemuck::cast_slice` on a one-element array borrows a temporary
        // array, which is fine: `write_buffer` copies immediately and the borrow
        // does not outlive the statement.
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[*uniforms]));

        let size = self.size;
        let (width, height) = size;

        // Round the workgroup count *up* in each axis. The kernel guards every
        // invocation with a bounds check, so the overhang of the last workgroup
        // is discarded on the GPU rather than clamped on the CPU. Rounding down
        // would leave the right and bottom edges unrendered after any resize
        // that is not a multiple of the workgroup size.
        let groups_x = width.div_ceil(WORKGROUP_SIZE);
        let groups_y = height.div_ceil(WORKGROUP_SIZE);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("steel-pulse:plot"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("steel-pulse:domain_coloring:pass"),
                // Only written when timing is on. `None` otherwise, so the pass
                // costs exactly what it costs without instrumentation. A
                // pass-level timestamp write needs `Features::TIMESTAMP_QUERY`
                // and *not* the non-default `TIMESTAMP_QUERY_INSIDE_ENCODERS`,
                // which is only for writing timestamps on the encoder itself.
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

        self.queue.submit(std::iter::once(encoder.finish()));

        // The query resolve goes in its own submission rather than at the tail of
        // the frame's command buffer. wgpu-core does track query-set writes and
        // would accept either, but a separate submit is unambiguous under every
        // reading of the spec and costs nothing here, because this whole block
        // only exists when the app layer opted in to timing.
        self.encode_timestamp_resolve();

        let texture = self.ensure_registered();
        let gpu_ms = self.resolve_timestamps_ms();

        RenderOutcome {
            texture,
            size,
            gpu_ms,
        }
    }

    /// Current image size, `(width, height)`, in physical pixels.
    ///
    /// This is the value to copy into `uniforms.resolution`; see the module
    /// documentation for why getting it wrong is undefined behaviour rather
    /// than a cosmetic error.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// A one-line description of the GPU stack, for the UI's status bar.
    ///
    /// Built from the real [`wgpu::AdapterInfo`] captured at construction rather
    /// than from a compile-time constant, so it reflects the adapter eframe
    /// actually selected — which is the only thing a bug report needs. Reads as
    /// `"wgpu 30.0.1 / Metal / Apple M-series"`.
    ///
    /// `Backend` and `DeviceType` implement neither `Display` nor a lowercase
    /// name, so they are spelled out explicitly.
    pub fn backend_description(&self) -> String {
        format!(
            "wgpu {WGPU_VERSION_LABEL} / {} / {}",
            backend_label(self.adapter_info.backend),
            device_type_label(self.adapter_info.device_type),
        )
    }

    /// Every adapter wgpu could see at startup, one per line, for the UI's
    /// diagnostics block.
    ///
    /// Rendered once in [`Renderer::new`] and cached as a string: the list cannot
    /// change while the process runs, and the value is only ever read to paint
    /// a read-only panel. Returns `"<none reported>"` on targets where eframe
    /// does not publish the list, which today means only wasm.
    pub fn adapter_summary(&self) -> String {
        self.adapter_list.clone()
    }

    /// The uniform buffer, for tests and for debug tooling that wants to read
    /// back what the last frame uploaded.
    pub fn uniform_buffer(&self) -> &wgpu::Buffer {
        &self.uniform_buffer
    }

    /// Tries to turn on GPU timing for the compute pass.
    ///
    /// Returns `true` if timing is now available.
    ///
    /// # This returns `false` with a default eframe configuration
    ///
    /// Device features are fixed at device-creation time and immutable
    /// afterwards. eframe's built-in `WgpuSetupCreateNew` builds its
    /// `DeviceDescriptor` with `..Default::default()`, and `Default` for
    /// `required_features` is `Features::empty()` — so **eframe does not request
    /// `TIMESTAMP_QUERY`**, and this returns `false` unless the app layer
    /// supplies its own device descriptor. This is not faked and not worked
    /// around: a `QuerySet` cannot be created on a device that lacks the
    /// feature, and `RenderOutcome::gpu_ms` stays `None`.
    ///
    /// The app layer can enable it by overriding the descriptor on the way into
    /// eframe, which must happen **before** `App::new`:
    ///
    /// ```ignore
    /// let mut options = eframe::NativeOptions { ..Default::default() };
    /// let ef::egui_wgpu::WgpuSetup::CreateNew(mut create_new) =
    ///     ef::egui_wgpu::WgpuSetup::without_display_handle()
    /// else {
    ///     unreachable!("eframe's own default is WgpuSetup::CreateNew");
    /// };
    /// create_new.device_descriptor = std::sync::Arc::new(|adapter: &wgpu::Adapter| {
    ///     wgpu::DeviceDescriptor {
    ///         required_features: wgpu::Features::TIMESTAMP_QUERY,
    ///         // Mirrors egui-wgpu's own default, which caps the 2D texture
    ///         // dimension so a 4k+ surface fits.
    ///         required_limits: wgpu::Limits {
    ///             max_texture_dimension_2d: 8192,
    ///             ..wgpu::Limits::default()
    ///         },
    ///         ..Default::default()
    ///     }
    /// });
    /// options.wgpu_options.wgpu_setup =
    ///     ef::egui_wgpu::WgpuSetup::CreateNew(create_new);
    /// ```
    ///
    /// Requesting the feature unconditionally would be wrong in the other
    /// direction: `request_device` *fails* when asked for a feature the adapter
    /// lacks, so a hard request would stop the app from starting on a machine
    /// that could otherwise run it perfectly well. The feature check here is the
    /// honest version of that gate, applied as late as the immutable API allows.
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

        // A two-slot `QueryType::Timestamp` set is unconditionally valid for a
        // device holding the feature, so no speculative creation and no error
        // scope is needed.
        self.timestamp_query_set = Some(self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("steel-pulse:timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        }));

        // `QUERY_RESOLVE` and `MAP_READ` on one buffer lets the resolve target be
        // mapped directly, with no staging copy. Resolution writes
        // `QUERY_SIZE` bytes per query at an offset aligned to
        // `QUERY_RESOLVE_BUFFER_ALIGNMENT`; offset 0 and this size satisfy both.
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

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    /// Registers [`Renderer::sampled_view`] with egui if it is not already, and
    /// returns the handle.
    ///
    /// Registration allocates a fresh [`egui::TextureId`] and a fresh bind group
    /// inside egui's renderer, so it happens once per texture — that is, the
    /// first frame after construction and the first frame after each
    /// [`Renderer::resize`] — and never again. Calling it per frame would leak
    /// one texture-id slot and one bind group per frame.
    ///
    /// The lock is held for the duration of a bind-group allocation and nothing
    /// else: no GPU work, no allocation of the image, no blocking wait. That
    /// matters because eframe takes the same lock while painting, and because
    /// epaint's `RwLock` has a debug-build deadlock detector that panics rather
    /// than waiting indefinitely.
    ///
    /// The lock cannot be poisoned — `epaint::mutex::RwLock` wraps
    /// `parking_lot`, which has no poisoning state — so this cannot fail and
    /// always returns `Some`. The lock is not held across any call that can
    /// panic, so a deadlock here is not reachable either.
    fn ensure_registered(&mut self) -> Option<egui::TextureId> {
        if let Some(id) = self.registered {
            return Some(id);
        }

        let mut egui_renderer = self.egui_renderer.write();
        let id = egui_renderer.register_native_texture_with_sampler_options(
            &self.device,
            &self.sampled_view,
            PLOT_SAMPLER,
        );
        drop(egui_renderer);

        self.registered = Some(id);
        Some(id)
    }

    /// Asks egui to drop its handle on the current view, so the bind group
    /// referencing it can be collected before the texture is replaced.
    fn unregister(&mut self) {
        let Some(id) = self.registered.take() else {
            return;
        };
        // egui stores `texture: None` for natively-registered textures, so this
        // releases the bind group and leaves our `wgpu::Texture` alone for the
        // caller to drop. The write guard is scoped tightly for the same reason
        // as in `ensure_registered`.
        self.egui_renderer.write().free_texture(&id);
    }

    /// Encodes and submits the query-set resolve for this frame's compute pass.
    ///
    /// A no-op unless timestamps are enabled. Kept separate from
    /// [`Renderer::render`]'s command buffer so the resolve is unambiguously
    /// ordered after the pass that wrote the queries.
    fn encode_timestamp_resolve(&self) {
        let (Some(query_set), Some(readback)) = (
            self.timestamp_query_set.as_ref(),
            self.timestamp_readback.as_ref(),
        ) else {
            return;
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("steel-pulse:timestamps:resolve"),
            });
        encoder.resolve_query_set(query_set, 0..2, readback, 0);
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Decodes the most recent GPU duration for the compute pass, in
    /// milliseconds, initiating at most one readback at a time.
    ///
    /// Deliberately **non-blocking**. The readback maps a buffer the GPU wrote a
    /// frame or two ago; waiting on it with `PollType::Wait` would serialise CPU
    /// and GPU and destroy the pipelining that a plot this size depends on. So
    /// this kicks off at most one readback at a time, gives the callback a single
    /// non-blocking poll to run, and returns whatever the previous readback
    /// produced — a frame stale, and therefore useless for synchronisation but
    /// exactly right for telemetry.
    fn resolve_timestamps_ms(&mut self) -> Option<f64> {
        let readback = self.timestamp_readback.clone()?;

        {
            let mut state = lock_timestamps(&self.timestamp_state);
            if state.in_flight {
                // A readback is outstanding; report the last decoded value rather
                // than trying to map a buffer that is already mapped.
                return state.latest.map(|d| d.as_secs_f64() * 1_000.0);
            }
            state.in_flight = true;
        }

        // Converts raw query ticks to seconds. It is 1.0 on the web and around
        // 1.0 on most native backends, but reading it is the only correct way to
        // do the arithmetic.
        let period = f64::from(self.queue.get_timestamp_period());

        let readback_for_callback = readback.clone();
        let state = Arc::clone(&self.timestamp_state);
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let mut state = lock_timestamps(&state);
                // Always clear the in-flight flag, even on failure, or no further
                // readback would ever be attempted.
                state.in_flight = false;

                if result.is_err() {
                    state.dropped += 1;
                    return;
                }

                let mapped = readback_for_callback.slice(..).get_mapped_range();
                let seconds = match mapped {
                    Ok(view) => {
                        // Two little-endian u64s in submission order. The buffer
                        // is exactly two queries wide, so `chunks_exact` yields
                        // exactly two words and anything malformed shows up as a
                        // short iterator rather than as a panic.
                        let mut words = view.chunks_exact(8).map(|word| {
                            u64::from_le_bytes([
                                word[0], word[1], word[2], word[3], word[4], word[5], word[6],
                                word[7],
                            ])
                        });
                        let start = words.next();
                        let end = words.next();
                        drop(view);
                        // Unmap before returning: the next `map_async` is only
                        // legal on an unmapped buffer.
                        readback_for_callback.unmap();

                        match (start, end) {
                            // `saturating_sub` rather than `checked_sub`: the
                            // query set holds whatever the last completed frame
                            // wrote, so a wrapped value means this frame's pass
                            // did not run and the reading is meaningless. It is
                            // filtered out below rather than reported.
                            (Some(start), Some(end)) => {
                                Some(end.saturating_sub(start) as f64 * period)
                            }
                            _ => None,
                        }
                    }
                    Err(_) => None,
                };

                match seconds {
                    // Zero means the query set was never written this frame, not
                    // a zero-cost pass. The upper bound rejects a nonsense value
                    // before `Duration::from_secs_f64` can panic on it.
                    Some(s) if s > 0.0 && s.is_finite() && s < 1.0 => {
                        state.latest = Some(Duration::from_secs_f64(s));
                    }
                    _ => state.dropped += 1,
                }
            });

        // One non-blocking poll so a readback that already completed is decoded on
        // this call rather than the next. `Poll` never waits, so it cannot stall
        // the frame loop. A device error here is not actionable and is reported
        // through the uncaptured-error handler anyway.
        let _ = self.device.poll(wgpu::PollType::Poll);

        lock_timestamps(&self.timestamp_state)
            .latest
            .map(|d| d.as_secs_f64() * 1_000.0)
    }
}

/// Allocates a `width` x `height` `rgba8unorm` image, its two views, and the
/// bind group that writes it.
///
/// A free function rather than a method so [`Renderer::new`] can build the
/// target from local handles and [`Renderer::resize`] from `self`'s, without
/// either path needing placeholder values to satisfy struct initialisation.
///
/// `width` and `height` are clamped rather than trusted: a `0x0` `create_texture`
/// is a validation error that poisons the device, and a minimised window really
/// does report zero, so the invariant is enforced at the one place that allocates
/// rather than trusted from every call site.
fn build_storage_target(
    device: &wgpu::Device,
    compute_layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
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
        // `TEXTURE_BINDING` for egui's `texture_2d<f32>`.
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

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
                    // `UNIFORM_BUFFER_SIZE` and stays correct if the struct ever
                    // grows.
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&storage_view),
            },
        ],
    });

    StorageTarget {
        texture,
        storage_view,
        sampled_view,
        compute_bind_group,
    }
}

/// Human-readable name for a wgpu backend.
///
/// `Backend` implements `Debug` but not `Display`, and the `Debug` rendering
/// happens to be the spelling we want for most variants; the two that are not
/// are spelled out here.
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
/// Worth spelling out because the distinction is the point of asking eframe for
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

/// Renders the adapter list once, for the UI's diagnostics panel.
///
/// `RenderState::available_adapters` is `#[cfg]`-gated to non-wasm targets, and
/// this crate targets macOS and Windows only (see `AGENTS.md`), so the field is
/// read unconditionally. The `cfg` is noted rather than encoded because a
/// second code path for a platform this project does not ship to would be
/// untestable here.
fn describe_adapters(state: &egui_wgpu::RenderState) -> String {
    let adapters = &state.available_adapters;
    if adapters.is_empty() {
        return "<none reported>".to_owned();
    }
    let mut out = String::new();
    for adapter in adapters {
        if !out.is_empty() {
            out.push('\n');
        }
        let info = adapter.get_info();
        out.push_str(&format!(
            "{} / {} / {}",
            info.name,
            backend_label(info.backend),
            device_type_label(info.device_type),
        ));
    }
    out
}

// Compile-time guards for the invariants this module documents but cannot
// enforce at a call site. A failure here names the exact assumption that broke,
// which is worth more than the first validation error it would otherwise cause.
const _: () = {
    assert!(
        WORKGROUP_SIZE > 0,
        "a zero workgroup size dispatches nothing"
    );
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
