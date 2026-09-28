// ============================================================================
//  blit.wgsl  --  fullscreen pass that presents the compute output
// ============================================================================
//
//  The compute kernel in `domain_coloring.wgsl` writes one rgba8unorm pixel per
//  invocation into a storage texture. Storage textures cannot be sampled, so
//  this pass does the presenting: it draws a single fullscreen triangle, reads
//  the storage texture through a *separate* sampled `texture_2d<f32>` view of
//  the same image, and writes it to the swapchain.
//
//  ---------------------------------------------------------------------------
//  BINDING LAYOUT  (the Rust side MUST match this exactly)
//  ---------------------------------------------------------------------------
//      @group(0) @binding(0)   var input_texture : texture_2d<f32>
//      @group(0) @binding(1)   var linear_sampler : sampler
//
//  Entry points:
//      vs_main     @vertex    -- no vertex buffer, no index buffer
//      fs_main     @fragment  -- one colour output at @location(0)
//
//  There is deliberately no uniform buffer in this pass. The triangle covers
//  the whole viewport, so the only thing the shader needs to know is its own
//  position, and the rasteriser supplies that for free via position and uv. The
//  two passes therefore have different binding layouts at group 0, which is
//  why they live in separate shader modules: a pipeline layout is shared by all
//  entry points in one module, so combining them would force a dummy binding
//  here and a stray one in the compute shader.
//
//  ---------------------------------------------------------------------------
//  WHY A TRIANGLE AND NOT A QUAD
//  ---------------------------------------------------------------------------
//  A quad needs 6 vertices (or 4 plus an index buffer) and draws two triangles,
//  which means the rasteriser evaluates the shared diagonal twice -- a seam can
//  appear along it if the fragment shader is not perfectly continuous. One
//  oversized triangle, generated from `vertex_index` with no vertex buffer at
//  all, covers the viewport exactly once with no seam and no extra memory
//  traffic. This is the standard "fullscreen triangle" and it is a single
//  primitive, so there is no diagonal at all.
//
//  The three clip-space positions are the corners of the viewport:
//
//        (-1, -1)  ( 3, -1)  (-1,  3)
//
//  The vertex at (3, 3)-equivalent position is the third one; it sits outside
//  the clip volume, and the GPU clips it away. The surviving triangle's
//  hypotenuse is the line x + y == 2, which passes exactly through the corner
//  (1, 1) of clip space. Because interpolation is perspective-correct, the uv
//  attribute -- also given with that same 0..2 overhang -- arrives at every
//  fragment of the viewport as a clean 0..1 across the whole screen. That is
//  why the uv values below extend past 1.0 rather than being clamped to the
//  corners: the overhang is deliberate and load-bearing.
//
//  ---------------------------------------------------------------------------
//  SAMPLING
//  ---------------------------------------------------------------------------
//  The texture is sampled with a `filtering` sampler rather than read through
//  `textureLoad`, so the presentation is a smooth bilinear (or, depending on
//  the Rust-side sampler descriptor, trilinear) resample from the offscreen
//  texture to the swapchain. That matters whenever the two differ in size --
//  window resize, or a swapchain chosen smaller than the render target to save
//  fill rate. `texture_2d<f32>` and `texture_storage_2d` are different binding
//  types in WebGPU, so the Rust side must create a *view* of the storage
//  texture with a sampled usage as well as the storage usage, and bind that
//  view at binding 0.
// ============================================================================

/// The offscreen render target, viewed as a sampleable float texture.
@group(0) @binding(0) var input_texture: texture_2d<f32>;

/// Filtering sampler. The Rust side should create this with
/// `SamplerDescriptor { mag_filter: Linear, min_filter: Linear, .. }` and
/// `address_mode_u/v: ClampToEdge` -- clamping is what keeps the uv overhang at
/// the triangle's far corner from wrapping around and smearing the opposite
/// edge of the image across the screen.
@group(0) @binding(1) var linear_sampler: sampler;

/// Colour output of the fragment stage. `rgba8unorm` matches the
/// `CanvasFormat::Rgba8Unorm` swapchain the host configures; if the host uses
/// `Rgba8UnormSrgb` or `Bgra8Unorm` the Rust side must adjust this type to
/// match, because WGSL requires the fragment output format to agree with the
/// pipeline target format.
///
/// Named `Varyings` rather than the conventional `VertexOutput`/`in`/`out`:
///
///   * `in` and `out` are reserved words in the WGSL specification even though
///     naga's own list happens not to carry them, so a stricter front-end such
///     as Tint or Dawn would reject the shader while naga accepted it. Since
///     the only WGSL front-end in play here is naga, and driver front-ends are
///     outside this crate's control, the shader stays on the safe side of the
///     spec rather than on the lenient side of the implementation.
///   * The struct is used as both the vertex output and the fragment input, so
///     a neutral name is more honest than either role's name.
struct Varyings {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

/// Generate the fullscreen triangle from `vertex_index` alone. No vertex buffer
/// is bound, so the only per-vertex data is this counter.
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> Varyings {
    var vout: Varyings;

    // vertex_index picks the corner: 0 -> bottom-left, 1 -> bottom-right,
    // 2 -> top-left.
    var clip = vec2<f32>(-1.0, -1.0);
    if (vertex_index == 1u) {
        clip = vec2<f32>(3.0, -1.0);
    } else if (vertex_index == 2u) {
        clip = vec2<f32>(-1.0, 3.0);
    }

    vout.clip_position = vec4<f32>(clip, 0.0, 1.0);

    // Map clip space to texture space. Clip +y is up, texture v is down, hence
    // the `1.0 -` on the v term; without it the image is presented upside down.
    // The same 0..2 overhang as the positions keeps this linear across the
    // triangle, so the far corner samples uv = (2, 2) and the viewport as a
    // whole lands exactly on 0..1.
    vout.uv = vec2<f32>(
        (clip.x + 1.0) * 0.5,
        (1.0 - clip.y) * 0.5,
    );

    return vout;
}

/// Sample the offscreen texture and emit it. The computed uv is already in
/// 0..1 over the visible viewport, so no clamping is needed; the sampler's
/// ClampToEdge address mode is the belt-and-braces for the degenerate far
/// corner.
@fragment
fn fs_main(vin: Varyings) -> @location(0) vec4<f32> {
    return textureSampleLevel(input_texture, linear_sampler, vin.uv, 0.0);
}
