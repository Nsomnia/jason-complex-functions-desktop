//! Shared GPU ABI: the single source of truth for the uniform block consumed by
//! `shaders/domain_coloring.wgsl`.
//!
//! # Why this file exists
//!
//! The CPU side (Rust) and the GPU side (WGSL) must agree byte-for-byte on the
//! layout of the uniform buffer, and the *function dispatch table* must agree on
//! integer identifiers. This module owns both contracts. Nothing else in the
//! crate is allowed to invent a field, reorder one, or renumber a function.
//!
//! Changing anything here requires a matching edit in the WGSL, and vice versa.
//! See `agents/ABI.md` for the full rules.
//!
//! # Memory layout rules (wgpu uniform address space)
//!
//! `Uniforms` is 80 bytes and satisfies `align_of == 16`, so it can be uploaded
//! with a single `device.queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[u]))`.
//! Any added or removed field must keep the total size a multiple of 16, or wgpu
//! will reject the buffer at creation time rather than at first draw, which is a
//! miserable way to find out.

/// Uniform block shared by the compute kernel and the blit pass.
///
/// Field-for-field this mirrors the `Uniforms` struct in
/// `shaders/domain_coloring.wgsl`. Keep the two in lockstep.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    /// Render-target size in pixels, `(width, height)`. Used to normalise
    /// pixel coordinates to the `[-1, 1]` quad before mapping to the complex plane.
    pub resolution: [f32; 2],
    /// Centre of the viewport in the complex plane, `(re, im)`.
    pub center: [f32; 2],
    /// Half-height of the viewport measured in complex units. The visible
    /// region is a rectangle of aspect `resolution.x / resolution.y` and height
    /// `2 * scale`, centred on `center`.
    pub scale: f32,
    /// Iteration cap for escape-time and series-based functions. Values that
    /// do not converge within this many steps are treated as "at infinity".
    pub max_iter: u32,
    /// Function selector. Must be one of the discriminants declared by
    /// [`crate::functions::FunctionId`]. The WGSL side dispatches on this with
    /// a `switch`.
    pub func_id: u32,
    /// Rotates the hue by this many turns (0.0 - 1.0). Purely cosmetic; use it
    /// to bring a given colour band to the front while reasoning about a plot.
    pub phase: f32,
    /// How many iso-modulus contour bands to draw per e-fold. 0.0 disables
    /// modulus contours entirely.
    pub modulus_contour_density: f32,
    /// How many iso-phase (constant argument) contour bands to draw per turn.
    /// 0.0 disables phase contours.
    pub phase_contour_density: f32,
    /// Additional lightening exponent applied to the modulus term. Values below
    /// 1.0 exaggerate the shading, 0.0 flattens it to a pure hue map.
    pub modulus_shading: f32,
    /// Draws unit circles and axes over the domain in world space when non-zero.
    pub grid_enabled: u32,
    /// Monotonic frame counter. Lets the kernel derive cheap deterministic
    /// dithering and lets telemetry correlate GPU work with CPU work.
    pub frame: u32,
    /// When non-zero, the selected function is applied repeatedly
    /// `max_iter` times instead of once. This is what turns an arbitrary
    /// function into a Julia-style set, and it is the single most important
    /// control in the program.
    pub iterate: u32,
    /// Explicit padding to round the struct up to 64 bytes. Never read this; it
    /// exists so the size stays a multiple of 16, which is the alignment WGSL
    /// demands of anything in the uniform address space.
    pub _pad: [u32; 2],
}

impl Default for Uniforms {
    /// Defaults describe a 3x2 view of the complex plane centred on the origin,
    /// colouring the identity map. This is the state the app opens in.
    fn default() -> Self {
        Self {
            resolution: [1.0, 1.0],
            center: [0.0, 0.0],
            scale: 1.5,
            max_iter: 256,
            func_id: 0,
            phase: 0.0,
            modulus_contour_density: 8.0,
            phase_contour_density: 16.0,
            modulus_shading: 1.0,
            grid_enabled: 1,
            frame: 0,
            iterate: 0,
            _pad: [0; 2],
        }
    }
}

// Compile-time guards. If someone adds a field and breaks the layout, the build
// fails here with a readable message instead of at runtime.
//
// Note the asymmetry: the Rust struct aligns to 4 (all members are 32-bit
// scalars), but WGSL rounds a uniform struct's alignment up to 16. That is fine,
// because a standalone uniform buffer starts at offset 0. The only real
// requirement is that the total size be a multiple of 16, which is what
// `_pad` buys us and what these guards enforce.
const _: () = assert!(std::mem::size_of::<Uniforms>() % 16 == 0);
const _: () = assert!(std::mem::size_of::<Uniforms>() == 64);

/// Number of bytes the uniform buffer must occupy.
pub const UNIFORM_BUFFER_SIZE: u64 = std::mem::size_of::<Uniforms>() as u64;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_is_exactly_sixty_four_bytes() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 64);
    }

    #[test]
    fn struct_size_is_a_multiple_of_sixteen() {
        assert_eq!(std::mem::size_of::<Uniforms>() % 16, 0);
    }

    #[test]
    fn field_offsets_match_the_documented_wgsl_layout() {
        // These offsets are load-bearing: the WGSL struct declares its members
        // in this order, and a mismatch silently mis-colours the entire plot
        // rather than failing loudly. Assert every one of them.
        let u = Uniforms::default();
        let base = &u as *const Uniforms as usize;
        let offset = |p: *const u8| p as usize - base;

        assert_eq!(offset(&u.resolution[0] as *const f32 as *const u8), 0);
        assert_eq!(offset(&u.center[0] as *const f32 as *const u8), 8);
        assert_eq!(offset(&u.scale as *const f32 as *const u8), 16);
        assert_eq!(offset(&u.max_iter as *const u32 as *const u8), 20);
        assert_eq!(offset(&u.func_id as *const u32 as *const u8), 24);
        assert_eq!(offset(&u.phase as *const f32 as *const u8), 28);
        assert_eq!(
            offset(&u.modulus_contour_density as *const f32 as *const u8),
            32
        );
        assert_eq!(
            offset(&u.phase_contour_density as *const f32 as *const u8),
            36
        );
        assert_eq!(offset(&u.modulus_shading as *const f32 as *const u8), 40);
        assert_eq!(offset(&u.grid_enabled as *const u32 as *const u8), 44);
        assert_eq!(offset(&u.frame as *const u32 as *const u8), 48);
        assert_eq!(offset(&u.iterate as *const u32 as *const u8), 52);
        assert_eq!(offset(&u._pad[0] as *const u32 as *const u8), 56);
    }

    #[test]
    fn default_view_is_sane() {
        let u = Uniforms::default();
        assert_eq!(u.center, [0.0, 0.0]);
        assert!(u.scale > 0.0, "scale must be positive or the mapping flips");
        assert!(u.max_iter > 0);
        assert_eq!(u.func_id, 0, "identity map is function 0");
    }

    #[test]
    fn uniform_buffer_size_constant_matches_layout() {
        assert_eq!(UNIFORM_BUFFER_SIZE, 64);
    }
}
