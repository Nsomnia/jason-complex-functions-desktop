//! STEEL-PULSE v2.0-TURBO
//!
//! A hardware-accelerated native desktop recreation of Samuel J. Li's WebGL
//! Complex Function Plotter (https://samuelj.li/complex-function-plotter/),
//! expanded with Riemann-surface exploration.
//!
//! # Module map
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`uniforms`] | The shared CPU/GPU ABI. Single source of truth for buffer layout. |
//! | [`complex`] | Complex arithmetic and the function library, with tests. |
//! | [`camera`] | Viewport state: pan, zoom, pixel-to-complex mapping. |
//! | [`renderer`] | wgpu device, compute pipeline, render pipeline, frame loop. |
//! | [`telemetry`] | Frame timing and per-stage GPU/CPU statistics. |
//! | [`theme`] | The neon "3AM tweaker" visual language. |
//! | [`panel`] | Docked control panel bound to application state. |
//! | [`app`] | The `eframe::App` that wires all of the above together. |

mod app;
mod camera;
mod complex;
mod panel;
mod renderer;
mod telemetry;
mod theme;
mod uniforms;

fn main() -> eframe::Result<()> {
    app::run()
}
