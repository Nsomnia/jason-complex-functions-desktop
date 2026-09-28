//! Integration layer: the `eframe::App` that wires the camera, the control
//! panel, the GPU renderer and the telemetry into one loop.
//!
//! # Frame structure
//!
//! ```text
//!   App::update
//!     |-- Telemetry::begin_frame
//!     |-- egui_dock::DockArea::show_inside          (borrows &mut dock)
//!     |     |-- Tab::Controls -> panel::controls    (mutates Uniforms)
//!     |     \-- Tab::Plot     -> draw_plot
//!     |           |-- input: drag to pan, wheel to zoom -> Camera
//!     |           |-- Renderer::resize(physical pixels of the plot rect)
//!     |           |-- uniforms.resolution <- renderer.size()   <-- see below
//!     |           \-- Renderer::render -> TextureId -> paint
//!     |-- Camera::update(dt)  (ease targets, detect settled)
//     \-- Telemetry::end_frame
//! ```
//!
//! # The resolution invariant
//!
//! `Uniforms::resolution` **must** equal the renderer's storage-texture size
//! exactly, in physical pixels. The WGSL kernel bounds-checks its dispatch
//! against the uniform and then `textureStore`s into the image. If the uniform
//! is larger the shader writes out of bounds, which is undefined behaviour
//! rather than a dropped pixel; if it is smaller the right and bottom edges
//! are never written.
//!
//! Three plausible sources of that number all disagree on a Retina display:
//! the window surface size in physical pixels, egui's logical points, and the
//! plot rectangle's available size. The plot rectangle scaled by
//! [`egui::Context::pixels_per_point`] is the authority, because that is the
//! texture egui will actually sample into. See `agents/ABI.md`.
//!
//! # Why the compute pass is submitted from here
//!
//! The compute pass is encoded and submitted inside `App::update`, outside
//! eframe's own render pass. That looks wrong and is not: both submissions go
//! to the same queue, so the compute pass is guaranteed to execute before the
//! render pass that samples the resulting texture. It is legal and the
//! ordering is exactly what is wanted.

use std::time::{Duration, Instant};

use crate::camera::Camera;
use crate::panel::{PanelAction, TelemetryView};
use crate::renderer::{RenderOutcome, Renderer};
use crate::telemetry::Telemetry;
use crate::theme;
use crate::uniforms::Uniforms;

/// Zoom applied per wheel notch, as a multiplicative factor.
///
/// `exp(-0.0015 * pixels)` is used instead of a fixed factor so that a
/// trackpad (which emits many small, fractional wheel deltas) and a notched
/// mouse wheel (which emits few large ones) both feel linear to the hand.
const ZOOM_SENSITIVITY: f64 = 0.0015;

/// Longest frame delta fed to the camera's easing. A window dragged from
/// another monitor or un-hidden from a stall produces one enormous delta,
/// which would otherwise snap the camera to its target in a single step.
const MAX_FRAME_DELTA_SECONDS: f64 = 1.0 / 15.0;

/// Launch the application.
///
/// Builds the native window with the project's own dark styling applied, then
/// hands control to eframe. All of the real work happens in [`App`].
pub fn run() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("STEEL-PULSE v2.0-TURBO")
            .with_inner_size([1440.0, 980.0])
            .with_min_inner_size([640.0, 480.0]),
        ..Default::default()
    };

    eframe::run_native(
        "STEEL-PULSE v2.0-TURBO",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)) as Box<dyn eframe::App>)),
    )
}

/// The tabs of the dock layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Tab {
    /// The domain-colouring plot. Fills whatever rectangle it is given.
    Plot,
    /// The neon control panel.
    Controls,
}

impl Tab {
    /// Short, all-caps title for the tab strip.
    fn title(self) -> &'static str {
        match self {
            Tab::Plot => "PLOT",
            Tab::Controls => "CONTROLS",
        }
    }
}

/// Everything that is not the dock layout.
///
/// This is a separate struct purely so that [`DockApp::viewer`] can hold a
/// `&mut Core` while `DockArea` holds a `&mut DockState` over the same parent
/// struct. Splitting one `App` into two fields makes those borrows disjoint,
/// which the borrow checker needs and which nesting cannot express.
struct Core {
    camera: Camera,
    uniforms: Uniforms,
    telemetry: Telemetry,
    /// `None` when the GPU path is unavailable - eframe is running its `glow`
    /// renderer, or the pipelines failed to build. The window still opens and
    /// says so, rather than dying with no explanation.
    renderer: Option<Renderer>,
    /// Why the GPU path is unavailable, when it is.
    renderer_error: Option<String>,
    /// The egui handle for the most recently rendered plot texture.
    ///
    /// Stable between frames, but invalidated by a resize, so it is refreshed
    /// from every [`RenderOutcome`] rather than trusted.
    plot_texture: Option<egui::TextureId>,
    /// Everything the GPU needs to know, captured at the last render. When it
    /// is unchanged the frame skips the GPU entirely.
    last_rendered: Option<RenderKey>,
    /// The first panel action seen this frame, kept because the panel is
    /// called from inside the dock borrow and the action is handled after it.
    pending_action: Option<PanelAction>,
    /// Set when the window or plot is resized, since a stale texture would
    /// otherwise be stretched.
    size_dirty: bool,
    /// When the previous frame began, for the frame delta. `None` on the very
    /// first frame.
    last_frame_start: Option<Instant>,
    /// A screenshot has been asked for and the readback is still in flight.
    /// The readback is non-blocking, so this persists across frames until the
    /// GPU copy actually lands.
    screenshot_requested: bool,
    /// Where the most recent screenshot went, or why it failed. Shown as a
    /// transient overlay in the plot area.
    last_screenshot: Option<String>,
    /// Quit once a screenshot has been written. Only ever set together with
    /// `screenshot_requested`, so the flag cannot be left set by hand.
    exit_after_capture: bool,
}

/// The subset of [`Uniforms`] plus the target size that determines what the
/// GPU would produce. Compared field-by-field each frame to decide whether a
/// re-render is needed at all.
///
/// This is deliberately explicit rather than a memcmp of the struct: the
/// struct contains padding and a frame counter that changes by design, and a
/// byte comparison would either miss real changes or fire every frame.
#[derive(Clone, Copy, PartialEq, Debug)]
struct RenderKey {
    center: [f32; 2],
    scale: f32,
    size: [u32; 2],
    max_iter: u32,
    func_id: u32,
    phase: f32,
    modulus_contour_density: f32,
    phase_contour_density: f32,
    modulus_shading: f32,
    grid_enabled: u32,
    iterate: u32,
}

impl RenderKey {
    /// Capture the key for the current state. `size` is the renderer's actual
    /// texture size, not the requested one, because that is what the kernel
    /// will be writing into.
    fn capture(uniforms: &Uniforms, size: (u32, u32)) -> Self {
        Self {
            center: uniforms.center,
            scale: uniforms.scale,
            size: [size.0, size.1],
            max_iter: uniforms.max_iter,
            func_id: uniforms.func_id,
            phase: uniforms.phase,
            modulus_contour_density: uniforms.modulus_contour_density,
            phase_contour_density: uniforms.phase_contour_density,
            modulus_shading: uniforms.modulus_shading,
            grid_enabled: uniforms.grid_enabled,
            iterate: uniforms.iterate,
        }
    }
}

impl Core {
    fn new() -> Self {
        // Setting `STEEL_PULSE_CAPTURE` writes one frame to a PPM and quits.
        // It exists because OS screenshotting cannot verify this project on
        // some machines, so the app has to be able to inspect its own output
        // unattended. Unset by default: an interactive session never
        // auto-quits, which would be infuriating.
        let capture_on_exit = std::env::var_os("STEEL_PULSE_CAPTURE").is_some();
        Self {
            camera: Camera::new(),
            uniforms: Uniforms::default(),
            telemetry: Telemetry::new(),
            renderer: None,
            renderer_error: None,
            plot_texture: None,
            last_rendered: None,
            pending_action: None,
            size_dirty: true,
            last_frame_start: None,
            last_screenshot: None,
            screenshot_requested: capture_on_exit,
            exit_after_capture: capture_on_exit,
        }
    }

    /// Fold the camera's current state into the uniform block.
    ///
    /// The camera owns the view; the uniform block is only ever a transport for
    /// it. The two are kept apart so the camera can stay in `f64` for
    /// pan/zoom accumulation and cross into `f32` exactly once, here.
    fn sync_uniforms_from_camera(&mut self) {
        self.uniforms.center = self.camera.as_uniform_center();
        self.uniforms.scale = self.camera.as_uniform_scale();
    }

    /// Handle one action raised by the control panel.
    fn apply_action(&mut self, action: PanelAction) {
        // The panel has already written the new values into `Uniforms` for the
        // colour and view resets, so most of these only need to move the
        // camera, which the panel does not own.
        match action {
            PanelAction::ResetView | PanelAction::ResetAll => {
                self.camera
                    .set_target(0.0, 0.0, crate::camera::DEFAULT_SCALE);
            }
            PanelAction::ResetColor => {
                // The panel already restored the colour fields. Nothing to do
                // here beyond invalidating the render, which the key compare
                // handles on its own.
            }
            PanelAction::RandomizeJulia => {
                // The panel chose the seed and set the uniform. Reset the view
                // so a new seed is not judged from wherever the user left the
                // camera, which would usually be deep inside the old set.
                self.camera
                    .set_target(0.0, 0.0, crate::camera::DEFAULT_SCALE);
            }
            PanelAction::Screenshot => {
                self.screenshot_requested = true;
            }
        }
    }

    /// Perform a requested screenshot, if one is pending.
    ///
    /// The readback is non-blocking on the GPU side: [`Renderer::read_ppm`]
    /// returns `None` until the copy has landed, so this is retried across
    /// frames rather than stalling the UI thread waiting for it. The plot
    /// carries on rendering throughout.
    fn service_screenshot(&mut self) {
        if !self.screenshot_requested {
            return;
        }
        let Some(renderer) = self.renderer.as_mut() else {
            self.screenshot_requested = false;
            return;
        };
        let Some(ppm) = renderer.read_ppm() else {
            // Not ready yet. Ask for another frame so the readback can land.
            return;
        };
        self.screenshot_requested = false;
        self.last_screenshot = Some(self.write_screenshot(&ppm));
    }

    /// Write a PPM to disk and return a human-readable description of where.
    ///
    /// PPM rather than PNG deliberately: encoding a PNG needs a compression
    /// dependency, and a raw dump is also the more honest artefact when
    /// debugging a colour pipeline, because nothing has re-encoded the bytes
    /// between the GPU and the file. macOS's `sips` converts it if a viewable
    /// image is wanted.
    fn write_screenshot(&self, ppm: &[u8]) -> String {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let path =
            std::env::temp_dir().join(format!("steel-pulse-{stamp}.ppm"));
        match std::fs::write(&path, ppm) {
            Ok(()) => format!("saved {}", path.display()),
            Err(err) => format!("screenshot failed: {err}"),
        }
    }

    /// Snapshot the telemetry into the shape the panel renders.
    fn telemetry_view(&self) -> TelemetryView {
        let (resolution, backend) = match &self.renderer {
            Some(r) => {
                let (w, h) = r.size();
                ([w, h], r.backend_description())
            }
            None => (
                [0, 0],
                self.renderer_error
                    .clone()
                    .unwrap_or_else(|| "no GPU renderer".to_string()),
            ),
        };

        let snapshot = self.telemetry.snapshot();
        TelemetryView::new(
            snapshot.fps,
            snapshot.frame_ms,
            snapshot.gpu_ms,
            snapshot.cpu_ms,
            resolution,
            backend,
        )
        // The panel's sparkline wants milliseconds, oldest first, and the
        // history stores exactly that in its native order.
        .with_history(
            self.telemetry
                .history()
                .iter()
                .map(|f| f64::from(f))
                .collect(),
        )
    }
}

/// The application.
struct App {
    /// The dock tree. A separate field from [`Core`] so the tab viewer can
    /// borrow both at once - see [`Core`].
    dock: egui_dock::DockState<Tab>,
    core: Core,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        theme::apply(&cc.egui_ctx);

        let dock = egui_dock::DockState::new(vec![Tab::Plot, Tab::Controls]);

        let mut core = Core::new();

        // eframe owns the adapter, the device and the queue. We borrow its
        // render state rather than creating a second device, which would be a
        // different GPU context and would not be able to share the texture we
        // are about to produce. `None` means eframe is on its `glow` backend.
        match cc.wgpu_render_state.as_ref() {
            Some(state) => match Renderer::new(state) {
                Ok(renderer) => core.renderer = Some(renderer),
                Err(err) => core.renderer_error = Some(err.to_string()),
            },
            None => core.renderer_error = Some("eframe is not using the wgpu backend".to_string()),
        }

        Self { dock, core }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.core.telemetry.begin_frame();

        let dt = self.frame_delta_seconds();
        self.core.sync_uniforms_from_camera();
        self.core.camera.update(dt);

        // The dock borrows `self.dock`; the viewer borrows `self.core`. They
        // are disjoint fields, so this is the whole reason `Core` is separate.
        egui_dock::DockArea::new(&mut self.dock).show_inside(
            ui,
            &mut Viewer {
                core: &mut self.core,
            },
        );

        if let Some(action) = self.core.pending_action.take() {
            self.core.apply_action(action);
        }

        // Retry any in-flight screenshot. Non-blocking: returns immediately
        // until the GPU copy has landed, and asks for another frame meanwhile.
        self.core.service_screenshot();
        if self.core.screenshot_requested {
            ui.ctx().request_repaint();
        } else if self.core.exit_after_capture {
            // Unattended capture mode: the frame is on disk, so leave. Without
            // this the process would sit in an event loop with nothing to do.
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }

        // Keep repainting while anything is still moving. Once the camera has
        // settled and the plot matches what is on screen, the app idles and
        // costs no GPU time, which is the behaviour a static plot should have.
        let moving = !self.core.camera.is_settled() || self.core.size_dirty;
        if moving {
            ui.ctx().request_repaint();
        }

        self.core.telemetry.end_frame();
    }
}

impl App {
    /// Elapsed wall-clock time since the previous frame, clamped.
    ///
    /// egui's own `input` carries prediction and jitter-correction; using it
    /// for a frame timer would make the telemetry report a value the user
    /// cannot reproduce. A monotonic wall clock is what an oscilloscope
    /// actually wants. The first frame has no predecessor and reports zero,
    /// so nothing eases on startup.
    fn frame_delta_seconds(&mut self) -> f64 {
        let now = Instant::now();
        let dt = match self.core.last_frame_start {
            Some(previous) => now.duration_since(previous).as_secs_f64(),
            None => 0.0,
        };
        self.core.last_frame_start = Some(now);
        if dt.is_finite() && dt > 0.0 {
            dt.min(MAX_FRAME_DELTA_SECONDS)
        } else {
            0.0
        }
    }
}

/// Draws the contents of whichever dock tab is on screen.
struct Viewer<'a> {
    core: &'a mut Core,
}

impl egui_dock::TabViewer for Viewer<'_> {
    type Tab = Tab;

    fn id(&mut self, tab: &mut Tab) -> egui::Id {
        match tab {
            Tab::Plot => egui::Id::new("steel_pulse.plot"),
            Tab::Controls => egui::Id::new("steel_pulse.controls"),
        }
    }

    fn title(&mut self, tab: &mut Tab) -> egui::WidgetText {
        tab.title().into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Tab) {
        match tab {
            Tab::Plot => draw_plot(ui, self.core),
            Tab::Controls => {
                let telemetry = self.core.telemetry_view();
                if let Some(action) = crate::panel::panel(ui, &mut self.core.uniforms, &telemetry) {
                    // First action wins for the frame; see `panel::controls`.
                    if self.core.pending_action.is_none() {
                        self.core.pending_action = Some(action);
                    }
                }
            }
        }
    }
}

/// Draw the plot: consume input, size the GPU image, render it, paint it.
fn draw_plot(ui: &mut egui::Ui, core: &mut Core) {
    let ctx = ui.ctx().clone();
    let rect = ui.available_rect_before_wrap();

    // Allocate the whole area so the response covers the plot, then interact.
    ui.allocate_exact_size(rect.size(), egui::Sense::hover());
    let response = ui.interact(
        rect,
        ui.id().with("plot_area"),
        egui::Sense::click_and_drag(),
    );

    if response.hovered() {
        // Drag pans. `pan` is exact: it samples the pixel-to-complex mapping
        // twice and applies the negated difference, so the point under the
        // cursor tracks the image with no zoom-dependent drift.
        if response.dragged() {
            let d = response.drag_delta();
            core.camera.pan(
                d.x as f64,
                d.y as f64,
                plot_width_logical(rect),
                plot_height_logical(rect),
            );
        }

        // Wheel zooms about the cursor. The `exp` form makes a trackpad's
        // fractional deltas feel the same as a notched wheel's integers.
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            let factor = (-scroll as f64 * ZOOM_SENSITIVITY).exp();
            if let Some(hover) = response.hover_pos() {
                core.camera.zoom_at(
                    factor,
                    ((hover.x - rect.min.x) as f64, (hover.y - rect.min.y) as f64),
                    plot_width_logical(rect),
                    plot_height_logical(rect),
                );
            }
        }
    }

    // Report the failure modes instead of showing a blank plot that looks
    // like a bug in the mathematics.
    if core.renderer.is_none() {
        let message = core
            .renderer_error
            .clone()
            .unwrap_or_else(|| "GPU renderer unavailable".to_string());
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            format!("GPU UNAVAILABLE\n\n{message}"),
            egui::FontId::monospace(14.0),
            theme::Theme::AMBER,
        );
        return;
    }

    // Physical pixels: the texture is sampled into a logical-point rectangle,
    // so it must be allocated at the physical size or it renders soft.
    let ppp = ctx.pixels_per_point();
    let want = (
        (rect.width() * ppp).round().max(1.0) as u32,
        (rect.height() * ppp).round().max(1.0) as u32,
    );

    let renderer = core.renderer.as_mut().expect("checked above");
    renderer.resize(want.0, want.1);
    let actual = renderer.size();
    if actual != want {
        core.size_dirty = true;
    }

    // The invariant. Set the uniform from the renderer's actual size, never
    // from the request and never from egui's logical points.
    core.uniforms.resolution = [actual.0 as f32, actual.1 as f32];
    core.uniforms.center = core.camera.as_uniform_center();
    core.uniforms.scale = core.camera.as_uniform_scale();

    let key = RenderKey::capture(&core.uniforms, actual);
    let needs_render = core.last_rendered != Some(key) || core.size_dirty;

    if needs_render {
        let cpu_start = Instant::now();
        core.uniforms.frame = core.uniforms.frame.wrapping_add(1);
        let outcome: RenderOutcome = renderer.render(&core.uniforms);
        core.telemetry.record_cpu_time(cpu_start.elapsed());

        if let Some(gpu_ms) = outcome.gpu_ms {
            core.telemetry
                .record_gpu_time(Duration::from_secs_f64(gpu_ms / 1000.0));
        }

        // Always take the handle from this frame's outcome: `resize` frees the
        // previous one, and a freed handle's bind group points at a destroyed
        // view. Trusting a cached id across a resize is a use-after-free.
        core.plot_texture = outcome.texture;
        core.last_rendered = Some(key);
        core.size_dirty = false;
    }

    if let Some(texture) = core.plot_texture {
        egui::Image::from_texture(egui::load::SizedTexture::new(texture, rect.size()))
            .paint_at(ui, rect);
    }

    // Report where a screenshot went, so the user does not have to go looking
    // in the temp directory for it. Click to dismiss.
    if let Some(message) = core.last_screenshot.clone() {
        let galley = ui.painter().layout_no_wrap(
            message,
            egui::FontId::monospace(12.0),
            theme::Theme::TEXT,
        );
        let margin = 6.0;
        let pad = 4.0;
        let size = galley.size();
        let box_rect = egui::Rect::from_min_size(
            egui::pos2(
                rect.left() + 8.0,
                rect.bottom() - size.y - 2.0 * pad - 8.0,
            ),
            egui::vec2(size.x + 2.0 * margin, size.y + 2.0 * pad),
        );
        let response =
            ui.interact(box_rect, ui.id().with("screenshot_notice"), egui::Sense::click());
        if response.clicked() {
            core.last_screenshot = None;
        }
        ui.painter().rect_filled(box_rect, 2.0, theme::Theme::VOID);
        ui.painter().rect_stroke(
            box_rect,
            2.0,
            egui::Stroke::new(1.0, theme::Theme::CYAN_DIM),
            egui::StrokeKind::Inside,
        );
        ui.painter()
            .galley(box_rect.min + egui::vec2(margin, pad), galley, theme::Theme::TEXT);
    }
}

/// Width of the plot rectangle in the logical points the camera expects.
fn plot_width_logical(rect: egui::Rect) -> f32 {
    rect.width().max(1.0)
}

/// Height of the plot rectangle in logical points.
fn plot_height_logical(rect: egui::Rect) -> f32 {
    rect.height().max(1.0)
}
