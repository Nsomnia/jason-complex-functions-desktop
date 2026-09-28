//! `panel.rs` — the docked control surface.
//!
//! # What this module is for
//!
//! One vertical strip of hairline-ruled, dense, monospace controls bound
//! directly to [`Uniforms`], plus a read-only telemetry block. It is the only
//! place in the program where application state is mutated from user input.
//!
//! # Contract with the app layer
//!
//! The app layer owns everything. This module owns nothing: it is handed a
//! `&mut Uniforms` and a read-only [`TelemetryView`], mutates the uniforms in
//! place, and hands back a *request* rather than performing it.
//!
//! ```no_run
//! // Inside a `egui_dock` tab, a `SidePanel`, or a `Window`:
//! if let Some(action) = panel::controls(ui, &mut self.uniforms, &self.telemetry_view()) {
//!     self.handle(action);
//! }
//! ```
//!
//! # The function table
//!
//! [`FUNCTION_TABLE`] below is a **hand-maintained mirror** of the `FUNCTIONS`
//! constant in `src/functions.rs`. `panel.rs` deliberately does not import it:
//! the panel must keep compiling and the selector must keep rendering even
//! while the function library is being written, and — more importantly — the
//! UI needs a *presentation* order and a *search* index that the library has
//! no reason to know about. If the two ever disagree, the shader dispatches
//! the wrong map and the picture silently changes, so
//! [`tests::table_is_in_sync_with_the_library`] exists purely to fail the build
//! the moment a row is added, renamed or renumbered over there.
//!
//! # On `#[allow(dead_code)]`
//!
//! This crate is a binary, so `rustc` treats nothing in it as a library and
//! reports anything not reachable from `main` as dead. The items below marked
//! `#[allow(dead_code)]` exist *for* the app layer, which lives in a different
//! file. They are not dead code; they are un-called code, which is a different
//! thing. The attribute is confined to those items rather than applied to the
//! module, so genuine dead code inside this file is still reported.
//!
//! # Style
//!
//! Everything here is custom-painted. There are no default `DragValue`s
//! floating loose and no `CollapsingHeader`s: a tool this dense needs a fixed
//! row rhythm, right-aligned tabular numerals, and section rules that a user
//! can navigate by shape alone in peripheral vision.

use egui::{
    epaint::PathStroke, pos2, vec2, Align2, Color32, CornerRadius, FontFamily, FontId, Margin,
    Rect, Sense, Shape, Stroke, TextEdit, Ui,
};

use crate::theme::{self, Theme, FPS_BAD, FPS_WARN, FRAME_BUDGET_MS};
use crate::uniforms::Uniforms;

// ===========================================================================
// Public API — the surface the app layer needs verbatim
// ===========================================================================

/// Read-only view of the frame loop's measurements.
///
/// The app layer builds one of these per frame out of its own telemetry
/// counters. It is deliberately a plain value with public fields and a
/// [`Default`], so it can be constructed either by
/// [`TelemetryView::new`] or by struct literal.
///
/// # `frame_history`
///
/// An optional ring of recent frame times in milliseconds, **oldest first**.
/// This is what the sparkline draws. If it is empty or has fewer than two
/// samples the sparkline is replaced by an explicit `NO TRACE` readout rather
/// than a misleading flat line — see [`frame_sparkline`]. Nothing here is
/// required for the panel to work; the telemetry block degrades cleanly.
#[allow(dead_code)]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TelemetryView {
    /// Smoothed frames per second. Drives the colour of the `FPS` readout.
    pub fps: f64,
    /// Wall-clock milliseconds per frame, end to end.
    pub frame_ms: f64,
    /// Milliseconds spent inside the GPU (timestamp-query derived, if the
    /// adapter supports it; otherwise an estimate).
    pub gpu_ms: f64,
    /// Milliseconds spent on the CPU side of the frame: uniform packing,
    /// camera, encode, submit.
    pub cpu_ms: f64,
    /// Render-target size in pixels, `(width, height)`.
    pub resolution: [u32; 2],
    /// Graphics backend name, e.g. `"Metal / Apple M3 Max"`. Shown as-is.
    pub backend: String,
    /// Recent frame times in milliseconds, oldest first. Drives the sparkline.
    pub frame_history: Vec<f64>,
}

#[allow(dead_code)]
impl TelemetryView {
    /// Construct from the six required measurements, with no sparkline history.
    ///
    /// Attach a trace afterwards with [`TelemetryView::with_history`].
    pub fn new(
        fps: f64,
        frame_ms: f64,
        gpu_ms: f64,
        cpu_ms: f64,
        resolution: [u32; 2],
        backend: String,
    ) -> Self {
        Self {
            fps,
            frame_ms,
            gpu_ms,
            cpu_ms,
            resolution,
            backend,
            frame_history: Vec::new(),
        }
    }

    /// Attach a frame-time trace, oldest sample first.
    #[must_use]
    pub fn with_history(mut self, history: Vec<f64>) -> Self {
        self.frame_history = history;
        self
    }

    /// The green→amber→red health colour for this snapshot.
    ///
    /// Delegates to [`theme::status_color`], so the panel and the theme can
    /// never disagree about what "good" means.
    pub fn status_color(&self) -> Color32 {
        theme::status_color(self.fps)
    }
}

/// Something the user asked for that the panel cannot do by itself.
///
/// The panel owns [`Uniforms`] but not the window, the camera, or the disk, so
/// a handful of commands have to travel upward. The app layer matches on this
/// and does the work.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PanelAction {
    /// Recentre on the origin and restore the default half-height.
    ///
    /// Note: the panel has *already* applied this to [`Uniforms`] before
    /// returning, so the app layer does not need to. The action exists so the
    /// app can also reset anything the panel does not own — the pixel-ratio
    /// cache, an accumulated zoom animation, a "dirty" flag.
    ResetView,
    /// Restore the default colour mapping (phase, contour densities, shading
    /// exponent). Already applied to [`Uniforms`].
    ResetColor,
    /// Pick a fresh Julia seed and switch to iterated mode.
    ///
    /// The panel has already written the new seed into
    /// [`Uniforms::center`] and set [`Uniforms::iterate`]. The app layer may
    /// additionally reset the view or clear any cached render.
    RandomizeJulia,
    /// Capture the current plot to a file. Not performed here: the app layer
    /// owns the surface and the filesystem.
    Screenshot,
    /// Reset view *and* colour. Already applied to [`Uniforms`].
    ResetAll,
}

/// Draw the whole control panel into `ui`.
///
/// * `state` — the live uniform block. Mutated in place as the user drags.
/// * `telemetry` — read-only measurements for the telemetry block. Never
///   written to, and never retained.
///
/// Returns the first [`PanelAction`] raised this frame, if any. The panel
/// checks at most one action per frame: if the user somehow triggers two in
/// the same frame (impossible in practice, since they are distinct buttons in
/// different frames of the layout) the earlier one in document order wins.
/// That is deterministic and therefore testable, which matters more than
/// losing a redundant second click.
///
/// # Layout
///
/// Lays out top to bottom and expects to be given a `Ui` of the width it
/// should use. Either wrap it yourself:
///
/// ```no_run
/// egui::ScrollArea::vertical().show(ui, |ui| {
///     let _ = panel::controls(ui, &mut uniforms, &telemetry);
/// });
/// ```
///
/// or use [`panel`], which does exactly that.
///
/// `#[allow(dead_code)]`: called by the app layer, not from within this module.
#[allow(dead_code)]
pub fn controls(
    ui: &mut Ui,
    state: &mut Uniforms,
    telemetry: &TelemetryView,
) -> Option<PanelAction> {
    let mut action: Option<PanelAction> = None;

    masthead(ui, &mut action, state, telemetry);

    section_header(ui, "VIEW");
    view_section(ui, state, telemetry, &mut action);

    section_header(ui, "FUNCTION");
    function_section(ui, state);

    section_header(ui, "COLOR");
    color_section(ui, state, &mut action);

    section_header(ui, "GRID");
    grid_section(ui, state);

    section_header(ui, "TELEMETRY");
    telemetry_section(ui, telemetry);

    action
}

/// Convenience wrapper: the same panel, inside a vertical [`egui::ScrollArea`]
/// and a hairline-framed body, clipped to whatever space is available.
///
/// Use this from a dock tab or a side panel. If you need to embed the controls
/// in a layout of your own, call [`controls`] directly instead — it must not be
/// nested in a scroll area twice, or the wheel events will fight.
///
/// `#[allow(dead_code)]`: the app layer picks one of [`panel`] and [`controls`],
/// so exactly one of them can ever be called.
#[allow(dead_code)]
pub fn panel(ui: &mut Ui, state: &mut Uniforms, telemetry: &TelemetryView) -> Option<PanelAction> {
    // Paint the body onto the void rather than inheriting whatever the dock
    // tab or side panel happens to be filled with. The panel is a fixed
    // instrument, not a document, and its background must not move.
    egui::Frame::new()
        .fill(Theme::VOID)
        .inner_margin(Margin::symmetric(GUTTER, 4))
        .show(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("steel_pulse_panel")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.add_space(2.0);
                    controls(ui, state, telemetry)
                })
                .inner
        })
        .inner
}

// ===========================================================================
// Function table — hand-maintained mirror of `src/functions.rs`
// ===========================================================================
//
// KEEP IN SYNC WITH `FUNCTIONS` IN src/functions.rs.
// `ids` are the shader's `switch` discriminants, written into
// `Uniforms::func_id`. Changing one here without changing one there draws the
// wrong map with no error anywhere. `tests::table_is_in_sync_with_the_library`
// compares this table against the real one.

/// The kind of map, used purely to group the selector.
///
/// Mirrors `functions::FunctionGroup` variant for variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FunctionKind {
    /// Powers, shifts, and the identity.
    Polynomial,
    /// Maps whose point is what happens when you iterate them.
    Iterated,
    /// Maps with a pole, `1/z` and friends.
    Reciprocal,
    /// `sin`, `cos`, `sinc`.
    Trigonometric,
    /// `sinh`, and by extension `cosh`/`tanh`.
    Hyperbolic,
    /// `e^z` and `e^{iz}`.
    Exponential,
    /// `log` and `sqrt`, where the branch cuts live.
    Transcendental,
    /// Linear fractional (Moebius) transformations.
    Mobius,
}

impl FunctionKind {
    /// Presentation order of the groups in the selector.
    ///
    /// Not the declaration order of the library's enum, and not the numeric
    /// order of the ids: this is ordered by how a person thinks about picking
    /// something, which is "start with the simple algebraic ones, then the
    /// pole, then the transcendental, then the iterating ones".
    pub const ORDER: [FunctionKind; 8] = [
        FunctionKind::Polynomial,
        FunctionKind::Iterated,
        FunctionKind::Reciprocal,
        FunctionKind::Trigonometric,
        FunctionKind::Hyperbolic,
        FunctionKind::Exponential,
        FunctionKind::Transcendental,
        FunctionKind::Mobius,
    ];

    /// The section heading shown above the group's rows.
    pub const fn label(self) -> &'static str {
        match self {
            FunctionKind::Polynomial => "POLYNOMIAL",
            FunctionKind::Iterated => "ITERATED",
            FunctionKind::Reciprocal => "RECIPROCAL",
            FunctionKind::Trigonometric => "TRIGONOMETRIC",
            FunctionKind::Hyperbolic => "HYPERBOLIC",
            FunctionKind::Exponential => "EXPONENTIAL",
            FunctionKind::Transcendental => "TRANSCENDENTAL",
            FunctionKind::Mobius => "MOEBIUS",
        }
    }

    /// A one-line note explaining what the group is *for*.
    ///
    /// Shown under the selector for whichever function is selected. This is the
    /// part that turns a list into an instrument: the names in the table are
    /// terse on purpose, and the terseness needs paying back somewhere.
    pub const fn note(self) -> &'static str {
        match self {
            FunctionKind::Polynomial => "algebraic: no iteration needed to see structure",
            FunctionKind::Iterated => "iterate to resolve the fractal boundary",
            FunctionKind::Reciprocal => "pole at 0; the plane folds through the origin",
            FunctionKind::Trigonometric => "periodic in the real direction, unbounded above",
            FunctionKind::Hyperbolic => "grows along the real axis, periodic on the imaginary",
            FunctionKind::Exponential => "never returns; modulus encodes Re z",
            FunctionKind::Transcendental => "branch cuts and boundary conditions dominate",
            FunctionKind::Mobius => "linear fractional; acts on the Riemann sphere",
        }
    }
}

/// One row of the function selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FunctionRow {
    /// Dispatch id written into [`Uniforms::func_id`].
    pub id: u32,
    /// Terse lowercase name, e.g. `"basilica"`.
    pub name: &'static str,
    /// Display formula, exactly as the library spells it.
    pub formula: &'static str,
    /// Which group this row belongs to.
    pub kind: FunctionKind,
}

impl FunctionRow {
    /// Case-insensitive substring match against the name, the formula, and the
    /// bare id, so `"z3"`, `"3"`, `"cube"` and `"z^3"` all find the same row.
    pub fn matches(&self, query: &str) -> bool {
        let q = query.trim();
        if q.is_empty() {
            return true;
        }
        let q = q.to_ascii_lowercase();
        self.name.contains(&q)
            || self.formula.to_ascii_lowercase().contains(&q)
            || self.id.to_string() == q
            || self.kind.label().to_ascii_lowercase().contains(&q)
    }
}

/// The 16 dispatchable maps, in `id` order.
///
/// **Mirror of `functions::FUNCTIONS`.** See the file-level note.
pub const FUNCTION_TABLE: [FunctionRow; 16] = [
    FunctionRow {
        id: 0,
        name: "identity",
        formula: "z",
        kind: FunctionKind::Polynomial,
    },
    FunctionRow {
        id: 1,
        name: "square",
        formula: "z^2",
        kind: FunctionKind::Polynomial,
    },
    FunctionRow {
        id: 2,
        name: "cube",
        formula: "z^3",
        kind: FunctionKind::Polynomial,
    },
    FunctionRow {
        id: 3,
        name: "reciprocal",
        formula: "1/z",
        kind: FunctionKind::Reciprocal,
    },
    FunctionRow {
        id: 4,
        name: "z^2 - 1",
        formula: "z^2 - 1",
        kind: FunctionKind::Polynomial,
    },
    FunctionRow {
        id: 5,
        name: "z^3 - 1",
        formula: "z^3 - 1",
        kind: FunctionKind::Polynomial,
    },
    FunctionRow {
        id: 6,
        name: "basilica",
        formula: "z^3 - 2z",
        kind: FunctionKind::Iterated,
    },
    FunctionRow {
        id: 7,
        name: "sin",
        formula: "sin z",
        kind: FunctionKind::Trigonometric,
    },
    FunctionRow {
        id: 8,
        name: "cos",
        formula: "cos z",
        kind: FunctionKind::Trigonometric,
    },
    FunctionRow {
        id: 9,
        name: "sinc",
        formula: "sin z / z",
        kind: FunctionKind::Trigonometric,
    },
    FunctionRow {
        id: 10,
        name: "sinh",
        formula: "sinh z",
        kind: FunctionKind::Hyperbolic,
    },
    FunctionRow {
        id: 11,
        name: "exp",
        formula: "e^z",
        kind: FunctionKind::Exponential,
    },
    FunctionRow {
        id: 12,
        name: "log",
        formula: "ln z",
        kind: FunctionKind::Transcendental,
    },
    FunctionRow {
        id: 13,
        name: "sqrt",
        formula: "sqrt z",
        kind: FunctionKind::Transcendental,
    },
    FunctionRow {
        id: 14,
        name: "mobius",
        formula: "(z - 1) / (z + 1)",
        kind: FunctionKind::Mobius,
    },
    FunctionRow {
        id: 15,
        name: "julia",
        formula: "z^2 + c",
        kind: FunctionKind::Iterated,
    },
];

/// Number of dispatchable functions. Mirrors `FUNCTIONS.len()`.
pub const FUNCTION_COUNT: usize = FUNCTION_TABLE.len();

/// Narrowest the panel is designed to be, in points.
///
/// Set this as the `egui_dock` tab's or `SidePanel`'s `min_size` in the app
/// layer. The panel does not enforce it itself because a widget that silently
/// overflows its own pane is worse than a pane that refuses to get narrower.
///
/// The floor is set by two `egui` widgets with intrinsic minimum widths rather
/// than by anything in this file: a pair of six-decimal `DragValue`s side by
/// side (the Re/Im centre fields) needs about 226 pt, and the two buttons
/// flanking the phase field need about 200 pt. The layout is otherwise
/// fluid — every other row sizes itself from `Ui::available_width`, and the
/// notes wrap.
///
/// `#[allow(dead_code)]`: the app layer is the consumer, and it lives in a
/// different file.
#[allow(dead_code)]
pub const MIN_PANEL_W: f32 = 236.0;

/// Look up a selector row by dispatch id, or `None` if the id is out of range.
///
/// A `None` here means the uniform block holds an id the shader could never
/// have produced, which the panel renders as an explicit `ID n/a` rather than
/// silently highlighting the wrong row.
pub fn row_for_id(id: u32) -> Option<&'static FunctionRow> {
    FUNCTION_TABLE.get(id as usize)
}

/// Number of functions matching `query`, for the `n/16` counter.
pub fn match_count(query: &str) -> usize {
    FUNCTION_TABLE.iter().filter(|r| r.matches(query)).count()
}

// ===========================================================================
// Layout constants
// ===========================================================================

/// Height of a control row. Matches the theme's `interact_size.y`.
const ROW_H: f32 = 16.0;
/// Width reserved for a field label on the left of a control row. Clamped
/// against the panel width by [`field_row`], so a narrow dock shrinks the label
/// column before it starts truncating the numbers.
const LABEL_W: f32 = 66.0;
/// Gap between widgets inside a control row.
const ROW_GAP: f32 = 4.0;
/// Horizontal padding inside a rectangular button.
const BTN_PAD: f32 = 14.0;
/// Height of a telemetry readout row.
const READOUT_H: f32 = 13.0;
/// Height of a function-selector row.
const FN_ROW_H: f32 = 15.0;
/// Maximum height of the scrolling function list before it gets its own
/// scrollbar.
///
/// 168 px holds about nine rows plus a group heading, which is enough to scan
/// the table by group without pushing the colour controls below the fold on a
/// 1000 px window. The list scrolls for the rest, which is the right trade:
/// the *shape* of the whole panel stays visible, and only the list is paged.
const FN_LIST_MAX_H: f32 = 168.0;
/// Height of the frame-time sparkline.
const SPARK_H: f32 = 40.0;
/// Left gutter, in points, inside every framed block.
const GUTTER: i8 = 5;

/// Font used for tracked, uppercase section furniture.
fn label_font() -> FontId {
    FontId::new(Theme::SIZE_SMALL, FontFamily::Monospace)
}

/// Font used for numerals. Always monospace so digits are tabular and do not
/// shift the row as they change.
fn value_font() -> FontId {
    FontId::new(Theme::SIZE_BODY, FontFamily::Monospace)
}

/// U+2009 THIN SPACE.
///
/// `Painter::text` takes a plain string, not a [`RichText`], so it has no
/// letter-spacing knob. Tracking is therefore *baked into the string* as thin
/// spaces between glyphs. At these sizes one thin space is a little over two
/// points, which is the tracking this design wants, and because the face is
/// monospace the result still lands on the character grid.
///
/// Only *labels* are ever tracked. Values are left untracked, because
/// inserting a variable-width space into a right-aligned number would destroy
/// the tabular alignment the whole layout depends on.
const THIN_SPACE: char = '\u{2009}';

/// How many thin spaces per gap [`tracked`] inserts, derived from
/// [`Theme::TRACKING`].
#[inline]
fn tracked_steps() -> usize {
    // One thin space measures a little over two points in the monospace face at
    // the sizes this panel uses, so a 2-point tracking request is one space and
    // anything below it is none. Keeping the derivation here means the
    // palette constant is the thing that actually controls the result.
    usize::from(Theme::TRACKING >= 2.0)
}

/// Uppercase label text with standard tracking.
fn tracked(text: &str) -> String {
    tracked_n(text, tracked_steps())
}

/// Label text with `n` thin spaces between each pair of glyphs.
/// `n == 0` returns the text unchanged.
fn tracked_n(text: &str, n: usize) -> String {
    if n == 0 {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len() * (n + 1));
    for ch in text.chars() {
        for _ in 0..n {
            out.push(THIN_SPACE);
        }
        out.push(ch);
    }
    out
}

// ===========================================================================
// Ephemeral UI state
// ===========================================================================
//
// The panel is a pure function of (`Uniforms`, `TelemetryView`) plus a couple
// of purely-visual bits of state: the search query and the PRNG cursor behind
// the Julia randomiser. Both live in `egui`'s `IdTypeMap` under fixed ids, so
// the three-argument [`controls`] signature stays honest and the app layer
// does not have to own anything on the panel's behalf. [`reset_ephemeral`]
// clears both, and is the only reason the ids are public.

/// `IdTypeMap` key for the function-search query.
///
/// `egui::Id::new` is not a `const fn` (it hashes), so these are functions
/// rather than associated constants. They cost one short-string hash, only on
/// the frames where the search box or the randomiser actually runs.
#[allow(dead_code)]
pub fn search_state_id() -> egui::Id {
    egui::Id::new("steel_pulse/fn_search")
}

/// `IdTypeMap` key for the Julia-seed PRNG cursor. See [`search_state_id`].
#[allow(dead_code)]
pub fn seed_state_id() -> egui::Id {
    egui::Id::new("steel_pulse/julia_seed")
}

/// Forget the search query and the Julia-seed cursor.
///
/// Call after a "reset everything" action so the next session of use starts
/// from a clean slate. Note that [`controls`] does *not* call this for you:
/// clearing the search box out from under a user who is mid-search is a worse
/// bug than a stale query surviving a reset.
///
/// `#[allow(dead_code)]`: called by the app layer if it wants it; nothing in
/// this module needs it.
#[allow(dead_code)]
pub fn reset_ephemeral(ctx: &egui::Context) {
    ctx.data_mut(|d| {
        d.remove_temp::<String>(search_state_id());
        d.remove_temp::<u64>(seed_state_id());
    });
}

// ===========================================================================
// Section 0 — masthead
// ===========================================================================

/// Title bar: identity on the left, the two global actions on the right.
fn masthead(
    ui: &mut Ui,
    action: &mut Option<PanelAction>,
    state: &mut Uniforms,
    t: &TelemetryView,
) {
    let h = 20.0;
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), h), Sense::hover());
    if ui.is_rect_visible(rect) {
        let p = ui.painter().with_clip_rect(rect.intersect(ui.clip_rect()));
        p.text(
            rect.left_center(),
            Align2::LEFT_CENTER,
            tracked("STEEL-PULSE"),
            FontId::new(Theme::SIZE_BODY, FontFamily::Monospace),
            Theme::TEXT,
        );
        // The live backend tag, right-aligned against the title. A long adapter
        // name is the one place truncation matters, so it is clipped to the
        // row rather than allowed to run under the title.
        p.text(
            pos2(rect.right(), rect.center().y),
            Align2::RIGHT_CENTER,
            t.backend.as_str(),
            label_font(),
            Theme::TEXT_FAINT,
        );
    }

    let (rule, _) = ui.allocate_exact_size(vec2(ui.available_width(), 1.0), Sense::hover());
    ui.painter().hline(
        rule.x_range(),
        rule.center().y,
        Theme::accent_line(Theme::CYAN_DIM),
    );
    ui.add_space(4.0);

    // Wrapped, not `horizontal`: three buttons in a row force a 250 px
    // minimum, which is wider than a useful dock tab. Wrapping lets the
    // masthead compress to the width of its widest single button.
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = vec2(ROW_GAP, 0.0);
        if small_button(ui, "RESET ALL").clicked() {
            apply_reset_all(state);
            fire(action, PanelAction::ResetAll);
        }
        if small_button(ui, "SHOT").clicked() {
            fire(action, PanelAction::Screenshot);
        }
        if small_button(ui, "RANDOM c").clicked() {
            randomize_julia(ui, state);
            fire(action, PanelAction::RandomizeJulia);
        }
    });
}

// ===========================================================================
// Section 1 — VIEW
// ===========================================================================

/// Viewport controls plus the derived, aspect-correct readouts.
fn view_section(
    ui: &mut Ui,
    state: &mut Uniforms,
    t: &TelemetryView,
    action: &mut Option<PanelAction>,
) {
    // Two drag values side by side. `speed` is value-change-per-logical-pixel,
    // so 0.0008 means a 100 px drag crosses one unit of the complex plane -
    // slow enough to land exactly on a point, fast enough to cross the view in
    // one gesture.
    field_row(ui, "CENTRE", |ui| {
        // The two fields split whatever the row has left, so a dock dragged
        // narrow truncates the numbers rather than pushing a neighbour off the
        // edge of the panel.
        let w = each_field(ui, 2);
        ui.add_sized(
            [w, ROW_H],
            egui::DragValue::new(&mut state.center[0])
                .speed(0.0008)
                .range(-1.0e6..=1.0e6)
                .fixed_decimals(6)
                .update_while_editing(true),
        );
        ui.add_sized(
            [w, ROW_H],
            egui::DragValue::new(&mut state.center[1])
                .speed(0.0008)
                .range(-1.0e6..=1.0e6)
                .fixed_decimals(6)
                .update_while_editing(true),
        );
    });

    field_row(ui, "SCALE", |ui| {
        // Zoom spans fourteen orders of magnitude: from the whole plane down to
        // a point of the order 1e-6 across. A linear speed would be useless at
        // one end or the other, so the drag step is proportional and the typed
        // entry is what handles the extremes.
        let w = (ui.available_width() - small_button_width(ui, "1.5") - ROW_GAP).max(40.0);
        ui.add_sized(
            [w, ROW_H],
            egui::DragValue::new(&mut state.scale)
                .speed(0.0005)
                .range(1.0e-7..=1.0e7)
                .fixed_decimals(6)
                .update_while_editing(true),
        );
        if small_button(ui, "1.5")
            .on_hover_text("Reset scale to the default 1.5")
            .clicked()
        {
            state.scale = DEFAULT_SCALE;
        }
    });

    inline_rule(ui);

    // Aspect-correct derived readouts. `scale` is the half-*height* in complex
    // units, so the visible width is `2 * scale * aspect`. Reporting the
    // on-screen pixels-per-complex-unit is the number that actually matters
    // when judging whether a contour band is a hairline or a blob.
    let aspect = aspect_ratio(t.resolution);
    let half_h = state.scale.max(f32::MIN_POSITIVE) as f64;
    let half_w = half_h * aspect;
    let height_px = t.resolution[1].max(1) as f64;
    let ppu = height_px / (2.0 * half_h);

    readout_row(
        ui,
        "SPAN",
        format!("{:.4} x {:.4}", 2.0 * half_w, 2.0 * half_h),
        Theme::TEXT,
    );
    readout_row(ui, "PPU", format!("{ppu:.1} px/unit"), Theme::TEXT);
    readout_row(ui, "ASPECT", format!("{aspect:.4}"), Theme::TEXT_DIM);

    ui.add_space(2.0);
    if small_button(ui, "RESET VIEW").clicked() {
        apply_reset_view(state);
        fire(action, PanelAction::ResetView);
    }
}

/// The default half-height of the viewport, mirroring `Uniforms::default`.
const DEFAULT_SCALE: f32 = 1.5;

// ===========================================================================
// Section 2 — FUNCTION
// ===========================================================================

/// The grouped, searchable function selector plus the iteration controls.
fn function_section(ui: &mut Ui, state: &mut Uniforms) {
    // -- search -----------------------------------------------------------
    let mut query = read_search(ui.ctx());
    let response = ui.add(
        TextEdit::singleline(&mut query)
            .font(egui::FontSelection::FontId(FontId::new(
                Theme::SIZE_SMALL,
                FontFamily::Monospace,
            )))
            .desired_width(ui.available_width())
            .frame(
                egui::Frame::new()
                    .fill(Theme::SUNKEN)
                    .stroke(Theme::hairline_strong())
                    .corner_radius(CornerRadius::same(Theme::RADIUS))
                    .inner_margin(Margin::symmetric(GUTTER, 3)),
            )
            .hint_text("filter 16 maps by name, id or formula"),
    );
    if response.changed() {
        write_search(ui.ctx(), query.clone());
    }
    if !query.is_empty()
        && small_button(ui, "CLEAR")
            .on_hover_text("Clear the filter")
            .clicked()
    {
        let cleared = String::new();
        write_search(ui.ctx(), cleared.clone());
        query = cleared;
    }
    ui.add_space(3.0);

    // -- the list ---------------------------------------------------------
    let total = FUNCTION_COUNT;
    let hits = match_count(&query);
    let selected_row = row_for_id(state.func_id).copied();

    egui::ScrollArea::vertical()
        .id_salt("steel_pulse_fn_list")
        .max_height(FN_LIST_MAX_H)
        .auto_shrink([false, true])
        .show(ui, |ui| {
            let mut any = false;
            for kind in FunctionKind::ORDER {
                let matching: Vec<FunctionRow> = FUNCTION_TABLE
                    .iter()
                    .copied()
                    .filter(|r| r.kind == kind && r.matches(&query))
                    .collect();
                if matching.is_empty() {
                    continue;
                }
                any = true;
                group_heading(ui, kind.label());
                for row in matching {
                    let selected = state.func_id == row.id;
                    if function_row(ui, &row, selected).clicked() {
                        state.func_id = row.id;
                    }
                }
                ui.add_space(2.0);
            }
            if !any {
                let (rect, _) =
                    ui.allocate_exact_size(vec2(ui.available_width(), 22.0), Sense::hover());
                if ui.is_rect_visible(rect) {
                    let p = ui.painter().with_clip_rect(rect.intersect(ui.clip_rect()));
                    p.text(
                        rect.left_center(),
                        Align2::LEFT_CENTER,
                        format!("no map matches \"{}\"", query.trim()),
                        value_font(),
                        Theme::AMBER,
                    );
                }
            }
        });
    ui.add_space(3.0);

    // -- what is selected --------------------------------------------------
    match selected_row {
        Some(row) => {
            readout_row(
                ui,
                "SELECTED",
                format!("[{:02}] {}", row.id, row.name),
                Theme::CYAN,
            );
            note_line(ui, &format!("{}   {}", row.formula, row.kind.note()));
        }
        None => {
            readout_row(
                ui,
                "SELECTED",
                format!("ID {} n/a", state.func_id),
                Theme::RED,
            );
            note_line(
                ui,
                "id is outside the dispatch table; the shader will fall through",
            );
        }
    }
    ui.add_space(2.0);
    readout_row(
        ui,
        "FILTER",
        format!("{hits}/{total} shown"),
        Theme::TEXT_DIM,
    );

    inline_rule(ui);

    // -- iterate ----------------------------------------------------------
    // The single most important control in the program: it is what turns any
    // map into a Julia-style set. So it gets a full-width state chip rather
    // than a checkbox, and turns magenta when engaged - magenta is reserved
    // for "this changes the math", and this is the only control that does.
    let on = state.iterate != 0;
    if state_chip(ui, "ITERATE", on, on.then_some(Theme::MAGENTA)).clicked() {
        state.iterate = u32::from(on);
    }
    ui.add_space(2.0);
    note_line(
        ui,
        if on {
            "applying the map max_iter times; centre is the constant c"
        } else {
            "single application; turn on for Julia-style iteration"
        },
    );

    ui.add_space(4.0);
    field_row(ui, "MAX ITER", |ui| {
        // A `Slider` has an intrinsic minimum of roughly its default rail width
        // plus the value, which is ~275 px — by far the widest thing in the
        // panel. The rail is shrunk for this one row, where it shares the width
        // with a label, rather than globally, where it is the whole control.
        ui.spacing_mut().slider_width = 84.0;
        ui.add_sized(
            [ui.available_width(), ROW_H],
            egui::Slider::new(&mut state.max_iter, MIN_ITER..=MAX_ITER)
                .logarithmic(true)
                .clamping(egui::SliderClamping::Edits)
                .drag_value_speed(1.0)
                .show_value(true)
                .text_color(Theme::TEXT)
                .handle_shape(egui::style::HandleShape::Rect { aspect_ratio: 0.22 }),
        );
    });
    ui.add_space(1.0);
    note_line(ui, "16 - 4096, logarithmic; the cap mirrors the library");
}

/// Minimum iteration count offered by the slider. Below 16 nothing in the table
/// resolves anything worth looking at, and a 4-step orbit is just a smear.
const MIN_ITER: u32 = 16;
/// Maximum iteration count, matching `functions::ITERATION_CAP`.
const MAX_ITER: u32 = 4096;

// The note printed under the slider spells the range out in words rather than
// formatting the constants, because a string is a string. This assertion is
// what stops the two from drifting: change the slider bounds and the build
// fails until the sentence is updated too.
const _: () = assert!(MIN_ITER == 16 && MAX_ITER == 4096);

// ===========================================================================
// Section 3 — COLOR
// ===========================================================================

/// Phase rotation, contour densities and the shading exponent.
fn color_section(ui: &mut Ui, state: &mut Uniforms, action: &mut Option<PanelAction>) {
    field_row(ui, "PHASE", |ui| {
        // Phase is in turns, 0..=1, so one drag across the field is a full
        // colour cycle. Four decimals is ~1/10000 of a turn, which is finer
        // than the eye can resolve on a 1000 px band but coarse enough that
        // the digits stop flickering.
        let w = (ui.available_width() - small_button_width(ui, "+1/8") - ROW_GAP).max(40.0);
        ui.add_sized(
            [w, ROW_H],
            egui::DragValue::new(&mut state.phase)
                .speed(0.002)
                .range(0.0..=1.0)
                .fixed_decimals(4)
                .update_while_editing(true),
        );
        if small_button(ui, "+1/8")
            .on_hover_text("Advance the hue by an eighth turn")
            .clicked()
        {
            state.phase = (state.phase + 0.125).fract();
        }
    });

    field_row(ui, "MOD", |ui| {
        // Bands per e-fold. Zero disables the contours entirely, which is a
        // legitimate and frequently useful state, so the range starts at 0.
        ui.add_sized(
            [num_w(ui), ROW_H],
            egui::DragValue::new(&mut state.modulus_contour_density)
                .speed(0.02)
                .range(0.0..=64.0)
                .fixed_decimals(2)
                .update_while_editing(true),
        );
    });

    field_row(ui, "PHASE", |ui| {
        // Bands per turn of argument. 16 is the default: at the default zoom a
        // turn spans roughly a third of the viewport, so 16 bands puts one
        // every ~20 px, which is the density at which a ray structure reads as
        // rays rather than as a moire wash.
        ui.add_sized(
            [num_w(ui), ROW_H],
            egui::DragValue::new(&mut state.phase_contour_density)
                .speed(0.02)
                .range(0.0..=64.0)
                .fixed_decimals(2)
                .update_while_editing(true),
        );
    });

    field_row(ui, "SHADING", |ui| {
        // 0 flattens the image to a pure hue map, which is the correct choice
        // when judging boundary geometry and the wrong choice when judging
        // depth, so the whole range is reachable.
        ui.add_sized(
            [num_w(ui), ROW_H],
            egui::DragValue::new(&mut state.modulus_shading)
                .speed(0.004)
                .range(0.0..=4.0)
                .fixed_decimals(3)
                .update_while_editing(true),
        );
    });

    ui.add_space(3.0);
    if small_button(ui, "RESET COLOR").clicked() {
        apply_reset_color(state);
        fire(action, PanelAction::ResetColor);
    }
}

// ===========================================================================
// Section 4 — GRID
// ===========================================================================

/// The world-space unit-circle / axis overlay.
fn grid_section(ui: &mut Ui, state: &mut Uniforms) {
    let on = state.grid_enabled != 0;
    if state_chip(ui, "GRID", on, on.then_some(Theme::CYAN)).clicked() {
        state.grid_enabled = u32::from(on);
    }
    ui.add_space(2.0);
    note_line(ui, "unit circles and the real axis, drawn in world space");
    // The grid is only meaningful once you can see where a unit is, so say so
    // rather than leaving the user to wonder why nothing appeared.
    if on && state.scale > 1.5 {
        note_line(ui, "zoomed out past |z| = 1; the overlay is off screen");
    }
}

// ===========================================================================
// Section 4b — TELEMETRY
// ===========================================================================

/// The read-only instrument block.
fn telemetry_section(ui: &mut Ui, t: &TelemetryView) {
    frame_sparkline(ui, &t.frame_history, t.frame_ms);

    inline_rule(ui);

    // The whole point of this panel being right-aligned and monospace: these
    // numbers are read while they change, several times a second, for hours.
    // If a digit were proportional the entire right edge would shimmer.
    let status = t.status_color();
    readout_row(ui, "FPS", format!("{:.1}", t.fps), status);
    readout_row(
        ui,
        "FRAME",
        format!("{:.2} ms", t.frame_ms),
        theme::status_color_from_frame_ms(t.frame_ms),
    );
    readout_row(ui, "GPU", format!("{:.2} ms", t.gpu_ms), Theme::TEXT);
    readout_row(ui, "CPU", format!("{:.2} ms", t.cpu_ms), Theme::TEXT);
    readout_row(ui, "GPU SHARE", gpu_share(t), Theme::TEXT_DIM);
    readout_row(
        ui,
        "VIEWPORT",
        format!("{} x {}", t.resolution[0], t.resolution[1]),
        Theme::TEXT,
    );
    readout_row(ui, "RAMP", ramp_caption(t.fps), Theme::TEXT_FAINT);
}

/// The frame-time sparkline.
///
/// Three things are drawn, and each is there because it answers a question a
/// number cannot:
///
/// 1. **The trace itself** - did a frame spike, and was it one frame or ten?
/// 2. **A hairline at the 60 Hz budget** - so "16.7 ms" has a visual meaning.
/// 3. **Magenta ticks at dropped frames** - anything past twice the budget is
///    a frame the user actually saw stutter, and marking them in the hot
///    accent is worth more than any smoothing.
///
/// A trace with fewer than two samples is not drawn as a flat line, because a
/// flat line is indistinguishable from a genuinely constant frame time. It is
/// drawn as an explicit `NO TRACE` state instead.
fn frame_sparkline(ui: &mut Ui, samples: &[f64], current_ms: f64) {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), SPARK_H), Sense::hover());
    if !ui.is_rect_visible(rect) {
        return;
    }
    let p = ui.painter().with_clip_rect(rect.intersect(ui.clip_rect()));

    // Recessed well behind the trace.
    p.rect_filled(rect, CornerRadius::same(Theme::RADIUS), Theme::SUNKEN);
    p.rect_stroke(
        rect,
        CornerRadius::same(Theme::RADIUS),
        Theme::hairline(),
        egui::StrokeKind::Inside,
    );

    let plot = Rect::from_min_max(
        pos2(rect.left() + 3.0, rect.top() + 3.0),
        pos2(rect.right() - 3.0, rect.bottom() - 10.0),
    );

    // -- the budget line ---------------------------------------------------
    let ceiling = spark_ceiling(samples, current_ms);
    let budget_y = map_y(plot, FRAME_BUDGET_MS, ceiling);
    p.hline(
        plot.x_range(),
        budget_y,
        Theme::accent_line(Theme::alpha(Theme::AMBER, 0.55)),
    );
    p.text(
        pos2(rect.right() - 4.0, plot.bottom() + 1.0),
        Align2::LEFT_BOTTOM,
        format!("{FRAME_BUDGET_MS:.1} BUDGET"),
        label_font(),
        Theme::alpha(Theme::AMBER, 0.75),
    );

    if samples.len() < 2 {
        p.text(
            plot.left_center(),
            Align2::LEFT_CENTER,
            format!("NO TRACE   {current_ms:.2} ms"),
            label_font(),
            Theme::TEXT_FAINT,
        );
        return;
    }

    // -- the trace ---------------------------------------------------------
    let n = samples.len();
    let step = plot.width() / (n.max(2) - 1) as f32;
    let mut line = Vec::with_capacity(n);
    for (i, &ms) in samples.iter().enumerate() {
        let x = plot.left() + step * i as f32;
        line.push(pos2(x, map_y(plot, ms, ceiling)));
    }

    // Fill first, stroke second, so the outline is never overdrawn.
    let mut fill = line.clone();
    fill.push(pos2(line[line.len() - 1].x, plot.bottom()));
    fill.push(pos2(line[0].x, plot.bottom()));
    p.add(Shape::convex_polygon(
        fill,
        Theme::alpha(Theme::CYAN, 0.16),
        PathStroke::NONE,
    ));
    p.add(Shape::line(line.clone(), Stroke::new(1.0, Theme::CYAN)));

    // -- dropped-frame ticks ----------------------------------------------
    for (i, pt) in line.iter().enumerate() {
        if samples[i] > FRAME_BUDGET_MS * 2.0 {
            p.vline(
                pt.x,
                plot.y_range(),
                Theme::accent_line(Theme::alpha(Theme::MAGENTA, 0.5)),
            );
        }
    }

    // -- the newest sample, marked ----------------------------------------
    if let Some(last) = line.last() {
        p.rect_filled(
            Rect::from_center_size(*last, vec2(2.0, plot.height())),
            CornerRadius::same(Theme::RADIUS),
            Theme::CYAN,
        );
    }
    p.text(
        pos2(rect.right() - 4.0, plot.bottom() + 1.0),
        Align2::RIGHT_BOTTOM,
        format!("PEAK {:.1}", ceiling),
        label_font(),
        Theme::TEXT_FAINT,
    );
}

/// The upper bound of the sparkline's vertical scale, in milliseconds.
///
/// The scale is anchored on the 60 Hz budget rather than on the data, because
/// a trace that autoscales to its own maximum has no absolute meaning: a
/// 1 ms frame and a 16 ms frame would draw the same shape. If the data
/// exceeds the budget, the scale grows to fit it with 10% headroom.
fn spark_ceiling(samples: &[f64], current_ms: f64) -> f64 {
    let peak = samples
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(current_ms.max(0.0), f64::max);
    let anchor = FRAME_BUDGET_MS * 1.25;
    if peak <= anchor {
        anchor
    } else {
        peak * 1.1
    }
}

/// Map a frame time in milliseconds to a y coordinate. Larger ms is higher.
///
/// Non-finite input is pinned to the bottom of the plot rather than allowed
/// through: `f64::clamp` passes `NaN` straight out, and a `NaN` vertex in a
/// `Shape` makes the tessellator drop the shape — which on this panel would
/// mean the entire trace silently vanishing on the one frame the user was
/// trying to read it.
fn map_y(plot: Rect, ms: f64, ceiling: f64) -> f32 {
    let ceiling = if ceiling.is_finite() && ceiling > 0.0 {
        ceiling
    } else {
        1.0
    };
    let t = if ms.is_finite() {
        (ms / ceiling).clamp(0.0, 1.0)
    } else {
        0.0
    } as f32;
    (plot.bottom() - t * plot.height()).clamp(plot.top(), plot.bottom())
}

/// What the ramp is currently saying, in words.
fn ramp_caption(fps: f64) -> String {
    if !fps.is_finite() {
        return "NO SIGNAL".to_owned();
    }
    if fps >= theme::FPS_GOOD {
        "GREEN".to_owned()
    } else if fps <= FPS_BAD {
        "RED".to_owned()
    } else if fps >= FPS_WARN {
        "AMBER-G".to_owned()
    } else {
        "RED-AMBER".to_owned()
    }
}

/// GPU time as a percentage of the frame, when the numbers allow it.
fn gpu_share(t: &TelemetryView) -> String {
    if !t.frame_ms.is_finite() || t.frame_ms <= 0.0 || !t.gpu_ms.is_finite() {
        return "-".to_owned();
    }
    let pct = (t.gpu_ms / t.frame_ms).clamp(0.0, 1.0) * 100.0;
    format!("{pct:.0}%")
}

// ===========================================================================
// Section 5 — actions
// ===========================================================================

/// Default half-height of the viewport. Mirrors `Uniforms::default::scale`.
const DEFAULT_VIEW: [f32; 2] = [0.0, 0.0];

/// Restore the default camera. Mirrors `Uniforms::default`.
fn apply_reset_view(state: &mut Uniforms) {
    state.center = DEFAULT_VIEW;
    state.scale = DEFAULT_SCALE;
}

/// Restore the default colour mapping. Mirrors `Uniforms::default`.
fn apply_reset_color(state: &mut Uniforms) {
    let d = Uniforms::default();
    state.phase = d.phase;
    state.modulus_contour_density = d.modulus_contour_density;
    state.phase_contour_density = d.phase_contour_density;
    state.modulus_shading = d.modulus_shading;
}

/// Restore the default camera *and* the default colour mapping.
fn apply_reset_all(state: &mut Uniforms) {
    apply_reset_view(state);
    apply_reset_color(state);
}

/// Pick a fresh Julia seed.
///
/// Writes the seed into [`Uniforms::center`] - the iterated map reads its
/// constant from there - and turns iteration on, because a seed with iteration
/// off draws the same parabola every time and the button would look broken.
///
/// The seed is drawn from the square `[-0.8, 0.8]^2`. That box is not
/// arbitrary: it contains the main cardioid and the period-2 bulb of the
/// Mandelbrot set, so a random pick is connected roughly three quarters of the
/// time, which is the interesting case. Sampling the whole plane would give
/// dust most of the time.
///
/// There is no `rand` dependency, and none is wanted: a xorshift64* seeded
/// from the wall clock, with the cursor kept in the `IdTypeMap` between
/// presses, is four lines and is impossible to misconfigure.
fn randomize_julia(ui: &Ui, state: &mut Uniforms) {
    let ctx = ui.ctx();
    // Mix in a changing source so two presses inside the same frame - or two
    // presses of the key repeat - cannot return the same seed.
    let entropy = ui.input(|i| (i.time * 1.0e6) as u64 ^ ((i.events.len() as u64) << 40));
    let seed = ctx.data_mut(|d| {
        let cursor: &mut u64 = d.get_temp_mut_or_default(seed_state_id());
        *cursor ^= entropy ^ 0x9E37_79B9_7F4A_7C15;
        // xorshift64*: one shift-multiply-xor round is plenty for a
        // 4-decimal visual jitter.
        *cursor ^= *cursor >> 12;
        *cursor ^= *cursor << 25;
        *cursor ^= *cursor >> 27;
        cursor.wrapping_mul(0x2545_F491_4F6C_DD1D)
    });

    // Two independent 53-bit slices of one u64.
    let a = (seed >> 11) as f64 / (1u64 << 53) as f64;
    let b = (seed.rotate_left(31) >> 11) as f64 / (1u64 << 53) as f64;
    state.center = [(a * 1.6 - 0.8) as f32, (b * 1.6 - 0.8) as f32];
    state.iterate = 1;
}

// ===========================================================================
// Primitive widgets
// ===========================================================================

/// A hairline rule with a cyan tick and a tracked, uppercase title.
///
/// This is the only structural element in the panel, and it is deliberately
/// *not* a `CollapsingHeader`: a dense instrument should have the same shape
/// every frame, so a user can find a control without reading it.
fn section_header(ui: &mut Ui, title: &str) {
    ui.add_space(5.0);
    let (rule, _) = ui.allocate_exact_size(vec2(ui.available_width(), 1.0), Sense::hover());
    if ui.is_rect_visible(rule) {
        ui.painter()
            .hline(rule.x_range(), rule.center().y, Theme::hairline_strong());
    }
    ui.add_space(3.0);
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 11.0), Sense::hover());
    if ui.is_rect_visible(rect) {
        let p = ui.painter().with_clip_rect(rect.intersect(ui.clip_rect()));
        p.vline(rect.left(), rect.y_range(), Theme::accent_line(Theme::CYAN));
        p.text(
            pos2(rect.left() + 5.0, rect.center().y),
            Align2::LEFT_CENTER,
            tracked(title),
            label_font(),
            Theme::CYAN,
        );
    }
    ui.add_space(2.0);
}

/// A sub-heading inside the function list, marking a group of rows.
fn group_heading(ui: &mut Ui, label: &str) {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 11.0), Sense::hover());
    if ui.is_rect_visible(rect) {
        let p = ui.painter().with_clip_rect(rect.intersect(ui.clip_rect()));
        p.hline(rect.x_range(), rect.top(), Theme::hairline());
        p.text(
            pos2(rect.left() + 5.0, rect.center().y),
            Align2::LEFT_CENTER,
            tracked(label),
            label_font(),
            Theme::TEXT_FAINT,
        );
    }
    ui.add_space(1.0);
}

/// A label-and-control row. The label is painted by hand so that it can be
/// tracked and dimmed without dragging a `Label`'s layout and padding along
/// with it.
fn field_row<R>(ui: &mut Ui, label: &str, add: impl FnOnce(&mut Ui) -> R) -> R {
    // The label column never takes more than 42% of the row, so a dock dragged
    // down to 150 px still shows numbers rather than a truncated "SHADING".
    let label_w = LABEL_W.min(ui.available_width() * 0.42);
    let inner = ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing = vec2(ROW_GAP, 0.0);
        let (rect, _) = ui.allocate_exact_size(vec2(label_w, ROW_H), Sense::hover());
        if ui.is_rect_visible(rect) {
            let p = ui.painter().with_clip_rect(rect.intersect(ui.clip_rect()));
            p.text(
                rect.left_center(),
                Align2::LEFT_CENTER,
                tracked(label),
                label_font(),
                Theme::TEXT_DIM,
            );
        }
        add(ui)
    });
    inner.inner
}

/// A read-only `label ......... value` row, value hard against the right edge.
///
/// Monospace plus right alignment is what makes the digits tabular: the
/// glyphs all have the same advance width, so the value column never moves.
fn readout_row(ui: &mut Ui, label: &str, value: String, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), READOUT_H), Sense::hover());
    if !ui.is_rect_visible(rect) {
        return;
    }
    let p = ui.painter().with_clip_rect(rect.intersect(ui.clip_rect()));
    p.text(
        rect.left_center(),
        Align2::LEFT_CENTER,
        tracked(label),
        label_font(),
        Theme::TEXT_FAINT,
    );
    p.text(
        pos2(rect.right(), rect.center().y),
        Align2::RIGHT_CENTER,
        value,
        value_font(),
        color,
    );
}

/// A wrapped explanatory line. This is where the panel pays back the terseness
/// of its labels; it is also the only place in the panel that is allowed to
/// use a dimmer, smaller voice than the data around it.
fn note_line(ui: &mut Ui, text: &str) {
    let galley = ui.painter().layout(
        text.to_owned(),
        label_font(),
        Theme::TEXT_FAINT,
        ui.available_width().floor(),
    );
    let (rect, _) = ui.allocate_exact_size(
        vec2(ui.available_width(), galley.size().y.max(11.0)),
        Sense::hover(),
    );
    if ui.is_rect_visible(rect) {
        let p = ui.painter().with_clip_rect(rect.intersect(ui.clip_rect()));
        p.galley(rect.min, galley, Theme::TEXT_FAINT);
    }
}

/// A compact rectangular button.
///
/// `Button` is not used because its minimum width would force the three
/// masthead actions onto two lines in a narrow dock, and because a default
/// button's rounded frame is exactly the softness this design forbids.
fn small_button(ui: &mut Ui, label: &str) -> egui::Response {
    let text = tracked_n(label, 1);
    let galley = ui
        .painter()
        .layout_no_wrap(text.clone(), label_font(), Theme::TEXT_DIM);
    let width = galley.size().x + 14.0;
    let (rect, response) = ui.allocate_exact_size(vec2(width, ROW_H), Sense::click());

    if ui.is_rect_visible(rect) {
        let p = ui.painter().with_clip_rect(rect.intersect(ui.clip_rect()));
        let (fill, stroke, fg) = if response.is_pointer_button_down_on() {
            (
                Theme::RAISED_ACTIVE,
                Theme::accent_line(Theme::CYAN),
                Theme::CYAN,
            )
        } else if response.hovered() {
            (
                Theme::RAISED_HOVER,
                Theme::accent_line(Theme::CYAN_DIM),
                Theme::CYAN,
            )
        } else {
            (Theme::RAISED, Theme::hairline_strong(), Theme::TEXT_DIM)
        };
        p.rect_filled(rect, CornerRadius::same(Theme::RADIUS), fill);
        p.rect_stroke(
            rect,
            CornerRadius::same(Theme::RADIUS),
            stroke,
            egui::StrokeKind::Inside,
        );
        // The galley is top-left anchored, so offset it to the row's centre.
        p.galley(
            pos2(rect.left() + 7.0, rect.center().y - galley.size().y * 0.5),
            galley,
            fg,
        );
    }
    response
}

/// A full-width on/off state chip.
///
/// The only widget in the panel with a saturated border, and it is used exactly
/// twice: `ITERATE` and `GRID`. `ITERATE` goes magenta because it is the one
/// control that changes what is being computed; `GRID` goes cyan because it is
/// the same idea at a different weight. Everything else stays dark.
fn state_chip(ui: &mut Ui, label: &str, on: bool, accent: Option<Color32>) -> egui::Response {
    let h = ROW_H + 2.0;
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), h), Sense::click());

    if ui.is_rect_visible(rect) {
        let p = ui.painter().with_clip_rect(rect.intersect(ui.clip_rect()));
        let accent = accent.unwrap_or(Theme::CYAN);
        let (fill, text_color) = if on {
            (Theme::RAISED_ACTIVE, accent)
        } else if response.hovered() {
            (Theme::RAISED_HOVER, Theme::TEXT)
        } else {
            (Theme::RAISED, Theme::TEXT_DIM)
        };
        p.rect_filled(rect, CornerRadius::same(Theme::RADIUS), fill);
        p.rect_stroke(
            rect,
            CornerRadius::same(Theme::RADIUS),
            if on {
                Theme::accent_line(accent)
            } else {
                Theme::hairline_strong()
            },
            egui::StrokeKind::Inside,
        );
        if on {
            p.vline(rect.left(), rect.y_range(), Theme::accent_line(accent));
        }
        let galley = p.layout_no_wrap(tracked(label), label_font(), text_color);
        p.galley(
            pos2(rect.left() + 9.0, rect.center().y - galley.size().y * 0.5),
            galley,
            text_color,
        );
        // The state word, hard right. Tabular: "ON" and "OFF" are both three
        // and two characters in a fixed cell, so the row does not twitch.
        p.text(
            pos2(rect.right() - 8.0, rect.center().y),
            Align2::RIGHT_CENTER,
            if on { "ON" } else { "OFF" },
            label_font(),
            if on { accent } else { Theme::TEXT_FAINT },
        );
    }
    response
}

/// One row of the function selector: `NN  name   formula`.
///
/// Painted by hand rather than with `selectable_label` for two reasons: the id
/// needs its own dim column so that a glance at the left edge is a list of
/// dispatch numbers, and the selected state needs a left tick plus a dim
/// wash rather than egui's flat selection colour.
fn function_row(ui: &mut Ui, row: &FunctionRow, selected: bool) -> egui::Response {
    let (rect, response) =
        ui.allocate_exact_size(vec2(ui.available_width(), FN_ROW_H), Sense::click());

    if ui.is_rect_visible(rect) {
        let p = ui.painter().with_clip_rect(rect.intersect(ui.clip_rect()));
        if selected {
            p.rect_filled(rect, CornerRadius::same(Theme::RADIUS), Theme::SELECTION);
        } else if response.hovered() {
            p.rect_filled(rect, CornerRadius::same(Theme::RADIUS), Theme::RAISED_HOVER);
        }
        if selected {
            p.vline(rect.left(), rect.y_range(), Theme::accent_line(Theme::CYAN));
        }

        let name_color = if selected { Theme::CYAN } else { Theme::TEXT };
        let cy = rect.center().y;
        p.text(
            pos2(rect.left() + 6.0, cy),
            Align2::LEFT_CENTER,
            format!("{:02}", row.id),
            label_font(),
            if selected {
                Theme::CYAN_DIM
            } else {
                Theme::TEXT_FAINT
            },
        );
        p.text(
            pos2(rect.left() + 24.0, cy),
            Align2::LEFT_CENTER,
            row.name,
            value_font(),
            name_color,
        );
        p.text(
            pos2(rect.right() - 6.0, cy),
            Align2::RIGHT_CENTER,
            row.formula,
            label_font(),
            if selected {
                Theme::TEXT_DIM
            } else {
                Theme::TEXT_FAINT
            },
        );
    }
    response
}

/// The width [`small_button`] will occupy for `label`.
///
/// Exists so a row can size the widget *next* to a button without adding the
/// button first, which is what keeps the row from overflowing a narrow panel.
fn small_button_width(ui: &Ui, label: &str) -> f32 {
    let text = tracked_n(label, tracked_steps());
    ui.painter()
        .layout_no_wrap(text, label_font(), Theme::TEXT_DIM)
        .size()
        .x
        + BTN_PAD
}

/// Width for one of `n` fields sharing a row, after the inter-field gaps.
#[inline]
fn each_field(ui: &Ui, n: usize) -> f32 {
    let gaps = ROW_GAP * n.saturating_sub(1) as f32;
    ((ui.available_width() - gaps) / n as f32).max(36.0)
}

/// Width for the single-value colour fields, so that their digits line up down
/// the column. Clamped against the row so a narrow dock still fits.
#[inline]
fn num_w(ui: &Ui) -> f32 {
    each_field(ui, 1).clamp(36.0, 96.0)
}

/// A hairline between two parts of one section, with breathing room either
/// side so it reads as a rule rather than as a border.
fn inline_rule(ui: &mut Ui) {
    ui.add_space(3.0);
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 1.0), Sense::hover());
    if ui.is_rect_visible(rect) {
        ui.painter()
            .hline(rect.x_range(), rect.center().y, Theme::hairline());
    }
    ui.add_space(4.0);
}

/// Record the first action raised this frame, if none has been yet.
fn fire(slot: &mut Option<PanelAction>, action: PanelAction) {
    if slot.is_none() {
        *slot = Some(action);
    }
}

/// Viewport aspect ratio, with a sane floor.
fn aspect_ratio(resolution: [u32; 2]) -> f64 {
    if resolution[1] == 0 {
        1.0
    } else {
        f64::from(resolution[0]) / f64::from(resolution[1])
    }
}

/// Read the current function-search query out of the `IdTypeMap`.
fn read_search(ctx: &egui::Context) -> String {
    ctx.data(|d| d.get_temp::<String>(search_state_id()).unwrap_or_default())
}

/// Write the function-search query back into the `IdTypeMap`.
fn write_search(ctx: &egui::Context, value: String) {
    ctx.data_mut(|d| *d.get_temp_mut_or_default::<String>(search_state_id()) = value);
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// The one test that actually matters: this hand-maintained mirror must
    /// agree with the library, id for id, name for name, formula for formula.
    #[test]
    fn table_is_in_sync_with_the_library() {
        use crate::complex::functions::{FunctionEntry, FUNCTIONS};

        assert_eq!(
            FUNCTION_TABLE.len(),
            FUNCTIONS.len(),
            "panel::FUNCTION_TABLE and functions::FUNCTIONS disagree on length"
        );

        for (panel_row, lib_row) in FUNCTION_TABLE.iter().zip(FUNCTIONS.iter()) {
            let FunctionEntry {
                id,
                name,
                formula,
                group,
            } = lib_row;
            assert_eq!(panel_row.id, *id, "id mismatch");
            assert_eq!(panel_row.name, *name, "name mismatch for id {id}");
            assert_eq!(panel_row.formula, *formula, "formula mismatch for id {id}");
            assert_eq!(
                panel_row.kind,
                mirror_group(*group),
                "group mismatch for id {id}"
            );
        }
    }

    /// Translate the library's group enum into the panel's, failing loudly if
    /// the library grows a variant nobody mirrored.
    fn mirror_group(g: crate::complex::functions::FunctionGroup) -> FunctionKind {
        use crate::complex::functions::FunctionGroup as G;
        match g {
            G::Polynomial => FunctionKind::Polynomial,
            G::Iterated => FunctionKind::Iterated,
            G::Reciprocal => FunctionKind::Reciprocal,
            G::Trigonometric => FunctionKind::Trigonometric,
            G::Hyperbolic => FunctionKind::Hyperbolic,
            G::Exponential => FunctionKind::Exponential,
            G::Transcendental => FunctionKind::Transcendental,
            G::Mobius => FunctionKind::Mobius,
        }
    }

    /// Ids must be exactly `0..N` and dense, because the shader switches on
    /// them and a gap would fall through to a default case.
    #[test]
    fn ids_are_dense_and_sequential() {
        for (i, row) in FUNCTION_TABLE.iter().enumerate() {
            assert_eq!(row.id as usize, i, "id {i} is out of order");
        }
        assert_eq!(row_for_id(16), None);
        assert_eq!(row_for_id(15).map(|r| r.name), Some("julia"));
    }

    /// Every group must be non-empty, or a heading appears with nothing under
    /// it.
    #[test]
    fn every_group_has_members() {
        for kind in FunctionKind::ORDER {
            assert!(
                FUNCTION_TABLE.iter().any(|r| r.kind == kind),
                "group {kind:?} is empty"
            );
        }
    }

    /// The filter must find a row by every handle a user might type, and must
    /// not match a row it should not.
    #[test]
    fn search_finds_by_name_id_and_formula() {
        assert_eq!(match_count("cube"), 1);
        assert_eq!(match_count("CUBE"), 1);
        assert_eq!(match_count("  cube  "), 1);
        assert_eq!(match_count("z^3"), 3, "z^3, z^3 - 1 and z^3 - 2z");
        assert_eq!(match_count("5"), 1, "bare id 5");
        assert_eq!(match_count("trigonometric"), 3);
        assert_eq!(match_count(""), FUNCTION_COUNT);
        assert_eq!(match_count("definitely-not-a-function"), 0);
    }

    /// The sparkline scale must be anchored on the frame budget, not on the
    /// data, or the trace has no absolute meaning.
    #[test]
    fn sparkline_ceiling_is_anchored_and_bounded() {
        let quiet = [2.0, 2.1, 2.0, 2.2];
        let ceiling = spark_ceiling(&quiet, 2.0);
        assert!(
            ceiling > FRAME_BUDGET_MS,
            "a fast trace must still be legible against the budget line"
        );
        assert!(ceiling < FRAME_BUDGET_MS * 1.5);

        let janky = [90.0, 12.0, 130.0];
        let ceiling = spark_ceiling(&janky, 12.0);
        assert!(ceiling >= 130.0, "a spike must not be clipped off the top");
    }

    /// Non-finite samples must not produce NaN geometry, which in a tessellator
    /// is a dropped frame with no explanation.
    #[test]
    fn sparkline_ignores_non_finite_samples() {
        let ceiling = spark_ceiling(&[f64::NAN, 3.0, f64::INFINITY, 4.0], 3.0);
        assert!(ceiling.is_finite() && ceiling > 0.0);
        for ms in [f64::NAN, f64::INFINITY, -5.0, 0.0, 1.0e9] {
            let plot = Rect::from_min_max(pos2(0.0, 0.0), pos2(100.0, 40.0));
            assert!(map_y(plot, ms, ceiling).is_finite(), "{ms} mapped to NaN");
        }
    }

    /// A zero-height render target must not produce a divide-by-zero aspect.
    #[test]
    fn aspect_ratio_survives_degenerate_resolutions() {
        assert_eq!(aspect_ratio([0, 0]), 1.0);
        assert_eq!(aspect_ratio([1920, 0]), 1.0);
        assert!((aspect_ratio([1600, 900]) - 16.0 / 9.0).abs() < 1.0e-9);
    }

    /// A zero scale must not turn the readouts into `inf` or `NaN`.
    #[test]
    fn view_readouts_survive_a_degenerate_scale() {
        let u = Uniforms {
            scale: 0.0,
            ..Uniforms::default()
        };
        assert!(aspect_ratio([800, 600]).is_finite());
        // `map_y` is the only place the scale is used as a divisor for display.
        let y = map_y(
            Rect::from_min_max(pos2(0.0, 0.0), pos2(10.0, 10.0)),
            u.scale as f64,
            1.0,
        );
        assert!(y.is_finite());
    }

    /// Resets must produce exactly the documented default view, and must not
    /// touch anything they do not own.
    #[test]
    fn resets_restore_defaults_and_nothing_else() {
        let mut u = Uniforms {
            center: [3.0, -4.0],
            scale: 0.001,
            phase: 0.7,
            modulus_contour_density: 40.0,
            phase_contour_density: 3.0,
            modulus_shading: 2.5,
            func_id: 9,
            grid_enabled: 0,
            iterate: 1,
            ..Uniforms::default()
        };

        apply_reset_all(&mut u);
        let d = Uniforms::default();
        assert_eq!(u.center, d.center);
        assert_eq!(u.scale, d.scale);
        assert_eq!(u.phase, d.phase);
        assert_eq!(u.modulus_contour_density, d.modulus_contour_density);
        assert_eq!(u.phase_contour_density, d.phase_contour_density);
        assert_eq!(u.modulus_shading, d.modulus_shading);
        // Untouched by a reset.
        assert_eq!(u.func_id, 9);
        assert_eq!(u.iterate, 1);
        assert_eq!(u.grid_enabled, 0);
    }

    /// `ResetView` and `ResetColor` must not bleed into each other.
    #[test]
    fn resets_are_independent() {
        let mut u = Uniforms {
            phase: 0.42,
            center: [1.0, 1.0],
            ..Uniforms::default()
        };
        apply_reset_view(&mut u);
        assert_eq!(u.phase, 0.42, "resetting the view touched the colour");
        assert_eq!(u.center, [0.0, 0.0]);

        u.center = [2.0, 2.0];
        apply_reset_color(&mut u);
        assert_eq!(
            u.center,
            [2.0, 2.0],
            "resetting the colour touched the view"
        );
    }

    /// Only the first action in a frame is reported, and it is the earlier one
    /// in document order.
    #[test]
    fn only_the_first_action_per_frame_is_reported() {
        let mut slot: Option<PanelAction> = None;
        assert!(slot.is_none());
        fire(&mut slot, PanelAction::ResetAll);
        fire(&mut slot, PanelAction::Screenshot);
        fire(&mut slot, PanelAction::RandomizeJulia);
        assert_eq!(slot, Some(PanelAction::ResetAll));
    }

    /// The public surface must be constructible from outside the module. This
    /// is a compile-time check that also documents the intended construction.
    #[test]
    fn public_types_are_constructible_externally() {
        let t = TelemetryView {
            fps: 60.0,
            frame_ms: 16.6,
            gpu_ms: 12.0,
            cpu_ms: 4.6,
            resolution: [2560, 1440],
            backend: String::from("Metal"),
            frame_history: vec![16.0, 16.4, 17.1],
        };
        assert_eq!(t.resolution[1], 1440);
        assert_eq!(t.status_color(), Theme::GREEN);

        // And via the constructor, which is the form the app layer will use.
        let t2 = TelemetryView::new(30.0, 33.3, 30.0, 3.3, [800, 600], String::from("Vulkan"))
            .with_history(vec![33.0]);
        assert_eq!(t2.backend, "Vulkan");
        assert_eq!(t2.frame_history.len(), 1);

        // Struct-literal construction with defaults.
        let t3 = TelemetryView {
            fps: 1.0,
            ..Default::default()
        };
        assert_eq!(t3.resolution, [0, 0]);
        assert!(t3.frame_history.is_empty());

        // PanelAction is a plain fieldless enum: matchable and Copy.
        let a = PanelAction::RandomizeJulia;
        let b = a;
        assert_eq!(a, b);
        assert_eq!(
            format!("{:?}", PanelAction::ResetColor),
            "ResetColor".to_owned()
        );

        // FunctionRow is a plain record.
        let row = FunctionRow {
            id: 0,
            name: "identity",
            formula: "z",
            kind: FunctionKind::Polynomial,
        };
        assert!(row.matches(""));
        assert!(row.matches("IDENT"));

        // And the free functions.
        assert_eq!(row_for_id(7).map(|r| r.name), Some("sin"));
        assert_eq!(match_count("polynomial"), 5);
        assert_eq!(FUNCTION_COUNT, 16);
        assert_eq!(FUNCTION_TABLE[9].name, "sinc");
        assert_eq!(FunctionKind::Mobius.label(), "MOEBIUS");
        assert_eq!(FunctionKind::ORDER.len(), 8);
        let _ = search_state_id();
        let _ = seed_state_id();
    }

    /// Smoke test: drive the real panel through real egui frames, with the real
    /// theme and real fonts, across a range of panel widths and states.
    ///
    /// This is the only test that actually executes the layout and painting
    /// code, and it is here because that code is mostly hand-written
    /// `allocate_exact_size` + `Painter` calls, which is exactly the kind of
    /// code that panics on a degenerate rect and produces an invisible widget
    /// on a narrow one. A `debug_assert` failure here is a real defect.
    ///
    /// The width sweep matters: the panel is designed for a dock tab or a side
    /// panel, and both can be dragged narrower than the content wants to be.
    #[test]
    fn panel_survives_a_real_frame_at_every_width() {
        for width in [150.0_f32, 200.0, 260.0, 320.0, 480.0] {
            for with_history in [false, true] {
                for iterate in [0_u32, 1] {
                    let mut u = Uniforms {
                        iterate,
                        ..Uniforms::default()
                    };
                    let history = if with_history {
                        (0..64)
                            .map(|i| 12.0 + (i % 7) as f64 * 1.5)
                            .collect::<Vec<f64>>()
                    } else {
                        Vec::new()
                    };
                    let t = TelemetryView::new(
                        58.0,
                        17.2,
                        13.0,
                        4.2,
                        [1920, 1080],
                        String::from("Metal / Intel UHD Graphics 617"),
                    )
                    .with_history(history);

                    run_frame(width, &mut u, &t, true);
                    run_frame(width, &mut u, &t, false);
                }
            }
        }
    }

    /// The panel must not be wider than [`MIN_PANEL_W`] claims, or the app
    /// layer's `min_size` is a lie and the pane silently clips.
    ///
    /// This measures the real laid-out content width at the declared minimum,
    /// so adding a wide row — a long button label, a fourth masthead action, a
    /// seventh decimal on the centre field — fails the build instead of
    /// producing a panel that quietly overflows.
    #[test]
    fn panel_fits_its_declared_minimum_width() {
        let ctx = egui::Context::default();
        crate::theme::apply(&ctx);
        let t = TelemetryView::new(58.0, 17.2, 13.0, 4.2, [1920, 1080], String::from("Metal"))
            .with_history((0..64).map(|i| 12.0 + (i % 7) as f64 * 1.5).collect());
        let content = measure_content_width(&ctx, MIN_PANEL_W, &t);
        assert!(
            content <= MIN_PANEL_W,
            "panel needs {content:.0} pt but MIN_PANEL_W says {MIN_PANEL_W:.0} pt"
        );
    }

    /// The same, with hostile inputs: a zero-sized render target, a NaN frame
    /// time, a `func_id` that is not in the table, and a search query that
    /// matches nothing. Each of these has a designed fallback in the panel, and
    /// this is what proves those fallbacks are reachable without panicking.
    #[test]
    fn panel_survives_hostile_input() {
        let hostile = [
            Uniforms {
                func_id: 99,
                scale: 0.0,
                center: [f32::NAN, f32::INFINITY],
                max_iter: 0,
                phase: -3.0,
                modulus_contour_density: 1.0e9,
                ..Uniforms::default()
            },
            Uniforms {
                resolution: [0.0, 0.0],
                ..Uniforms::default()
            },
        ];
        for u in hostile {
            let mut u = u;
            let t = TelemetryView {
                fps: f64::NAN,
                frame_ms: f64::NAN,
                gpu_ms: f64::NAN,
                cpu_ms: f64::INFINITY,
                resolution: [0, 0],
                backend: String::new(),
                frame_history: vec![f64::NAN, f64::INFINITY, -1.0, 0.0, 1.0e12],
            };
            run_frame(220.0, &mut u, &t, true);
            run_frame(220.0, &mut u, &t, false);
        }

        // A query that matches nothing, and one that matches one thing.
        let mut u = Uniforms::default();
        let t = TelemetryView::default();
        write_search(&egui::Context::default(), "zzzz-no-such-map".to_owned());
        // The search lives in the context, so the context must be the same one
        // the frame runs in; use the helper rather than a fresh one.
        let ctx = egui::Context::default();
        crate::theme::apply(&ctx);
        write_search(&ctx, "zzzz-no-such-map".to_owned());
        frame_in_ctx(&ctx, 220.0, &mut u, &t, true);
        write_search(&ctx, "julia".to_owned());
        frame_in_ctx(&ctx, 220.0, &mut u, &t, true);
        // And confirm the filter state is actually readable back.
        assert_eq!(read_search(&ctx), "julia");
        reset_ephemeral(&ctx);
        assert_eq!(read_search(&ctx), "");
    }

    /// One headless frame, in its own themed context.
    fn run_frame(width: f32, u: &mut Uniforms, t: &TelemetryView, use_wrapper: bool) {
        let ctx = egui::Context::default();
        crate::theme::apply(&ctx);
        frame_in_ctx(&ctx, width, u, t, use_wrapper);
    }

    /// Lay the panel out at `width` and return the width its content actually
    /// needed, which is `>= width` if anything overflowed.
    fn measure_content_width(ctx: &egui::Context, width: f32, t: &TelemetryView) -> f32 {
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(
                pos2(0.0, 0.0),
                vec2(width + 60.0, 4000.0),
            )),
            viewport_id: egui::ViewportId::ROOT,
            ..egui::RawInput::default()
        };
        // `run_ui` takes an `FnMut(&mut Ui)`, so the measurement has to be
        // captured by reference rather than returned from the closure.
        let mut measured = 0.0f32;
        let _ = ctx.run_ui(input, |root| {
            measured = root
                .allocate_ui_with_layout(
                    vec2(width, 4000.0),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        let _ = controls(ui, &mut Uniforms::default(), t);
                    },
                )
                .response
                .rect
                .width();
        });
        measured
    }

    /// Lay the panel out inside a fixed-width central panel and run one frame.
    fn frame_in_ctx(
        ctx: &egui::Context,
        width: f32,
        u: &mut Uniforms,
        t: &TelemetryView,
        wrap: bool,
    ) {
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(
                pos2(0.0, 0.0),
                vec2(width + 40.0, 1400.0),
            )),
            viewport_id: egui::ViewportId::ROOT,
            ..egui::RawInput::default()
        };
        let output = ctx.run_ui(input, |root| {
            root.allocate_ui_with_layout(
                vec2(width, 1200.0),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    if wrap {
                        let _ = panel(ui, u, t);
                    } else {
                        let _ = controls(ui, u, t);
                    }
                },
            );
        });
        // The panel must actually have painted something. An empty shape list
        // would mean the layout silently produced nothing, which is the one
        // failure mode a no-panic assertion cannot see.
        assert!(
            !output.shapes.is_empty(),
            "the panel painted nothing at width {width}"
        );
    }
}
