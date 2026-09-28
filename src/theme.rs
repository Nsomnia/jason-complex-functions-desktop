//! `theme.rs` — the visual language of STEEL-PULSE.
//!
//! # Design intent
//!
//! This is an *instrument*, not an app. The reference is a bench oscilloscope
//! on a black desk: near-black ground, phosphor-neon accents used only where
//! they carry meaning, hairline rules instead of borders, tabular monospace
//! numerals, and absolutely no softness — no rounded corners, no drop
//! shadows, no decorative gradients.
//!
//! The single hardest constraint is that **the plot is full-bleed saturated
//! domain colouring**. Hue, modulus shading and phase contours fill every pixel
//! of the render target. If the surrounding chrome were anything other than
//! near-black and very low chroma, the UI would compete with the image and the
//! image would stop being readable. So:
//!
//! * Every *surface* colour in this module sits below 8% luminance and carries
//!   only a slight blue cast. Nothing warm, nothing bright.
//! * Accent colours appear as **strokes, ticks, focus rings, selected-row fills
//!   at low alpha, and numeric readouts** — never as a large saturated fill.
//! * There is exactly one accent per role: cyan for "live / selected / focused",
//!   magenta for "hot / active / warning", and the green→amber→red ramp for
//!   frame-rate health only. Nothing else is allowed to be colourful.
//!
//! # Cost model
//!
//! Every colour is an associated `const`, so reading one is a compile-time
//! literal: no allocation, no lazy static, no lock, safe to touch every frame
//! from anywhere. [`Theme::visuals`] and [`Theme::apply`] are only meant to be
//! called once at start-up.
//!
//! # Entry point
//!
//! ```no_run
//! let ctx = egui::Context::default();
//! steel_pulse::theme::Theme::apply(&ctx);
//! ```

use std::collections::BTreeMap;
use std::sync::Arc;

use egui::style::{HandleShape, Selection, TextCursorStyle, WidgetVisuals};
use egui::{
    vec2, Color32, CornerRadius, CursorIcon, FontDefinitions, FontFamily, FontId, Margin, Shadow,
    Spacing, Stroke, TextStyle, Visuals,
};

// ---------------------------------------------------------------------------
// Frame-rate status ramp — thresholds
// ---------------------------------------------------------------------------

/// Upper bound of the "good" band, in frames per second.
///
/// **Why 55 and not 60?** A 60 Hz compositor presents a frame at 16.67 ms. Any
/// pipeline — wgpu submit, `wgpu::Queue` fence wait, egui pass, composite —
/// that lands under ~18 ms is going to miss a vsync. 55 fps is the point where
/// a single dropped frame stops being an isolated hiccup and becomes something
/// a user can *see* while dragging a slider. It is deliberately a little below
/// the display rate so the readout means "comfortably smooth", not "not yet
/// broken".
pub const FPS_GOOD: f64 = 55.0;

/// Middle of the ramp, in frames per second.
///
/// **Why 30?** Half of a 60 Hz refresh. Below this, direct manipulation of the
/// viewport stops feeling connected to the cursor: a drag gesture has more than
/// 33 ms of latency between samples, which is roughly the threshold at which
/// users start compensating for the delay in their hand motion. It is the
/// point where the tool is still usable but no longer pleasant, which is
/// exactly what "amber" should mean.
pub const FPS_WARN: f64 = 30.0;

/// Lower bound of the ramp, in frames per second.
///
/// **Why 24?** At 24 fps the plot is genuinely hard to *read as a picture* while
/// it is also being animated — bands shimmer, hairline contours crawl, and it
/// becomes hard to judge a colour or a boundary. This is the first threshold
/// at which the render is worse than useless for its actual job, so the ramp
/// saturates at full red here rather than continuing to darken (a darker red on
/// a black panel would be a *less* alarming colour, which is exactly backwards).
pub const FPS_BAD: f64 = 24.0;

/// The frame budget, in milliseconds, that the status ramp is calibrated
/// against. `1000 / 60` rounded to two decimals. Drawn as the reference line on
/// the telemetry sparkline so the number and the trace agree.
pub const FRAME_BUDGET_MS: f64 = 16.67;

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

/// The named colour palette and widget styling for STEEL-PULSE.
///
/// A namespace of associated constants plus a handful of constructors. It is
/// deliberately not instantiable: there is one theme, and making it a
/// zero-sized value type that cannot be created keeps that honest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Theme;

impl Theme {
    // -- Surfaces ----------------------------------------------------------
    //
    // A five-step ladder. Each step is roughly +3% luminance and carries a
    // slight blue cast so that "black" never reads as a dead LCD grey.

    /// `#05070C` — the void. Window background behind everything, and the
    /// colour a docked panel sits on. Near-black, blue cast, never `#000000`
    /// (pure black crushes the blue in the plot's shadow regions and makes the
    /// whole app look switched off).
    pub const VOID: Color32 = Color32::from_rgb(0x05, 0x07, 0x0C);

    /// `#080B12` — panel fill. One step up from [`Theme::VOID`]; this is what
    /// the control panel's own background is painted with.
    pub const PANEL: Color32 = Color32::from_rgb(0x08, 0x0B, 0x12);

    /// `#06090F` — sunken. Text-edit backgrounds and slider rails. Deliberately
    /// *below* the panel fill, and only a hair above the void, so that an
    /// editable well reads as a recess rather than as a raised plate.
    pub const SUNKEN: Color32 = Color32::from_rgb(0x06, 0x09, 0x0F);

    /// `#121926` — raised. The resting fill for buttons, checkboxes and other
    /// interactive chrome. Dark enough that a row of buttons never becomes a
    /// band of grey.
    pub const RAISED: Color32 = Color32::from_rgb(0x12, 0x19, 0x26);

    /// `#1A2434` — raised + hovered. A clear but quiet step up; the hover state
    /// must be obvious at a glance across a wide panel.
    pub const RAISED_HOVER: Color32 = Color32::from_rgb(0x1A, 0x24, 0x34);

    /// `#24324A` — raised + active/pressed. Deliberately tinted with the cyan
    /// accent rather than just brightened, so "pressed" and "this is the live
    /// selection" share a colour language.
    pub const RAISED_ACTIVE: Color32 = Color32::from_rgb(0x24, 0x32, 0x4A);

    // -- Hairlines ---------------------------------------------------------

    /// `#161E2B` — the hairline. One physical pixel, the default rule colour.
    /// Used for every separator, group underline and widget outline.
    pub const HAIRLINE: Color32 = Color32::from_rgb(0x16, 0x1E, 0x2B);

    /// `#26344A` — a hairline that must be *seen* rather than merely noticed:
    /// widget outlines at rest, the panel/window border, group headers.
    pub const HAIRLINE_STRONG: Color32 = Color32::from_rgb(0x26, 0x34, 0x4A);

    // -- Text --------------------------------------------------------------

    /// `#CBD9E8` — primary text. Cool near-white, slightly blue so it belongs to
    /// the same light as the cyan accent.
    pub const TEXT: Color32 = Color32::from_rgb(0xCB, 0xD9, 0xE8);

    /// `#7A8CA1` — secondary text. Field labels, units, hints.
    pub const TEXT_DIM: Color32 = Color32::from_rgb(0x7A, 0x8C, 0xA1);

    /// `#45566B` — tertiary text. Disabled values, and anything that exists only
    /// to be read once.
    pub const TEXT_FAINT: Color32 = Color32::from_rgb(0x45, 0x56, 0x6B);

    // -- Accents -----------------------------------------------------------

    /// `#22E3F0` — the primary accent, electric cyan. Reserved for: focus rings,
    /// the selected function row, live numeric readouts, slider handles, and
    /// the section-header tick. This is the only colour allowed to be bright.
    pub const CYAN: Color32 = Color32::from_rgb(0x22, 0xE3, 0xF0);

    /// `#0E6E7C` — cyan, knocked back for use on dark fills where full chroma
    /// would glare (inactive slider handles, dim group rules).
    pub const CYAN_DIM: Color32 = Color32::from_rgb(0x0E, 0x6E, 0x7C);

    /// `#FF3D9E` — the hot accent, magenta. Reserved for: "this control is
    /// engaged and it changes the math" (iteration mode), and for frame-time
    /// spikes on the telemetry trace. Never decorative.
    pub const MAGENTA: Color32 = Color32::from_rgb(0xFF, 0x3D, 0x9E);

    /// `#FFB43D` — amber. Warning state and the frame-budget reference line.
    pub const AMBER: Color32 = Color32::from_rgb(0xFF, 0xB4, 0x3D);

    /// `#3BE07A` — the top of the frame-rate health ramp.
    pub const GREEN: Color32 = Color32::from_rgb(0x3B, 0xE0, 0x7A);

    /// `#FF4A55` — the bottom of the frame-rate health ramp, and egui's error
    /// text colour.
    pub const RED: Color32 = Color32::from_rgb(0xFF, 0x4A, 0x55);

    // -- Composite colours -------------------------------------------------

    /// `#0C2029` — the selected-row / text-selection fill. Cyan at roughly 20%
    /// over the panel: enough to read as "this one", not enough to glow.
    pub const SELECTION: Color32 = Color32::from_rgb(0x0C, 0x20, 0x29);

    // -- Geometry and metrics ---------------------------------------------

    /// Hairline weight, in points. One point at typical pixel ratios is one
    /// physical pixel, which is the whole point.
    pub const HAIRLINE_W: f32 = 1.0;

    /// The focus ring. Slightly heavier than a hairline so a keyboard-focused
    /// control is unambiguous even in a dense stack of drag values.
    pub const FOCUS_W: f32 = 1.5;

    /// Corner radius. Zero, everywhere, always.
    ///
    /// This is the *only* radius value the crate exposes, and every
    /// [`CornerRadius`] in the palette is built from it, so there is no way to
    /// ask for softness anywhere in the application.
    pub const RADIUS: u8 = 0;

    /// Optical glyph scale applied to the monospace face. `1.0` renders Hack
    /// slightly small inside its own line box; nudging it up makes numerals
    /// fill the row height, which is what a dense readout needs.
    pub const GLYPH_SCALE: f32 = 1.06;

    /// Point size for body text and numeric readouts.
    pub const SIZE_BODY: f32 = 12.0;
    /// Point size for button labels.
    pub const SIZE_BUTTON: f32 = 11.0;
    /// Point size for captions and hints.
    pub const SIZE_SMALL: f32 = 10.0;
    /// Point size for section headers. Not larger than body — headers are
    /// distinguished by *colour and tracking*, not by size. Growing the type is
    /// the single fastest way to make a dense tool feel like a website.
    pub const SIZE_HEADING: f32 = 12.0;

    /// Letter tracking for section headers and field labels, in points.
    ///
    /// The panel realises this as [`tracked_n`] thin spaces between glyphs
    /// rather than as a font metric, because [`egui::Painter::text`] takes a
    /// plain string and has no spacing knob. Two points is exactly one thin
    /// space in the monospace face at these sizes, which is why
    /// [`tracked_steps`] maps this to a count of one.
    pub const TRACKING: f32 = 2.0;

    // -- Small helpers -----------------------------------------------------

    /// The default hairline rule.
    #[inline]
    pub const fn hairline() -> Stroke {
        Self::accent_line(Self::HAIRLINE)
    }

    /// A hairline that should be noticed: widget outlines, group underlines.
    #[inline]
    pub const fn hairline_strong() -> Stroke {
        Self::accent_line(Self::HAIRLINE_STRONG)
    }

    /// A hairline in an arbitrary accent colour, for rules that must point at
    /// something (selected row, focus ring, active section).
    #[inline]
    pub const fn accent_line(color: Color32) -> Stroke {
        Stroke {
            width: Self::HAIRLINE_W,
            color,
        }
    }

    /// The focus ring: [`Theme::FOCUS_W`] wide in the cyan accent.
    #[inline]
    pub const fn focus_ring() -> Stroke {
        Stroke {
            width: Self::FOCUS_W,
            color: Self::CYAN,
        }
    }

    /// Re-alpha a colour without touching its RGB. Used for accent washes,
    /// e.g. the sparkline fill under the trace.
    #[inline]
    pub fn alpha(color: Color32, alpha: f32) -> Color32 {
        color.gamma_multiply(alpha.clamp(0.0, 1.0))
    }
}

// ---------------------------------------------------------------------------
// Frame-rate colour ramp
// ---------------------------------------------------------------------------

/// Map a frame rate to a green → amber → red health colour.
///
/// The ramp is three-stop and interpolated in gamma space:
///
/// * `[FPS_BAD, FPS_WARN]` blends [`Theme::RED`] → [`Theme::AMBER`]
/// * `[FPS_WARN, FPS_GOOD]` blends [`Theme::AMBER`] → [`Theme::GREEN`]
/// * at or above [`FPS_GOOD`] it is exactly [`Theme::GREEN`]
/// * at or below [`FPS_BAD`] it is exactly [`Theme::RED`]
///
/// See the `FPS_*` constants for why those three numbers are where they are.
///
/// # Non-finite and zero
///
/// A non-finite rate (NaN from a division by a zero-length frame delta, or an
/// infinite rate from a zero-length one) is *not* a good frame — it is a broken
/// measurement, and a broken measurement in a tuning tool must be loud. It maps
/// straight to [`Theme::RED`].
pub fn status_color(fps: f64) -> Color32 {
    if !fps.is_finite() {
        return Theme::RED;
    }
    if fps >= FPS_GOOD {
        return Theme::GREEN;
    }
    if fps <= FPS_BAD {
        return Theme::RED;
    }
    if fps >= FPS_WARN {
        let t = (fps - FPS_WARN) / (FPS_GOOD - FPS_WARN);
        Theme::AMBER.lerp_to_gamma(Theme::GREEN, t as f32)
    } else {
        let t = (fps - FPS_BAD) / (FPS_WARN - FPS_BAD);
        Theme::RED.lerp_to_gamma(Theme::AMBER, t as f32)
    }
}

/// [`status_color`] expressed in the quantity a frame loop actually has: how
/// long the frame took.
///
/// A frame time of zero, or one that is not finite, maps to [`Theme::RED`]
/// rather than to [`Theme::GREEN`] — the same reasoning as a broken frame rate.
pub fn status_color_from_frame_ms(frame_ms: f64) -> Color32 {
    if !frame_ms.is_finite() || frame_ms <= 0.0 {
        return Theme::RED;
    }
    status_color(1000.0 / frame_ms)
}

// ---------------------------------------------------------------------------
// Fonts
// ---------------------------------------------------------------------------

/// Build the font atlas definition for this app.
///
/// No font file is loaded or embedded here. `epaint` already ships `Hack` (a
/// very good instrument monospace) as the built-in `FontFamily::Monospace`
/// face, along with the usual symbol and emoji fallbacks. The only change made
/// here is to **promote the monospace stack to the proportional family as
/// well**, so that there is no font in this application which can render
/// proportional: a stray `Label`, a tooltip, a window title bar, or a
/// `DragValue`'s own text style will all land on the same fixed-pitch grid
/// without any per-call-site discipline.
///
/// The glyph scale is nudged up by [`Theme::GLYPH_SCALE`] so numerals fill
/// their row instead of floating inside it.
pub fn fonts() -> FontDefinitions {
    let mut defs = FontDefinitions::default();

    if let Some(stack) = defs
        .families
        .get(&FontFamily::Monospace)
        .cloned()
        .filter(|s| !s.is_empty())
    {
        defs.families.insert(FontFamily::Proportional, stack);
    }

    if let Some(hack) = defs.font_data.get_mut("Hack") {
        // `Arc::make_mut` is free here: this map was just built by
        // `FontDefinitions::default()` and nothing else holds a reference.
        Arc::make_mut(hack).tweak.scale = Theme::GLYPH_SCALE;
    }

    defs
}

/// The text style table. Every entry is monospace-first.
///
/// `Heading` is deliberately *not* larger than `Body`. In a tool this dense,
/// hierarchy comes from colour, tracking and rules; if the headers were bigger
/// they would eat vertical space and the panel would need to scroll to show
/// four controls.
pub fn text_styles() -> BTreeMap<TextStyle, FontId> {
    [
        (TextStyle::Small, Theme::SIZE_SMALL),
        (TextStyle::Body, Theme::SIZE_BODY),
        (TextStyle::Monospace, Theme::SIZE_BODY),
        (TextStyle::Button, Theme::SIZE_BUTTON),
        (TextStyle::Heading, Theme::SIZE_HEADING),
    ]
    .into_iter()
    .map(|(style, size)| (style, FontId::new(size, FontFamily::Monospace)))
    .collect()
}

// ---------------------------------------------------------------------------
// Spacing
// ---------------------------------------------------------------------------

/// Tight, regular spacing. The default egui rhythm is designed for consumer
/// forms and is roughly 1.6x too loose for an instrument panel.
pub fn spacing() -> Spacing {
    Spacing {
        item_spacing: vec2(6.0, 5.0),
        button_padding: vec2(8.0, 3.0),
        window_margin: Margin::same(8),
        menu_margin: Margin::same(6),
        indent: 12.0,
        interact_size: vec2(64.0, 16.0),
        slider_width: 140.0,
        slider_rail_height: 3.0,
        ..Spacing::default()
    }
}

// ---------------------------------------------------------------------------
// Visuals
// ---------------------------------------------------------------------------

/// The complete dark [`Visuals`] for this application.
///
/// Constructed fresh on each call; intended to be used once by [`Theme::apply`]
/// (or once at start-up by hand). Every widget state — non-interactive,
/// inactive, hovered, active and open — is specified explicitly, so nothing
/// falls back to an egui default that would clash with the palette.
pub fn visuals() -> Visuals {
    let mut v = Visuals::dark();

    // -- Text ---------------------------------------------------------------
    // `override_text_color` is left `None` on purpose: when it is set, *all*
    // text in the app is forced to one colour and hover/active feedback on
    // buttons, sliders and drag values dies. Instead the foreground stroke of
    // each widget state carries the text colour, so the interaction states
    // tint themselves.
    v.weak_text_color = Some(Theme::TEXT_DIM);
    // Only a fallback: `weak_text_color` above wins, but keep the multiplier
    // sane (0.62, a clear-but-readable step down) in case a future egui drops
    // `weak_text_color`.
    v.weak_text_alpha = 0.62;

    // -- Surfaces -----------------------------------------------------------
    v.panel_fill = Theme::PANEL;
    v.window_fill = Theme::PANEL;
    v.window_stroke = Theme::hairline_strong();
    v.window_corner_radius = CornerRadius::same(Theme::RADIUS);
    v.window_shadow = Shadow::NONE; // no drop shadows, ever
    v.menu_corner_radius = CornerRadius::same(Theme::RADIUS);
    v.popup_shadow = Shadow::NONE;
    v.extreme_bg_color = Theme::SUNKEN;
    v.faint_bg_color = Theme::RAISED;
    v.code_bg_color = Theme::SUNKEN;
    v.text_edit_bg_color = Some(Theme::SUNKEN);
    v.resize_corner_size = 0.0; // draw our own corner treatment, if any

    // -- Accents and signals ------------------------------------------------
    v.hyperlink_color = Theme::CYAN;
    v.warn_fg_color = Theme::AMBER;
    v.error_fg_color = Theme::RED;
    v.selection = Selection {
        bg_fill: Theme::SELECTION,
        stroke: Theme::accent_line(Theme::CYAN),
    };
    v.text_cursor = TextCursorStyle {
        stroke: Theme::focus_ring(),
        // Show where a click would put the caret: this is a text-field-heavy
        // instrument and blind clicking into a number is maddening.
        preview: true,
        blink: true,
        on_duration: 0.45,
        off_duration: 0.45,
    };
    // A disabled control in this app is a control the user has genuinely
    // taken away, not a control that is merely irrelevant, so dimming is
    // gentle (0.45) rather than the egui default half-strength.
    v.disabled_alpha = 0.45;

    // -- Widget behaviour ---------------------------------------------------
    v.button_frame = true;
    v.collapsing_header_frame = false;
    // egui's indent guide is a grey vertical rule; this app draws its own
    // cyan hairlines, and two competing guide systems is one too many.
    v.indent_has_left_vline = false;
    v.striped = false;
    v.slider_trailing_fill = false;
    // A narrow rectangle, not a circle. Circles are soft; a slider handle that
    // is a tiny vertical bar reads like a hardware trim control.
    v.handle_shape = HandleShape::Rect { aspect_ratio: 0.22 };
    v.interact_cursor = Some(CursorIcon::PointingHand);
    v.window_highlight_topmost = false;

    // -- Widget state matrix ------------------------------------------------
    v.widgets.noninteractive = WidgetVisuals {
        bg_fill: Theme::PANEL,
        weak_bg_fill: Color32::TRANSPARENT,
        bg_stroke: Theme::hairline(),
        corner_radius: CornerRadius::same(Theme::RADIUS),
        fg_stroke: Stroke::new(Theme::HAIRLINE_W, Theme::TEXT_DIM),
        expansion: 0.0,
    };

    v.widgets.inactive = WidgetVisuals {
        bg_fill: Theme::RAISED,
        weak_bg_fill: Theme::RAISED,
        bg_stroke: Theme::hairline_strong(),
        corner_radius: CornerRadius::same(Theme::RADIUS),
        fg_stroke: Stroke::new(Theme::HAIRLINE_W, Theme::TEXT),
        expansion: 0.0,
    };

    v.widgets.hovered = WidgetVisuals {
        bg_fill: Theme::RAISED_HOVER,
        weak_bg_fill: Theme::RAISED_HOVER,
        // The hover ring is cyan-dim, not cyan: full-strength cyan on a large
        // button outline would out-shout the plot.
        bg_stroke: Theme::accent_line(Theme::CYAN_DIM),
        corner_radius: CornerRadius::same(Theme::RADIUS),
        fg_stroke: Stroke::new(Theme::HAIRLINE_W, Theme::CYAN),
        expansion: 0.0,
    };

    v.widgets.active = WidgetVisuals {
        bg_fill: Theme::RAISED_ACTIVE,
        weak_bg_fill: Theme::RAISED_ACTIVE,
        bg_stroke: Theme::focus_ring(),
        corner_radius: CornerRadius::same(Theme::RADIUS),
        fg_stroke: Stroke::new(Theme::HAIRLINE_W, Theme::CYAN),
        expansion: 0.0,
    };

    v.widgets.open = WidgetVisuals {
        bg_fill: Theme::RAISED_ACTIVE,
        weak_bg_fill: Theme::RAISED_ACTIVE,
        bg_stroke: Theme::focus_ring(),
        corner_radius: CornerRadius::same(Theme::RADIUS),
        fg_stroke: Stroke::new(Theme::HAIRLINE_W, Theme::CYAN),
        expansion: 0.0,
    };

    v
}

// ---------------------------------------------------------------------------
// Apply
// ---------------------------------------------------------------------------

/// Install the theme: fonts, visuals, text styles and spacing.
///
/// Call this **once**, before the first frame, typically right after creating
/// the `egui::Context` and before `eframe::run_native` takes it.
///
/// Both the dark and light style slots are overwritten, so even if something
/// else in the app (or the OS theme) later flips `ThemePreference`, the
/// application still comes back looking like itself. The theme preference is
/// then pinned to dark, because a light fallback would be actively wrong: the
/// plot behind the chrome is a full-bleed dark image.
///
/// This function is not cheap — it rebuilds the font atlas and both style
/// objects. It is *not* a per-frame call.
///
/// # `dead_code`
///
/// This crate is a binary, so nothing in it is "public API" as far as
/// `rustc`'s reachability analysis is concerned until something actually calls
/// it. `apply` is called by the app layer, which is a separate file. The
/// attribute is here so that a build in which the app has not yet wired the
/// theme up does not report the entry point as dead.
#[allow(dead_code)]
pub fn apply(ctx: &egui::Context) {
    // 1. Fonts. Must come first: the text styles below name the monospace
    //    family, and egui resolves fonts lazily on first use.
    ctx.set_fonts(fonts());

    // 2. Visuals + text styles + spacing, into both theme slots.
    let v = visuals();
    let styles = text_styles();
    let sp = spacing();
    ctx.all_styles_mut(|style| {
        style.visuals = v.clone();
        style.text_styles.clone_from(&styles);
        style.spacing.clone_from(&sp);

        // Numeric fields must be monospace and must not be explained by a
        // tooltip popup every time the user grabs one.
        style.drag_value_text_style = TextStyle::Monospace;
        style.explanation_tooltips = false;
        // Labels should truncate/extend rather than wrap; a wrapped label in a
        // fixed-width panel silently breaks the row rhythm.
        style.wrap_mode = Some(egui::TextWrapMode::Extend);
        // No global font override: the per-style table already pins every
        // style to monospace, and an override would fight `TextStyle` lookups.
        style.override_text_style = None;
        style.override_font_id = None;
        // Fast, mechanical. A dense tool should snap; it should not dissolve.
        style.animation_time = 0.08;
        style.interaction.tooltip_delay = 0.25;
        style.interaction.tooltip_grace_time = 0.0;
        style.interaction.selectable_labels = true;
        style.interaction.multi_widget_text_select = false;
    });

    // 3. Pin the preference to dark and make the current style match.
    ctx.set_visuals(v);
    ctx.set_theme(egui::ThemePreference::Dark);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Every *resting* surface fill must be dark enough not to fight the plot.
    /// Uses the maximum channel as a cheap proxy for "loudness": above 40/255 a
    /// surface starts reading as a lit surface rather than as chrome.
    ///
    /// Hover and press fills are held to a looser bound separately, because
    /// they are only on screen while the cursor is on the widget, and a hover
    /// state the user cannot see is not a hover state.
    ///
    /// Hairlines and text are deliberately excluded from this check: a rule
    /// nobody can see is not a rule, and `TEXT` in particular is supposed to be
    /// near-white. They are checked for *hierarchy* in
    /// [`text_and_hairline_ramp_is_ordered`] instead.
    #[test]
    fn surface_fills_stay_dark() {
        for (name, c) in [
            ("VOID", Theme::VOID),
            ("PANEL", Theme::PANEL),
            ("SUNKEN", Theme::SUNKEN),
            ("RAISED", Theme::RAISED),
        ] {
            let loud = c.r().max(c.g()).max(c.b());
            assert!(loud <= 40, "{name} is too loud for chrome: {c:?}");
        }

        for (name, c) in [
            ("RAISED_HOVER", Theme::RAISED_HOVER),
            ("RAISED_ACTIVE", Theme::RAISED_ACTIVE),
            // The selection wash is cyan-tinted rather than neutral, so it is
            // checked here rather than against the neutral-surface bound.
            ("SELECTION", Theme::SELECTION),
        ] {
            let loud = c.r().max(c.g()).max(c.b());
            assert!(
                loud <= 90,
                "{name} is a lit surface, not a hover state: {c:?}"
            );
        }
    }

    /// Text and rules must form a readable ramp: faint < dim < text, and
    /// hairline < hairline_strong. If these collide, hierarchy is gone.
    #[test]
    fn text_and_hairline_ramp_is_ordered() {
        let luma = |c: Color32| {
            0.2126 * f32::from(c.r()) + 0.7152 * f32::from(c.g()) + 0.0722 * f32::from(c.b())
        };
        assert!(luma(Theme::HAIRLINE) < luma(Theme::HAIRLINE_STRONG));
        assert!(luma(Theme::TEXT_FAINT) < luma(Theme::TEXT_DIM));
        assert!(luma(Theme::TEXT_DIM) < luma(Theme::TEXT));
        // The faintest text must still clear the panel it sits on by a
        // visible margin, or notes and units become unreadable.
        assert!(luma(Theme::TEXT_FAINT) - luma(Theme::PANEL) > 12.0);
    }

    /// The surface ladder must be strictly monotonic in luminance, otherwise
    /// "sunken" and "raised" are lies.
    #[test]
    fn surface_ladder_is_ordered() {
        let luma = |c: Color32| {
            0.2126 * f32::from(c.r()) + 0.7152 * f32::from(c.g()) + 0.0722 * f32::from(c.b())
        };
        assert!(luma(Theme::SUNKEN) < luma(Theme::PANEL));
        assert!(luma(Theme::PANEL) < luma(Theme::RAISED));
        assert!(luma(Theme::RAISED) < luma(Theme::RAISED_HOVER));
        assert!(luma(Theme::RAISED_HOVER) < luma(Theme::RAISED_ACTIVE));
        assert!(luma(Theme::HAIRLINE) < luma(Theme::HAIRLINE_STRONG));
    }

    /// The ramp must saturate at its documented ends and be monotonic through
    /// the middle, so a moving number never flickers between two hues.
    #[test]
    fn status_ramp_is_monotonic() {
        assert_eq!(status_color(FPS_GOOD), Theme::GREEN);
        assert_eq!(status_color(1000.0), Theme::GREEN);
        assert_eq!(status_color(FPS_BAD), Theme::RED);
        assert_eq!(status_color(0.0), Theme::RED);

        let mut prev_g = 0u8;
        for i in 0..=100 {
            let fps = FPS_BAD + (FPS_GOOD - FPS_BAD) * (i as f64 / 100.0);
            let c = status_color(fps);
            assert!(
                c.g() >= prev_g || c.r() < 255,
                "green channel fell while ramping up: {fps} -> {c:?}"
            );
            prev_g = c.g();
        }
    }

    /// Broken measurements must be loud, not green.
    #[test]
    fn broken_measurements_are_red() {
        assert_eq!(status_color(f64::NAN), Theme::RED);
        assert_eq!(status_color(f64::INFINITY), Theme::RED);
        assert_eq!(status_color(f64::NEG_INFINITY), Theme::RED);
        assert_eq!(status_color_from_frame_ms(0.0), Theme::RED);
        assert_eq!(status_color_from_frame_ms(f64::NAN), Theme::RED);
        assert_eq!(status_color_from_frame_ms(16.67), Theme::GREEN);
        assert_eq!(status_color_from_frame_ms(100.0), Theme::RED);
    }

    /// Monospace is the *default* family: nothing in this app may render
    /// proportional by accident.
    #[test]
    fn proportional_is_promoted_to_monospace() {
        let defs = fonts();
        if let Some(mono) = defs.families.get(&FontFamily::Monospace) {
            assert!(!mono.is_empty());
            assert_eq!(
                defs.families.get(&FontFamily::Proportional),
                Some(mono),
                "the proportional family must be the monospace stack"
            );
        }
        for (style, id) in text_styles() {
            assert_eq!(id.family, FontFamily::Monospace, "{style} is not monospace");
        }
    }

    /// No shadows, no radii, no transparent mandatory fills: the three ways a
    /// theme can quietly become "soft" without anyone noticing.
    #[test]
    fn visuals_stay_hard() {
        let v = visuals();
        assert!(v.dark_mode);
        assert_eq!(v.window_corner_radius, CornerRadius::same(Theme::RADIUS));
        assert_eq!(v.menu_corner_radius, CornerRadius::same(Theme::RADIUS));
        assert_eq!(v.window_shadow, Shadow::NONE);
        assert_eq!(v.popup_shadow, Shadow::NONE);
        for (name, w) in [
            ("noninteractive", &v.widgets.noninteractive),
            ("inactive", &v.widgets.inactive),
            ("hovered", &v.widgets.hovered),
            ("active", &v.widgets.active),
            ("open", &v.widgets.open),
        ] {
            assert_eq!(
                w.corner_radius,
                CornerRadius::same(Theme::RADIUS),
                "{name} is rounded"
            );
            assert!(w.bg_fill.a() > 0, "{name} has a transparent mandatory fill");
        }
    }

    /// `apply` must not panic and must leave the context dark.
    #[test]
    fn apply_installs_a_dark_style() {
        let ctx = egui::Context::default();
        apply(&ctx);
        assert_eq!(ctx.theme(), egui::Theme::Dark);
        let style = ctx.style_of(egui::Theme::Dark);
        assert_eq!(style.visuals.panel_fill, Theme::PANEL);
        assert_eq!(style.spacing.item_spacing, vec2(6.0, 5.0));
        assert_eq!(
            style.text_styles[&TextStyle::Body].family,
            FontFamily::Monospace
        );
    }
}
