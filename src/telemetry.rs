//! Frame timing and per-stage CPU/GPU statistics.
//!
//! The brief for this program calls for "sub-millisecond frame telemetry" and
//! an oscilloscope view of jitter, so this is an instrument, not a stopwatch.
//! Three things come out of it:
//!
//! * An **oscilloscope**: [`FrameHistory`], a fixed-capacity ring of the last
//!   128 frame durations, in milliseconds, oldest first. The panel plots this
//!   directly.
//! * A **worst-case needle**: the maximum over the retained window, which is
//!   the number that actually catches a hitch. A mean never notices one.
//! * Two **frame-rate estimates**: an exponential moving average (`fps_instant`,
//!   responsive, tracks a change within a few frames) and a windowed average
//!   over the whole ring (`fps_average`, stable, immune to a single spike).
//!   Disagreement between them *is* the jitter signal.
//!
//! # GPU timing is optional
//!
//! `record_gpu_time` is fed by the renderer's timestamp-query result, and only
//! when the adapter actually exposes `TimestampQuery`. It may never be called on
//! a machine without that feature, so `gpu_ms` reads `0.0` rather than `NaN`
//! and the panel can show "GPU n/a" off [`Telemetry::has_gpu_timing`].
//!
//! # Deliberate design choices
//!
//! * **`std::time::Instant`, not the `instant` crate.** The crate is already
//!   built against a real wgpu/eframe stack on a desktop target, where
//!   `std::time::Instant` is a real monotonic clock. Adding a dependency to get
//!   the same clock on platforms this binary does not run on is not a trade
//!   worth making.
//! * **`snapshot()` returns an owned [`TelemetrySnapshot`] with a `Vec<f32>`
//!   rather than borrowing the ring.** The obvious alternative is a guard type
//!   that borrows the tracker, but that forces the caller to hold the borrow
//!   across the whole draw of the plot, which in `egui` means borrowing the
//!   `Telemetry` for the duration of a closure - awkward to combine with a
//!   `begin_frame`/`end_frame` pair in the same pass. The cost is one
//!   128-element allocation per frame, about 512 bytes, once per 16 ms: roughly
//!   0.02% of the frame budget, and invisible next to the work already being
//!   done. Correctness and borrow ergonomics win a trade that cheap.
//! * **There is exactly one way to read a number.** [`Telemetry::snapshot`] is
//!   the only read path, and the per-field accessors that used to duplicate it
//!   (`frame_ms`, `cpu_ms`, `gpu_ms`, `fps_instant`, `sample_count`) are gone.
//!   Two ways to read one value is two values that can disagree, and a
//!   statistics type whose point is to be believed should not have that. The
//!   statistics that are *not* in the snapshot — the windowed frame rate and
//!   whether a GPU timestamp has ever landed — keep their own accessors, because
//!   they are derived on demand rather than stored.

/// Number of frame durations retained by default.
///
/// 128 is a power of two so the ring's index arithmetic is a mask rather than a
/// division, and at 60 Hz it is a little over two seconds of history, which is
/// the shortest window in which a periodic stutter is recognisable as periodic
/// rather than as noise.
pub const FRAME_HISTORY_CAPACITY: usize = 128;

/// Rate of the FPS exponential moving average, in units of `1/s`.
///
/// A 0.25 s time constant. Matches the reasoning in
/// [`crate::camera::Camera::update`]: the blend factor is `1 - exp(-k * dt)`,
/// so the estimate advances by the same proportion per *second* no matter what
/// the frame rate is. The naive `k * dt` form would make the readout's
/// sensitivity a function of the machine it is running on, which is precisely
/// the wrong property for an instrument used to compare machines.
const FPS_SMOOTHING_RATE: f64 = 4.0;

/// The exponential-moving-average blend factor for a frame lasting `dt`
/// seconds.
///
/// `1 - exp(-k * dt)` rather than the naive `k * dt`: the naive form is a
/// first-order expansion of the right one, valid only while `k * dt` is small,
/// and its step size is per *frame* - so the same code would react twice as
/// slowly on a 30 Hz machine as on a 60 Hz one, and overshoot outright once
/// `k * dt` exceeds 1. That is the wrong property for an instrument whose whole
/// job is to compare machines.
fn fps_blend(dt: f64) -> f64 {
    1.0 - (-FPS_SMOOTHING_RATE * dt).exp()
}

/// A fixed-capacity ring buffer of recent frame durations, in milliseconds.
///
/// Oldest-to-newest, wrapping, and `O(1)` per [`FrameHistory::push`] with no
/// allocation after construction. The capacity is fixed at
/// [`FRAME_HISTORY_CAPACITY`] rather than being configurable: this buffer is
/// read by the plot every frame, and a capacity that cannot change means the
/// index arithmetic can never be wrong for a value the caller chose.
#[derive(Debug, Clone)]
pub struct FrameHistory {
    /// Storage. Fixed at `CAPACITY`; entries are milliseconds.
    slots: Box<[f32]>,
    /// Index of the next slot to write. Also the oldest entry once full.
    head: usize,
    /// Number of live entries, saturating at `CAPACITY`.
    len: usize,
}

impl FrameHistory {
    /// Fixed number of retained frame durations.
    pub const CAPACITY: usize = FRAME_HISTORY_CAPACITY;

    /// Creates an empty history with a fixed capacity of
    /// [`FrameHistory::CAPACITY`] milliseconds.
    ///
    /// Allocation happens once, here; see the module docs for why the capacity
    /// is fixed rather than a parameter.
    pub fn new() -> Self {
        Self {
            slots: vec![0.0; Self::CAPACITY].into_boxed_slice(),
            head: 0,
            len: 0,
        }
    }

    /// Records one frame duration, given in **seconds**, and evicts the oldest
    /// entry if the ring is full.
    ///
    /// `O(1)`, allocation-free, and overwrites in place. Storing milliseconds
    /// rather than seconds keeps the plot's `f32` pipeline free of a multiply
    /// per point, and milliseconds is the unit the panel wants to draw.
    ///
    /// A negative duration is clamped to `0.0`, and a non-finite one is dropped
    /// entirely. Both would otherwise be permanent: one `NaN` in the ring
    /// poisons [`FrameHistory::mean_ms`] and [`FrameHistory::worst_ms`] for as
    /// long as it takes 128 healthy frames to rotate it out, which is long
    /// enough that the instrument looks broken.
    pub fn push(&mut self, dt_seconds: f64) {
        debug_assert!(self.slots.len() == Self::CAPACITY);
        if !dt_seconds.is_finite() {
            return;
        }
        let ms = (if dt_seconds > 0.0 { dt_seconds } else { 0.0 }) * 1000.0;
        self.slots[self.head] = ms as f32;
        self.head = (self.head + 1) % Self::CAPACITY;
        if self.len < Self::CAPACITY {
            self.len += 1;
        }
    }

    /// Number of retained frame durations, saturating at
    /// [`FrameHistory::CAPACITY`].
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no frame has been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Iterates the retained durations in milliseconds, **oldest first**.
    ///
    /// Once the ring is full the oldest entry is the one `push` is about to
    /// overwrite, which is `head`; before that, entry zero. Both cases are
    /// resolved once, up front, so iteration is a straight walk rather than a
    /// per-element special case.
    pub fn iter(&self) -> impl Iterator<Item = f32> + '_ {
        let start = if self.len == Self::CAPACITY {
            self.head
        } else {
            0
        };
        (0..self.len).map(move |offset| self.slots[(start + offset) % Self::CAPACITY])
    }

    /// Longest retained frame in milliseconds, or `0.0` when empty.
    ///
    /// Returned as the stored `f32` because this is plotted alongside the
    /// history it came from, and reading the same bits out of both avoids a
    /// discrepancy between the needle and the trace.
    pub fn worst_ms(&self) -> f32 {
        let mut worst = 0.0_f32;
        for ms in self.iter() {
            if ms > worst {
                worst = ms;
            }
        }
        worst
    }

    /// Mean retained frame duration in milliseconds, or `0.0` when empty.
    ///
    /// Accumulated in `f64` even though the samples are `f32`: this value is
    /// fed straight to a reciprocal to produce [`Telemetry::fps_average`], and
    /// `f32`'s 24-bit mantissa shows up in a frame rate as a visible wobble in
    /// the last digit.
    pub fn mean_ms(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        let total: f64 = self.iter().map(f64::from).sum();
        total / self.len as f64
    }

    /// Drops every retained sample and rewinds the ring, without reallocating.
    pub fn clear(&mut self) {
        self.slots.iter_mut().for_each(|slot| *slot = 0.0);
        self.head = 0;
        self.len = 0;
    }
}

impl Default for FrameHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// An owned, self-contained reading of the telemetry, for the UI layer.
///
/// Deliberately a plain value: the panel can hold one across a layout pass
/// without borrowing the [`Telemetry`] it came from, and two snapshots can be
/// compared directly in a test. See the module docs for why the history is
/// copied rather than borrowed.
///
/// Every field is finite, including on a brand-new tracker: an empty reading is
/// all zeros, never `NaN`. A `NaN` in a plot is not a cosmetic problem - egui
/// silently drops the geometry, so the scope renders empty with no clue why.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TelemetrySnapshot {
    /// Instantaneous frame rate in frames per second, the exponential moving
    /// average of `1 / dt`. The responsive number; use it for the headline
    /// readout. The windowed alternative is [`Telemetry::fps_average`].
    pub fps: f64,
    /// Wall-clock duration of the most recently completed frame, in
    /// milliseconds.
    pub frame_ms: f64,
    /// CPU-side cost of the most recently measured frame, in milliseconds
    /// (`0.0` if the renderer never reported one). Distinct from
    /// [`TelemetrySnapshot::frame_ms`], which includes the GPU wait.
    pub cpu_ms: f64,
    /// GPU-side cost from the renderer's timestamp query, in milliseconds
    /// (`0.0` when the adapter has no timestamp query, or none has landed yet).
    pub gpu_ms: f64,
    /// Longest frame in the retained window, in milliseconds; `0.0` when
    /// empty. The hitches are here, not in the mean.
    pub worst_frame_ms: f64,
    /// Number of samples in [`TelemetrySnapshot::history`].
    pub sample_count: usize,
    /// Recent frame durations in milliseconds, oldest first, at most
    /// [`FRAME_HISTORY_CAPACITY`] long. The oscilloscope trace.
    pub history: Vec<f32>,
}

/// Accumulates per-frame CPU and GPU timings and turns them into statistics.
///
/// The intended frame shape, from `app.rs`:
///
/// ```text
/// telemetry.begin_frame();
/// ... update camera, build uniforms, record_cpu_time(work_done_so_far) ...
/// renderer.render(&uniforms, &mut telemetry);   // calls record_gpu_time
/// telemetry.end_frame();
/// let reading = telemetry.snapshot();           // handed to the panel
/// ```
///
/// `end_frame` is what closes the wall-clock interval, so it must be the last
/// thing the frame does. Calling it without a matching `begin_frame` is not a
/// panic: it records nothing and returns `0.0`, because a mis-ordered call from
/// the app layer should degrade one frame's readout, not take the window down.
#[derive(Debug, Clone)]
pub struct Telemetry {
    /// The oscilloscope ring.
    history: FrameHistory,
    /// Start of the frame in flight, or `None` when no frame is open.
    frame_start: Option<std::time::Instant>,
    /// Wall-clock duration of the last completed frame, in milliseconds.
    frame_ms: f64,
    /// CPU cost reported for the frame in flight or just finished.
    cpu_ms: f64,
    /// GPU cost reported for the frame in flight or just finished.
    gpu_ms: f64,
    /// Exponential moving average of the instantaneous frame rate.
    fps_instant: f64,
    /// Whether the renderer has ever supplied a GPU timestamp, so the panel can
    /// say "n/a" instead of drawing a confident `0.00 ms`.
    gpu_timing_seen: bool,
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::new()
    }
}

impl Telemetry {
    /// Creates a tracker with an empty history and no frame in flight.
    pub fn new() -> Self {
        Self {
            history: FrameHistory::new(),
            frame_start: None,
            frame_ms: 0.0,
            cpu_ms: 0.0,
            gpu_ms: 0.0,
            fps_instant: 0.0,
            gpu_timing_seen: false,
        }
    }

    /// Opens a frame: stamps the start time and clears the per-frame stage
    /// timers.
    ///
    /// Clearing the stage timers here rather than in `end_frame` means the
    /// values in a snapshot always describe one specific frame. Leaving them
    /// alone would make a stage that stops being reported keep displaying its
    /// last result forever, which reads as a plausible number and is worse than
    /// a zero.
    pub fn begin_frame(&mut self) {
        self.frame_start = Some(std::time::Instant::now());
        self.cpu_ms = 0.0;
        self.gpu_ms = 0.0;
    }

    /// Closes the frame opened by [`Telemetry::begin_frame`], records its
    /// duration, and returns that duration in milliseconds.
    ///
    /// This is the only place wall-clock time enters the tracker. The sample is
    /// pushed onto the history and the instantaneous frame rate is updated
    /// with the same frame-rate-independent blend factor the camera uses, so
    /// the readout tracks a change just as quickly at 30 Hz as at 144 Hz.
    ///
    /// Returns `0.0` and records nothing if no frame is open.
    pub fn end_frame(&mut self) -> f64 {
        let Some(start) = self.frame_start.take() else {
            return 0.0;
        };
        let dt = start.elapsed().as_secs_f64();
        self.frame_ms = dt * 1000.0;
        self.history.push(dt);
        if dt > 0.0 {
            let instantaneous = 1.0 / dt;
            self.fps_instant = if self.fps_instant > 0.0 {
                let alpha = fps_blend(dt);
                self.fps_instant + (instantaneous - self.fps_instant) * alpha
            } else {
                // Seed the average with the first sample instead of pulling it
                // up from zero over the first second.
                instantaneous
            };
        }
        self.frame_ms
    }

    /// Records the CPU-side cost of the frame in flight.
    ///
    /// Called by the renderer with the time spent outside the GPU wait. Fed by
    /// the same timestamp that [`Telemetry::begin_frame`] took, so it is a real
    /// measurement rather than a guess; if the renderer has no such split it
    /// simply never calls this and `cpu_ms` stays `0.0`.
    pub fn record_cpu_time(&mut self, duration: std::time::Duration) {
        self.cpu_ms = duration.as_secs_f64() * 1000.0;
    }

    /// Records the GPU-side cost of the frame in flight, from the renderer's
    /// timestamp query.
    ///
    /// The result of a `wgpu::QuerySet` timestamp read, which is one to three
    /// frames stale by the time it can be mapped. That latency is inherent to
    /// the mechanism and is why the value is reported as-is: the alternative
    /// is to invent a latency compensation that is wrong whenever the queue
    /// depth changes.
    pub fn record_gpu_time(&mut self, duration: std::time::Duration) {
        self.gpu_ms = duration.as_secs_f64() * 1000.0;
        self.gpu_timing_seen = true;
    }

    /// Whether the renderer has ever supplied a GPU timestamp.
    ///
    /// Lets the panel distinguish "the GPU work took no time" from "this adapter
    /// has no timestamp query, so nobody is measuring". Use it to render
    /// "n/a" rather than `0.00 ms`.
    pub fn has_gpu_timing(&self) -> bool {
        self.gpu_timing_seen
    }

    /// Windowed average frame rate in frames per second, taken over the whole
    /// retained history. `0.0` when empty.
    ///
    /// The reciprocal of the mean frame time, so a single 200 ms stall in the
    /// window pulls it down exactly as much as 12 good frames would - which is
    /// the point. It is the number to quote when comparing runs; compare it
    /// against [`TelemetrySnapshot::fps`] and the gap between them is the
    /// jitter.
    ///
    /// Not in the snapshot because it is derived from the whole ring on demand
    /// rather than carried from frame to frame, and recomputing it once per
    /// frame is far cheaper than storing a number that can go stale.
    pub fn fps_average(&self) -> f64 {
        let mean = self.history.mean_ms();
        if mean > 0.0 {
            1000.0 / mean
        } else {
            0.0
        }
    }

    /// Longest frame in the retained window in milliseconds, or `0.0` when
    /// empty.
    ///
    /// The same value as [`TelemetrySnapshot::worst_frame_ms`]; kept as an
    /// accessor because the panel asks for it beside the frame rate rather than
    /// unpacking a whole snapshot for two numbers.
    pub fn worst_frame_ms(&self) -> f64 {
        f64::from(self.history.worst_ms())
    }

    /// The oscilloscope ring, for the panel to plot.
    ///
    /// The returned reference is a live view; prefer
    /// [`Telemetry::snapshot`] when the reading has to outlive the borrow.
    pub fn history(&self) -> &FrameHistory {
        &self.history
    }

    /// An owned, self-contained reading of the current state.
    ///
    /// Copies the ring into [`TelemetrySnapshot::history`] rather than
    /// borrowing, so the panel is not tied to the tracker's lifetime. The copy
    /// costs one small allocation per frame, which the module docs justify in
    /// full. Cheaper derived values - `worst_frame_ms`, `sample_count` - are
    /// computed on the way out so they cannot drift from the trace they came
    /// from.
    pub fn snapshot(&self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            fps: self.fps_instant,
            frame_ms: self.frame_ms,
            cpu_ms: self.cpu_ms,
            gpu_ms: self.gpu_ms,
            worst_frame_ms: f64::from(self.history.worst_ms()),
            sample_count: self.history.len(),
            history: self.history.iter().collect(),
        }
    }

    /// Discards all history and counters, and closes any frame in flight.
    ///
    /// For "reset the benchmark" in the UI, and after a stall that should not
    /// colour the next window.
    ///
    /// `#[allow(dead_code)]`: no caller yet. The panel has no "reset the
    /// benchmark" command — a new [`crate::panel::PanelAction`] would be the
    /// thing that called this, and inventing a button for it is a UI decision
    /// rather than a correctness fix, so the method stays. It is not dead
    /// weight: `FrameHistory::clear` has no other caller either, and the pair
    /// is the only way to rewind this tracker without dropping it.
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.history.clear();
        self.frame_start = None;
        self.frame_ms = 0.0;
        self.cpu_ms = 0.0;
        self.gpu_ms = 0.0;
        self.fps_instant = 0.0;
        self.gpu_timing_seen = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Fills `history` to exactly capacity with a recognisable pattern.
    fn fill_history(history: &mut FrameHistory) {
        for _ in 0..FrameHistory::CAPACITY {
            history.push(0.004);
        }
    }

    /// Asserts two `f64`s agree to within `tolerance`, with a message naming
    /// both so a failure is readable without a debugger.
    #[track_caller]
    /// Absolute-or-relative closeness.
    ///
    /// A fixed *absolute* tolerance is the wrong tool for any quantity that
    /// can grow: one `f64` ulp is about `1.8e-9` at a magnitude of `1e7`, so a
    /// `1e-9` absolute tolerance is unmeetable by construction and fails on
    /// arithmetic that is exactly right. The tolerance is therefore taken as
    /// the larger of the absolute floor and a relative share of the larger
    /// magnitude, so the assertion stays strict near zero (where the absolute
    /// term binds) and stays meaningful for large values (where the relative
    /// term binds).
    fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
        let difference = (actual - expected).abs();
        let magnitude = actual.abs().max(expected.abs());
        let effective = tolerance.max(tolerance * magnitude);
        assert!(
            difference <= effective,
            "{what}: {actual} != {expected} (abs diff {difference:e}, \
             tolerance {tolerance:e}, effective {effective:e})"
        );
    }

    /// Drives `n` frames of a synthetic `dt` straight into the history.
    ///
    /// The public path for this is `begin_frame`/`end_frame`, which can only be
    /// as fast as the test host actually runs; this reaches the statistics
    /// under test without a 128-frame sleep. The wiring itself is covered by
    /// `end_frame_records_the_wall_time`.
    fn fill(telemetry: &mut Telemetry, dt_seconds: f64, n: usize) {
        for _ in 0..n {
            telemetry.history.push(dt_seconds);
        }
    }

    #[test]
    fn default_capacity_is_a_power_of_two_of_128() {
        assert_eq!(FRAME_HISTORY_CAPACITY, 128);
        assert!(FRAME_HISTORY_CAPACITY.is_power_of_two());
        assert_eq!(FrameHistory::CAPACITY, 128);
    }

    #[test]
    fn new_history_is_empty_and_reports_its_capacity() {
        let history = FrameHistory::new();
        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
        assert_eq!(history.iter().count(), 0);
    }

    #[test]
    fn push_stores_milliseconds() {
        let mut history = FrameHistory::new();
        history.push(0.016);
        assert_eq!(history.len(), 1);
        let only = history.iter().next().expect("one sample was pushed");
        assert_close(f64::from(only), 16.0, 1e-3, "milliseconds");
    }

    #[test]
    fn push_yields_oldest_to_newest_order() {
        let mut history = FrameHistory::new();
        for step in 0..5 {
            history.push(0.001 * (step + 1) as f64);
        }
        let got: Vec<f32> = history.iter().collect();
        let want = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(got, want);
    }

    #[test]
    fn push_wraps_at_capacity_without_growing() {
        let mut history = FrameHistory::new();
        for step in 0..(FrameHistory::CAPACITY * 3) {
            history.push(0.001 * (step + 1) as f64);
        }
        assert_eq!(
            history.len(),
            FrameHistory::CAPACITY,
            "the ring must saturate, not grow"
        );
        assert_eq!(history.iter().count(), FrameHistory::CAPACITY);

        // The survivors are the most recent `CAPACITY` samples, in order. Each
        // push was `n` milliseconds for `n` in 1..=total.
        let total = FrameHistory::CAPACITY * 3;
        let oldest_kept = total - FrameHistory::CAPACITY + 1;
        for (offset, ms) in history.iter().enumerate() {
            let want = (oldest_kept + offset) as f64;
            assert_close(f64::from(ms), want, 1e-3, "wrapped sample");
        }
    }

    #[test]
    fn ordering_survives_a_wrap() {
        let mut history = FrameHistory::new();
        let cap = FrameHistory::CAPACITY;
        for step in 0..(cap + 1) {
            history.push(0.001 * (step + 1) as f64);
        }
        // 129 pushes into a 128-slot ring: sample 1 is gone, 2..=129 remain.
        let got: Vec<f32> = history.iter().collect();
        assert_eq!(got.len(), cap);
        assert_close(f64::from(got[0]), 2.0, 1e-3, "oldest after wrap");
        assert_close(f64::from(got[cap - 1]), 129.0, 1e-3, "newest after wrap");
        let ascending = got.windows(2).all(|pair| pair[0] < pair[1]);
        assert!(ascending, "order broke across the wrap");
    }

    #[test]
    fn push_ignores_non_finite_and_clamps_negative_durations() {
        let mut history = FrameHistory::new();
        history.push(f64::NAN);
        history.push(f64::INFINITY);
        history.push(-1.0);
        assert_eq!(history.len(), 1, "NaN is dropped, the negative is clamped");
        let got: Vec<f32> = history.iter().collect();
        assert_eq!(got, vec![0.0_f32]);
        assert!(history.worst_ms().is_finite());
        assert!(history.mean_ms().is_finite());
    }

    #[test]
    fn worst_and_mean_of_an_empty_history_are_zero() {
        let history = FrameHistory::new();
        assert_eq!(history.worst_ms(), 0.0);
        assert_eq!(history.mean_ms(), 0.0);
    }

    #[test]
    fn mean_and_worst_track_the_window() {
        let mut history = FrameHistory::new();
        for _ in 0..10 {
            history.push(0.010);
        }
        history.push(0.050);
        // (10 * 10 + 50) / 11
        assert_close(history.mean_ms(), 150.0 / 11.0, 1e-3, "mean with one spike");
        assert_close(f64::from(history.worst_ms()), 50.0, 1e-3, "worst");
    }

    #[test]
    fn clear_empties_the_ring() {
        let mut history = FrameHistory::new();
        fill_history(&mut history);
        assert_eq!(history.len(), FrameHistory::CAPACITY);
        history.clear();
        assert!(history.is_empty());
        assert_eq!(history.iter().count(), 0);
        assert_eq!(history.worst_ms(), 0.0);
        // Usable again from the start.
        history.push(0.002);
        assert_close(
            f64::from(history.iter().next().expect("one sample")),
            2.0,
            1e-3,
            "ms",
        );
    }

    #[test]
    fn default_agrees_with_new() {
        let mut a = Telemetry::new();
        fill(&mut a, 0.01, 4);
        let mut b = Telemetry::default();
        fill(&mut b, 0.01, 4);
        assert_eq!(a.snapshot(), b.snapshot());
    }

    #[test]
    fn snapshot_of_an_empty_tracker_is_zeros_not_nan() {
        let telemetry = Telemetry::new();
        let s = telemetry.snapshot();
        assert_eq!(s.fps, 0.0);
        assert_eq!(s.frame_ms, 0.0);
        assert_eq!(s.cpu_ms, 0.0);
        assert_eq!(s.gpu_ms, 0.0);
        assert_eq!(s.worst_frame_ms, 0.0);
        assert_eq!(s.sample_count, 0);
        assert!(s.history.is_empty());
        for value in [s.fps, s.frame_ms, s.cpu_ms, s.gpu_ms, s.worst_frame_ms] {
            assert!(value.is_finite(), "{value} is not finite");
        }
        assert!(s.fps.is_finite());
        assert!(telemetry.fps_average().is_finite());
        assert_eq!(telemetry.fps_average(), 0.0);
        assert!(!telemetry.has_gpu_timing());
    }

    #[test]
    fn fps_average_is_sane_for_a_constant_frame_time() {
        let mut telemetry = Telemetry::new();
        fill(&mut telemetry, 1.0 / 60.0, FrameHistory::CAPACITY);
        assert_close(telemetry.fps_average(), 60.0, 0.01, "60 Hz average");

        let mut display = Telemetry::new();
        fill(&mut display, 1.0 / 144.0, FrameHistory::CAPACITY);
        assert_close(display.fps_average(), 144.0, 0.01, "144 Hz average");
    }

    #[test]
    fn fps_average_uses_the_whole_window_not_just_the_newest() {
        // 12 frames of 10 ms and 12 of 30 ms average to 20 ms, i.e. 50 fps.
        // A tracker that only looked at the last frame would say 33.
        let mut telemetry = Telemetry::new();
        fill(&mut telemetry, 0.010, 12);
        fill(&mut telemetry, 0.030, 12);
        assert_close(telemetry.fps_average(), 50.0, 0.01, "windowed average");
    }

    #[test]
    fn worst_frame_tracks_the_maximum_of_the_window() {
        let mut telemetry = Telemetry::new();
        fill(&mut telemetry, 0.016, 20);
        assert_close(telemetry.worst_frame_ms(), 16.0, 0.1, "steady state");

        telemetry.history.push(0.250);
        assert_close(telemetry.worst_frame_ms(), 250.0, 0.1, "after the spike");

        // And it forgets the spike once it has rotated out.
        fill(&mut telemetry, 0.016, FrameHistory::CAPACITY);
        assert_close(
            telemetry.worst_frame_ms(),
            16.0,
            0.1,
            "after the window moves on",
        );
    }

    #[test]
    fn worst_frame_zero_before_any_frame() {
        let telemetry = Telemetry::new();
        assert_eq!(telemetry.worst_frame_ms(), 0.0);
        assert_eq!(telemetry.snapshot().sample_count, 0);
    }

    #[test]
    fn end_frame_records_the_wall_time() {
        let mut telemetry = Telemetry::new();
        telemetry.begin_frame();
        std::thread::sleep(Duration::from_millis(4));
        let ms = telemetry.end_frame();

        assert!(ms >= 3.0, "4 ms sleep reported {ms} ms");
        assert!(ms < 1000.0, "implausible frame time {ms} ms");
        assert_eq!(telemetry.snapshot().sample_count, 1);
        assert_close(
            telemetry.snapshot().frame_ms,
            ms,
            1e-12,
            "frame_ms mirrors end_frame",
        );
        assert_close(telemetry.history.mean_ms(), ms, 1e-3, "sample recorded");
        let fps = telemetry.snapshot().fps;
        assert!(fps > 0.0 && fps.is_finite());
    }

    #[test]
    fn end_frame_without_begin_frame_records_nothing() {
        let mut telemetry = Telemetry::new();
        assert_eq!(telemetry.end_frame(), 0.0);
        assert_eq!(telemetry.snapshot().sample_count, 0);
        // And the tracker is still usable afterwards.
        telemetry.begin_frame();
        assert!(telemetry.end_frame() >= 0.0);
        assert_eq!(telemetry.snapshot().sample_count, 1);
    }

    #[test]
    fn the_first_frame_seeds_the_instantaneous_rate() {
        let mut telemetry = Telemetry::new();
        telemetry.begin_frame();
        let first_ms = telemetry.end_frame();
        assert!(first_ms > 0.0, "the clock did not advance");
        // Seeded from the sample, not ramped up from zero: the first reading is
        // exactly 1/dt. Ramping would make the first second of every run read
        // as a ramp from nothing, which is exactly when it is being watched.
        assert_close(
            telemetry.snapshot().fps,
            1000.0 / first_ms,
            1e-6,
            "seeded fps",
        );
    }

    #[test]
    fn later_frames_smooth_the_instantaneous_rate() {
        let mut telemetry = Telemetry::new();
        telemetry.begin_frame();
        let first_ms = telemetry.end_frame();
        telemetry.begin_frame();
        let second_ms = telemetry.end_frame();

        // The average must move a known fraction of the way toward the second
        // frame's own rate, by the exponential factor for that frame.
        let start = 1000.0 / first_ms;
        let target = 1000.0 / second_ms;
        let alpha = fps_blend(second_ms / 1000.0);
        assert!((0.0..1.0).contains(&alpha), "blend {alpha} out of range");
        assert_close(
            telemetry.snapshot().fps,
            start + (target - start) * alpha,
            1e-9,
            "smoothed fps",
        );
    }

    #[test]
    fn the_fps_blend_is_frame_rate_independent() {
        // One 100 ms frame and two 50 ms frames must leave the same average.
        // With the naive `k * dt` factor the two-step version overshoots the
        // one-step version, and the gap is a direct function of frame rate.
        let one_step = fps_blend(0.1);
        let two_step_factor = 1.0 - (1.0 - fps_blend(0.05)).powi(2);
        assert_close(two_step_factor, one_step, 1e-12, "blend factors");
        assert!(one_step < 1.0 && one_step > 0.0);
    }

    #[test]
    fn fps_average_follows_a_change_in_frame_time() {
        // The windowed average has to actually move; a tracker stuck on the last
        // value would pass every constant-dt test above.
        let mut telemetry = Telemetry::new();
        fill(&mut telemetry, 0.016, 10);
        assert!(telemetry.fps_average() > 60.0);
        fill(&mut telemetry, 0.100, 10);
        assert!(
            telemetry.fps_average() < 30.0,
            "the average must follow the data, got {}",
            telemetry.fps_average()
        );
    }

    #[test]
    fn stage_timers_are_reported() {
        let mut telemetry = Telemetry::new();
        telemetry.begin_frame();
        telemetry.record_cpu_time(Duration::from_micros(250));
        telemetry.record_gpu_time(Duration::from_micros(1800));
        let s = telemetry.snapshot();
        assert_close(s.cpu_ms, 0.25, 1e-9, "cpu_ms");
        assert_close(s.gpu_ms, 1.8, 1e-9, "gpu_ms");
        assert!(telemetry.has_gpu_timing());
    }

    #[test]
    fn begin_frame_clears_the_stage_timers() {
        // Otherwise a stage that stops being reported shows its last value for
        // ever, which looks like a real measurement.
        let mut telemetry = Telemetry::new();
        telemetry.begin_frame();
        telemetry.record_cpu_time(Duration::from_millis(2));
        telemetry.record_gpu_time(Duration::from_millis(3));
        telemetry.end_frame();
        assert_close(telemetry.snapshot().cpu_ms, 2.0, 1e-9, "cpu reported");

        telemetry.begin_frame();
        let cleared = telemetry.snapshot();
        assert_eq!(cleared.cpu_ms, 0.0, "cpu must be cleared");
        assert_eq!(cleared.gpu_ms, 0.0, "gpu must be cleared");
    }

    #[test]
    fn snapshot_history_is_a_copy_not_a_live_view() {
        let mut telemetry = Telemetry::new();
        fill(&mut telemetry, 0.010, 8);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.sample_count, 8);
        assert_eq!(snapshot.history.len(), 8);

        // The tracker moves on; the snapshot must not.
        fill(&mut telemetry, 0.200, 4);
        assert_eq!(snapshot.history.len(), 8, "snapshot aliased the ring");
        assert_close(f64::from(snapshot.history[0]), 10.0, 1e-3, "first sample");
        assert_eq!(telemetry.snapshot().sample_count, 12);
    }

    #[test]
    fn snapshot_history_is_oldest_first() {
        let mut telemetry = Telemetry::new();
        // Six distinct samples, `n` milliseconds for `n` in 1..=6, so the
        // ordering is visible in the values rather than only in their count.
        for n in 1..=6 {
            telemetry.history.push(0.001 * f64::from(n));
        }
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.sample_count, 6);
        for (index, ms) in snapshot.history.iter().enumerate() {
            assert_close(
                f64::from(*ms),
                f64::from(index as u16 + 1),
                1e-3,
                "snapshot order",
            );
        }
    }

    /// The snapshot is the only read path, so this is the test that it agrees
    /// with the *ring* underneath it and with the accessors that are still
    /// worth having: the sample count, the worst frame and the trace must all
    /// describe the same window.
    #[test]
    fn snapshot_agrees_with_the_ring() {
        let mut telemetry = Telemetry::new();
        fill(&mut telemetry, 0.020, FrameHistory::CAPACITY);
        telemetry.history.push(0.400);
        let s = telemetry.snapshot();
        assert_eq!(s.sample_count, telemetry.history().len());
        assert_eq!(s.history.len(), telemetry.history().iter().count());
        assert_close(s.worst_frame_ms, telemetry.worst_frame_ms(), 1e-12, "worst");
        assert_close(s.fps, 0.0, 1e-12, "no real frames means no fps yet");
    }

    #[test]
    fn reset_clears_everything() {
        let mut telemetry = Telemetry::new();
        telemetry.begin_frame();
        telemetry.record_cpu_time(Duration::from_millis(1));
        telemetry.end_frame();
        fill(&mut telemetry, 0.010, 5);
        telemetry.fps_instant = 60.0;

        telemetry.reset();
        let s = telemetry.snapshot();
        assert_eq!(s, TelemetrySnapshot::default());
        assert_eq!(s.sample_count, 0);
        assert_eq!(s.fps, 0.0);
        assert_eq!(telemetry.fps_average(), 0.0);
        assert!(!telemetry.has_gpu_timing());
        // Still usable afterwards.
        telemetry.begin_frame();
        assert!(telemetry.end_frame() >= 0.0);
    }
}
