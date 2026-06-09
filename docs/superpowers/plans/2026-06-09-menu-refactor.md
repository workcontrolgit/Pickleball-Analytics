# Menu Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the confusing three-mode menu with three clearly-named, mutually-exclusive modes — Video Analysis, Split Rallies, and Detect Serve — each producing exactly one type of output.

**Architecture:** Add three named mode constants to `process_video.py` and restructure the main loop as an explicit if/elif dispatcher so each mode's steps are visible at a glance. Update `main.py` to use the new labels, show a per-mode description, and display mode-appropriate results.

**Tech Stack:** Python 3, customtkinter, OpenCV, existing BallTracker / PlayerTracker / CourtDetector / Analytics / ServeDetector / OllamaServeAnalyzer.

**Spec:** `docs/superpowers/specs/2026-06-09-menu-refactor-design.md`

---

## File Map

| File | What changes |
|---|---|
| `process_video.py` | Add 3 mode constants; conditional `__init__`; restructure `process_video()` as dispatcher |
| `main.py` | New mode labels; mode description label; updated mode_map import; per-mode results card |
| `tests/test_video_processor_modes.py` | New — unit tests for mode constants and dispatcher guard rails |

---

## Task 1: Add mode constants to `process_video.py`

**Files:**
- Modify: `process_video.py:36-50` (after imports, before `PROJECT_DIR`)
- Test: `tests/test_video_processor_modes.py` (create)

- [ ] **Step 1: Write the failing test**

Open `tests/test_video_processor_modes.py` and write:

```python
"""Tests for process_video mode constants and VideoProcessor mode guard."""
import pytest
from process_video import (
    MODE_VIDEO_ANALYSIS,
    MODE_SPLIT_RALLIES,
    MODE_DETECT_SERVE,
)


def test_mode_constants_are_distinct_strings():
    modes = [MODE_VIDEO_ANALYSIS, MODE_SPLIT_RALLIES, MODE_DETECT_SERVE]
    assert len(set(modes)) == 3
    assert all(isinstance(m, str) for m in modes)


def test_mode_constants_have_expected_values():
    assert MODE_VIDEO_ANALYSIS == "video_analysis"
    assert MODE_SPLIT_RALLIES  == "split_rallies"
    assert MODE_DETECT_SERVE   == "detect_serve"
```

- [ ] **Step 2: Run test to confirm it fails**

```
cd c:/apps/pickleball/Pickleball-Analytics
.venv/Scripts/pytest tests/test_video_processor_modes.py -v
```

Expected: `ImportError: cannot import name 'MODE_VIDEO_ANALYSIS'`

- [ ] **Step 3: Add mode constants to `process_video.py`**

In `process_video.py`, replace the block starting at line 37 that reads:
```python
# ==============================================================================
# Module‑level constants (easy to tweak and reuse)
# ==============================================================================
PROJECT_DIR: str = os.path.dirname(os.path.abspath(__file__))
```

with:
```python
# ==============================================================================
# Processing Modes
# ==============================================================================
# Pick exactly one per run. Each mode runs court/player/ball detection,
# then diverges based on what output it produces.

MODE_VIDEO_ANALYSIS = "video_analysis"
# Produces: Main_overlay.mp4 — annotated video with player/ball/court
#           overlays, bird's-eye view, and analytics panels.

MODE_SPLIT_RALLIES = "split_rallies"
# Produces: rally_01.mp4, rally_02.mp4, … — one raw clip per long rally
#           (5+ net crossings). No annotated video.

MODE_DETECT_SERVE = "detect_serve"
# Produces: serve_report.json — serve candidates scored by Ollama vision.
#           No video output. Fastest mode.

# ==============================================================================
# Module‑level constants (easy to tweak and reuse)
# ==============================================================================
PROJECT_DIR: str = os.path.dirname(os.path.abspath(__file__))
```

- [ ] **Step 4: Run test to confirm it passes**

```
.venv/Scripts/pytest tests/test_video_processor_modes.py::test_mode_constants_are_distinct_strings tests/test_video_processor_modes.py::test_mode_constants_have_expected_values -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add process_video.py tests/test_video_processor_modes.py
git commit -m "Add MODE_VIDEO_ANALYSIS, MODE_SPLIT_RALLIES, MODE_DETECT_SERVE constants"
```

---

## Task 2: Conditional `__init__` in `VideoProcessor`

**Files:**
- Modify: `process_video.py:99-117` (`VideoProcessor.__init__`)
- Test: `tests/test_video_processor_modes.py`

**Background:** Currently `__init__` always creates `Analytics`, `ServeDetector`, and
`OllamaServeAnalyzer` regardless of mode. After this task, each object is only created
when the mode actually needs it. This prevents Ollama workers from spinning up during
a Split Rallies run, and prevents Analytics from allocating heatmap buffers during a
Detect Serve run.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_video_processor_modes.py`:

```python
from unittest.mock import patch, MagicMock


def _make_processor(mode):
    """Create a VideoProcessor with all heavy dependencies mocked out."""
    with patch("process_video.BallTracker"), \
         patch("process_video.PlayerTracker"), \
         patch("process_video.CourtDetector"), \
         patch("process_video.Analytics"), \
         patch("process_video.ServeDetector"), \
         patch("process_video.OllamaServeAnalyzer"), \
         patch("process_video.VideoProcessor._make_output_dir", return_value="/tmp/run"), \
         patch("process_video.setup_logger"):
        from process_video import VideoProcessor
        return VideoProcessor.__new__(VideoProcessor)


def test_video_analysis_has_analytics_no_serve_detector():
    """VIDEO_ANALYSIS mode: analytics present, serve objects absent."""
    with patch("process_video.BallTracker"), \
         patch("process_video.PlayerTracker"), \
         patch("process_video.CourtDetector"), \
         patch("process_video.Analytics") as MockAnalytics, \
         patch("process_video.ServeDetector") as MockServeDetector, \
         patch("process_video.OllamaServeAnalyzer") as MockAnalyzer, \
         patch("process_video.VideoProcessor._make_output_dir", return_value="/tmp/run"), \
         patch("process_video.setup_logger"):
        from process_video import VideoProcessor, MODE_VIDEO_ANALYSIS
        p = VideoProcessor("fake.mp4", {}, mode=MODE_VIDEO_ANALYSIS)
        MockAnalytics.assert_called_once()
        MockServeDetector.assert_not_called()
        MockAnalyzer.assert_not_called()


def test_split_rallies_has_analytics_no_serve_detector():
    """SPLIT_RALLIES mode: analytics present, serve objects absent."""
    with patch("process_video.BallTracker"), \
         patch("process_video.PlayerTracker"), \
         patch("process_video.CourtDetector"), \
         patch("process_video.Analytics") as MockAnalytics, \
         patch("process_video.ServeDetector") as MockServeDetector, \
         patch("process_video.OllamaServeAnalyzer") as MockAnalyzer, \
         patch("process_video.VideoProcessor._make_output_dir", return_value="/tmp/run"), \
         patch("process_video.setup_logger"):
        from process_video import VideoProcessor, MODE_SPLIT_RALLIES
        p = VideoProcessor("fake.mp4", {}, mode=MODE_SPLIT_RALLIES)
        MockAnalytics.assert_called_once()
        MockServeDetector.assert_not_called()
        MockAnalyzer.assert_not_called()


def test_detect_serve_has_serve_detector_no_analytics():
    """DETECT_SERVE mode: serve objects present, analytics absent."""
    with patch("process_video.BallTracker"), \
         patch("process_video.PlayerTracker"), \
         patch("process_video.CourtDetector"), \
         patch("process_video.Analytics") as MockAnalytics, \
         patch("process_video.ServeDetector") as MockServeDetector, \
         patch("process_video.OllamaServeAnalyzer") as MockAnalyzer, \
         patch("process_video.VideoProcessor._make_output_dir", return_value="/tmp/run"), \
         patch("process_video.setup_logger"):
        from process_video import VideoProcessor, MODE_DETECT_SERVE
        p = VideoProcessor("fake.mp4", {}, mode=MODE_DETECT_SERVE)
        MockAnalytics.assert_not_called()
        MockServeDetector.assert_called_once()
        MockAnalyzer.assert_called_once()
```

- [ ] **Step 2: Run tests to confirm they fail**

```
.venv/Scripts/pytest tests/test_video_processor_modes.py -v -k "analytics or serve_detector"
```

Expected: 3 failed (Analytics and Serve objects always constructed right now)

- [ ] **Step 3: Rewrite `VideoProcessor.__init__`**

Replace the current `__init__` method (lines 100–117) with:

```python
def __init__(self, video_path: str, filters: dict, mode: str = MODE_VIDEO_ANALYSIS):
    self.video_path = video_path
    self.filters = self._apply_default_filters(filters)
    self.mode = mode

    # ── Detectors used by all modes ────────────────────────────────────
    self.ball_tracker   = BallTracker(os.path.join(MODELS_DIR, "ball_tracking.pt"))
    self.player_tracker = PlayerTracker(os.path.join(MODELS_DIR, "player_tracking.pt"))
    self.court_mapper   = CourtDetector(os.path.join(MODELS_DIR, "court_detection.pt"))
    self._cached_kps = None   # court keypoints cache (court is static)
    self._cached_Hmg = None

    # ── Analytics — Video Analysis and Split Rallies only ──────────────
    if self.mode in (MODE_VIDEO_ANALYSIS, MODE_SPLIT_RALLIES):
        self.analytics = Analytics(self.filters)

    # ── Serve pipeline — Detect Serve only ────────────────────────────
    if self.mode == MODE_DETECT_SERVE:
        self.serve_detector  = ServeDetector()
        self.serve_analyzer  = OllamaServeAnalyzer(model="qwen2.5vl:7b", workers=2)

    self.output_dir = self._make_output_dir()
    setup_logger(self.output_dir)
    self._start_time = time.time()
    logger.bind(video_path=self.video_path, mode=self.mode).info("run_started")
```

- [ ] **Step 4: Run tests to confirm they pass**

```
.venv/Scripts/pytest tests/test_video_processor_modes.py -v
```

Expected: all 5 tests pass

- [ ] **Step 5: Commit**

```bash
git add process_video.py tests/test_video_processor_modes.py
git commit -m "Conditional __init__: only construct objects needed by each mode"
```

---

## Task 3: Restructure `process_video()` as an explicit dispatcher

**Files:**
- Modify: `process_video.py:122-206` (the `process_video` method)

**Background:** Currently the method has `if self.mode != 'rallies_only'` and
`if self.mode in (...)` guards scattered through one tangled block. After this task,
each mode has a clearly labelled `if/elif` section in the frame loop and a matching
section in the `finally` block.

The `_render_analytics_grid` method currently calls
`self.serve_analyzer.get_results()` (line 423). After this task, that call only
happens in `MODE_VIDEO_ANALYSIS` — add a guard there or pass results explicitly.

- [ ] **Step 1: Replace the `process_video` method**

Replace the entire `process_video` method (lines 122–206) with:

```python
def process_video(self, progress_callback=None) -> str:
    cap = self._open_capture(self.video_path)
    total_frames, src_w, src_h, fps = self._read_video_meta(cap)
    out_w, out_h, main_w, be_w, grid_w, panel_w, panel_h = self._compute_layout(src_w, src_h)

    # Working dimensions — match per-frame downscale so fallback coords are correct
    work_w, work_h = src_w, src_h
    if src_h > 1080:
        scale = 1080 / src_h
        work_w = int(src_w * scale)
        work_h = 1080

    if self.mode in (MODE_VIDEO_ANALYSIS, MODE_SPLIT_RALLIES):
        self.analytics.set_canvas_size(work_w, work_h)
        self.analytics.set_video_context(total_frames=total_frames, fps=fps)

    writer = None
    out_path = None
    if self.mode == MODE_VIDEO_ANALYSIS:
        out_path, writer = self._create_writer(out_w, out_h, fps)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Downscale 4K+ frames to 1080p
            if frame.shape[0] > 1080:
                scale = 1080 / frame.shape[0]
                frame = cv2.resize(
                    frame,
                    (int(frame.shape[1] * scale), 1080),
                    interpolation=cv2.INTER_AREA,
                )

            # ── Common detections (all modes) ──────────────────────────
            if frame_idx % 30 == 0 or self._cached_kps is None:
                kps, Hmg = self.court_mapper.get_keypoints_and_homography(frame)
                if kps is not None:
                    self._cached_kps, self._cached_Hmg = kps, Hmg
            else:
                kps, Hmg = self._cached_kps, self._cached_Hmg

            if frame_idx == 0:
                logger.bind(frame_idx=frame_idx, homography_valid=Hmg is not None).info("court_detected")

            players, proj_players = self.player_tracker.detect_and_project(frame, Hmg)
            ball_det  = self.ball_tracker.detect_frame(frame)
            ball_bbox, ball_proj = self.ball_tracker.process_and_project(ball_det, frame, Hmg)

            # ── Video Analysis ─────────────────────────────────────────
            if self.mode == MODE_VIDEO_ANALYSIS:
                self._update_analytics_geometry(kps, Hmg, work_w, work_h)
                self.analytics.update_counters(frame_idx, proj_players, ball_proj)
                main_col = self._render_main_view(frame, players, ball_bbox, kps, (main_w, out_h))
                bird_col = self._render_birdseye(work_w, work_h, kps, Hmg, proj_players, ball_proj, (be_w, out_h))
                grid_col = self._render_analytics_grid((panel_w, panel_h), (grid_w, out_h), bird_reference=bird_col)
                writer.write(cv2.hconcat([main_col, bird_col, grid_col]))

            # ── Split Rallies ──────────────────────────────────────────
            elif self.mode == MODE_SPLIT_RALLIES:
                self._update_analytics_geometry(kps, Hmg, work_w, work_h)
                self.analytics.update_counters(frame_idx, proj_players, ball_proj)
                # No rendering — detection only

            # ── Detect Serve ───────────────────────────────────────────
            elif self.mode == MODE_DETECT_SERVE:
                candidate = self.serve_detector.update(frame_idx, frame, ball_proj, players)
                if candidate is not None:
                    logger.bind(frame_idx=frame_idx, player_id=candidate.player_id).info("serve_candidate_detected")
                    self.serve_analyzer.submit(candidate)

            frame_idx += 1
            if frame_idx % 500 == 0 and total_frames > 0:
                logger.bind(
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    progress_pct=round(frame_idx / total_frames * 100, 1),
                ).debug("frame_milestone")
            self._report_progress(progress_callback, frame_idx, total_frames)

    finally:
        cap.release()

        # ── Video Analysis finalization ────────────────────────────────
        if self.mode == MODE_VIDEO_ANALYSIS:
            if writer:
                writer.release()
            if self.analytics._rally_active:
                self.analytics._finalize_current_rally(frame_idx=frame_idx)
            self.analytics.save_outputs()

        # ── Split Rallies finalization ─────────────────────────────────
        elif self.mode == MODE_SPLIT_RALLIES:
            if self.analytics._rally_active:
                self.analytics._finalize_current_rally(frame_idx=frame_idx)
            self.analytics.save_outputs()

        # ── Detect Serve finalization ──────────────────────────────────
        elif self.mode == MODE_DETECT_SERVE:
            self.serve_analyzer.shutdown()
            self._save_serve_report()

        rally_count = len(self.analytics._long_rallies) if hasattr(self, "analytics") else 0
        elapsed_s = round(time.time() - self._start_time, 2)
        logger.bind(total_frames=total_frames, rally_count=rally_count, elapsed_s=elapsed_s).info("run_completed")
        self._report_progress(progress_callback, total_frames, total_frames)

    if self.mode == MODE_SPLIT_RALLIES:
        self._clip_long_rallies(total_frames, fps)

    if out_path:
        print(f"Saved: {out_path}")
    return out_path
```

- [ ] **Step 2: Fix `_render_analytics_grid` — guard `serve_analyzer` access**

`_render_analytics_grid` is only called in `MODE_VIDEO_ANALYSIS` after this refactor,
but `serve_analyzer` doesn't exist on the object in that mode. Replace line 423:

```python
# Before
sv = self.analytics.panel_serve_summary(self.serve_analyzer.get_results(), (panel_w, panel_h))

# After
serve_results = self.serve_analyzer.get_results() if hasattr(self, "serve_analyzer") else []
sv = self.analytics.panel_serve_summary(serve_results, (panel_w, panel_h))
```

- [ ] **Step 3: Run the full existing test suite to confirm nothing broke**

```
.venv/Scripts/pytest tests/ -v
```

Expected: all tests pass (test_serve_detector.py, test_serve_analyzer.py,
test_analytics_serve_panel.py, test_video_processor_modes.py)

- [ ] **Step 4: Commit**

```bash
git add process_video.py
git commit -m "Restructure process_video() as explicit mode dispatcher"
```

---

## Task 4: Update `main.py` — mode selector and description label

**Files:**
- Modify: `main.py:54` (mode_var default)
- Modify: `main.py:157-193` (`_build_mode_card`)

- [ ] **Step 1: Update the mode_var default value and add the import**

At the top of `main.py`, replace:
```python
from process_video import VideoProcessor
```
with:
```python
from process_video import (
    VideoProcessor,
    MODE_VIDEO_ANALYSIS,
    MODE_SPLIT_RALLIES,
    MODE_DETECT_SERVE,
)
```

Then on line 54, replace:
```python
self.mode_var = ctk.StringVar(value="Full Analytics")
```
with:
```python
self.mode_var = ctk.StringVar(value="Video Analysis")
```

- [ ] **Step 2: Rewrite `_build_mode_card` to use new labels and add description label**

Replace the entire `_build_mode_card` method with:

```python
# Mode descriptions shown below the selector
MODE_DESCRIPTIONS = {
    "Video Analysis": "Produces an annotated video with overlays and analytics",
    "Split Rallies":  "Finds long rallies and saves each one as a clip",
    "Detect Serve":   "Scores serves using AI vision — fastest mode",
}

def _build_mode_card(self, parent):
    card = self._make_card(parent, 0, 1)
    card.grid_columnconfigure(0, weight=1)
    card.grid_rowconfigure(0, weight=0)
    card.grid_rowconfigure(1, weight=0)
    card.grid_rowconfigure(2, weight=0)
    card.grid_rowconfigure(3, weight=1)
    card.grid_rowconfigure(4, weight=0)

    ctk.CTkLabel(
        card,
        text="Processing Mode",
        font=ctk.CTkFont(size=14, weight="bold"),
        text_color=BODY_TEXT,
    ).grid(row=0, column=0, padx=16, pady=(16, 6), sticky="w")

    self.mode_selector = ctk.CTkSegmentedButton(
        card,
        values=["Video Analysis", "Split Rallies", "Detect Serve"],
        variable=self.mode_var,
        state="disabled",
        selected_color=CTA,
        selected_hover_color="#00B8D9",
        unselected_color=CARD_BORDER,
        font=ctk.CTkFont(size=12),
        command=self._on_mode_changed,
    )
    self.mode_selector.grid(row=1, column=0, padx=16, pady=8, sticky="ew")

    self.mode_desc_label = ctk.CTkLabel(
        card,
        text=MODE_DESCRIPTIONS["Video Analysis"],
        font=ctk.CTkFont(size=11),
        text_color=MUTED_TEXT,
        wraplength=280,
        justify="left",
    )
    self.mode_desc_label.grid(row=2, column=0, padx=16, pady=(0, 8), sticky="w")

    self.process_btn = ctk.CTkButton(
        card,
        text="Process Video",
        fg_color=CTA,
        text_color="black",
        font=ctk.CTkFont(size=13, weight="bold"),
        corner_radius=8,
        state="disabled",
        command=self.process_video_thread,
    )
    self.process_btn.grid(row=4, column=0, padx=16, pady=(0, 16), sticky="ew")
```

Note: `MODE_DESCRIPTIONS` is a module-level dict — place it just above the `App` class
definition (before line 44).

- [ ] **Step 3: Add `_on_mode_changed` callback to the `App` class**

Add this method to the `App` class (after `_set_state_idle`, around line 305):

```python
def _on_mode_changed(self, value: str) -> None:
    """Update the description label when the user picks a different mode."""
    self.mode_desc_label.configure(text=MODE_DESCRIPTIONS.get(value, ""))
```

- [ ] **Step 4: Run the app to verify selector and description work**

```
cd c:/apps/pickleball/Pickleball-Analytics
.venv/Scripts/python main.py
```

Expected:
- Selector shows "Video Analysis | Split Rallies | Detect Serve"
- Default selection is "Video Analysis"
- Description label reads "Produces an annotated video with overlays and analytics"
- Clicking "Split Rallies" changes description to "Finds long rallies and saves each one as a clip"
- Clicking "Detect Serve" changes description to "Scores serves using AI vision — fastest mode"

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "Update mode selector: new labels, description label, _on_mode_changed"
```

---

## Task 5: Update `main.py` — mode_map, results card, remove badges card

**Files:**
- Modify: `main.py:392-422` (`_process_video` and `_set_state_complete`)
- Modify: `main.py:104-116` (`_build_card_area` — remove badges card call, fix layout)
- Delete method: `_build_badges_card` (~lines 196-225)
- Modify: `_build_results_card` — span both columns in row 1

- [ ] **Step 1: Update `_process_video` to use mode constants**

Replace the `mode_map` block inside `_process_video` (currently lines 404–409):

```python
# Before
mode_map = {
    "Full Analytics": "full",
    "Rallies Only": "rallies_only",
    "Full + Rallies": "full_and_rallies",
}
mode = mode_map[mode_label]
processor = VideoProcessor(self.video_path, filters, mode=mode)

# After
mode_map = {
    "Video Analysis": MODE_VIDEO_ANALYSIS,
    "Split Rallies":  MODE_SPLIT_RALLIES,
    "Detect Serve":   MODE_DETECT_SERVE,
}
mode = mode_map[mode_label]
processor = VideoProcessor(self.video_path, filters, mode=mode)
```

Also update the result extraction block (currently lines 413–414) to handle modes that
don't have analytics or serve_analyzer:

```python
# Before
rally_count = len(processor.analytics._long_rallies)
out_dir = processor.output_dir

# After
rally_count = len(processor.analytics._long_rallies) if hasattr(processor, "analytics") else 0
serve_count = len(processor.serve_analyzer.get_results()) if hasattr(processor, "serve_analyzer") else 0
serve_avg   = (
    round(
        sum(r.score for r in processor.serve_analyzer.get_results())
        / serve_count, 1
    )
    if serve_count > 0 else 0
)
out_dir = processor.output_dir
```

Update the `self.after(...)` call to pass the new data to `_set_state_complete`:

```python
# Before
self.after(
    0, lambda: self._set_state_complete(rally_count, out_dir, mode)
)

# After
self.after(
    0,
    lambda rc=rally_count, sc=serve_count, sa=serve_avg, od=out_dir, m=mode:
        self._set_state_complete(rc, sc, sa, od, m),
)
```

- [ ] **Step 2: Update `_set_state_complete` to show per-mode results**

Replace the entire `_set_state_complete` method:

```python
def _set_state_complete(self, rally_count, serve_count, serve_avg, out_dir, mode):
    self.browse_btn.configure(state="normal")
    self.mode_selector.configure(state="normal")
    self.process_btn.configure(state="normal")
    self.progress_bar.set(1.0)
    self.status_label.configure(text="Done!", text_color=SUCCESS)

    if mode == MODE_VIDEO_ANALYSIS:
        badge_text = "Output saved"
    elif mode == MODE_SPLIT_RALLIES:
        badge_text = (
            f"{rally_count} long {'rally' if rally_count == 1 else 'rallies'} found"
            if rally_count > 0
            else "No long rallies detected"
        )
    elif mode == MODE_DETECT_SERVE:
        badge_text = (
            f"{serve_count} {'serve' if serve_count == 1 else 'serves'} detected"
            f" · avg score {serve_avg}/10"
            if serve_count > 0
            else "No serves detected"
        )
    else:
        badge_text = "Processing complete"

    self.rally_badge_label.configure(text=badge_text)
    self.output_path_label.configure(text=out_dir)
    self.out_dir = out_dir
    self._reveal_results()
```

- [ ] **Step 3: Remove the badges card and fix the layout**

**3a.** In `_build_card_area`, replace the entire method body:

```python
# Before
def _build_card_area(self):
    area = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
    area.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 10))
    area.grid_columnconfigure(0, weight=1)
    area.grid_columnconfigure(1, weight=1)
    area.grid_rowconfigure(0, weight=1)
    area.grid_rowconfigure(1, weight=1)

    self._build_file_card(area)
    self._build_mode_card(area)
    self._build_badges_card(area)
    self._build_results_card(area)

# After
def _build_card_area(self):
    area = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
    area.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 10))
    area.grid_columnconfigure(0, weight=1)
    area.grid_columnconfigure(1, weight=1)
    area.grid_rowconfigure(0, weight=1)
    area.grid_rowconfigure(1, weight=1)

    self._build_file_card(area)
    self._build_mode_card(area)
    self._build_results_card(area)
```

**3b.** Delete the entire `_build_badges_card` method (approximately lines 196–225).

**3c.** In `_build_results_card`, change the `_make_card` call so results spans both
columns in row 1:

```python
# Before
self.results_card = self._make_card(parent, 1, 1, height=0)

# After — span both columns so it fills the full bottom row
self.results_card = self._make_card(parent, 1, 0, height=0)
self.results_card.grid(
    row=1, column=0, columnspan=2, padx=8, pady=8, sticky="nsew"
)
```

Note: `_make_card` calls `card.grid(...)` internally, so the second `.grid()` call
above overrides the placement with `columnspan=2`.

- [ ] **Step 4: Run the app end-to-end with each mode**

```
.venv/Scripts/python main.py
```

Verify:
1. Select `Sample3.mp4`
2. **Split Rallies** → process → results badge shows "N rallies found" or "No long rallies detected"
3. Select video again → **Detect Serve** → process → results badge shows "N serves detected · avg score X/10"
4. Select video again → **Video Analysis** → process → results badge shows "Output saved"

- [ ] **Step 5: Run full test suite**

```
.venv/Scripts/pytest tests/ -v
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "Per-mode results card, mode_map uses constants, remove badges card"
```

---

## Self-Review

**Spec coverage:**
- ✅ 3 mode constants added (Task 1)
- ✅ Conditional `__init__` (Task 2)
- ✅ Dispatcher structure (Task 3)
- ✅ Mode selector new labels + description label (Task 4)
- ✅ mode_map uses imported constants (Task 5)
- ✅ Per-mode results card (Task 5)
- ✅ Badges card removed — inaccurate across modes (Task 5)
- ✅ Results card spans full bottom row after badges removal (Task 5)

**Placeholder scan:** None found — all steps have complete code.

**Type consistency:**
- `_set_state_complete` signature changes from `(rally_count, out_dir, mode)` to `(rally_count, serve_count, serve_avg, out_dir, mode)` — verified the lambda in Task 5 Step 1 and the method definition in Task 5 Step 2 match exactly.
- `MODE_DESCRIPTIONS` dict defined at module level before `App` class — referenced in `_build_mode_card` and `_on_mode_changed`, both inside `App`. Consistent.
