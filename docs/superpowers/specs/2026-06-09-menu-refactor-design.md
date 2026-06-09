# Menu Refactor — Design Spec

> **Scope:** `process_video.py` and `main.py` only. Flat file structure is preserved.
> Full migration to `src/` layout is a separate future step.

---

## Goal

Replace the confusing three-mode menu ("Full Analytics", "Rallies Only", "Full + Rallies")
with three clearly-named, mutually-exclusive modes. Each mode produces exactly one type
of output and the menu label makes that obvious.

---

## Three Modes

| UI Label | Internal Constant | Produces |
|---|---|---|
| Video Analysis | `MODE_VIDEO_ANALYSIS` | `Main_overlay.mp4` — annotated video with player/ball/court overlays, bird's-eye, and analytics panels |
| Split Rallies | `MODE_SPLIT_RALLIES` | `rally_01.mp4`, `rally_02.mp4`, … — one raw clip per long rally (5+ net crossings) |
| Detect Serve | `MODE_DETECT_SERVE` | `serve_report.json` — serve candidates scored by Ollama vision. Fastest mode, no video output |

Modes are mutually exclusive. Pick one per run.

---

## `process_video.py` Changes

### 1. Mode constants (top of file, replacing old string literals)

```python
# ── Processing Modes ──────────────────────────────────────────────────────────
# Pick exactly one per run. Each mode runs court/player/ball detection,
# then diverges based on what output it produces.

MODE_VIDEO_ANALYSIS = "video_analysis"
# Produces: Main_overlay.mp4 — annotated video with overlays and analytics.

MODE_SPLIT_RALLIES = "split_rallies"
# Produces: rally_01.mp4, rally_02.mp4, … — one raw clip per long rally.

MODE_DETECT_SERVE = "detect_serve"
# Produces: serve_report.json — serve candidates scored by Ollama vision.
```

### 2. `VideoProcessor.__init__` — conditional construction

Only instantiate what each mode needs:

```python
# All modes need these:
self.ball_tracker   = BallTracker(...)
self.player_tracker = PlayerTracker(...)
self.court_mapper   = CourtDetector(...)

# Rally counting needed by Video Analysis and Split Rallies:
if self.mode in (MODE_VIDEO_ANALYSIS, MODE_SPLIT_RALLIES):
    self.analytics = Analytics(self.filters)

# Serve detection needed by Detect Serve only:
if self.mode == MODE_DETECT_SERVE:
    self.serve_detector = ServeDetector()
    self.serve_analyzer = OllamaServeAnalyzer(model="qwen2.5vl:7b", workers=2)
```

### 3. `process_video()` — dispatcher structure

```python
def process_video(self, progress_callback=None):
    # Setup (all modes)
    ...

    try:
        while True:
            frame = read_next_frame()

            # Common detections — all modes
            kps, Hmg  = self._detect_court(frame_idx)
            players, proj_players = self.player_tracker.detect_and_project(frame, Hmg)
            ball_bbox, ball_proj  = self._detect_ball(frame, Hmg)

            # ── Video Analysis ──────────────────────────────────────────
            if self.mode == MODE_VIDEO_ANALYSIS:
                self._update_analytics_geometry(kps, Hmg, work_w, work_h)
                self.analytics.update_counters(frame_idx, proj_players, ball_proj)
                main_col = self._render_main_view(frame, players, ball_bbox, kps, ...)
                bird_col = self._render_birdseye(work_w, work_h, kps, Hmg, proj_players, ball_proj, ...)
                grid_col = self._render_analytics_grid(...)
                writer.write(cv2.hconcat([main_col, bird_col, grid_col]))

            # ── Split Rallies ───────────────────────────────────────────
            elif self.mode == MODE_SPLIT_RALLIES:
                self._update_analytics_geometry(kps, Hmg, work_w, work_h)
                self.analytics.update_counters(frame_idx, proj_players, ball_proj)
                # No rendering — detection only

            # ── Detect Serve ────────────────────────────────────────────
            elif self.mode == MODE_DETECT_SERVE:
                candidate = self.serve_detector.update(
                    frame_idx, frame, ball_proj, players
                )
                if candidate is not None:
                    self.serve_analyzer.submit(candidate)

    finally:
        cap.release()

        # ── Video Analysis finalization ─────────────────────────────────
        if self.mode == MODE_VIDEO_ANALYSIS:
            writer.release()

        # ── Split Rallies finalization ──────────────────────────────────
        elif self.mode == MODE_SPLIT_RALLIES:
            if self.analytics._rally_active:
                self.analytics._finalize_current_rally(frame_idx)
            self._clip_long_rallies(total_frames, fps)

        # ── Detect Serve finalization ───────────────────────────────────
        elif self.mode == MODE_DETECT_SERVE:
            self.serve_analyzer.shutdown()
            self._save_serve_report()
```

---

## `main.py` Changes

### 1. Mode selector values

```python
# Before
values=["Full Analytics", "Rallies Only", "Full + Rallies"]

# After
values=["Video Analysis", "Split Rallies", "Detect Serve"]
```

Default value: `"Video Analysis"`

### 2. Mode description label

Add a `CTkLabel` below the segmented button in the Mode card.
Updates whenever the user changes the selection:

| Selection | Description |
|---|---|
| Video Analysis | Produces an annotated video with overlays and analytics |
| Split Rallies | Finds long rallies and saves each one as a clip |
| Detect Serve | Scores serves using AI vision — fastest mode |

### 3. mode_map in `_process_video()`

```python
mode_map = {
    "Video Analysis": MODE_VIDEO_ANALYSIS,
    "Split Rallies":  MODE_SPLIT_RALLIES,
    "Detect Serve":   MODE_DETECT_SERVE,
}
```

Import the constants from `process_video`:
```python
from process_video import VideoProcessor, MODE_VIDEO_ANALYSIS, MODE_SPLIT_RALLIES, MODE_DETECT_SERVE
```

### 4. Results card — per-mode display

Update `_set_state_complete()` to show output relevant to the completed mode:

| Mode | Badge text |
|---|---|
| Video Analysis | "Output saved" |
| Split Rallies | "N rallies found" or "No long rallies detected" |
| Detect Serve | "N serves detected · avg score X/10" or "No serves detected" |

All modes show the output path and "Open Folder" button.

### 5. Analytics badges card — rename

Change the card title from `"Analytics (Always Included)"` to
`"Detection Capabilities"` — accurate for all three modes.

---

## Files Changed

| File | Change |
|---|---|
| `process_video.py` | Add 3 mode constants; refactor `__init__` for conditional construction; restructure `process_video()` as explicit dispatcher |
| `main.py` | Update mode selector values and default; add description label; update mode_map import; update results card logic; rename badges card title |

## Files NOT Changed

`analytics.py`, `ball_tracker.py`, `player_tracker.py`, `court_detection.py`,
`serve_detector.py`, `serve_analyzer.py` — no modifications needed.

---

## Testing

After implementation, verify each mode end-to-end with `Sample3.mp4`:

1. **Video Analysis** → `video_outputs/run_*/Main_overlay.mp4` exists and opens
2. **Split Rallies** → `video_outputs/run_*/rally_01.mp4` exists (or log shows 0 rallies)
3. **Detect Serve** → `video_outputs/run_*/serve_report.json` exists with serve entries
