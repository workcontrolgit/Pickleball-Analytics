# Rally Clip Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect rallies with 5+ net crossings and export each as a raw MP4 clip, with a UI mode selector to choose between full analytics, rallies only, or both.

**Architecture:** During the main processing loop, `Analytics.update_counters()` detects net crossings and records qualifying rally frame ranges in `_long_rallies`. After the loop, `VideoProcessor._clip_long_rallies()` seeks the source video and writes one clip per qualifying rally. A `CTkSegmentedButton` in `main.py` controls which mode runs.

**Tech Stack:** Python 3.12, OpenCV (`cv2`), NumPy, customtkinter, pytest

---

## File Map

| File | Change |
|------|--------|
| `analytics.py` | Add net crossing state + detection logic in `update_counters()`; update `_net_y` in `update_kitchen_from_keypoints()` |
| `process_video.py` | Accept `mode` param; extract analytics geometry updates to `_update_analytics_geometry()`; add `_clip_long_rallies()`; conditionally skip rendering in `rallies_only` mode |
| `main.py` | Add `CTkSegmentedButton` for mode selection; pass mode to `VideoProcessor` |
| `tests/test_analytics_crossings.py` | New — unit tests for net crossing detection |
| `tests/test_clip_long_rallies.py` | New — unit tests for clip logic |

---

## Task 1: Add net crossing state to `Analytics`

**Files:**
- Modify: `analytics.py` (`__init__` and `update_kitchen_from_keypoints`)
- Test: `tests/test_analytics_crossings.py`

- [ ] **Step 1: Create test file with failing tests**

Create `tests/test_analytics_crossings.py`:

```python
import pytest
from analytics import Analytics


def make_analytics():
    a = Analytics({"player_heatmap": True, "ball_heatmap": True})
    a.set_canvas_size(400, 900)
    a.set_video_context(total_frames=900, fps=30)
    # Manually set kitchen bounds so net_y is defined
    a._kitchen_y_min = 300.0
    a._kitchen_y_max = 600.0
    a._net_y = 450.0
    a._court_bounds = (0.0, 0.0, 400.0, 900.0)
    return a


def test_initial_crossing_state():
    a = Analytics({"player_heatmap": True, "ball_heatmap": True})
    assert a._net_y is None
    assert a._ball_last_side is None
    assert a._rally_net_crossings == 0
    assert a._rally_start_frame is None
    assert a._long_rallies == []


def test_net_y_set_from_kitchen_bounds():
    a = Analytics({"player_heatmap": True, "ball_heatmap": True})
    a.set_canvas_size(400, 900)
    a._kitchen_y_min = 300.0
    a._kitchen_y_max = 600.0
    # Simulate kitchen update by calling internal update
    import numpy as np
    # Construct 12 keypoints with rows at y=100, y=300, y=600, y=800
    # Row 0: y=100, Row 1: y=300, Row 2: y=600, Row 3: y=800
    kpts = np.array([
        [50, 100], [200, 100], [350, 100],
        [50, 300], [200, 300], [350, 300],
        [50, 600], [200, 600], [350, 600],
        [50, 800], [200, 800], [350, 800],
    ], dtype=np.float32)
    a.update_kitchen_from_keypoints(kpts)
    assert a._net_y is not None
    assert abs(a._net_y - (a._kitchen_y_min + a._kitchen_y_max) / 2.0) < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd c:/apps/pickleball/Pickleball-Analytics
python -m pytest tests/test_analytics_crossings.py::test_initial_crossing_state tests/test_analytics_crossings.py::test_net_y_set_from_kitchen_bounds -v
```

Expected: `AttributeError: 'Analytics' object has no attribute '_net_y'`

- [ ] **Step 3: Add net crossing state to `Analytics.__init__`**

In `analytics.py`, add these fields after the `_rallies` line (around line 108):

```python
        # --- Net crossing / rally clip tracking ---
        self._net_y = None                   # net midpoint in bird coords (set from kitchen bounds)
        self._ball_last_side = None          # 'above' | 'below' | None
        self._rally_net_crossings = 0        # crossings in current rally
        self._rally_start_frame = None       # frame index when current rally began
        self._long_rallies = []              # list of (start_frame, end_frame, crossings)
```

- [ ] **Step 4: Update `_net_y` inside `update_kitchen_from_keypoints`**

At the end of `update_kitchen_from_keypoints()` in `analytics.py`, after the EMA update block, add:

```python
        self._net_y = (self._kitchen_y_min + self._kitchen_y_max) / 2.0
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_analytics_crossings.py::test_initial_crossing_state tests/test_analytics_crossings.py::test_net_y_set_from_kitchen_bounds -v
```

Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add analytics.py tests/test_analytics_crossings.py
git commit -m "Add net crossing state fields to Analytics"
```

---

## Task 2: Implement net crossing detection in `update_counters`

**Files:**
- Modify: `analytics.py` (`update_counters`)
- Test: `tests/test_analytics_crossings.py`

- [ ] **Step 1: Add failing tests for crossing detection**

Append to `tests/test_analytics_crossings.py`:

```python
def test_no_crossing_ball_stays_above():
    a = make_analytics()
    for i in range(10):
        a.update_counters(i, [], (200.0, 100.0))  # y=100 < net_y=450 → always above
    assert a._rally_net_crossings == 0


def test_single_crossing_detected():
    a = make_analytics()
    a.update_counters(0, [], (200.0, 100.0))  # above
    a.update_counters(1, [], (200.0, 800.0))  # below → 1 crossing
    assert a._rally_net_crossings == 1


def test_five_crossings_recorded_as_long_rally():
    a = make_analytics()
    # Alternate sides to produce 5 crossings
    sides = [100.0, 800.0, 100.0, 800.0, 100.0, 800.0]  # 5 crossings
    for i, y in enumerate(sides):
        a.update_counters(i, [], (200.0, y))
    # End rally with gap
    for i in range(len(sides), len(sides) + 20):
        a.update_counters(i, [], None)
    assert len(a._long_rallies) == 1
    start, end, crossings = a._long_rallies[0]
    assert crossings == 5
    assert start == 0


def test_four_crossings_not_recorded():
    a = make_analytics()
    sides = [100.0, 800.0, 100.0, 800.0, 100.0]  # 4 crossings
    for i, y in enumerate(sides):
        a.update_counters(i, [], (200.0, y))
    for i in range(len(sides), len(sides) + 20):
        a.update_counters(i, [], None)
    assert len(a._long_rallies) == 0


def test_missing_ball_detection_holds_last_side():
    a = make_analytics()
    a.update_counters(0, [], (200.0, 100.0))   # above
    a.update_counters(1, [], None)              # no ball — side held
    a.update_counters(2, [], (200.0, 100.0))   # above again — no crossing
    assert a._rally_net_crossings == 0


def test_rally_start_frame_recorded():
    a = make_analytics()
    a.update_counters(5, [], (200.0, 100.0))   # rally starts at frame 5
    assert a._rally_start_frame == 5


def test_video_ends_mid_rally_with_5_crossings():
    a = make_analytics()
    sides = [100.0, 800.0, 100.0, 800.0, 100.0, 800.0]  # 5 crossings
    for i, y in enumerate(sides):
        a.update_counters(i, [], (200.0, y))
    # No gap — rally still active; force finalize
    a._finalize_current_rally(frame_idx=len(sides))
    assert len(a._long_rallies) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_analytics_crossings.py -v -k "crossing or long_rally or missing_ball or start_frame or mid_rally"
```

Expected: multiple failures — logic not yet implemented.

- [ ] **Step 3: Add `_finalize_current_rally` helper to `Analytics`**

Add this method to `analytics.py` (after `update_counters`):

```python
    def _finalize_current_rally(self, frame_idx: int) -> None:
        """Finalize the current rally: record if it qualifies, then reset state."""
        if self._rally_frames > 0:
            self._rallies.append(self._rally_frames)
            if self._rally_net_crossings >= 5 and self._rally_start_frame is not None:
                end_frame = frame_idx
                self._long_rallies.append(
                    (self._rally_start_frame, end_frame, self._rally_net_crossings)
                )
        self._rally_active = False
        self._rally_frames = 0
        self._gap_frames = 0
        self._rally_start_frame = None
        self._rally_net_crossings = 0
        self._ball_last_side = None
```

- [ ] **Step 4: Update `update_counters` to detect crossings and use `_finalize_current_rally`**

Replace the `--- Rally state update ---` block in `update_counters` (from `self._elapsed_frames += 1` to the end of the method) with:

```python
        # --- Rally state update ---
        self._elapsed_frames += 1

        in_play = self._ball_in_bounds(ball_proj)
        if in_play:
            # Set start frame on first active frame of this rally
            if not self._rally_active:
                self._rally_start_frame = frame_idx
                self._rally_net_crossings = 0
                self._ball_last_side = None

            self._rally_active = True
            self._rally_frames += 1
            self._gap_frames = 0

            # Net crossing detection
            if self._net_y is not None and ball_proj is not None:
                current_side = 'above' if ball_proj[1] < self._net_y else 'below'
                if self._ball_last_side is not None and current_side != self._ball_last_side:
                    self._rally_net_crossings += 1
                self._ball_last_side = current_side
        else:
            if self._rally_active:
                self._gap_frames += 1
                if self._gap_frames >= self._gap_threshold:
                    self._finalize_current_rally(frame_idx=frame_idx - self._gap_frames)
```

- [ ] **Step 5: Run all crossing tests**

```bash
python -m pytest tests/test_analytics_crossings.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add analytics.py tests/test_analytics_crossings.py
git commit -m "Implement net crossing detection and long rally recording in Analytics"
```

---

## Task 3: Extract analytics geometry updates from `_render_birdseye`

**Files:**
- Modify: `process_video.py` (`_render_birdseye` and new `_update_analytics_geometry`)

The three analytics update calls (`update_kitchen_from_keypoints`, `update_court_bounds_from_keypoints`, `update_zones_from_keypoints`) currently live inside `_render_birdseye`. They must run in all modes, including `rallies_only` where rendering is skipped.

- [ ] **Step 1: Add `_update_analytics_geometry` method to `VideoProcessor`**

Add this method to `process_video.py` after `_render_birdseye`:

```python
    def _update_analytics_geometry(
        self,
        keypoints: Optional[np.ndarray],
        Hmg: Optional[np.ndarray],
        src_w: int,
        src_h: int,
    ) -> None:
        """Project keypoints and update analytics zone geometry. Called every frame regardless of mode."""
        if keypoints is None or Hmg is None:
            return
        pts = np.array(keypoints, dtype=np.float32).reshape(-1, 1, 2)
        proj_kps = cv2.perspectiveTransform(pts, Hmg).reshape(-1, 2)
        self.analytics.update_kitchen_from_keypoints(proj_kps)
        self.analytics.update_court_bounds_from_keypoints(proj_kps)
        self.analytics.update_zones_from_keypoints(proj_kps)
```

- [ ] **Step 2: Remove the three analytics update calls from `_render_birdseye`**

In `_render_birdseye`, delete these three lines (inside the `if keypoints is not None and Hmg is not None:` block):

```python
            # Teach analytics dynamic zones based on projected keypoints
            self.analytics.update_kitchen_from_keypoints(proj_kps)
            self.analytics.update_court_bounds_from_keypoints(proj_kps)
            self.analytics.update_zones_from_keypoints(proj_kps)
```

- [ ] **Step 3: Call `_update_analytics_geometry` in the main loop**

In `process_video()`, after `ball_bbox, ball_proj = ...` and before `self.analytics.update_counters(...)`, add:

```python
                self._update_analytics_geometry(kps, Hmg, src_w, src_h)
```

- [ ] **Step 4: Verify existing behavior is unchanged**

Run a quick smoke test to ensure the import chain is intact:

```bash
python -c "from process_video import VideoProcessor; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add process_video.py
git commit -m "Extract analytics geometry updates out of _render_birdseye into own method"
```

---

## Task 4: Add `mode` parameter and `_clip_long_rallies` to `VideoProcessor`

**Files:**
- Modify: `process_video.py`
- Test: `tests/test_clip_long_rallies.py`

- [ ] **Step 1: Write failing tests for `_clip_long_rallies`**

Create `tests/test_clip_long_rallies.py`:

```python
import os
import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
from process_video import VideoProcessor


def make_processor(tmp_path, mode='full'):
    filters = {"player_heatmap": True, "ball_heatmap": True,
                "kitchen_detection": True, "court_zone": True}
    with patch.object(VideoProcessor, '__init__', lambda self, vp, f, m: None):
        p = VideoProcessor.__new__(VideoProcessor)
    p.video_path = str(tmp_path / "fake.mp4")
    p.filters = filters
    p.mode = mode
    p.output_dir = str(tmp_path)
    p.analytics = MagicMock()
    p.analytics._long_rallies = []
    return p


def make_fake_video(path, num_frames=60, fps=30, w=64, h=64):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for _ in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_no_clips_when_no_long_rallies(tmp_path):
    p = make_processor(tmp_path)
    p.analytics._long_rallies = []
    p._clip_long_rallies(total_frames=60, fps=30)
    clips = list(tmp_path.glob("rally_*.mp4"))
    assert len(clips) == 0


def test_one_clip_written_for_one_long_rally(tmp_path):
    video_path = tmp_path / "fake.mp4"
    make_fake_video(video_path, num_frames=120, fps=30)
    p = make_processor(tmp_path)
    p.video_path = str(video_path)
    p.analytics._long_rallies = [(10, 50, 6)]
    p._clip_long_rallies(total_frames=120, fps=30)
    clips = list(tmp_path.glob("rally_*.mp4"))
    assert len(clips) == 1
    assert (tmp_path / "rally_01.mp4").exists()


def test_two_clips_written_for_two_long_rallies(tmp_path):
    video_path = tmp_path / "fake.mp4"
    make_fake_video(video_path, num_frames=300, fps=30)
    p = make_processor(tmp_path)
    p.video_path = str(video_path)
    p.analytics._long_rallies = [(10, 40, 6), (100, 150, 8)]
    p._clip_long_rallies(total_frames=300, fps=30)
    assert (tmp_path / "rally_01.mp4").exists()
    assert (tmp_path / "rally_02.mp4").exists()


def test_buffer_clamped_to_zero(tmp_path):
    video_path = tmp_path / "fake.mp4"
    make_fake_video(video_path, num_frames=60, fps=30)
    p = make_processor(tmp_path)
    p.video_path = str(video_path)
    # Rally starts at frame 5 — buffer of 60 frames would go negative
    p.analytics._long_rallies = [(5, 30, 6)]
    # Should not raise
    p._clip_long_rallies(total_frames=60, fps=30)
    assert (tmp_path / "rally_01.mp4").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_clip_long_rallies.py -v
```

Expected: `AttributeError` — `_clip_long_rallies` not yet defined.

- [ ] **Step 3: Add `mode` param to `VideoProcessor.__init__`**

In `process_video.py`, update `__init__`:

```python
    def __init__(self, video_path: str, filters: dict, mode: str = 'full'):
        self.video_path = video_path
        self.filters = self._apply_default_filters(filters)
        self.mode = mode

        self.ball_tracker = BallTracker(os.path.join(MODELS_DIR, "ball_tracking.pt"))
        self.player_tracker = PlayerTracker(os.path.join(MODELS_DIR, "player_tracking.pt"))
        self.court_mapper = CourtDetector(os.path.join(MODELS_DIR, "court_detection.pt"))
        self.analytics = Analytics(self.filters)

        self.output_dir = self._make_output_dir()
```

- [ ] **Step 4: Add `_clip_long_rallies` method to `VideoProcessor`**

Add after `_create_writer` in `process_video.py`:

```python
    def _clip_long_rallies(self, total_frames: int, fps: int) -> None:
        """Re-open source video and write a raw clip for each qualifying long rally."""
        if not self.analytics._long_rallies:
            return

        buffer = fps * 2
        cap = self._open_capture(self.video_path)

        for idx, (start_frame, end_frame, crossings) in enumerate(self.analytics._long_rallies, start=1):
            clip_start = max(0, start_frame - buffer)
            clip_end = min(total_frames - 1, end_frame + buffer)

            cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]
            clip_path = os.path.join(self.output_dir, f"rally_{idx:02d}.mp4")
            writer = cv2.VideoWriter(clip_path, FOURCC, fps, (w, h))
            writer.write(frame)

            for _ in range(clip_end - clip_start):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)

            writer.release()
            print(f"Saved rally clip: {clip_path} ({crossings} net crossings)")

        cap.release()
```

- [ ] **Step 5: Run clip tests**

```bash
python -m pytest tests/test_clip_long_rallies.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add process_video.py tests/test_clip_long_rallies.py
git commit -m "Add mode param and _clip_long_rallies to VideoProcessor"
```

---

## Task 5: Wire mode into `process_video()` main loop

**Files:**
- Modify: `process_video.py` (`process_video` method)

- [ ] **Step 1: Update `process_video()` to respect `self.mode`**

Replace the entire `process_video` method body with:

```python
    def process_video(self, progress_callback=None) -> str:
        cap = self._open_capture(self.video_path)
        total_frames, src_w, src_h, fps = self._read_video_meta(cap)
        out_w, out_h, main_w, be_w, grid_w, panel_w, panel_h = self._compute_layout(src_w, src_h)

        self.analytics.set_canvas_size(src_w, src_h)
        self.analytics.set_video_context(total_frames=total_frames, fps=fps)

        writer = None
        out_path = None
        if self.mode != 'rallies_only':
            out_path, writer = self._create_writer(out_w, out_h, fps)

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                kps, Hmg = self.court_mapper.get_keypoints_and_homography(frame)
                players, proj_players = self.player_tracker.detect_and_project(frame, Hmg)
                ball_det = self.ball_tracker.detect_frame(frame)
                ball_bbox, ball_proj = self.ball_tracker.process_and_project(ball_det, frame, Hmg)

                self._update_analytics_geometry(kps, Hmg, src_w, src_h)
                self.analytics.update_counters(frame_idx, proj_players, ball_proj)

                if self.mode != 'rallies_only':
                    main_col = self._render_main_view(frame, players, ball_bbox, kps, (main_w, out_h))
                    bird_col = self._render_birdseye(src_w, src_h, kps, Hmg, proj_players, ball_proj, (be_w, out_h))
                    grid_col = self._render_analytics_grid((panel_w, panel_h), (grid_w, out_h), bird_reference=bird_col)
                    composite = cv2.hconcat([main_col, bird_col, grid_col])
                    writer.write(composite)

                frame_idx += 1
                self._report_progress(progress_callback, frame_idx, total_frames)
        finally:
            cap.release()
            if writer:
                writer.release()
            # Finalize any rally still active at video end
            if self.analytics._rally_active:
                self.analytics._finalize_current_rally(frame_idx=frame_idx)
            self.analytics.save_outputs()
            self._report_progress(progress_callback, total_frames, total_frames)

        if self.mode in ('rallies_only', 'full_and_rallies'):
            self._clip_long_rallies(total_frames, fps)

        if out_path:
            print(f"Saved: {out_path}")
        return out_path
```

- [ ] **Step 2: Verify import chain is intact**

```bash
python -c "from process_video import VideoProcessor; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add process_video.py
git commit -m "Wire mode into process_video main loop; finalize mid-video rallies"
```

---

## Task 6: Add mode selector UI to `main.py`

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add `mode_var` and `CTkSegmentedButton` to `App.__init__` and `create_widgets`**

In `App.__init__`, add after `self.video_path = None`:

```python
        self.mode_var = ctk.StringVar(value="Full Analytics")
```

In `create_widgets`, replace the existing `self.process_btn` grid call and add the segmented button. Change the rows so:
- Row 0: Select Video button (unchanged)
- Row 1: Mode selector (new)
- Row 2: Process Video button (was row 1)
- Row 3: Status frame (was row 2)

```python
        self.select_btn = ctk.CTkButton(left_frame, text="Select Video", command=self.select_video)
        self.select_btn.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        self.mode_selector = ctk.CTkSegmentedButton(
            left_frame,
            values=["Full Analytics", "Rallies Only", "Full + Rallies"],
            variable=self.mode_var,
            state="disabled",
        )
        self.mode_selector.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.process_btn = ctk.CTkButton(
            left_frame, text="Process Video", command=self.process_video_thread, state="disabled"
        )
        self.process_btn.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        status_frame = ctk.CTkFrame(left_frame, corner_radius=10, fg_color="#2a2d2e")
        status_frame.grid(row=3, column=0, padx=20, pady=(10, 20), sticky="nsew")
```

- [ ] **Step 2: Enable `mode_selector` in `select_video`**

In `select_video`, add after `self.process_btn.configure(state="normal")`:

```python
            self.mode_selector.configure(state="normal")
```

- [ ] **Step 3: Pass mode to `VideoProcessor` in `process_video`**

In `App.process_video`, replace the `processor = VideoProcessor(...)` line:

```python
        mode_map = {
            "Full Analytics": "full",
            "Rallies Only": "rallies_only",
            "Full + Rallies": "full_and_rallies",
        }
        mode = mode_map[self.mode_var.get()]
        processor = VideoProcessor(self.video_path, filters, mode=mode)
```

- [ ] **Step 4: Verify UI launches without error**

```bash
python main.py
```

Expected: window opens, mode selector appears disabled until a video is selected.

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "Add mode selector UI for rally clip export"
```

---

## Task 7: Run full test suite and verify

**Files:** none new

- [ ] **Step 1: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 2: Smoke test import of all modules**

```bash
python -c "from analytics import Analytics; from process_video import VideoProcessor; import main; print('All imports OK')"
```

Expected: `All imports OK`

- [ ] **Step 3: Final commit if anything was fixed**

```bash
git add -A
git status
# Only commit if there are changes
git commit -m "Fix any issues found during final test run"
```
