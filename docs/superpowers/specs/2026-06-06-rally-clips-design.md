# Rally Clip Export — Design Spec
**Date:** 2026-06-06

## Overview

Detect "long rallies" (5+ net crossings) in a pickleball video and export each as a separate raw footage MP4 clip. A new UI mode selector lets the user choose between full analytics, rallies only, or both.

---

## Requirements

- A rally qualifies as "long" when the ball crosses the net **5 or more times** in a single rally.
- Each qualifying rally is saved as a separate raw (no overlay) MP4 clip in the existing run output folder: `video_outputs/run_<timestamp>/rally_01.mp4`, `rally_02.mp4`, etc.
- Each clip includes a **2-second buffer** of raw footage before the first crossing and after the last shot.
- The UI presents three processing modes: **Full Analytics**, **Rallies Only**, **Full + Rallies**.
- In **Rallies Only** mode, the composite `Main_overlay.mp4` is not produced — only rally clips.

---

## Architecture

### Approach: Record Indices + Post-Process Seek (Option C)

During the main processing loop (which already runs to produce `Main_overlay.mp4`), rally metadata is accumulated in `Analytics`. After the loop, `VideoProcessor` re-opens the source video and seeks directly to each qualifying rally's frame range to write clips.

No frame buffering in memory. Seeks only to qualifying rally segments.

---

## Component Changes

### 1. `analytics.py` — Net Crossing Detection

**New internal state:**

| Field | Type | Purpose |
|-------|------|---------|
| `_net_y` | `float \| None` | Net midpoint in bird coords (`(kitchen_y_min + kitchen_y_max) / 2`). `None` until court is detected. |
| `_ball_last_side` | `str \| None` | `'above'` or `'below'` net. `None` until first detection. |
| `_rally_net_crossings` | `int` | Crossing count for the current active rally. |
| `_rally_start_frame` | `int \| None` | Frame index when current rally began. |
| `_long_rallies` | `list[tuple]` | List of `(start_frame, end_frame, crossings)` for completed qualifying rallies. |

**`update_counters()` additions:**

1. Derive `_net_y` from `(_kitchen_y_min + _kitchen_y_max) / 2` each frame (once kitchen bounds are learned).
2. When ball is detected and `_net_y` is known, compute current side (`'above'` if `ball_y < _net_y`, else `'below'`).
3. If `_ball_last_side` is set and current side differs → increment `_rally_net_crossings`, update `_ball_last_side`.
4. When a rally begins (`_rally_active` transitions to `True`): record `_rally_start_frame = frame_idx`, reset `_rally_net_crossings = 0`, reset `_ball_last_side = None`.
5. When a rally ends (gap threshold exceeded): if `_rally_net_crossings >= 5`, append `(start_frame, end_frame, crossings)` to `_long_rallies`.

**Missing ball detections:** If ball is not detected in a frame, `_ball_last_side` is held — no crossing counted for that frame.

**`update_kitchen_from_keypoints()` side effect:** `_net_y` is updated whenever kitchen bounds change.

---

### 2. `process_video.py` — Mode Support & Clipping

**`VideoProcessor.__init__()` change:**
- Accept a `mode: str` parameter (`'full'`, `'rallies_only'`, `'full_and_rallies'`). Default: `'full'`.

**`process_video()` changes:**

- `mode='full'`: existing behavior unchanged.
- `mode='rallies_only'`: skip `_render_main_view`, `_render_birdseye`, `_render_analytics_grid`, and `VideoWriter`. Still runs all detection (court/player/ball), calls `analytics.update_kitchen_from_keypoints()`, `update_court_bounds_from_keypoints()`, `update_zones_from_keypoints()`, and `update_counters()` per frame (these are currently called inside `_render_birdseye` — they must be extracted to run in all modes). After loop, call `_clip_long_rallies()`.
- `mode='full_and_rallies'`: existing composite rendering + after loop call `_clip_long_rallies()`.

**New method `_clip_long_rallies(total_frames, fps)`:**

```
for each (start_frame, end_frame, crossings) in analytics._long_rallies:
    clip_start = max(0, start_frame - fps * 2)
    clip_end   = min(total_frames - 1, end_frame + fps * 2)
    seek source video to clip_start
    open VideoWriter → output_dir/rally_NN.mp4  (original resolution, same fps)
    read frames clip_start..clip_end, write raw frames
    release writer
```

Clips are numbered sequentially by rally order (`rally_01`, `rally_02`, …).

**`_create_writer()` guard:** In `rallies_only` mode, no composite writer is opened.

---

### 3. `main.py` — Mode Selector UI

**New widget:** `CTkSegmentedButton` with values `["Full Analytics", "Rallies Only", "Full + Rallies"]`, default `"Full Analytics"`. Placed between "Select Video" and "Process Video" buttons in the left panel. Disabled until a video is selected.

**Mode mapping:**

| UI Label | `mode` value passed to `VideoProcessor` |
|----------|------------------------------------------|
| Full Analytics | `'full'` |
| Rallies Only | `'rallies_only'` |
| Full + Rallies | `'full_and_rallies'` |

---

## Edge Cases

| Case | Behavior |
|------|----------|
| No long rallies detected | `_clip_long_rallies()` exits silently, no empty files written |
| Court not detected in early frames | `_net_y` is `None`; crossing detection skipped until court is found |
| Video ends mid-rally with 5+ crossings | Rally is finalized and included |
| Two qualifying rallies with overlapping buffers | Clips written independently, no merging |
| Seek imprecision (codec-dependent) | 2-second buffer absorbs drift |

---

## Output

- `video_outputs/run_<timestamp>/Main_overlay.mp4` — composite (if mode is `full` or `full_and_rallies`)
- `video_outputs/run_<timestamp>/rally_01.mp4`, `rally_02.mp4`, … — raw clips (if mode is `rallies_only` or `full_and_rallies`)
