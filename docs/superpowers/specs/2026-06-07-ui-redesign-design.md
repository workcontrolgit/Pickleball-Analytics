# UI Redesign + Logging Design Spec

## Goal

Redesign the customtkinter desktop app with a modern, sporty card-based layout and add structured JSON logging (loguru) to disk per run for troubleshooting.

---

## Section 1 — Layout

### Window

- Size: ~900×600, non-resizable
- Background: `#0F1117` (near-black)
- Title: "Pickleball Analytics"

### Three-Zone Structure

**Zone 1 — Header Strip** (top, ~50px)
- App title on the left: "Pickleball Analytics" in white, bold
- Video filename chip on the right: pill-shaped label showing the selected filename (e.g., "Sample2.mp4"). Hidden until file is selected. Chip bg `#2A2D3A`, text white.

**Zone 2 — Main Card Area** (center, fills remaining vertical space)
Four cards arranged in a 2×2 grid:

| Card | Contents |
|------|----------|
| **File Card** | Drop zone placeholder text + "Browse Video" CTK button (cyan `#00D4FF`) |
| **Mode Card** | CTkSegmentedButton with options: "Full Analytics", "Rallies Only", "Full + Rallies" |
| **Analytics Badges Card** | Static 2×2 grid of four colored pills: Player Heatmap (teal), Ball Heatmap (indigo), Kitchen Detection (amber), Rally Length (purple) — always on, informational only |
| **Results Card** | Hidden at height=0 until processing completes. Reveals via animation. Shows rally count badge (orange) + output path text + "📂 Open Folder" button |

Card style: surface `#1A1D27`, border `#2A2D3A`, corner radius 12px.

**Zone 3 — Footer** (bottom, ~60px)
- Progress bar (full width)
- Status label below progress bar (left-aligned)

---

## Section 2 — Colors & Motion

### Color Palette

| Role | Hex | Usage |
|------|-----|-------|
| Background | `#0F1117` | Window background |
| Card surface | `#1A1D27` | Card backgrounds |
| Card border | `#2A2D3A` | Card outlines, chips |
| CTA / active | `#00D4FF` | Browse button, active tab |
| Rally stats | `#FF6B35` | Rally count badge |
| Success | `#00C49A` | Completion status |
| Error | `#FF4757` | Error status text |
| Body text | `#E0E0E0` | Normal text |
| Muted text | `#6B7280` | Placeholder, hints |

### Results Card Reveal Animation

When processing completes, reveal the Results card via a height animation:
- Start: `card.configure(height=0)` (hidden)
- End: `card.configure(height=90)`
- Mechanism: `after()` loop — each tick increments height by 10px, 30ms interval → 300ms total
- Easing: linear (uniform steps)
- Only triggered on success; error state shows error inline in footer status, Results card stays hidden

```python
def _reveal_results(self, step=0):
    target = 90
    increment = 10
    delay = 30
    current = step * increment
    if current <= target:
        self.results_card.configure(height=current)
        self.after(delay, lambda: self._reveal_results(step + 1))
```

---

## Section 3 — State Machine

Five states govern all widget enable/disable behavior and visual feedback.

### States

**Idle** (app startup)
- Select Video: enabled
- Mode selector: disabled
- Process Video: disabled
- Header chip: hidden
- Status: "Select a video to begin" (muted gray)
- Progress bar: 0, hidden or grayed

**File Selected** (after Browse/file chosen)
- Select Video: enabled (re-select allowed)
- Mode selector: enabled
- Process Video: enabled
- Header chip: visible, shows filename
- Status: filename or "Ready to process"

**Processing** (after Process Video clicked)
- All controls: disabled
- Progress bar: animating (indeterminate or frame-based)
- Status: "Analyzing… frame X / Y" (updates via `progress_callback`)

**Complete** (processing finished successfully)
- Process Video: enabled (re-process allowed)
- Mode selector: enabled
- Select Video: enabled
- Results card: revealed via animation
  - Rally count badge (orange `#FF6B35`): "N rallies detected" or "No long rallies"
  - Output path: truncated path string
  - "📂 Open Folder" button → `os.startfile(out_dir)`
- Status: "Done!" in emerald `#00C49A`

**Error** (exception during processing)
- Process Video: enabled (retry allowed)
- Mode selector: enabled
- Select Video: enabled
- Results card: stays hidden (no animation)
- Status: `f"Error: {str(e)}"` in red `#FF4757`

### State Transitions

```
Idle → File Selected        (file chosen)
File Selected → Processing  (Process Video clicked)
Processing → Complete       (success)
Processing → Error          (exception)
Complete → File Selected    (new file selected)
Complete → Processing       (Process Video clicked again)
Error → File Selected       (new file selected)
Error → Processing          (Process Video clicked, retry)
```

---

## Section 4 — Logging Architecture

### Library

`loguru` with `serialize=True` for structured JSON output.

### Log File Location

`video_outputs/<run_YYYYMMDD_HHMMSS>/run.log`

One log file per run, co-located with output videos.

### Setup

`setup_logger(run_dir: str)` is called from `VideoProcessor.__init__` immediately after `self.output_dir` is determined.

```python
# In process_video.py — VideoProcessor.__init__
from loguru import logger
import sys

def setup_logger(run_dir: str):
    logger.remove()  # Remove default stderr sink
    log_path = os.path.join(run_dir, "run.log")
    logger.add(log_path, serialize=True, level="DEBUG", enqueue=True)
```

`enqueue=True` makes writes non-blocking (safe on background thread).

### Events to Log

| Event | Level | Extra fields |
|-------|-------|-------------|
| `run_started` | INFO | `video_path`, `mode` |
| `court_detected` | INFO | `frame_idx`, `homography_valid` |
| `frame_milestone` | DEBUG | `frame_idx`, `total_frames`, `progress_pct` — every 500 frames |
| `rally_candidate` | DEBUG | `start_frame`, `end_frame`, `crossing_count` |
| `rally_clip_saved` | INFO | `rally_num`, `clip_path`, `duration_s`, `crossing_count` |
| `run_completed` | INFO | `total_frames`, `rally_count`, `elapsed_s` |
| Exceptions | ERROR | via `logger.exception("message")` inside except blocks |

### Sample Log Entry

```json
{
  "record": {
    "elapsed": {"repr": "0:00:45.123", "seconds": 45.123},
    "level": {"name": "INFO"},
    "message": "rally_clip_saved",
    "time": {"repr": "2026-06-07 06:44:44.000000+00:00"},
    "extra": {
      "rally_num": 3,
      "clip_path": "video_outputs/run_20260607_064444/rally_03.mp4",
      "duration_s": 12.4,
      "crossing_count": 8
    }
  }
}
```

### Logging Pattern

Use `logger.bind(**extra).info("event_name")` to attach structured context:

```python
logger.bind(rally_num=rally_num, clip_path=str(clip_path), duration_s=round(duration, 1)).info("rally_clip_saved")
```

---

## Files Changed

| File | Change |
|------|--------|
| `main.py` | Full rewrite — 3-zone layout, 4 cards, state machine, results reveal animation |
| `process_video.py` | Add `setup_logger(self.output_dir)` call in `__init__` + log events throughout |

No other files change. Analytics, trackers, and court detection are untouched.

---

## Out of Scope

- No batch processing (single video at a time)
- No in-app log viewer panel
- No dark/light mode toggle
- No drag-and-drop (browse button only)
