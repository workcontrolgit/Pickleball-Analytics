# Rally Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect pickleball rallies by anchoring to serve events and the two-bounce rule, write `rally_report.json`, then let the user confirm before clipping into individual video files.

**Architecture:** A new `RallyDetector` FSM class (`rally_detector.py`) tracks each rally through five states (IDLE → SERVE_DETECTED → BOUNCE_1_PENDING → BOUNCE_2_PENDING → OPEN_PLAY → ENDED). Two new modes are added to `VideoProcessor`: `MODE_DETECT_RALLIES` (analysis pass → JSON report) and `MODE_CLIP_FROM_REPORT` (reads report → writes clips). The Tkinter UI gains a "Detect Rallies" mode with a two-step confirm flow.

**Tech Stack:** Python 3.11, OpenCV (`cv2`), NumPy, customtkinter, pytest, existing `ServeDetector` class

**Spec:** `docs/superpowers/specs/2026-06-10-rally-detection-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `rally_detector.py` | **Create** | `RallyDetector` FSM + `RallyRecord` dataclass |
| `tests/test_rally_detector.py` | **Create** | Unit tests for `RallyDetector` |
| `process_video.py` | **Modify** | Add `MODE_DETECT_RALLIES`, `MODE_CLIP_FROM_REPORT`, wire `RallyDetector` |
| `tests/test_video_processor_modes.py` | **Modify** | Add mode constant + init tests for the two new modes |
| `main.py` | **Modify** | Add "Detect Rallies" to mode selector, two-step confirm UI |

---

## Task 1: `RallyRecord` dataclass and FSM skeleton

**Files:**
- Create: `rally_detector.py`
- Create: `tests/test_rally_detector.py`

- [ ] **Step 1.1: Write failing tests for RallyRecord and initial FSM state**

```python
# tests/test_rally_detector.py
import pytest
from rally_detector import RallyDetector, RallyRecord

def test_rally_record_fields():
    r = RallyRecord(
        rally_num=1,
        start_frame=10,
        end_frame=50,
        fps=30,
        net_crossings=4,
        end_reason="out",
        two_bounce_complete=True,
    )
    assert r.start_sec == pytest.approx(10 / 30)
    assert r.end_sec == pytest.approx(50 / 30)
    assert r.duration_sec == pytest.approx((50 - 10) / 30)

def test_detector_starts_idle():
    det = RallyDetector(fps=30)
    assert det.state == RallyDetector.IDLE
    assert det.get_rallies() == []
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && .venv/Scripts/python -m pytest tests/test_rally_detector.py -v
```

Expected: `ModuleNotFoundError: No module named 'rally_detector'`

- [ ] **Step 1.3: Create `rally_detector.py` with dataclass and skeleton**

```python
"""
Rally detection module

Purpose
-------
Detects pickleball rallies by anchoring to serve events and the two-bounce rule.
Tracks each rally through a five-state FSM and emits RallyRecord objects.

States
------
IDLE → SERVE_DETECTED → BOUNCE_1_PENDING → BOUNCE_2_PENDING → OPEN_PLAY → ENDED

Inputs (per frame via update())
------
- frame_idx: int
- ball_proj: Optional[tuple]   — (x, y) in bird's-eye coords
- ball_y2:   Optional[float]   — ball bbox bottom in raw frame pixels (bounce detection)
- serve_event                  — ServeCandidate or None
- net_y:     Optional[float]   — net Y in bird space
- court_bounds: Optional[tuple]— (xmin, ymin, xmax, ymax) in bird space

Outputs
-------
- get_rallies() → list of RallyRecord (completed rallies only)
- get_report()  → dict suitable for json.dump
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RallyRecord:
    rally_num: int
    start_frame: int
    end_frame: int
    fps: float
    net_crossings: int
    end_reason: str          # "out" | "net" | "fault"
    two_bounce_complete: bool

    @property
    def start_sec(self) -> float:
        return round(self.start_frame / self.fps, 2)

    @property
    def end_sec(self) -> float:
        return round(self.end_frame / self.fps, 2)

    @property
    def duration_sec(self) -> float:
        return round((self.end_frame - self.start_frame) / self.fps, 2)


class RallyDetector:
    # FSM state constants
    IDLE              = "idle"
    SERVE_DETECTED    = "serve_detected"
    BOUNCE_1_PENDING  = "bounce_1_pending"
    BOUNCE_2_PENDING  = "bounce_2_pending"
    OPEN_PLAY         = "open_play"

    # Tunable constants
    NET_HIT_RADIUS_PX  = 40    # ball within this many bird-space px of net_y counts as net-hit zone
    NET_HIT_GONE_FRAMES = 15   # frames absent after net-zone last seen → net hit
    FAULT_GONE_FRAMES  = 30    # frames absent after in-bounds last seen → fault/catch
    MAX_GAP_FRAMES     = 5     # tolerate this many consecutive missing detections
    BOUNDS_AREA_RATIO  = 2.0   # reject court bounds update if area > this × reference

    def __init__(self, fps: float = 30.0):
        self._fps = fps
        self._rallies: list[RallyRecord] = []
        self._reset()

    def _reset(self):
        self.state = self.IDLE
        self._start_frame: Optional[int] = None
        self._net_crossings = 0
        self._ball_last_side: Optional[str] = None   # 'near' | 'far'
        self._gap_frames = 0
        self._last_inbounds_frame: Optional[int] = None
        self._last_near_net_frame: Optional[int] = None
        self._two_bounce_complete = False
        # Bounce detection
        self._y2_history: list[float] = []            # last N raw-frame y2 values
        self._bounce_count = 0
        self._net_crossings_at_last_bounce = 0

    def get_rallies(self) -> list[RallyRecord]:
        return list(self._rallies)

    def get_report(self, video_path: str = "") -> dict:
        rallies = self.get_rallies()
        return {
            "video_path": video_path,
            "fps": self._fps,
            "total_rallies": len(rallies),
            "rallies": [
                {
                    "rally_num": r.rally_num,
                    "start_frame": r.start_frame,
                    "end_frame": r.end_frame,
                    "start_sec": r.start_sec,
                    "end_sec": r.end_sec,
                    "duration_sec": r.duration_sec,
                    "net_crossings": r.net_crossings,
                    "end_reason": r.end_reason,
                    "two_bounce_complete": r.two_bounce_complete,
                }
                for r in rallies
            ],
        }
```

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && .venv/Scripts/python -m pytest tests/test_rally_detector.py::test_rally_record_fields tests/test_rally_detector.py::test_detector_starts_idle -v
```

Expected: 2 PASSED

- [ ] **Step 1.5: Commit**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && git add rally_detector.py tests/test_rally_detector.py && git commit -m "feat: add RallyRecord dataclass and RallyDetector FSM skeleton"
```

---

## Task 2: Court bounds sanity check

**Files:**
- Modify: `rally_detector.py` — add `_reference_court_area`, `_validate_and_set_court_bounds()`
- Modify: `tests/test_rally_detector.py`

- [ ] **Step 2.1: Write failing tests**

```python
# append to tests/test_rally_detector.py

def test_court_bounds_accepted_on_first_call():
    det = RallyDetector(fps=30)
    bounds = (0.0, 0.0, 100.0, 200.0)   # area = 100*200 = 20000
    det._validate_and_set_court_bounds(bounds)
    assert det._court_bounds == bounds
    assert det._reference_court_area == pytest.approx(20000.0)

def test_court_bounds_rejected_when_too_large():
    det = RallyDetector(fps=30)
    small = (0.0, 0.0, 100.0, 200.0)    # area 20000
    big   = (0.0, 0.0, 300.0, 200.0)    # area 60000 — 3× reference
    det._validate_and_set_court_bounds(small)
    det._validate_and_set_court_bounds(big)
    assert det._court_bounds == small    # rejected — held previous

def test_court_bounds_accepted_within_ratio():
    det = RallyDetector(fps=30)
    first  = (0.0, 0.0, 100.0, 200.0)  # area 20000
    second = (0.0, 0.0, 110.0, 200.0)  # area 22000 — 1.1× OK
    det._validate_and_set_court_bounds(first)
    det._validate_and_set_court_bounds(second)
    assert det._court_bounds == second
```

- [ ] **Step 2.2: Run to verify failure**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && .venv/Scripts/python -m pytest tests/test_rally_detector.py::test_court_bounds_accepted_on_first_call tests/test_rally_detector.py::test_court_bounds_rejected_when_too_large tests/test_rally_detector.py::test_court_bounds_accepted_within_ratio -v
```

Expected: 3 FAILED (`AttributeError`)

- [ ] **Step 2.3: Implement `_validate_and_set_court_bounds` in `rally_detector.py`**

Add to `RallyDetector.__init__` (inside `_reset` is wrong — court ref persists across rallies):

```python
def __init__(self, fps: float = 30.0):
    self._fps = fps
    self._rallies: list[RallyRecord] = []
    self._court_bounds: Optional[tuple] = None
    self._reference_court_area: Optional[float] = None
    self._net_y: Optional[float] = None
    self._reset()
```

Add method to `RallyDetector`:

```python
def _validate_and_set_court_bounds(self, bounds: tuple) -> None:
    """Accept new court bounds only if area is within BOUNDS_AREA_RATIO of the reference.
    On first call, accept unconditionally and set the reference area.
    Protects against tennis-court keypoints inflating the court bounds.
    """
    xmin, ymin, xmax, ymax = bounds
    area = max((xmax - xmin) * (ymax - ymin), 1.0)

    if self._reference_court_area is None:
        self._reference_court_area = area
        self._court_bounds = bounds
        return

    if area <= self._reference_court_area * self.BOUNDS_AREA_RATIO:
        self._court_bounds = bounds
    # else: silently hold last valid bounds
```

- [ ] **Step 2.4: Run tests to verify they pass**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && .venv/Scripts/python -m pytest tests/test_rally_detector.py -v
```

Expected: 5 PASSED

- [ ] **Step 2.5: Commit**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && git add rally_detector.py tests/test_rally_detector.py && git commit -m "feat: add court bounds sanity check to RallyDetector"
```

---

## Task 3: Net crossing and ball-in-bounds detection

**Files:**
- Modify: `rally_detector.py` — add `_ball_in_bounds()`, `_update_net_crossing()`
- Modify: `tests/test_rally_detector.py`

- [ ] **Step 3.1: Write failing tests**

```python
# append to tests/test_rally_detector.py

def _make_det_with_court(net_y=500.0):
    det = RallyDetector(fps=30)
    det._validate_and_set_court_bounds((0.0, 0.0, 400.0, 1000.0))
    det._net_y = net_y
    return det

def test_ball_in_bounds_inside():
    det = _make_det_with_court()
    assert det._ball_in_bounds((200.0, 500.0)) is True

def test_ball_in_bounds_outside():
    det = _make_det_with_court()
    assert det._ball_in_bounds((500.0, 500.0)) is False  # x > xmax=400

def test_ball_in_bounds_none():
    det = _make_det_with_court()
    assert det._ball_in_bounds(None) is False

def test_net_crossing_detected():
    det = _make_det_with_court(net_y=500.0)
    det.state = det.OPEN_PLAY
    det._ball_last_side = "near"
    det._update_net_crossing((200.0, 300.0))   # y=300 < net_y=500 → "far" side
    assert det._net_crossings == 1

def test_no_crossing_same_side():
    det = _make_det_with_court(net_y=500.0)
    det.state = det.OPEN_PLAY
    det._ball_last_side = "near"
    det._update_net_crossing((200.0, 700.0))   # still near side (y > net_y)
    assert det._net_crossings == 0
```

- [ ] **Step 3.2: Run to verify failure**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && .venv/Scripts/python -m pytest tests/test_rally_detector.py -k "in_bounds or net_crossing" -v
```

Expected: 5 FAILED

- [ ] **Step 3.3: Implement `_ball_in_bounds` and `_update_net_crossing`**

Add to `RallyDetector` class in `rally_detector.py`:

```python
def _ball_in_bounds(self, ball_proj: Optional[tuple]) -> bool:
    if ball_proj is None or self._court_bounds is None:
        return False
    x, y = ball_proj
    xmin, ymin, xmax, ymax = self._court_bounds
    tol = 2.0
    return (xmin - tol) <= x <= (xmax + tol) and (ymin - tol) <= y <= (ymax + tol)

def _update_net_crossing(self, ball_proj: tuple) -> None:
    """Track ball side relative to net; increment _net_crossings on each change."""
    if self._net_y is None:
        return
    x, y = ball_proj
    # From behind baseline: y increases toward camera (near end).
    # "near" = same side as camera (y > net_y), "far" = opposite side (y < net_y).
    side = "near" if y > self._net_y else "far"
    if self._ball_last_side is not None and side != self._ball_last_side:
        self._net_crossings += 1
    self._ball_last_side = side
```

- [ ] **Step 3.4: Run tests**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && .venv/Scripts/python -m pytest tests/test_rally_detector.py -v
```

Expected: 10 PASSED

- [ ] **Step 3.5: Commit**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && git add rally_detector.py tests/test_rally_detector.py && git commit -m "feat: add ball-in-bounds and net-crossing detection to RallyDetector"
```

---

## Task 4: Bounce detection with net-crossing fallback

**Files:**
- Modify: `rally_detector.py` — add `_update_bounce_detection()`, `_two_bounce_satisfied()`
- Modify: `tests/test_rally_detector.py`

- [ ] **Step 4.1: Write failing tests**

```python
# append to tests/test_rally_detector.py

def test_bounce_detected_on_y2_reversal():
    """Ball y2 rises then falls → local maximum = bounce."""
    det = RallyDetector(fps=30)
    # Feed increasing y2 (ball falling) then decreasing (ball rising after bounce)
    for y2 in [300.0, 320.0, 340.0, 355.0, 360.0]:   # falling
        det._update_bounce_detection(y2)
    for y2 in [350.0, 330.0]:                          # rising — bounce happened
        det._update_bounce_detection(y2)
    assert det._bounce_count >= 1

def test_no_bounce_on_monotone_fall():
    det = RallyDetector(fps=30)
    for y2 in [200.0, 250.0, 300.0, 350.0, 400.0]:
        det._update_bounce_detection(y2)
    assert det._bounce_count == 0

def test_two_bounce_satisfied_via_bounces():
    det = RallyDetector(fps=30)
    det._bounce_count = 2
    det._net_crossings = 1
    assert det._two_bounce_satisfied() is True

def test_two_bounce_satisfied_via_net_crossings_fallback():
    """If bounce count is low but net crossings >= 2, use crossings as proxy."""
    det = RallyDetector(fps=30)
    det._bounce_count = 0    # noisy tracking — no bounces detected
    det._net_crossings = 2   # but two crossings seen → implies two bounces occurred
    assert det._two_bounce_satisfied() is True

def test_two_bounce_not_satisfied():
    det = RallyDetector(fps=30)
    det._bounce_count = 0
    det._net_crossings = 1
    assert det._two_bounce_satisfied() is False
```

- [ ] **Step 4.2: Run to verify failure**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && .venv/Scripts/python -m pytest tests/test_rally_detector.py -k "bounce" -v
```

Expected: 5 FAILED

- [ ] **Step 4.3: Implement bounce detection in `rally_detector.py`**

Add to `RallyDetector` class:

```python
_Y2_HISTORY_LEN = 5          # sliding window for bounce detection
_BOUNCE_REVERSAL_PX = 5.0    # minimum y2 drop after peak to confirm bounce

def _update_bounce_detection(self, ball_y2: float) -> None:
    """Detect a bounce via local maximum in raw-frame y2 (ball bottom).
    From behind-the-baseline: y2 increases as ball falls, decreases as ball rises.
    A bounce = y2 peaks then drops by at least _BOUNCE_REVERSAL_PX.
    """
    self._y2_history.append(ball_y2)
    if len(self._y2_history) > self._Y2_HISTORY_LEN:
        self._y2_history.pop(0)

    if len(self._y2_history) < 3:
        return

    # Check if the second-to-last value is a local maximum
    prev, peak, curr = self._y2_history[-3], self._y2_history[-2], self._y2_history[-1]
    if peak > prev and peak > curr and (peak - curr) >= self._BOUNCE_REVERSAL_PX:
        self._bounce_count += 1

def _two_bounce_satisfied(self) -> bool:
    """True if both mandatory bounces of the two-bounce rule have occurred.
    Primary signal: 2+ detected bounces.
    Fallback: 2+ net crossings (implies serve bounced on far side, return bounced on near side).
    """
    if self._bounce_count >= 2:
        return True
    # Fallback: net crossings proxy
    # 1st crossing = serve going over, 2nd crossing = return going over
    # By the time the return crosses back, both mandatory bounces must have occurred.
    if self._net_crossings >= 2:
        return True
    return False
```

- [ ] **Step 4.4: Run tests**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && .venv/Scripts/python -m pytest tests/test_rally_detector.py -v
```

Expected: 15 PASSED

- [ ] **Step 4.5: Commit**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && git add rally_detector.py tests/test_rally_detector.py && git commit -m "feat: add bounce detection with net-crossing fallback"
```

---

## Task 5: Rally end detection

**Files:**
- Modify: `rally_detector.py` — add `_check_end_condition()`
- Modify: `tests/test_rally_detector.py`

- [ ] **Step 5.1: Write failing tests**

```python
# append to tests/test_rally_detector.py

def test_end_condition_out_of_bounds():
    det = _make_det_with_court(net_y=500.0)
    det.state = det.OPEN_PLAY
    det._start_frame = 0
    # Ball outside bounds for 3+ frames
    for fi in range(3):
        det._last_inbounds_frame = 0
        reason = det._check_end_condition(fi + 1, (600.0, 500.0))  # x=600 > xmax=400
    assert reason == "out"

def test_end_condition_net_hit():
    det = _make_det_with_court(net_y=500.0)
    det.state = det.OPEN_PLAY
    det._start_frame = 0
    # Ball last seen near net, then absent for NET_HIT_GONE_FRAMES
    det._last_near_net_frame = 10
    det._last_inbounds_frame = 10
    reason = det._check_end_condition(10 + RallyDetector.NET_HIT_GONE_FRAMES, None)
    assert reason == "net"

def test_end_condition_fault():
    det = _make_det_with_court(net_y=500.0)
    det.state = det.OPEN_PLAY
    det._start_frame = 0
    # Ball absent for FAULT_GONE_FRAMES with no near-net context
    det._last_near_net_frame = None
    det._last_inbounds_frame = 0
    reason = det._check_end_condition(RallyDetector.FAULT_GONE_FRAMES, None)
    assert reason == "fault"

def test_no_end_condition_when_active():
    det = _make_det_with_court(net_y=500.0)
    det.state = det.OPEN_PLAY
    det._start_frame = 0
    det._last_inbounds_frame = 10
    reason = det._check_end_condition(11, (200.0, 500.0))
    assert reason is None
```

- [ ] **Step 5.2: Run to verify failure**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && .venv/Scripts/python -m pytest tests/test_rally_detector.py -k "end_condition" -v
```

Expected: 4 FAILED

- [ ] **Step 5.3: Implement `_check_end_condition` in `rally_detector.py`**

Add to `RallyDetector` class:

```python
_OUT_CONSEC_FRAMES = 3   # ball must be OOB for this many frames to trigger "out"

def _check_end_condition(
    self, frame_idx: int, ball_proj: Optional[tuple]
) -> Optional[str]:
    """Return end reason string if rally should end, else None.
    Call this only when state is not IDLE.
    """
    in_bounds = self._ball_in_bounds(ball_proj)

    if in_bounds:
        self._last_inbounds_frame = frame_idx
        self._gap_frames = 0
        # Track near-net position
        if self._net_y is not None and ball_proj is not None:
            if abs(ball_proj[1] - self._net_y) <= self.NET_HIT_RADIUS_PX:
                self._last_near_net_frame = frame_idx
        return None

    # Ball not in bounds (or None)
    self._gap_frames += 1

    if ball_proj is None:
        # Ball missing — check net-hit and fault timers
        if (
            self._last_near_net_frame is not None
            and self._last_inbounds_frame is not None
            and (frame_idx - self._last_near_net_frame) >= self.NET_HIT_GONE_FRAMES
            and self._last_near_net_frame >= (self._last_inbounds_frame - 5)
        ):
            return "net"
        if (
            self._last_inbounds_frame is not None
            and (frame_idx - self._last_inbounds_frame) >= self.FAULT_GONE_FRAMES
        ):
            return "fault"
    else:
        # Ball detected but outside court bounds
        if self._gap_frames >= self._OUT_CONSEC_FRAMES:
            return "out"

    return None
```

- [ ] **Step 5.4: Run tests**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && .venv/Scripts/python -m pytest tests/test_rally_detector.py -v
```

Expected: 19 PASSED

- [ ] **Step 5.5: Commit**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && git add rally_detector.py tests/test_rally_detector.py && git commit -m "feat: add rally end condition detection (out/net/fault)"
```

---

## Task 6: Full FSM `update()` method and rally finalization

**Files:**
- Modify: `rally_detector.py` — add `update()`, `_finalize_rally()`
- Modify: `tests/test_rally_detector.py`

- [ ] **Step 6.1: Write failing tests**

```python
# append to tests/test_rally_detector.py
from serve_detector import ServeCandidate
import numpy as np

def _make_serve_event(frame_idx=0):
    return ServeCandidate(
        frame_idx=frame_idx,
        timestamp_sec=frame_idx / 30.0,
        player_id=1,
        ball_pos=(200.0, 800.0),
        frame_small=np.zeros((720, 1280, 3), dtype=np.uint8),
    )

def _run_full_rally(det, start=0, length=90):
    """Simulate a complete rally: serve → 2 net crossings → ball OOB."""
    net_y = 500.0
    det._net_y = net_y
    det._validate_and_set_court_bounds((0.0, 0.0, 400.0, 1000.0))

    # Serve at frame `start`
    det.update(start, (200.0, 800.0), 700.0, _make_serve_event(start), net_y, None)

    # Ball moves to far side (1st crossing)
    for fi in range(start + 1, start + 20):
        det.update(fi, (200.0, 300.0), 300.0, None, net_y, None)

    # Ball bounces far side (simulate y2 peak)
    for y2 in [280.0, 300.0, 320.0, 340.0, 345.0, 330.0, 310.0]:
        det.update(start + 20, (200.0, 300.0), y2, None, net_y, None)

    # Ball returns (2nd crossing)
    for fi in range(start + 21, start + 40):
        det.update(fi, (200.0, 700.0), 600.0, None, net_y, None)

    # Open play — several more crossings
    for fi in range(start + 40, start + length - 3):
        side_y = 300.0 if fi % 20 < 10 else 700.0
        det.update(fi, (200.0, side_y), 400.0, None, net_y, None)

    # Ball goes OOB for 3+ frames
    for fi in range(start + length - 3, start + length):
        det.update(fi, (600.0, 500.0), 400.0, None, net_y, None)

def test_full_rally_recorded():
    det = RallyDetector(fps=30)
    _run_full_rally(det, start=10, length=90)
    rallies = det.get_rallies()
    assert len(rallies) == 1
    r = rallies[0]
    assert r.rally_num == 1
    assert r.start_frame == 10
    assert r.end_reason == "out"
    assert r.two_bounce_complete is True

def test_two_rallies_recorded():
    det = RallyDetector(fps=30)
    _run_full_rally(det, start=0, length=90)
    _run_full_rally(det, start=200, length=60)
    assert len(det.get_rallies()) == 2
    assert det.get_rallies()[1].rally_num == 2

def test_no_rally_without_serve():
    det = RallyDetector(fps=30)
    net_y = 500.0
    det._net_y = net_y
    det._validate_and_set_court_bounds((0.0, 0.0, 400.0, 1000.0))
    # Ball in bounds but no serve event
    for fi in range(100):
        det.update(fi, (200.0, 400.0), 400.0, None, net_y, None)
    assert det.get_rallies() == []
```

- [ ] **Step 6.2: Run to verify failure**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && .venv/Scripts/python -m pytest tests/test_rally_detector.py -k "full_rally or two_rallies or without_serve" -v
```

Expected: 3 FAILED (`AttributeError: 'RallyDetector' object has no attribute 'update'`)

- [ ] **Step 6.3: Implement `update()` and `_finalize_rally()` in `rally_detector.py`**

Add to `RallyDetector` class:

```python
def update(
    self,
    frame_idx: int,
    ball_proj: Optional[tuple],
    ball_y2: Optional[float],
    serve_event,                       # ServeCandidate | None
    net_y: Optional[float],
    court_bounds: Optional[tuple],
) -> None:
    """Process one frame. Updates internal FSM state and emits RallyRecord on end."""
    # Update court bounds (with sanity check) and net position
    if court_bounds is not None:
        self._validate_and_set_court_bounds(court_bounds)
    if net_y is not None:
        self._net_y = net_y

    # --- IDLE: wait for a serve ---
    if self.state == self.IDLE:
        if serve_event is not None:
            self.state = self.SERVE_DETECTED
            self._start_frame = serve_event.frame_idx
            self._last_inbounds_frame = serve_event.frame_idx
            self._ball_last_side = "near"   # server is on near side (behind baseline)
        return

    # --- Active rally: update bounce + crossing signals ---
    if ball_proj is not None and self._ball_in_bounds(ball_proj):
        self._update_net_crossing(ball_proj)
        if ball_y2 is not None:
            self._update_bounce_detection(ball_y2)

    # --- Advance FSM based on net crossings ---
    if self.state == self.SERVE_DETECTED and self._net_crossings >= 1:
        self.state = self.BOUNCE_1_PENDING

    if self.state == self.BOUNCE_1_PENDING and self._net_crossings >= 2:
        self.state = self.BOUNCE_2_PENDING

    if self.state == self.BOUNCE_2_PENDING and self._two_bounce_satisfied():
        self.state = self.OPEN_PLAY
        self._two_bounce_complete = True

    # --- Check end condition ---
    reason = self._check_end_condition(frame_idx, ball_proj)
    if reason is not None:
        self._finalize_rally(frame_idx, reason)

def _finalize_rally(self, end_frame: int, end_reason: str) -> None:
    """Record the completed rally and reset FSM to IDLE."""
    record = RallyRecord(
        rally_num=len(self._rallies) + 1,
        start_frame=self._start_frame,
        end_frame=end_frame,
        fps=self._fps,
        net_crossings=self._net_crossings,
        end_reason=end_reason,
        two_bounce_complete=self._two_bounce_complete,
    )
    self._rallies.append(record)
    self._reset()
```

- [ ] **Step 6.4: Run all tests**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && .venv/Scripts/python -m pytest tests/test_rally_detector.py -v
```

Expected: 22 PASSED

- [ ] **Step 6.5: Commit**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && git add rally_detector.py tests/test_rally_detector.py && git commit -m "feat: implement RallyDetector FSM update() and rally finalization"
```

---

## Task 7: `MODE_DETECT_RALLIES` in `process_video.py`

**Files:**
- Modify: `process_video.py` — add constant, init branch, per-frame update, `_save_rally_report()`
- Modify: `tests/test_video_processor_modes.py`

- [ ] **Step 7.1: Write failing tests**

```python
# append to tests/test_video_processor_modes.py
from process_video import MODE_DETECT_RALLIES

def test_mode_detect_rallies_constant():
    assert MODE_DETECT_RALLIES == "detect_rallies"

def test_detect_rallies_has_rally_detector_no_analytics():
    """DETECT_RALLIES mode: RallyDetector present, Analytics absent."""
    with patch("process_video.BallTracker"), \
         patch("process_video.PlayerTracker"), \
         patch("process_video.CourtDetector"), \
         patch("process_video.Analytics") as MockAnalytics, \
         patch("process_video.ServeDetector"), \
         patch("process_video.OllamaServeAnalyzer"), \
         patch("process_video.RallyDetector") as MockRallyDetector, \
         patch("process_video.VideoProcessor._make_output_dir", return_value="/tmp/run"), \
         patch("process_video.setup_logger"):
        from process_video import VideoProcessor, MODE_DETECT_RALLIES
        p = VideoProcessor("fake.mp4", {}, mode=MODE_DETECT_RALLIES)
        MockAnalytics.assert_not_called()
        MockRallyDetector.assert_called_once()
        assert hasattr(p, "rally_detector")
        assert not hasattr(p, "analytics")
```

- [ ] **Step 7.2: Run to verify failure**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && .venv/Scripts/python -m pytest tests/test_video_processor_modes.py::test_mode_detect_rallies_constant tests/test_video_processor_modes.py::test_detect_rallies_has_rally_detector_no_analytics -v
```

Expected: 2 FAILED

- [ ] **Step 7.3: Add `MODE_DETECT_RALLIES` to `process_video.py`**

In `process_video.py`, after the existing mode constants (around line 48):

```python
MODE_DETECT_RALLIES = "detect_rallies"
# Produces: rally_report.json — one entry per detected rally with start/end frames.
# No video output. Fast pass.
```

Add the import at the top of `process_video.py` (after the existing local imports):

```python
from rally_detector import RallyDetector
```

In `VideoProcessor.__init__`, add a new branch after the serve pipeline branch (around line 136):

```python
# ── Rally detection pipeline — Detect Rallies only ───────────────────
if self.mode == MODE_DETECT_RALLIES:
    self.serve_detector  = ServeDetector()
    self.rally_detector  = RallyDetector(fps=30)   # fps updated in process_video
```

- [ ] **Step 7.4: Run tests**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && .venv/Scripts/python -m pytest tests/test_video_processor_modes.py -v
```

Expected: 5 PASSED (existing 3 + 2 new)

- [ ] **Step 7.5: Wire per-frame update and finalization**

In `VideoProcessor.process_video()`, inside the main frame loop, add after the existing `elif self.mode == MODE_DETECT_SERVE:` block:

```python
# ── Detect Rallies ─────────────────────────────────────────────────
elif self.mode == MODE_DETECT_RALLIES:
    serve_event = self.serve_detector.update(frame_idx, frame, ball_proj, players)
    # Derive net_y from cached court keypoints if available
    net_y = None
    court_bounds = None
    if self._cached_kps is not None and self._cached_Hmg is not None:
        pts = np.array(self._cached_kps, dtype=np.float32).reshape(-1, 1, 2)
        proj_kps = cv2.perspectiveTransform(pts, self._cached_Hmg).reshape(-1, 2)
        if proj_kps.shape[0] >= 12:
            # Net Y = average of row1 and row2 midpoints (same logic as analytics.py)
            row1_y = float(np.mean(proj_kps[3:6, 1]))
            row2_y = float(np.mean(proj_kps[6:9, 1]))
            net_y = (row1_y + row2_y) / 2.0
            xmin = float(np.min(proj_kps[:, 0]))
            xmax = float(np.max(proj_kps[:, 0]))
            ymin = float(np.min(proj_kps[:, 1]))
            ymax = float(np.max(proj_kps[:, 1]))
            court_bounds = (xmin, ymin, xmax, ymax)
    ball_y2 = float(ball_bbox[3]) if ball_bbox is not None else None
    self.rally_detector.update(frame_idx, ball_proj, ball_y2, serve_event, net_y, court_bounds)
```

In `VideoProcessor.process_video()`, in the `finally` block, add after the serve finalization section:

```python
# ── Detect Rallies finalization ────────────────────────────────────
elif self.mode == MODE_DETECT_RALLIES:
    # Finalize any open rally at end of video
    if self.rally_detector.state != RallyDetector.IDLE:
        self.rally_detector._finalize_rally(frame_idx, "fault")
    # Update fps now that we know it
    self.rally_detector._fps = fps
    self._save_rally_report()
```

Also update the fps on the `RallyDetector` after reading video meta. In `process_video()`, after `total_frames, src_w, src_h, fps = self._read_video_meta(cap)`, add:

```python
if self.mode == MODE_DETECT_RALLIES:
    self.rally_detector._fps = fps
```

Add `_save_rally_report()` method to `VideoProcessor`:

```python
def _save_rally_report(self) -> None:
    """Write rally_report.json to the run output directory."""
    import json
    from datetime import datetime

    report = self.rally_detector.get_report(video_path=self.video_path)
    report["generated"] = datetime.now().isoformat()

    report_path = os.path.join(self.output_dir, "rally_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.bind(path=report_path, total_rallies=report["total_rallies"]).info("rally_report_saved")
    print(f"Rally report saved: {report_path}")
```

- [ ] **Step 7.6: Run all tests**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && .venv/Scripts/python -m pytest tests/ -v
```

Expected: All existing tests + 2 new PASSED, no regressions

- [ ] **Step 7.7: Commit**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && git add process_video.py tests/test_video_processor_modes.py && git commit -m "feat: add MODE_DETECT_RALLIES to VideoProcessor"
```

---

## Task 8: `MODE_CLIP_FROM_REPORT` in `process_video.py`

**Files:**
- Modify: `process_video.py` — add constant, init branch, `_clip_from_report()`
- Modify: `tests/test_video_processor_modes.py`

- [ ] **Step 8.1: Write failing tests**

```python
# append to tests/test_video_processor_modes.py
from process_video import MODE_CLIP_FROM_REPORT

def test_mode_clip_from_report_constant():
    assert MODE_CLIP_FROM_REPORT == "clip_from_report"

def test_clip_from_report_init_no_detectors():
    """CLIP_FROM_REPORT mode: no detectors or analytics instantiated."""
    with patch("process_video.BallTracker") as MockBT, \
         patch("process_video.PlayerTracker") as MockPT, \
         patch("process_video.CourtDetector") as MockCD, \
         patch("process_video.Analytics") as MockAnalytics, \
         patch("process_video.ServeDetector") as MockSD, \
         patch("process_video.OllamaServeAnalyzer") as MockAnalyzer, \
         patch("process_video.RallyDetector") as MockRD, \
         patch("process_video.VideoProcessor._make_output_dir", return_value="/tmp/run"), \
         patch("process_video.setup_logger"):
        from process_video import VideoProcessor, MODE_CLIP_FROM_REPORT
        p = VideoProcessor("fake.mp4", {}, mode=MODE_CLIP_FROM_REPORT,
                           rally_report_path="/tmp/rally_report.json")
        MockAnalytics.assert_not_called()
        MockBT.assert_not_called()
        MockPT.assert_not_called()
        MockCD.assert_not_called()
        MockSD.assert_not_called()
        MockRD.assert_not_called()
        assert p.rally_report_path == "/tmp/rally_report.json"
```

- [ ] **Step 8.2: Run to verify failure**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && .venv/Scripts/python -m pytest tests/test_video_processor_modes.py::test_mode_clip_from_report_constant tests/test_video_processor_modes.py::test_clip_from_report_init_no_detectors -v
```

Expected: 2 FAILED

- [ ] **Step 8.3: Add `MODE_CLIP_FROM_REPORT` and update `__init__`**

In `process_video.py`, after `MODE_DETECT_RALLIES`:

```python
MODE_CLIP_FROM_REPORT = "clip_from_report"
# Reads: rally_report.json (path provided at init)
# Produces: rally_01.mp4, rally_02.mp4, … — one clip per rally in the report.
```

Update `VideoProcessor.__init__` signature to accept optional `rally_report_path`:

```python
def __init__(self, video_path: str, filters: dict, mode: str = MODE_VIDEO_ANALYSIS,
             rally_report_path: Optional[str] = None):
    self.video_path = video_path
    self.filters = self._apply_default_filters(filters)
    self.mode = mode
    self.rally_report_path = rally_report_path
```

Add branch at the end of the detector init section in `__init__` — `MODE_CLIP_FROM_REPORT` skips all detector construction:

```python
# ── Clip from report — no detectors needed ────────────────────────
if self.mode == MODE_CLIP_FROM_REPORT:
    pass   # all work done in process_video() via _clip_from_report()
```

Also guard the existing detector construction so it only runs for modes that need it. Replace the three existing detector instantiation lines:

```python
# ── Detectors used by detection modes ──────────────────────────────
if self.mode != MODE_CLIP_FROM_REPORT:
    self.ball_tracker   = BallTracker(os.path.join(MODELS_DIR, "ball_tracking.pt"))
    self.player_tracker = PlayerTracker(os.path.join(MODELS_DIR, "player_tracking.pt"))
    self.court_mapper   = CourtDetector(os.path.join(MODELS_DIR, "court_detection.pt"))
    self._cached_kps = None
    self._cached_Hmg = None
```

- [ ] **Step 8.4: Add `_clip_from_report()` method and wire it in `process_video()`**

Add method to `VideoProcessor`:

```python
def _clip_from_report(self, progress_callback=None) -> str:
    """Read rally_report.json and write one clip per rally. Returns output dir."""
    import json

    if not self.rally_report_path or not os.path.isfile(self.rally_report_path):
        raise RuntimeError(f"rally_report_path not found: {self.rally_report_path}")

    with open(self.rally_report_path) as f:
        report = json.load(f)

    rallies = report.get("rallies", [])
    fps = int(report.get("fps", 30))
    buffer = fps * 2   # 2-second padding on each side

    cap = self._open_capture(self.video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for idx, rally in enumerate(rallies, start=1):
        start_frame = max(0, rally["start_frame"] - buffer)
        end_frame   = min(total_frames - 1, rally["end_frame"] + buffer)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        clip_path = os.path.join(self.output_dir, f"rally_{idx:02d}.mp4")
        try:
            writer = cv2.VideoWriter(clip_path, FOURCC, fps, (w, h))
            writer.write(frame)
            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
            writer.release()
            duration_s = round((end_frame - start_frame) / fps, 1)
            logger.bind(rally_num=idx, clip_path=clip_path, duration_s=duration_s).info("rally_clip_saved")
            print(f"Saved rally clip: {clip_path} ({duration_s}s)")
        except Exception:
            logger.exception(f"Failed to write rally clip {idx}")

        self._report_progress(progress_callback, idx, len(rallies))

    cap.release()
    return self.output_dir
```

In `process_video()`, add an early-return path at the top of the method body (before `cap = self._open_capture(...)`):

```python
def process_video(self, progress_callback=None) -> str:
    if self.mode == MODE_CLIP_FROM_REPORT:
        return self._clip_from_report(progress_callback)

    cap = self._open_capture(self.video_path)
    # ... rest of existing method unchanged
```

- [ ] **Step 8.5: Run all tests**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && .venv/Scripts/python -m pytest tests/ -v
```

Expected: All PASSED, no regressions

- [ ] **Step 8.6: Commit**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && git add process_video.py tests/test_video_processor_modes.py && git commit -m "feat: add MODE_CLIP_FROM_REPORT to VideoProcessor"
```

---

## Task 9: UI — "Detect Rallies" mode with two-step confirm flow

**Files:**
- Modify: `main.py`

No unit tests for the UI — manual verification against `sample3.mp4`.

- [ ] **Step 9.1: Add mode constant, description, and mode_map entry**

In `main.py`, update the imports at the top:

```python
from process_video import (
    VideoProcessor,
    MODE_VIDEO_ANALYSIS,
    MODE_SPLIT_RALLIES,
    MODE_DETECT_SERVE,
    MODE_DETECT_RALLIES,
    MODE_CLIP_FROM_REPORT,
)
```

Add to `MODE_DESCRIPTIONS`:

```python
MODE_DESCRIPTIONS = {
    "Video Analysis":   "Produces an annotated video with overlays and analytics",
    "Split Rallies":    "Finds long rallies and saves each one as a clip",
    "Detect Serve":     "Scores serves using AI vision — fastest mode",
    "Detect Rallies":   "Detect rallies, review summary, then clip",
}
```

Update `mode_selector` values in `_build_mode_card()`:

```python
self.mode_selector = ctk.CTkSegmentedButton(
    card,
    values=["Video Analysis", "Split Rallies", "Detect Serve", "Detect Rallies"],
    ...
)
```

- [ ] **Step 9.2: Add rally report state variables and scrollable list widget**

Add instance variables to `App.__init__` after `self._reveal_id = None`:

```python
self._rally_report_path: Optional[str] = None
self._clip_btn = None
self._rally_list_box = None
```

Add import at top of `main.py`:

```python
from typing import Optional
```

- [ ] **Step 9.3: Add `_set_state_report_ready()` to show summary and Clip button**

Add method to `App`:

```python
def _set_state_report_ready(self, rally_report_path: str, rallies: list, out_dir: str):
    """Show rally summary after MODE_DETECT_RALLIES completes."""
    import json

    self._rally_report_path = rally_report_path
    self.browse_btn.configure(state="normal")
    self.mode_selector.configure(state="normal")
    self.process_btn.configure(state="normal")
    self.progress_bar.set(1.0)
    self.status_label.configure(text="Analysis complete — review rallies below", text_color=SUCCESS)
    self.out_dir = out_dir

    count = len(rallies)
    if count == 0:
        badge_text = "No rallies detected"
    else:
        durations = [r["duration_sec"] for r in rallies]
        avg_s = sum(durations) / len(durations)
        max_s = max(durations)
        badge_text = f"{count} {'rally' if count == 1 else 'rallies'} · avg {avg_s:.1f}s · longest {max_s:.1f}s"

    self.rally_badge_label.configure(text=badge_text)
    self.output_path_label.configure(text=out_dir)

    # Populate rally list textbox
    if self._rally_list_box is not None:
        self._rally_list_box.configure(state="normal")
        self._rally_list_box.delete("1.0", "end")
    else:
        self._rally_list_box = ctk.CTkTextbox(
            self.results_card,
            height=120,
            font=ctk.CTkFont(size=11, family="Courier"),
            fg_color=BG,
            text_color=BODY_TEXT,
            state="normal",
        )
        self._rally_list_box.grid(row=4, column=0, padx=16, pady=(0, 8), sticky="ew")

    lines = []
    for r in rallies:
        flag = "" if r["two_bounce_complete"] else " ⚠"
        lines.append(
            f"Rally {r['rally_num']:>2}  {r['start_sec']:6.1f}s – {r['end_sec']:6.1f}s"
            f"  ({r['duration_sec']:.1f}s)  {r['end_reason']}{flag}"
        )
    self._rally_list_box.insert("end", "\n".join(lines))
    self._rally_list_box.configure(state="disabled")

    # Clip Rallies button
    if self._clip_btn is None:
        self._clip_btn = ctk.CTkButton(
            self.results_card,
            text="Clip Rallies",
            fg_color=SUCCESS,
            text_color="black",
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=8,
            command=self._on_clip_rallies,
        )
        self._clip_btn.grid(row=5, column=0, padx=16, pady=(0, 16), sticky="ew")
    else:
        self._clip_btn.configure(state="normal")

    self._reveal_results_tall()
```

- [ ] **Step 9.4: Add `_set_state_clip_done()` and `_on_clip_rallies()`**

```python
def _on_clip_rallies(self):
    if not self._rally_report_path:
        return
    self._clip_btn.configure(state="disabled")
    self.status_label.configure(text="Clipping rallies…", text_color=MUTED_TEXT)
    threading.Thread(target=self._run_clip_from_report, daemon=True).start()

def _run_clip_from_report(self):
    def update_progress(value):
        self.after(0, lambda: self.progress_bar.set(value))
    try:
        processor = VideoProcessor(
            self.video_path, {}, mode=MODE_CLIP_FROM_REPORT,
            rally_report_path=self._rally_report_path,
        )
        processor.process_video(progress_callback=update_progress)
        out_dir = processor.output_dir
        self.after(0, lambda od=out_dir: self._set_state_clip_done(od))
    except Exception as e:
        err = str(e)
        self.after(0, lambda: self._set_state_error(err))

def _set_state_clip_done(self, out_dir: str):
    self.progress_bar.set(1.0)
    self.status_label.configure(text="Clips saved!", text_color=SUCCESS)
    self.rally_badge_label.configure(
        text=self.rally_badge_label.cget("text").split("·")[0].strip() + " — clips saved"
    )
    self.output_path_label.configure(text=out_dir)
    self.out_dir = out_dir
    if self._clip_btn:
        self._clip_btn.configure(state="disabled", text="Clipped")
```

- [ ] **Step 9.5: Add `_reveal_results_tall()` and wire `_process_video` for new mode**

```python
def _reveal_results_tall(self, step=0):
    """Taller reveal for rally report (needs room for list + clip button)."""
    target = 280
    increment = 20
    delay = 20
    current = step * increment
    if current <= target:
        self.results_card.configure(height=current)
        self._reveal_id = self.after(delay, lambda: self._reveal_results_tall(step + 1))
    else:
        self._reveal_id = None
```

In `_process_video()`, update `mode_map` and add handling for `MODE_DETECT_RALLIES`:

```python
mode_map = {
    "Video Analysis":  MODE_VIDEO_ANALYSIS,
    "Split Rallies":   MODE_SPLIT_RALLIES,
    "Detect Serve":    MODE_DETECT_SERVE,
    "Detect Rallies":  MODE_DETECT_RALLIES,
}
mode = mode_map[mode_label]
processor = VideoProcessor(self.video_path, filters, mode=mode)
processor.process_video(progress_callback=update_progress)

if mode == MODE_DETECT_RALLIES:
    import json, os
    report_path = os.path.join(processor.output_dir, "rally_report.json")
    with open(report_path) as f:
        report = json.load(f)
    rallies = report.get("rallies", [])
    self.after(
        0,
        lambda rp=report_path, rv=rallies, od=processor.output_dir:
            self._set_state_report_ready(rp, rv, od),
    )
    return

# existing completion path for other modes
rally_count   = len(processor.analytics._long_rallies) if hasattr(processor, "analytics") else 0
serve_results = processor.serve_analyzer.get_results() if hasattr(processor, "serve_analyzer") else []
serve_count   = len(serve_results)
serve_avg     = round(sum(r.score for r in serve_results) / serve_count, 1) if serve_count > 0 else 0
out_dir = processor.output_dir

self.after(
    0,
    lambda rc=rally_count, sc=serve_count, sa=serve_avg, od=out_dir, m=mode:
        self._set_state_complete(rc, sc, sa, od, m),
)
```

- [ ] **Step 9.6: Run all automated tests**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && .venv/Scripts/python -m pytest tests/ -v
```

Expected: All PASSED

- [ ] **Step 9.7: Manual smoke test**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && .venv/Scripts/python main.py
```

1. Browse to `sample3.mp4`
2. Select "Detect Rallies" mode
3. Click "Process Video" — progress bar runs
4. Results card expands showing rally list and "Clip Rallies" button
5. Verify rally list shows entries with timestamps and end reasons
6. Click "Clip Rallies" — progress bar runs again
7. Verify "Clips saved!" status and clips appear in output folder

- [ ] **Step 9.8: Commit**

```bash
cd c:/apps/pickleball/Pickleball-Analytics && git add main.py && git commit -m "feat: add Detect Rallies mode with two-step confirm UI"
```

---

## Self-Review

**Spec coverage:**
- ✅ `RallyDetector` FSM (5 states) — Tasks 1, 6
- ✅ Court bounds sanity check (tennis/pickleball shared court) — Task 2
- ✅ Bounce detection + net-crossing fallback — Task 4
- ✅ Three end conditions (out / net / fault) — Task 5
- ✅ `rally_report.json` schema — Task 7 (`get_report()`)
- ✅ `MODE_DETECT_RALLIES` — Task 7
- ✅ `MODE_CLIP_FROM_REPORT` — Task 8
- ✅ UI two-step flow (analysis → confirm → clip) — Task 9
- ✅ `two_bounce_complete` flag in report — Tasks 4, 6
- ✅ `end_reason` in report — Tasks 5, 6

**Placeholder scan:** No TBDs or "implement later" in any step. All code blocks are complete.

**Type consistency:**
- `RallyDetector.update()` signature used consistently across Tasks 6, 7
- `RallyRecord` properties (`start_sec`, `end_sec`, `duration_sec`) defined in Task 1, used in Task 7
- `_validate_and_set_court_bounds()` defined in Task 2, called in Task 6
- `_update_bounce_detection()` defined in Task 4, called in Task 6
- `_check_end_condition()` defined in Task 5, called in Task 6
- `MODE_DETECT_RALLIES` / `MODE_CLIP_FROM_REPORT` defined in Task 7/8, imported in Task 9
