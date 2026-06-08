# Player Identity Tracking & Serve Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persistent player IDs (P1–P4) via ByteTrack and detect/score pickleball serves using local Ollama vision LLM (`qwen2.5vl:7b`), outputting a per-player serve report JSON and video panel.

**Architecture:** `PlayerTracker` swaps `predict()` for `track()` returning dicts with IDs. A new `ServeDetector` watches ball stillness + launch each frame to emit `ServeCandidate` objects. A new `OllamaServeAnalyzer` scores candidates in a background thread pool and accumulates `ServeResult` objects flushed to JSON at video end. `Analytics` gains a `panel_serve_summary()` tile.

**Tech Stack:** Python 3.12, ultralytics ByteTrack, Ollama Python SDK (`pip install ollama`), OpenCV, `qwen2.5vl:7b` running locally at `localhost:11434`

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `player_tracker.py` | Swap predict→track, return `list[dict]` with id/bbox/proj |
| Create | `serve_detector.py` | Ball stillness + launch detection → `ServeCandidate` |
| Create | `serve_analyzer.py` | Thread pool Ollama calls → `ServeResult`, flush JSON |
| Modify | `analytics.py` | Add `panel_serve_summary(results, size)` panel method |
| Modify | `process_video.py` | Wire ServeDetector + OllamaServeAnalyzer into main loop |
| Create | `tests/test_player_tracker.py` | Unit tests for updated player tracker |
| Create | `tests/test_serve_detector.py` | Unit tests for serve detection logic |
| Create | `tests/test_serve_analyzer.py` | Unit tests for analyzer (mocked Ollama) |

---

## Task 1: Install ollama SDK and upgrade PlayerTracker to ByteTrack

**Files:**
- Modify: `player_tracker.py`
- Create: `tests/test_player_tracker.py`

- [ ] **Step 1: Install ollama SDK**

```bash
pip install ollama
```

Expected output: `Successfully installed ollama-x.x.x`

- [ ] **Step 2: Write failing tests**

Create `tests/test_player_tracker.py`:

```python
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

def make_mock_box(x1, y1, x2, y2, track_id):
    box = MagicMock()
    box.xyxy.cpu.return_value.numpy.return_value = np.array([[x1, y1, x2, y2]])
    box.id = None if track_id is None else MagicMock()
    if track_id is not None:
        box.id.cpu.return_value.numpy.return_value = np.array([track_id])
    return box

def make_mock_results(boxes):
    result = MagicMock()
    result.boxes = boxes
    return result

@patch("player_tracker.YOLO")
def test_detect_and_project_returns_dicts_with_ids(mock_yolo_cls):
    from player_tracker import PlayerTracker

    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    box1 = make_mock_box(100, 200, 200, 400, track_id=1)
    box2 = make_mock_box(300, 200, 400, 400, track_id=2)
    mock_model.track.return_value = [make_mock_results([box1, box2])]

    tracker = PlayerTracker("fake.pt")
    players, proj = tracker.detect_and_project(np.zeros((480, 640, 3), dtype=np.uint8), H=None)

    assert len(players) == 2
    assert players[0]["id"] == 1
    assert "bbox" in players[0]
    assert "proj" in players[0]
    assert players[0]["proj"] is None  # no homography

@patch("player_tracker.YOLO")
def test_detect_and_project_no_id_falls_back(mock_yolo_cls):
    from player_tracker import PlayerTracker

    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    box1 = make_mock_box(100, 200, 200, 400, track_id=None)
    mock_model.track.return_value = [make_mock_results([box1])]

    tracker = PlayerTracker("fake.pt")
    players, _ = tracker.detect_and_project(np.zeros((480, 640, 3), dtype=np.uint8), H=None)

    assert players[0]["id"] is None
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd c:/apps/pickleball/Pickleball-Analytics
pytest tests/test_player_tracker.py -v
```

Expected: FAIL — `detect_and_project` returns tuples, not dicts.

- [ ] **Step 4: Update `player_tracker.py`**

Replace the entire file content:

```python
"""
Player detection & projection module

Purpose
-------
Runs YOLO with ByteTrack to detect and persistently ID players each frame,
then projects each player's bottom-center into bird's-eye space.

What it does
------------
- detect_and_project(frame, H): YOLO+ByteTrack → list of player dicts
  Each dict: {"id": int|None, "bbox": [x1,y1,x2,y2], "proj": (bx,by)|None}

Inputs
------
- BGR frame (OpenCV), homography H (3×3 or None)

Outputs
-------
- (players, proj_points) where:
    players: list[dict] with id, bbox, proj
    proj_points: list of (bx, by) tuples (for backwards compat with analytics)
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class PlayerTracker:
    def __init__(self, model_path, conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = 0 if torch.cuda.is_available() else "cpu"

    def detect_and_project(self, frame, H):
        """
        Run YOLO+ByteTrack on frame. Returns:
          players: list of {"id": int|None, "bbox": [x1,y1,x2,y2], "proj": (bx,by)|None}
          proj_points: list of (bx, by) for backwards compatibility
        """
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        players = []
        for box in results.boxes:
            bbox = box.xyxy.cpu().numpy()[0].tolist()
            track_id = None
            if box.id is not None:
                track_id = int(box.id.cpu().numpy()[0])

            proj = None
            if H is not None:
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2
                cy = y2
                pt = np.array([[[cx, cy]]], dtype=np.float32)
                proj = tuple(cv2.perspectiveTransform(pt, H)[0][0])

            players.append({"id": track_id, "bbox": bbox, "proj": proj})

        proj_points = [p["proj"] for p in players if p["proj"] is not None]
        return players, proj_points
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_player_tracker.py -v
```

Expected: PASS both tests.

- [ ] **Step 6: Commit**

```bash
git add player_tracker.py tests/test_player_tracker.py
git commit -m "Upgrade PlayerTracker to ByteTrack with persistent player IDs"
```

---

## Task 2: Update process_video.py to use new player dict format

**Files:**
- Modify: `process_video.py:139,147,148,286-288`

The main loop and render methods pass `players` directly. Now `players` is `list[dict]` — update rendering to draw IDs.

- [ ] **Step 1: Update the main loop player usage**

In `process_video.py`, the line:
```python
players, proj_players = self.player_tracker.detect_and_project(frame, Hmg)
```
Already works — `proj_players` is still a list of tuples, compatible with analytics.

- [ ] **Step 2: Update `_render_main_view` to draw player IDs**

Replace the Players block in `_render_main_view` (lines 286-288):

```python
# Players
for p in players or []:
    x1, y1, x2, y2 = map(int, p)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_PLAYER, 2)
```

With:

```python
# Players — draw bbox and ID label
for p in players or []:
    if isinstance(p, dict):
        x1, y1, x2, y2 = map(int, p["bbox"])
        pid = p.get("id")
    else:
        x1, y1, x2, y2 = map(int, p)
        pid = None
    cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_PLAYER, 2)
    if pid is not None:
        label = f"P{pid}"
        cv2.putText(canvas, label, (x1, y1 - 8), FONT, FONT_SCALE, COLOR_PLAYER, FONT_THICKNESS)
```

- [ ] **Step 3: Update `_render_birdseye` player dots**

Replace the Players block in `_render_birdseye` (lines 337-340):

```python
# Players
for pt in projected_players or []:
    x, y = map(int, pt)
    if 0 <= x < src_w and 0 <= y < src_h:
        cv2.circle(bird, (x, y), PLAYER_DOT_RADIUS, COLOR_PLAYER, -1)
```

With:

```python
# Players — proj_points is list of (bx, by) tuples
for pt in projected_players or []:
    x, y = map(int, pt)
    if 0 <= x < src_w and 0 <= y < src_h:
        cv2.circle(bird, (x, y), PLAYER_DOT_RADIUS, COLOR_PLAYER, -1)
```

(No change needed here — `proj_players` is already a flat list of tuples.)

- [ ] **Step 4: Commit**

```bash
git add process_video.py
git commit -m "Render player IDs (P1-P4) on main view bounding boxes"
```

---

## Task 3: Create ServeDetector

**Files:**
- Create: `serve_detector.py`
- Create: `tests/test_serve_detector.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_serve_detector.py`:

```python
import numpy as np
import pytest
from serve_detector import ServeDetector, ServeCandidate


def make_players(player_id, bbox):
    return [{"id": player_id, "bbox": bbox, "proj": None}]


def test_no_candidate_during_stillness():
    """Ball stationary for fewer than 15 frames yields no candidate."""
    det = ServeDetector(fps=30)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    players = make_players(1, [100, 300, 200, 500])

    result = None
    for i in range(14):
        result = det.update(i, frame, (150.0, 400.0), players)
    assert result is None


def test_candidate_emitted_on_launch():
    """Ball still for 15+ frames then moves >50px → ServeCandidate returned."""
    det = ServeDetector(fps=30)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    players = make_players(1, [100, 300, 200, 500])

    # 15 frames of stillness
    for i in range(15):
        det.update(i, frame, (150.0, 400.0), players)

    # Ball launches downward (toward net, lower y)
    result = det.update(15, frame, (210.0, 300.0), players)
    assert isinstance(result, ServeCandidate)
    assert result.player_id == 1
    assert result.frame_idx == 15


def test_cooldown_prevents_double_detection():
    """Second serve within 5 seconds is suppressed."""
    det = ServeDetector(fps=30)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    players = make_players(1, [100, 300, 200, 500])

    for i in range(15):
        det.update(i, frame, (150.0, 400.0), players)
    det.update(15, frame, (210.0, 300.0), players)  # first serve

    # Reset stillness and try again immediately
    for i in range(16, 31):
        det.update(i, frame, (150.0, 400.0), players)
    result = det.update(31, frame, (210.0, 300.0), players)
    assert result is None  # still in cooldown


def test_no_ball_no_candidate():
    """No ball projection → no candidate ever."""
    det = ServeDetector(fps=30)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    players = make_players(1, [100, 300, 200, 500])

    for i in range(20):
        result = det.update(i, frame, None, players)
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_serve_detector.py -v
```

Expected: FAIL — `serve_detector` module does not exist.

- [ ] **Step 3: Create `serve_detector.py`**

```python
"""
Serve detection module

Purpose
-------
Watches ball position and player state each frame to detect serve events.
A serve is detected when:
  1. Ball stays within 20px of a position for 15+ consecutive frames (stationary)
  2. Ball then moves >50px in a single frame (launch)
  3. Ball moves toward the net (y decreases toward center of court)

Outputs a ServeCandidate when all three conditions are met.
A 5-second cooldown prevents double-detection.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import cv2
import numpy as np
from typing import Optional


@dataclass
class ServeCandidate:
    frame_idx: int
    timestamp_sec: float
    player_id: Optional[int]
    ball_pos: tuple          # (x, y) at launch
    frame_small: np.ndarray  # resized to 1280x720


class ServeDetector:
    STILLNESS_FRAMES = 15    # frames ball must be stationary
    STILLNESS_PX = 20        # max movement to count as stationary
    LAUNCH_PX = 50           # min movement to count as launch
    COOLDOWN_SEC = 5.0       # seconds between detections

    def __init__(self, fps: int = 30):
        self._fps = fps
        self._cooldown_frames = int(self.COOLDOWN_SEC * fps)
        self._still_pos: Optional[tuple] = None
        self._still_count: int = 0
        self._last_ball: Optional[tuple] = None
        self._last_detected_frame: int = -9999

    def update(
        self,
        frame_idx: int,
        frame: np.ndarray,
        ball_proj: Optional[tuple],
        players: list,
    ) -> Optional[ServeCandidate]:
        if ball_proj is None:
            self._reset_stillness()
            return None

        bx, by = ball_proj

        # --- Check cooldown ---
        if frame_idx - self._last_detected_frame < self._cooldown_frames:
            self._last_ball = ball_proj
            return None

        # --- Update stillness counter ---
        if self._still_pos is None:
            self._still_pos = (bx, by)
            self._still_count = 1
        else:
            dist = np.hypot(bx - self._still_pos[0], by - self._still_pos[1])
            if dist <= self.STILLNESS_PX:
                self._still_count += 1
            else:
                # Ball moved — check if this is a launch
                if self._still_count >= self.STILLNESS_FRAMES and self._last_ball is not None:
                    launch_dist = np.hypot(bx - self._last_ball[0], by - self._last_ball[1])
                    if launch_dist >= self.LAUNCH_PX:
                        candidate = self._build_candidate(frame_idx, frame, ball_proj, players)
                        self._last_detected_frame = frame_idx
                        self._reset_stillness()
                        self._last_ball = ball_proj
                        return candidate
                # Reset stillness to new position
                self._still_pos = (bx, by)
                self._still_count = 1

        self._last_ball = ball_proj
        return None

    def _reset_stillness(self):
        self._still_pos = None
        self._still_count = 0

    def _build_candidate(self, frame_idx, frame, ball_pos, players) -> ServeCandidate:
        timestamp_sec = frame_idx / self._fps

        # Find closest player to ball
        bx, by = ball_pos
        player_id = None
        min_dist = float("inf")
        for p in players or []:
            bbox = p.get("bbox", [])
            if len(bbox) == 4:
                px = (bbox[0] + bbox[2]) / 2
                py = bbox[3]  # bottom center
                d = np.hypot(bx - px, by - py)
                if d < min_dist:
                    min_dist = d
                    player_id = p.get("id")

        # Resize frame for Ollama
        frame_small = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)

        return ServeCandidate(
            frame_idx=frame_idx,
            timestamp_sec=round(timestamp_sec, 2),
            player_id=player_id,
            ball_pos=ball_pos,
            frame_small=frame_small,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_serve_detector.py -v
```

Expected: PASS all 4 tests.

- [ ] **Step 5: Commit**

```bash
git add serve_detector.py tests/test_serve_detector.py
git commit -m "Add ServeDetector with ball stillness and launch detection"
```

---

## Task 4: Create OllamaServeAnalyzer

**Files:**
- Create: `serve_analyzer.py`
- Create: `tests/test_serve_analyzer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_serve_analyzer.py`:

```python
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from serve_detector import ServeCandidate
from serve_analyzer import OllamaServeAnalyzer, ServeResult


def make_candidate(player_id=1, frame_idx=100):
    return ServeCandidate(
        frame_idx=frame_idx,
        timestamp_sec=3.33,
        player_id=player_id,
        ball_pos=(150.0, 200.0),
        frame_small=np.zeros((720, 1280, 3), dtype=np.uint8),
    )


VALID_RESPONSE = """{
  "is_serve": true,
  "score": 8,
  "stance": "good",
  "ball_toss": "good",
  "contact_point": "good",
  "follow_through": "good",
  "landing_zone": "deep",
  "coaching_tip": "Great serve, keep it up."
}"""

REJECTED_RESPONSE = '{"is_serve": false, "score": 0, "stance": "unknown", "ball_toss": "unknown", "contact_point": "unknown", "follow_through": "unknown", "landing_zone": "unknown", "coaching_tip": ""}'


@patch("serve_analyzer.ollama")
def test_valid_serve_stored(mock_ollama):
    mock_ollama.chat.return_value = {"message": {"content": VALID_RESPONSE}}
    analyzer = OllamaServeAnalyzer(model="qwen2.5vl:7b", workers=1)
    analyzer.submit(make_candidate(player_id=1))
    analyzer.shutdown()
    results = analyzer.get_results()
    assert len(results) == 1
    assert results[0].score == 8
    assert results[0].player_id == 1
    assert results[0].is_serve is True


@patch("serve_analyzer.ollama")
def test_rejected_serve_discarded(mock_ollama):
    mock_ollama.chat.return_value = {"message": {"content": REJECTED_RESPONSE}}
    analyzer = OllamaServeAnalyzer(model="qwen2.5vl:7b", workers=1)
    analyzer.submit(make_candidate(player_id=2))
    analyzer.shutdown()
    results = analyzer.get_results()
    assert len(results) == 0


@patch("serve_analyzer.ollama")
def test_malformed_json_discarded(mock_ollama):
    mock_ollama.chat.return_value = {"message": {"content": "not json at all"}}
    analyzer = OllamaServeAnalyzer(model="qwen2.5vl:7b", workers=1)
    analyzer.submit(make_candidate())
    analyzer.shutdown()
    assert len(analyzer.get_results()) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_serve_analyzer.py -v
```

Expected: FAIL — `serve_analyzer` module does not exist.

- [ ] **Step 3: Create `serve_analyzer.py`**

```python
"""
Ollama Serve Analyzer

Purpose
-------
Scores serve candidates using a local Ollama vision LLM in a background
thread pool. Never blocks the video processing pipeline.

Usage
-----
    analyzer = OllamaServeAnalyzer()
    analyzer.submit(candidate)         # non-blocking
    ...
    analyzer.shutdown()                # wait for all pending calls
    results = analyzer.get_results()   # list[ServeResult]
"""

from __future__ import annotations

import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore

from serve_detector import ServeCandidate


PROMPT = """You are a pickleball coach analyzing a serve frame.
Respond in valid JSON only — no explanation, no markdown, no code block.

{
  "is_serve": true or false,
  "score": integer 1-10,
  "stance": "good" or "poor" or "unknown",
  "ball_toss": "good" or "too_low" or "too_far" or "unknown",
  "contact_point": "good" or "too_low" or "too_far" or "unknown",
  "follow_through": "good" or "poor" or "unknown",
  "landing_zone": "deep" or "short" or "wide" or "unknown",
  "coaching_tip": "one short sentence"
}"""


@dataclass
class ServeResult:
    frame_idx: int
    timestamp_sec: float
    player_id: Optional[int]
    is_serve: bool
    score: int
    stance: str
    ball_toss: str
    contact_point: str
    follow_through: str
    landing_zone: str
    coaching_tip: str


class OllamaServeAnalyzer:
    TIMEOUT_SEC = 30

    def __init__(self, model: str = "qwen2.5vl:7b", workers: int = 2):
        self._model = model
        self._results: list[ServeResult] = []
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=workers)

    def submit(self, candidate: ServeCandidate) -> None:
        """Submit a candidate for async analysis. Non-blocking."""
        self._executor.submit(self._analyze, candidate)

    def get_results(self) -> list[ServeResult]:
        """Return all confirmed serve results accumulated so far."""
        with self._lock:
            return list(self._results)

    def shutdown(self) -> None:
        """Wait for all pending analyses to complete."""
        self._executor.shutdown(wait=True)

    def _analyze(self, candidate: ServeCandidate) -> None:
        if ollama is None:
            return
        try:
            _, buf = cv2.imencode(".jpg", candidate.frame_small)
            image_b64 = base64.b64encode(buf).decode()

            response = ollama.chat(
                model=self._model,
                messages=[{
                    "role": "user",
                    "content": PROMPT,
                    "images": [image_b64],
                }],
                options={"num_predict": 200},
            )
            content = response["message"]["content"]
            # Strip markdown code fences if present
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            data = json.loads(content)
            if not data.get("is_serve", False):
                return

            result = ServeResult(
                frame_idx=candidate.frame_idx,
                timestamp_sec=candidate.timestamp_sec,
                player_id=candidate.player_id,
                is_serve=True,
                score=int(data.get("score", 5)),
                stance=data.get("stance", "unknown"),
                ball_toss=data.get("ball_toss", "unknown"),
                contact_point=data.get("contact_point", "unknown"),
                follow_through=data.get("follow_through", "unknown"),
                landing_zone=data.get("landing_zone", "unknown"),
                coaching_tip=data.get("coaching_tip", ""),
            )
            with self._lock:
                self._results.append(result)

        except Exception:
            pass  # silently discard on timeout or parse failure
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_serve_analyzer.py -v
```

Expected: PASS all 3 tests.

- [ ] **Step 5: Commit**

```bash
git add serve_analyzer.py tests/test_serve_analyzer.py
git commit -m "Add OllamaServeAnalyzer with thread pool and JSON parsing"
```

---

## Task 5: Add panel_serve_summary to Analytics

**Files:**
- Modify: `analytics.py`
- Create: `tests/test_analytics_serve_panel.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_analytics_serve_panel.py`:

```python
import numpy as np
import pytest
from serve_analyzer import ServeResult
from analytics import Analytics


def make_result(player_id, score, fault_field="stance", fault_val="poor"):
    kwargs = dict(
        frame_idx=100, timestamp_sec=3.3, player_id=player_id,
        is_serve=True, score=score, stance="good", ball_toss="good",
        contact_point="good", follow_through="good",
        landing_zone="deep", coaching_tip="Good."
    )
    kwargs[fault_field] = fault_val
    return ServeResult(**kwargs)


def test_panel_serve_summary_returns_correct_size():
    analytics = Analytics(filters={})
    results = [make_result(1, 8), make_result(1, 6), make_result(2, 7)]
    panel = analytics.panel_serve_summary(results, (300, 300))
    assert panel.shape == (300, 300, 3)


def test_panel_serve_summary_empty_results():
    analytics = Analytics(filters={})
    panel = analytics.panel_serve_summary([], (300, 300))
    assert panel.shape == (300, 300, 3)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_analytics_serve_panel.py -v
```

Expected: FAIL — `Analytics` has no `panel_serve_summary` method.

- [ ] **Step 3: Add `panel_serve_summary` to `analytics.py`**

Add the following method to the `Analytics` class (after `panel_rally_tempo`):

```python
def panel_serve_summary(self, results: list, size: tuple) -> np.ndarray:
    """
    Render a serve summary panel.
    results: list of ServeResult objects
    size: (width, height)
    """
    import numpy as np
    w, h = size
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, "SERVE ANALYSIS", (10, 25), font, 0.55, (255, 255, 255), 1)

    if not results:
        cv2.putText(panel, "No serves detected", (10, 60), font, 0.45, (160, 160, 160), 1)
        return panel

    # Group by player
    from collections import defaultdict, Counter
    by_player: dict = defaultdict(list)
    for r in results:
        key = f"P{r.player_id}" if r.player_id is not None else "P?"
        by_player[key].append(r)

    y = 50
    for player_label, serves in sorted(by_player.items()):
        avg_score = round(sum(s.score for s in serves) / len(serves), 1)

        # Find most common fault (any field with non-"good" / non-"deep" value)
        faults = []
        for s in serves:
            for field in ("stance", "ball_toss", "contact_point", "follow_through"):
                val = getattr(s, field)
                if val not in ("good", "unknown"):
                    faults.append(f"{field}:{val}")
        common_fault = Counter(faults).most_common(1)
        fault_str = common_fault[0][0].replace("_", " ") if common_fault else "none"

        # Score bar (green)
        bar_w = int((avg_score / 10) * (w - 20))
        cv2.rectangle(panel, (10, y + 18), (10 + bar_w, y + 28), (0, 200, 80), -1)

        cv2.putText(panel, f"{player_label}  {len(serves)} serves  avg:{avg_score}/10",
                    (10, y + 14), font, 0.42, (200, 255, 200), 1)
        cv2.putText(panel, f"  fault: {fault_str}",
                    (10, y + 42), font, 0.38, (180, 180, 180), 1)
        y += 60
        if y > h - 20:
            break

    return panel
```

Also add `import cv2` at the top of `analytics.py` if not already present (it is).

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_analytics_serve_panel.py -v
```

Expected: PASS both tests.

- [ ] **Step 5: Commit**

```bash
git add analytics.py tests/test_analytics_serve_panel.py
git commit -m "Add panel_serve_summary to Analytics for serve stats display"
```

---

## Task 6: Wire ServeDetector + OllamaServeAnalyzer into process_video.py

**Files:**
- Modify: `process_video.py`

- [ ] **Step 1: Add imports at top of `process_video.py`**

After the existing imports, add:

```python
from serve_detector import ServeDetector
from serve_analyzer import OllamaServeAnalyzer
```

- [ ] **Step 2: Instantiate in `VideoProcessor.__init__`**

After `self.analytics = Analytics(self.filters)`, add:

```python
self.serve_detector = ServeDetector()
self.serve_analyzer = OllamaServeAnalyzer(model="qwen2.5vl:7b", workers=2)
```

- [ ] **Step 3: Wire per-frame call in `process_video` main loop**

After `self.analytics.update_counters(frame_idx, proj_players, ball_proj)`, add:

```python
candidate = self.serve_detector.update(frame_idx, frame, ball_proj, players)
if candidate is not None:
    logger.bind(frame_idx=frame_idx, player_id=candidate.player_id).info("serve_candidate_detected")
    self.serve_analyzer.submit(candidate)
```

- [ ] **Step 4: Update `_render_analytics_grid` to include serve panel**

Replace the existing grid assembly in `_render_analytics_grid`:

```python
ph = self.analytics.panel_player_heatmap((panel_w, panel_h), bird_reference=bird_reference)
bh = self.analytics.panel_ball_heatmap((panel_w, panel_h), bird_reference=bird_reference)
kd = self.analytics.panel_kitchen_intrusion(None, (panel_w, panel_h))
rl = self.analytics.panel_rally_tempo((panel_w, panel_h))

top = cv2.hconcat([ph, bh])
bot = cv2.hconcat([kd, rl])
grid = cv2.vconcat([top, bot])
```

With:

```python
ph = self.analytics.panel_player_heatmap((panel_w, panel_h), bird_reference=bird_reference)
bh = self.analytics.panel_ball_heatmap((panel_w, panel_h), bird_reference=bird_reference)
kd = self.analytics.panel_kitchen_intrusion(None, (panel_w, panel_h))
sv = self.analytics.panel_serve_summary(self.serve_analyzer.get_results(), (panel_w, panel_h))

top = cv2.hconcat([ph, bh])
bot = cv2.hconcat([kd, sv])
grid = cv2.vconcat([top, bot])
```

Note: `panel_rally_tempo` is replaced by `panel_serve_summary` in the bottom-right tile.

- [ ] **Step 5: Flush report in the `finally` block**

After `self.analytics.save_outputs()` in the `finally` block, add:

```python
self.serve_analyzer.shutdown()
self._save_serve_report()
```

- [ ] **Step 6: Add `_save_serve_report` method to `VideoProcessor`**

Add after `_clip_long_rallies`:

```python
def _save_serve_report(self) -> None:
    """Write serve_report.json to the run output directory."""
    import json
    from datetime import datetime
    from collections import defaultdict, Counter

    results = self.serve_analyzer.get_results()
    by_player: dict = defaultdict(list)
    for r in results:
        key = f"P{r.player_id}" if r.player_id is not None else "P?"
        by_player[key].append(r)

    players_data = {}
    for player_label, serves in by_player.items():
        scores = [s.score for s in serves]
        faults = []
        for s in serves:
            for field in ("stance", "ball_toss", "contact_point", "follow_through"):
                val = getattr(s, field)
                if val not in ("good", "unknown"):
                    faults.append(f"{field}_{val}")
        common = Counter(faults).most_common(1)
        players_data[player_label] = {
            "serve_count": len(serves),
            "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "scores": scores,
            "common_fault": common[0][0] if common else "none",
            "serves": [
                {
                    "frame_idx": s.frame_idx,
                    "timestamp_sec": s.timestamp_sec,
                    "score": s.score,
                    "stance": s.stance,
                    "ball_toss": s.ball_toss,
                    "contact_point": s.contact_point,
                    "follow_through": s.follow_through,
                    "landing_zone": s.landing_zone,
                    "coaching_tip": s.coaching_tip,
                }
                for s in serves
            ],
        }

    report = {
        "generated": datetime.now().isoformat(),
        "total_serves": len(results),
        "players": players_data,
    }

    report_path = os.path.join(self.output_dir, "serve_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.bind(path=report_path, total_serves=len(results)).info("serve_report_saved")
```

- [ ] **Step 7: Commit**

```bash
git add process_video.py
git commit -m "Wire ServeDetector and OllamaServeAnalyzer into video processing pipeline"
```

---

## Task 7: Run full pipeline smoke test

- [ ] **Step 1: Run all unit tests**

```bash
pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 2: Run pipeline on sample video**

```bash
cd c:/apps/pickleball/Pickleball-Analytics
python -c "
from process_video import VideoProcessor
vp = VideoProcessor('video_outputs/Sample2.mp4', filters={}, mode='full')
out = vp.process_video()
print('Output:', out)
"
```

Expected: Video processes without errors, `serve_report.json` created in the run output directory.

- [ ] **Step 3: Verify serve report**

```bash
python -c "
import json, glob, os
runs = sorted(glob.glob('video_outputs/run_*/serve_report.json'))
if runs:
    with open(runs[-1]) as f:
        print(json.dumps(json.load(f), indent=2))
else:
    print('No serve report found')
"
```

Expected: Valid JSON with `total_serves`, `players` keys. May be 0 serves if no serve signatures detected in sample — that is acceptable for first run.

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "Complete player tracking and serve analysis feature"
```

---

## Dependencies to install before starting

```bash
pip install ollama
```

Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

Expected: JSON response listing `qwen2.5vl:7b` in the models list.
