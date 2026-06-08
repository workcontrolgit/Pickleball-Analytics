# Player Identity Tracking & Serve Analysis — Design Spec

**Date:** 2026-06-08  
**Status:** Approved  

---

## Overview

Two interconnected features added to the Pickleball Analytics pipeline:

1. **Player Identity Tracking** — Assign persistent IDs (P1–P4) to players across frames using ByteTrack, surviving side switches and movement.
2. **Serve Detection + Quality Scoring** — Detect serve events using ball trajectory heuristics, confirm and score each serve using local Ollama vision LLM (`qwen2.5vl:7b`), output per-player serve reports.

---

## Architecture & Data Flow

```
Per-frame pipeline:
  CourtDetector          → homography H
  PlayerTracker          → boxes + persistent IDs [P1–P4]  ← upgraded
  BallTracker            → bbox + ball_proj

  ServeDetector.update(frame_idx, frame, ball_proj, players)
      → ServeCandidate | None

  If ServeCandidate:
      save frame as JPEG (resized 1280×720) to temp dir
      submit async → OllamaServeAnalyzer (ThreadPoolExecutor)

  OllamaServeAnalyzer (background, 2 workers):
      send frame + prompt → qwen2.5vl:7b @ localhost:11434
      parse JSON response → ServeResult
      accumulate in memory

After video completes:
      flush serve_report.json → video_outputs/run_xxx/
      render ServeReportPanel → analytics grid
```

---

## Section 1: Player Tracking (ByteTrack)

**File:** `player_tracker.py`

Single change: swap `model.predict()` for `model.track()`.

```python
results = self.model.track(
    frame, conf=self.conf_threshold,
    device=self.device, persist=True, tracker="bytetrack.yaml"
)[0]
```

ByteTrack assigns persistent integer IDs on first appearance, maintained across frames via appearance + motion continuity. IDs survive player movement and side switches.

**Updated return format from `detect_and_project()`:**

```python
# Each player entry
{"id": 1, "bbox": [x1, y1, x2, y2], "proj": (bx, by)}
```

**Display:** Player bounding boxes labeled `P1`, `P2`, etc. in the main view. Heatmaps remain combined in this phase.

---

## Section 2: Serve Detection

**File:** `serve_detector.py` (new)

Watches ball + player state per frame. A serve is detected when all 3 conditions are met:

1. **Ball stationary near a player** — ball position stays within 20px for 15+ consecutive frames (~0.5s at 30fps)
2. **Ball launches** — ball moves >50px in a single frame transition
3. **Ball moves toward net** — ball's projected y-coordinate moves away from server's baseline toward net

```python
class ServeDetector:
    def update(self, frame_idx, frame, ball_proj, players) -> ServeCandidate | None
```

**ServeCandidate contains:**
- Frame image (resized 1280×720)
- `frame_idx` + `timestamp_sec`
- Closest `player_id` (the server)
- Ball position at launch moment

**Edge cases:**
- 5-second cooldown after detection — prevents double-detection of the same serve
- Interpolated ball position used if tracker misses frames during stillness window

---

## Section 3: Ollama Serve Analyzer

**File:** `serve_analyzer.py` (new)

Runs in a background `ThreadPoolExecutor(max_workers=2)` — never blocks video processing.

**Prompt to `qwen2.5vl:7b`:**

```
You are a pickleball coach analyzing a serve.
Respond in valid JSON only, no other text.

{
  "is_serve": true/false,
  "score": 1-10,
  "stance": "good|poor|unknown",
  "ball_toss": "good|too_low|too_far|unknown",
  "contact_point": "good|too_low|too_far|unknown",
  "follow_through": "good|poor|unknown",
  "landing_zone": "deep|short|wide|unknown",
  "coaching_tip": "one sentence"
}
```

**Behavior:**
- `is_serve: false` → candidate discarded (LLM rejected it)
- Timeout: 30 seconds per call — skipped if Ollama is unresponsive
- Max 2 concurrent Ollama calls

```python
class OllamaServeAnalyzer:
    def __init__(self, model="qwen2.5vl:7b", workers=2)
    def submit(self, candidate: ServeCandidate) -> None
    def get_results() -> list[ServeResult]
    def flush() -> None  # called at end of video
```

---

## Section 4: Serve Report Output

### JSON — `video_outputs/run_xxx/serve_report.json`

```json
{
  "generated": "2026-06-08T12:00:00",
  "total_serves": 12,
  "players": {
    "P1": {
      "serve_count": 4,
      "avg_score": 7.2,
      "scores": [8, 7, 6, 8],
      "common_fault": "ball_toss_too_low",
      "serves": [
        {
          "frame_idx": 342,
          "timestamp_sec": 11.4,
          "score": 8,
          "stance": "good",
          "ball_toss": "good",
          "contact_point": "good",
          "follow_through": "good",
          "landing_zone": "deep",
          "coaching_tip": "Great power serve, maintain consistency."
        }
      ]
    }
  }
}
```

### Video Panel — `panel_serve_summary()`

New tile in the analytics grid showing:
- Per-player serve count and avg score (text/bar display)
- Most common fault per player
- Shows "Analyzing..." until Ollama responds for pending serves

---

## Section 5: Integration with Existing Files

### Files modified

| File | Change |
|------|--------|
| `player_tracker.py` | Swap predict → track, update return format |
| `process_video.py` | Instantiate ServeDetector + OllamaServeAnalyzer, wire per-frame calls, flush on complete |
| `analytics.py` | Add `panel_serve_summary()` method |

### New files

| File | Purpose |
|------|---------|
| `serve_detector.py` | Serve signature detection logic |
| `serve_analyzer.py` | Ollama thread pool integration |

### Files unchanged
`ball_tracker.py`, `court_detection.py`, `main.py`

---

## Dependencies

- `ultralytics` ByteTrack — already installed, built into ultralytics
- `ollama` Python SDK — `pip install ollama`
- `qwen2.5vl:7b` — already installed locally

---

## Success Criteria

- Players maintain consistent P1–P4 IDs across a full rally including side switches
- Serve detection fires within 1 second of actual serve in test footage
- Ollama confirms >70% of detected candidates as valid serves
- `serve_report.json` written for every video processed
- Serve summary panel visible in composite video output
