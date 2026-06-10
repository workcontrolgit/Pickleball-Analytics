# Rally Detection — Design Spec
**Date:** 2026-06-10  
**Branch:** feature/yolo26-upgrade  
**Status:** Approved

---

## Overview

Improve rally detection in the Pickleball Analytics pipeline so that rally start and end events are grounded in actual pickleball rules (serve triggers start, two-bounce rule validates the rally, ball out/net/fault triggers end). Produce a structured `rally_report.json` in a dedicated analysis pass, then let the user review and confirm before clipping into individual video files.

---

## Goals

- Accurate rally start detection anchored to the serve event.
- Two-bounce rule validation: a rally is only "complete" (worth saving) once both mandatory bounces have occurred.
- Three rally-end triggers: out of bounds, net hit, or extended ball absence (fault/catch).
- Two-step workflow: analysis pass → user reviews report → clip pass.
- UI confirmation step before any video files are written.

---

## Non-goals

- Real-time fault calling (NVZ violation, foot fault).
- Scoreboard tracking.
- Changes to `MODE_VIDEO_ANALYSIS`, `MODE_SPLIT_RALLIES`, or `MODE_DETECT_SERVE`.

---

## Architecture

### New Modes

| Mode constant | Output | Description |
|---|---|---|
| `MODE_DETECT_RALLIES` | `rally_report.json` | Analysis-only pass: runs detectors, logs rally events, no video written |
| `MODE_CLIP_FROM_REPORT` | `rally_01.mp4` … | Reads `rally_report.json`, writes one clip per rally |

The existing `MODE_SPLIT_RALLIES` is kept for backward compatibility.

### New File

`rally_detector.py` — `RallyDetector` class (see below). No changes to `analytics.py`.

### Modified Files

- `process_video.py` — add the two new modes, wire `RallyDetector`
- `main.py` — add "Detect Rallies" to mode selector, two-step confirm UI

---

## RallyDetector — State Machine

File: `rally_detector.py`

```
IDLE
  → SERVE_DETECTED      on: serve event from ServeDetector
  → BOUNCE_1_PENDING    on: ball crosses net (1st crossing after serve)
  → BOUNCE_2_PENDING    on: ball crosses net again (2nd crossing) OR bounce detected
  → OPEN_PLAY           on: ball bounces on receiver's side (two-bounce rule complete)
  → ENDED               on: end condition (see below)
```

After `ENDED`, state resets to `IDLE` and a rally record is appended.

### Rally Start

Reuse the existing `ServeDetector` (ball stationary ≥10 frames near a player → sudden launch ≥40px). `RallyDetector` subscribes to serve events; each serve triggers a transition from `IDLE` → `SERVE_DETECTED` and records `start_frame`.

### Bounce Detection

Camera angle: behind-the-baseline. A bounce is detected when:
1. Ball bbox bottom (`y2` in raw frame coordinates) reaches within N pixels of the projected court surface height at that horizontal position.
2. Ball vertical velocity reverses (downward → upward) within the next 3 frames.

If the ball tracking signal is too noisy to reliably detect bounces (fewer bounces detected than expected given net crossings), the detector falls back to using **net-crossing count as a proxy** — the second net crossing after the serve implies both mandatory bounces occurred.

### Rally End — Three Triggers

| Trigger | Condition | `end_reason` value |
|---|---|---|
| Out of bounds | Ball projected point exits court-bounds polygon with ≥3px margin for ≥3 consecutive frames | `"out"` |
| Net hit | Ball last detected within ±40px of `net_y` in bird space AND absent for ≥15 frames thereafter | `"net"` |
| Fault / catch | Ball absent for ≥30 frames after last in-bounds detection | `"fault"` |

### Two-bounce Rule Flag

`two_bounce_complete: true` when the FSM reached `OPEN_PLAY` before `ENDED`. Rallies with `false` are still written to the report (useful for diagnosing detection quality) but flagged clearly.

---

## rally_report.json Schema

```json
{
  "generated": "2026-06-10T14:32:00",
  "video_path": "sample3.mp4",
  "fps": 30,
  "total_rallies": 12,
  "rallies": [
    {
      "rally_num": 1,
      "start_frame": 142,
      "end_frame": 387,
      "start_sec": 4.73,
      "end_sec": 12.90,
      "duration_sec": 8.17,
      "net_crossings": 6,
      "end_reason": "out",
      "two_bounce_complete": true
    }
  ]
}
```

---

## UI Workflow (`main.py`)

### New Mode in Selector

Add **"Detect Rallies"** to the `CTkSegmentedButton` values alongside the existing three modes. Mode description: `"Detect rallies, review summary, then clip"`.

### Two-step State Machine

```
FILE_SELECTED
  → ANALYZING       on: "Process Video" click (runs MODE_DETECT_RALLIES in background thread)
  → REPORT_READY    on: analysis complete — results card shows rally summary + scrollable list
  → CLIPPING        on: "Clip Rallies" button click (runs MODE_CLIP_FROM_REPORT in background thread)
  → CLIP_DONE       on: clipping complete — results card updates with output folder
```

### Results Card — REPORT_READY State

Shows:
- Summary badge: `"12 rallies detected · avg 8.2s · longest 21.4s"`
- Scrollable list (one row per rally): `Rally 1 · 4.7s–12.9s (8.2s) · out`
- `two_bounce_complete: false` rallies shown with a warning color
- **"Clip Rallies"** button (enabled) — triggers the clip pass
- Output folder path (report location)

### Results Card — CLIP_DONE State

- Badge updates to: `"12 clips saved"`
- Output folder path updates to clip directory
- "Open Folder" button re-enabled

---

## Data Flow

```
VideoProcessor (MODE_DETECT_RALLIES)
  └── per frame:
        ServeDetector.update()  →  serve event
        RallyDetector.update(frame_idx, ball_proj, serve_event, net_y, court_bounds)
  └── on complete:
        RallyDetector.get_report()  →  rally_report.json

VideoProcessor (MODE_CLIP_FROM_REPORT)
  └── reads rally_report.json
  └── for each rally: seek to start_frame, write frames to rally_N.mp4
```

---

## Shared Court Line Handling (Tennis + Pickleball)

Amateur recordings frequently use tennis courts with pickleball lines painted or taped on top. The court detection model may lock onto tennis baseline/service box corners instead of pickleball keypoints, producing a homography that maps to a tennis-sized court (~3× larger area). This silently breaks out-of-bounds detection and kitchen/net position.

### Court Bounds Sanity Check

Every time `update_court_bounds_from_keypoints()` proposes new bounds, validate them before accepting:

1. **Aspect ratio**: pickleball court is 20ft × 44ft (width:length ≈ 0.45). Tennis is 36ft × 78ft (same ratio). Ratio alone cannot discriminate — use area instead.
2. **Area check**: compute pixel area of the proposed bounds in bird space. If the new area is more than **2× the current accepted area** (or more than 2× the first valid area), reject the update and hold the last valid bounds.
3. **First-frame bootstrap**: on the very first valid keypoint detection, accept the bounds unconditionally and record them as the `_reference_court_area`. All subsequent frames validate against this reference.

This prevents tennis-line contamination from corrupting the homography mid-rally. The existing EMA smoothing still applies to accepted updates.

The sanity check lives in `RallyDetector` (not `analytics.py`) so it is specific to the rally detection path and does not alter existing analytics behaviour.

---

## Edge Cases

- **Serve not detected**: if `ServeDetector` misses a serve, the rally will not be recorded. This is acceptable — it's a detection-quality issue, not a correctness issue.
- **Ball tracking gaps**: `RallyDetector` tolerates up to `MAX_GAP_FRAMES = 5` consecutive missing detections before triggering a fault end.
- **Overlapping serves**: if a serve is detected while a rally is already in `OPEN_PLAY`, the current rally is finalized before starting a new one.
- **Short rallies (< 2 net crossings)**: recorded in the report with `two_bounce_complete: false`. They appear in the report but are visually flagged.
- **Shared court lines**: handled by the court bounds sanity check above. If the first detection is a tennis court (no pickleball lines visible), all subsequent frames will be consistent — the error is uniform and the homography still produces a stable bird's-eye view, just at tennis-court scale. Rally detection still works correctly relative to whatever court was detected.

---

## Files Changed

| File | Change type |
|---|---|
| `rally_detector.py` | New |
| `process_video.py` | Modified — add 2 modes, wire RallyDetector |
| `main.py` | Modified — add mode, two-step UI |
