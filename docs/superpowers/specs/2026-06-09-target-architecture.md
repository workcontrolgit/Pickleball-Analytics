# Pickleball Analytics — Target Architecture

> **Status:** Reference document. Current code lives in a flat structure.
> This is the destination for a future migration, not the current state.

---

## Goal

Evolve from a flat-file prototype into a clean, layered coaching platform where each
feature (serves, rallies, volleys, …) is an independently testable module, domain
logic never depends on YOLO or Ollama directly, and new features are added as modules
rather than modifications to existing code.

---

## Technology Stack

| Area | Technology |
|---|---|
| Detection | YOLOv26 |
| Tracking | ByteTrack |
| Video processing | OpenCV |
| Clip generation | FFmpeg |
| AI coaching | Ollama vision (qwen2.5vl:7b) |
| Local storage | SQLite + JSON |
| Configuration | YAML |
| Testing | PyTest |

---

## Target Directory Layout

```
pickleball-analytics/
│
├── configs/
│   ├── local.yaml          # paths, output dirs
│   ├── models.yaml         # model weights locations
│   └── features.yaml       # feature flags (enable/disable per feature)
│
├── src/
│   │
│   ├── app/
│   │   ├── main.py         # CTk desktop UI — entry point
│   │   └── cli.py          # Optional headless CLI entry point
│   │
│   ├── core/
│   │   ├── config.py       # Load and validate YAML configs
│   │   ├── logging.py      # Loguru setup (replaces setup_logger in process_video.py)
│   │   └── constants.py    # Drawing styles, codec, layout ratios
│   │
│   ├── domain/
│   │   ├── entities.py     # Pure dataclasses: Frame, Player, Ball, Court, Rally, Serve
│   │   ├── events.py       # Event types: ServeEvent, RallyEvent, ShotEvent
│   │   └── metrics.py      # Computed metrics: ServeScore, RallyStats, PlayerStats
│   │
│   ├── pipelines/
│   │   ├── video_analysis.py   # MODE_VIDEO_ANALYSIS orchestration
│   │   ├── split_rallies.py    # MODE_SPLIT_RALLIES orchestration
│   │   └── detect_serve.py     # MODE_DETECT_SERVE orchestration
│   │
│   ├── features/
│   │   │
│   │   ├── base.py             # VideoFeature interface (detect / analyze / coach)
│   │   │
│   │   ├── highlight/          # Phase 1 — rally detection and clipping
│   │   │   ├── detector.py     # Rally state machine (from analytics.py)
│   │   │   ├── clipper.py      # Rally clip writer (from process_video._clip_long_rallies)
│   │   │   └── tests/
│   │   │
│   │   ├── serve/              # Phase 2 — serve detection and scoring
│   │   │   ├── detector.py     # ServeDetector (current serve_detector.py)
│   │   │   ├── analyzer.py     # OllamaServeAnalyzer (current serve_analyzer.py)
│   │   │   └── tests/
│   │   │
│   │   ├── return_shot/        # Phase 3
│   │   ├── volley/             # Phase 4
│   │   ├── overhead/           # Phase 5
│   │   ├── dink/               # Phase 6
│   │   └── footwork/           # Phase 6
│   │
│   ├── infrastructure/
│   │   │
│   │   ├── yolo/
│   │   │   ├── ball_tracker.py     # current ball_tracker.py
│   │   │   ├── player_tracker.py   # current player_tracker.py
│   │   │   └── court_detector.py   # current court_detection.py
│   │   │
│   │   ├── ollama/
│   │   │   └── client.py       # Thin wrapper around ollama.chat()
│   │   │
│   │   ├── opencv/
│   │   │   ├── renderer.py     # All cv2 drawing helpers (from process_video render methods)
│   │   │   └── video_io.py     # VideoCapture / VideoWriter helpers
│   │   │
│   │   └── ffmpeg/             # Future: FFmpeg clip generation
│   │
│   └── storage/
│       ├── json_store.py       # serve_report.json, rally_report.json writers
│       └── report_writer.py    # Human-readable coaching report generation
│
├── models/                     # NOT committed to git
│   ├── ball_tracking.pt
│   ├── player_tracking.pt
│   └── court_detection.pt
│
├── data/
│   ├── raw/                    # Input videos — NOT committed to git
│   ├── interim/                # Extracted frames / tracking cache — NOT committed
│   └── processed/              # Normalized events JSON — NOT committed
│
├── outputs/                    # All run outputs — NOT committed to git
│   ├── highlights/
│   ├── reports/
│   └── coaching/
│
└── tests/
    ├── unit/
    └── integration/
```

---

## Feature Interface

Every feature module implements the same three-method interface:

```python
class VideoFeature:
    def detect(self, context: FrameContext) -> list[Event]:
        """Identify events in this frame."""

    def analyze(self, context: FrameContext, events: list[Event]) -> Analysis:
        """Compute metrics from detected events."""

    def coach(self, context: FrameContext, analysis: Analysis) -> CoachingTip:
        """Generate a coaching tip (may call Ollama)."""
```

This means:
- Each feature can be tested without a real video (mock `context`)
- Ollama calls only happen inside `coach()` — never in `detect()` or `analyze()`
- New features are added by creating a new module, not editing existing code

---

## Design Principles

1. Domain logic (`domain/`) never imports from YOLO, Ollama, or OpenCV.
2. Infrastructure (`infrastructure/`) can be swapped without changing business logic.
3. Every pipeline stage produces a JSON output for debugging.
4. Videos and model weights are never committed to git.
5. New features are new modules — existing code is not modified.

---

## Development Roadmap

| Phase | Feature | Status |
|---|---|---|
| 1 | Highlights (rally detection + clips) | Implemented (flat structure) |
| 2 | Serve analysis (detection + Ollama scoring) | Implemented (flat structure) |
| 3 | Return shot analysis | Not started |
| 4 | Volley analysis | Not started |
| 5 | Overhead analysis | Not started |
| 6 | Advanced skills (dink, third-shot drop, footwork) | Not started |
| 7 | Player skill tracking over time | Not started |

---

## Migration Path (flat → target)

When ready to migrate:

1. Create `src/` tree with empty `__init__.py` files.
2. Move `ball_tracker.py` → `src/infrastructure/yolo/ball_tracker.py`
3. Move `player_tracker.py` → `src/infrastructure/yolo/player_tracker.py`
4. Move `court_detection.py` → `src/infrastructure/yolo/court_detector.py`
5. Move `serve_detector.py` → `src/features/serve/detector.py`
6. Move `serve_analyzer.py` → `src/features/serve/analyzer.py`
7. Extract rally state machine from `analytics.py` → `src/features/highlight/detector.py`
8. Extract render methods from `process_video.py` → `src/infrastructure/opencv/renderer.py`
9. Split `process_video.py` into three pipeline files under `src/pipelines/`.
10. Move `main.py` → `src/app/main.py`, update all imports.
11. Update `pyproject.toml` / `setup.py` with `src` as package root.
