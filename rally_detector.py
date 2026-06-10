"""
Rally detection module

Algorithm
---------
A rally is a continuous period of ball activity separated by gaps where the ball
is not detected. No court detection, serve detection, or direction tracking required.

1. When ball first appears → rally starts.
2. While ball is detected → rally continues (short gaps tolerated).
3. When ball missing for FAULT_FRAMES consecutive frames → rally ends.
4. Rallies shorter than MIN_RALLY_FRAMES are discarded (noise / scoreboard hits).

Inputs (per frame via update())
------
- frame_idx:  int
- ball_proj:  Optional[tuple]  — (x, y) ball position; None if not detected

Outputs
-------
- get_rallies() → list of RallyRecord (completed rallies)
- get_report()  → dict for json.dump
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class RallyRecord:
    rally_num: int
    start_frame: int
    end_frame: int
    fps: float
    end_reason: str    # "fault" | "video_end"

    @property
    def start_sec(self) -> float:
        return self.start_frame / self.fps

    @property
    def end_sec(self) -> float:
        return self.end_frame / self.fps

    @property
    def duration_sec(self) -> float:
        return (self.end_frame - self.start_frame) / self.fps


class RallyDetector:
    IDLE   = "idle"
    ACTIVE = "active"

    FAULT_FRAMES     = 45   # 1.5 s of missing ball → rally over
    MIN_RALLY_FRAMES = 30   # ignore segments shorter than 1 s (noise)

    def __init__(self, fps: float = 30):
        self._fps = fps
        self.state = self.IDLE
        self._rallies: list[RallyRecord] = []
        self._reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame_idx: int, ball_proj: Optional[tuple]) -> None:
        """Process one frame."""
        if self.state == self.IDLE:
            if ball_proj is not None:
                self.state = self.ACTIVE
                self._start_frame = frame_idx
                self._last_ball_frame = frame_idx
                self._missing = 0
            return

        # ACTIVE
        if ball_proj is not None:
            self._missing = 0
            self._last_ball_frame = frame_idx
        else:
            self._missing += 1
            if self._missing >= self.FAULT_FRAMES:
                self._finalize(self._last_ball_frame, "fault")

    def get_rallies(self) -> list[RallyRecord]:
        return list(self._rallies)

    def get_report(self, video_path: str = "") -> dict:
        import datetime
        rallies = self.get_rallies()
        return {
            "video_path": video_path,
            "fps": self._fps,
            "total_rallies": len(rallies),
            "rallies": [
                {
                    "rally_num":    r.rally_num,
                    "start_frame":  r.start_frame,
                    "end_frame":    r.end_frame,
                    "start_sec":    round(r.start_sec, 3),
                    "end_sec":      round(r.end_sec, 3),
                    "duration_sec": round(r.duration_sec, 3),
                    "end_reason":   r.end_reason,
                }
                for r in rallies
            ],
            "generated": datetime.datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _finalize(self, end_frame: int, end_reason: str) -> None:
        duration = end_frame - self._start_frame
        if duration >= self.MIN_RALLY_FRAMES:
            self._rallies.append(RallyRecord(
                rally_num=len(self._rallies) + 1,
                start_frame=self._start_frame,
                end_frame=end_frame,
                fps=self._fps,
                end_reason=end_reason,
            ))
        self.state = self.IDLE
        self._reset()

    def _reset(self) -> None:
        self._start_frame: Optional[int] = None
        self._last_ball_frame: Optional[int] = None
        self._missing: int = 0
