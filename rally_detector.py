"""
Rally detection module

Purpose
-------
Detects pickleball rallies anchored to serve events.
Counts exchanges (direction reversals of ball travel) as a proxy for shots.
Works with raw frame pixel coordinates — no court homography required.

Algorithm
---------
1. Wait for a ServeCandidate event → rally starts.
2. Track ball position. When ball travels MIN_TRAVEL_PX pixels from an anchor
   point and then reverses direction, count one exchange. Reset anchor.
3. Rally ends when ball is missing for FAULT_FRAMES consecutive frames.

Inputs (per frame via update())
------
- frame_idx:   int
- ball_proj:   Optional[tuple]  — (x, y) ball position (pixel coords)
- serve_event                   — ServeCandidate or None

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
    exchanges: int     # number of direction reversals (proxy for shots)
    end_reason: str    # "fault"

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

    FAULT_FRAMES      = 45   # 1.5 s of missing ball → rally over
    MIN_TRAVEL_PX     = 60   # min pixels to establish a travel direction
    EXCHANGE_COOLDOWN_F = 20 # min frames between valid exchanges (debounce)

    def __init__(self, fps: float = 30):
        self._fps = fps
        self.state = self.IDLE
        self._rallies: list[RallyRecord] = []
        self._reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame_idx: int,
        ball_proj: Optional[tuple],
        serve_event,               # ServeCandidate | None
    ) -> None:
        """Process one frame. Mutates internal state; emits RallyRecord on end."""
        if self.state == self.IDLE:
            if serve_event is not None:
                self.state = self.ACTIVE
                self._start_frame = serve_event.frame_idx
                self._ball_anchor = serve_event.ball_pos
            return

        # ACTIVE state
        if ball_proj is None:
            self._missing_count += 1
            if self._missing_count >= self.FAULT_FRAMES:
                self._finalize(frame_idx, "fault")
            return

        self._missing_count = 0
        self._update_exchanges(frame_idx, ball_proj)

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
                    "rally_num":   r.rally_num,
                    "start_frame": r.start_frame,
                    "end_frame":   r.end_frame,
                    "start_sec":   round(r.start_sec, 3),
                    "end_sec":     round(r.end_sec, 3),
                    "duration_sec": round(r.duration_sec, 3),
                    "exchanges":   r.exchanges,
                    "end_reason":  r.end_reason,
                }
                for r in rallies
            ],
            "generated": datetime.datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_exchanges(self, frame_idx: int, ball_proj: tuple) -> None:
        """Detect ball direction reversals and increment _exchanges."""
        if self._ball_anchor is None:
            self._ball_anchor = ball_proj
            return

        bx, _ = ball_proj
        ax, _ = self._ball_anchor
        dx = bx - ax

        if abs(dx) < self.MIN_TRAVEL_PX:
            return  # not enough travel yet

        new_dir = 1 if dx > 0 else -1

        if self._travel_dir is not None and new_dir != self._travel_dir:
            # Direction reversed — count as exchange if past cooldown
            if frame_idx - self._last_exchange_frame >= self.EXCHANGE_COOLDOWN_F:
                self._exchanges += 1
                self._last_exchange_frame = frame_idx

        self._travel_dir = new_dir
        self._ball_anchor = ball_proj  # reset anchor to current position

    def _finalize(self, end_frame: int, end_reason: str) -> None:
        self._rallies.append(RallyRecord(
            rally_num=len(self._rallies) + 1,
            start_frame=self._start_frame,
            end_frame=end_frame,
            fps=self._fps,
            exchanges=self._exchanges,
            end_reason=end_reason,
        ))
        self.state = self.IDLE
        self._reset()

    def _reset(self) -> None:
        self._start_frame: Optional[int] = None
        self._exchanges: int = 0
        self._missing_count: int = 0
        self._ball_anchor: Optional[tuple] = None
        self._travel_dir: Optional[int] = None   # +1 = right, -1 = left
        self._last_exchange_frame: int = -9999
