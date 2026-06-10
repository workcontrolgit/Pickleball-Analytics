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
        return self.start_frame / self.fps

    @property
    def end_sec(self) -> float:
        return self.end_frame / self.fps

    @property
    def duration_sec(self) -> float:
        return (self.end_frame - self.start_frame) / self.fps


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
        self._court_bounds: Optional[tuple] = None
        self._reference_court_area: Optional[float] = None
        self._net_y: Optional[float] = None
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

    def _ball_in_bounds(self, ball_proj: Optional[tuple]) -> bool:
        if ball_proj is None:
            return False
        if self._court_bounds is None:
            return True  # court not detected — assume in-bounds, rely on fault timer
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
