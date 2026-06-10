"""
Serve detection module

Purpose
-------
Watches ball position and player state each frame to detect serve events.
A serve is detected when:
  1. Ball stays within 20px of a position for 15+ consecutive frames (stationary)
  2. Ball then moves >50px in a single frame (launch)

Outputs a ServeCandidate when both conditions are met.
A 5-second cooldown prevents double-detection.
"""

from __future__ import annotations
from dataclasses import dataclass
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
    STILLNESS_FRAMES = 7     # frames ball must be stationary
    STILLNESS_PX = 25        # max movement to count as stationary
    LAUNCH_PX = 30           # min movement to count as launch
    MAX_LAUNCH_PX = 500      # max movement — filters wild false positives (e.g. tracker jumps)
    COOLDOWN_SEC = 5.0       # seconds between detections
    MAX_GAP_FRAMES = 5       # tolerate up to 5 consecutive missing detections
    MAX_PLAYER_DIST_PX = 300 # ball must be within this distance of a player to qualify

    def __init__(self, fps: int = 30):
        self._fps = fps
        self._cooldown_frames = int(self.COOLDOWN_SEC * fps)
        self._still_pos: Optional[tuple] = None
        self._still_count: int = 0
        self._last_ball: Optional[tuple] = None
        self._last_detected_frame: int = -9999
        self._missing_frames: int = 0  # consecutive frames with no ball

    def update(
        self,
        frame_idx: int,
        frame: np.ndarray,
        ball_proj: Optional[tuple],
        players: list,
    ) -> Optional[ServeCandidate]:
        if ball_proj is None:
            # Tolerate short gaps — ball detection is ~50% reliable
            self._missing_frames += 1
            if self._missing_frames > self.MAX_GAP_FRAMES:
                self._reset_stillness()
            return None
        self._missing_frames = 0

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
            elif dist > self.MAX_LAUNCH_PX:
                # Wildly far — false positive (scoreboard, noise). Ignore.
                pass
            else:
                # Ball moved a plausible distance — check if this is a launch
                if self._still_count >= self.STILLNESS_FRAMES and self._last_ball is not None:
                    launch_dist = np.hypot(bx - self._last_ball[0], by - self._last_ball[1])
                    if self.LAUNCH_PX <= launch_dist <= self.MAX_LAUNCH_PX:
                        # Only emit if ball was near a player during stillness
                        if self._ball_near_player(self._still_pos, players):
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

    def _ball_near_player(self, ball_pos: tuple, players: list) -> bool:
        """Return True if ball_pos is within MAX_PLAYER_DIST_PX of any player's center."""
        if not players:
            return True  # no player data — don't filter
        bx, by = ball_pos
        for p in players or []:
            bbox = p.get("bbox", [])
            if len(bbox) == 4:
                px = (bbox[0] + bbox[2]) / 2
                py = (bbox[1] + bbox[3]) / 2
                if np.hypot(bx - px, by - py) <= self.MAX_PLAYER_DIST_PX:
                    return True
        return False

    def _reset_stillness(self):
        self._still_pos = None
        self._still_count = 0
        self._missing_frames = 0

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
