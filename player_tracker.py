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
