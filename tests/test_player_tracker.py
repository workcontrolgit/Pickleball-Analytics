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
