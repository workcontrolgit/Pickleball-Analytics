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
