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


def test_serve_detected_with_small_launch_dist():
    """Ball still for STILLNESS_FRAMES then moves 35px → detected (threshold=30, was 40)."""
    det = ServeDetector(fps=30)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    players = make_players(1, [100, 300, 200, 500])

    for i in range(ServeDetector.STILLNESS_FRAMES):
        det.update(i, frame, (150.0, 400.0), players)

    # 35px launch — below old threshold (40), above new threshold (30)
    result = det.update(ServeDetector.STILLNESS_FRAMES, frame, (185.0, 400.0), players)
    assert isinstance(result, ServeCandidate), "35px launch after stillness should be detected"


def test_serve_detected_with_fewer_stillness_frames():
    """Ball still for 7 frames (< old threshold of 10) then launches 100px → detected."""
    det = ServeDetector(fps=30)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    players = make_players(1, [100, 300, 200, 500])

    # 7 frames of stillness — below old threshold (10), at new threshold (7)
    for i in range(7):
        det.update(i, frame, (150.0, 400.0), players)

    result = det.update(7, frame, (250.0, 400.0), players)
    assert isinstance(result, ServeCandidate), "100px launch after 7 still frames should be detected"


def test_wildly_large_launch_filtered():
    """Launch distance > MAX_LAUNCH_PX (e.g. 1300px) is rejected as a false detection."""
    det = ServeDetector(fps=30)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    players = make_players(1, [100, 300, 200, 500])

    for i in range(ServeDetector.STILLNESS_FRAMES):
        det.update(i, frame, (150.0, 400.0), players)

    # Ball jumps 1300px — clearly a false detection
    result = det.update(ServeDetector.STILLNESS_FRAMES, frame, (1450.0, 400.0), players)
    assert result is None, "1300px launch should be filtered as a false detection"
