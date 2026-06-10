import pytest
import numpy as np
from rally_detector import RallyDetector, RallyRecord
from serve_detector import ServeCandidate


def _make_serve(frame_idx=0, ball_pos=(300.0, 400.0)):
    return ServeCandidate(
        frame_idx=frame_idx,
        timestamp_sec=frame_idx / 30.0,
        player_id=1,
        ball_pos=ball_pos,
        frame_small=np.zeros((720, 1280, 3), dtype=np.uint8),
    )


# --- RallyRecord ---

def test_rally_record_fields():
    r = RallyRecord(
        rally_num=1,
        start_frame=10,
        end_frame=50,
        fps=30,
        exchanges=3,
        end_reason="fault",
    )
    assert r.start_sec == pytest.approx(10 / 30)
    assert r.end_sec == pytest.approx(50 / 30)
    assert r.duration_sec == pytest.approx(40 / 30)


# --- Initial state ---

def test_detector_starts_idle():
    det = RallyDetector(fps=30)
    assert det.state == RallyDetector.IDLE
    assert det.get_rallies() == []


def test_no_rally_without_serve():
    det = RallyDetector(fps=30)
    for fi in range(100):
        det.update(fi, (200.0, 400.0), None)
    assert det.get_rallies() == []


# --- Rally start ---

def test_rally_starts_on_serve():
    det = RallyDetector(fps=30)
    det.update(10, (300.0, 400.0), _make_serve(frame_idx=10))
    assert det.state == RallyDetector.ACTIVE


def test_ball_position_ignored_before_serve():
    """Ball detected before serve → still IDLE."""
    det = RallyDetector(fps=30)
    det.update(5, (300.0, 400.0), None)
    assert det.state == RallyDetector.IDLE


# --- Exchange counting ---

def test_exchange_counted_on_direction_reversal():
    """Ball travels left 80px then right 80px → 1 exchange."""
    det = RallyDetector(fps=30)
    det.update(0, (300.0, 400.0), _make_serve(0, (300.0, 400.0)))
    det.update(1, (220.0, 400.0), None)   # left 80px → direction = LEFT established
    det.update(2, (300.0, 400.0), None)   # right 80px from anchor → reversal → exchange
    assert det._exchanges == 1


def test_no_exchange_before_min_travel():
    """Ball moves less than MIN_TRAVEL_PX — no exchange even if it reverses."""
    det = RallyDetector(fps=30)
    det.update(0, (300.0, 400.0), _make_serve(0, (300.0, 400.0)))
    det.update(1, (270.0, 400.0), None)   # only 30px — below MIN_TRAVEL_PX
    det.update(2, (300.0, 400.0), None)   # back — still below threshold
    assert det._exchanges == 0


def test_exchange_cooldown_prevents_double_count():
    """Second reversal within EXCHANGE_COOLDOWN_F frames is suppressed."""
    det = RallyDetector(fps=30)
    det.update(0, (300.0, 400.0), _make_serve(0, (300.0, 400.0)))
    det.update(1, (220.0, 400.0), None)   # LEFT established
    det.update(2, (300.0, 400.0), None)   # RIGHT reversal → exchange #1 at frame 2
    det.update(3, (220.0, 400.0), None)   # LEFT reversal at frame 3 — within cooldown
    assert det._exchanges == 1


def test_multiple_exchanges_counted():
    """Each full back-and-forth beyond cooldown window is counted."""
    det = RallyDetector(fps=30)
    det.update(0, (300.0, 400.0), _make_serve(0, (300.0, 400.0)))
    det.update(1,  (220.0, 400.0), None)   # LEFT (80px)
    det.update(25, (300.0, 400.0), None)   # RIGHT reversal → exchange #1
    det.update(50, (220.0, 400.0), None)   # LEFT reversal → exchange #2
    det.update(75, (300.0, 400.0), None)   # RIGHT reversal → exchange #3
    assert det._exchanges == 3


# --- Fault / end conditions ---

def test_fault_ends_rally():
    """Ball missing for FAULT_FRAMES consecutive frames → rally finalized."""
    det = RallyDetector(fps=30)
    det.update(0, (300.0, 400.0), _make_serve(0))
    for fi in range(1, RallyDetector.FAULT_FRAMES + 1):
        det.update(fi, None, None)
    assert det.state == RallyDetector.IDLE
    assert len(det.get_rallies()) == 1
    assert det.get_rallies()[0].end_reason == "fault"


def test_ball_reappearance_resets_missing_counter():
    """Missing ball counter resets when ball is detected — rally stays active."""
    det = RallyDetector(fps=30)
    det.update(0, (300.0, 400.0), _make_serve(0))
    for fi in range(1, RallyDetector.FAULT_FRAMES - 1):
        det.update(fi, None, None)   # missing, but not enough to fault
    det.update(RallyDetector.FAULT_FRAMES, (300.0, 400.0), None)  # ball back
    assert det.state == RallyDetector.ACTIVE


# --- Full rally cycle ---

def test_full_rally_recorded():
    """Serve → exchanges → fault → one RallyRecord with correct fields."""
    det = RallyDetector(fps=30)
    det.update(10, (300.0, 400.0), _make_serve(10, (300.0, 400.0)))
    det.update(11, (220.0, 400.0), None)     # LEFT
    det.update(35, (300.0, 400.0), None)     # exchange #1
    det.update(60, (220.0, 400.0), None)     # exchange #2
    for fi in range(61, 61 + RallyDetector.FAULT_FRAMES):
        det.update(fi, None, None)
    rallies = det.get_rallies()
    assert len(rallies) == 1
    r = rallies[0]
    assert r.rally_num == 1
    assert r.start_frame == 10
    assert r.exchanges == 2
    assert r.end_reason == "fault"


def test_two_rallies_recorded():
    """Two serve-to-fault cycles produce two rally records."""
    det = RallyDetector(fps=30)
    # First rally
    det.update(0, (300.0, 400.0), _make_serve(0, (300.0, 400.0)))
    for fi in range(1, RallyDetector.FAULT_FRAMES + 1):
        det.update(fi, None, None)
    # Second rally
    start2 = RallyDetector.FAULT_FRAMES + 50
    det.update(start2, (300.0, 400.0), _make_serve(start2, (300.0, 400.0)))
    for fi in range(start2 + 1, start2 + RallyDetector.FAULT_FRAMES + 1):
        det.update(fi, None, None)
    rallies = det.get_rallies()
    assert len(rallies) == 2
    assert rallies[0].rally_num == 1
    assert rallies[1].rally_num == 2
