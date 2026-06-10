import pytest
from rally_detector import RallyDetector, RallyRecord


# --- RallyRecord ---

def test_rally_record_fields():
    r = RallyRecord(rally_num=1, start_frame=10, end_frame=50, fps=30, end_reason="fault")
    assert r.start_sec   == pytest.approx(10 / 30)
    assert r.end_sec     == pytest.approx(50 / 30)
    assert r.duration_sec == pytest.approx(40 / 30)


# --- Initial state ---

def test_detector_starts_idle():
    det = RallyDetector(fps=30)
    assert det.state == RallyDetector.IDLE
    assert det.get_rallies() == []


def test_no_rally_when_ball_never_detected():
    det = RallyDetector(fps=30)
    for fi in range(200):
        det.update(fi, None)
    assert det.get_rallies() == []


# --- Rally start ---

def test_rally_starts_when_ball_first_detected():
    det = RallyDetector(fps=30)
    det.update(0, None)
    det.update(1, None)
    det.update(2, (300.0, 400.0))   # ball appears
    assert det.state == RallyDetector.ACTIVE


# --- Rally end ---

def test_rally_ends_after_fault_frames_of_missing_ball():
    """Ball present then absent for FAULT_FRAMES → rally finalized."""
    det = RallyDetector(fps=30)
    # Start rally
    for fi in range(RallyDetector.MIN_RALLY_FRAMES + 5):
        det.update(fi, (300.0, 400.0))
    # Ball disappears
    start_missing = RallyDetector.MIN_RALLY_FRAMES + 5
    for fi in range(start_missing, start_missing + RallyDetector.FAULT_FRAMES):
        det.update(fi, None)
    assert det.state == RallyDetector.IDLE
    assert len(det.get_rallies()) == 1
    assert det.get_rallies()[0].end_reason == "fault"


def test_short_rally_below_min_duration_discarded():
    """Ball present for fewer than MIN_RALLY_FRAMES then gone → not recorded."""
    det = RallyDetector(fps=30)
    short = RallyDetector.MIN_RALLY_FRAMES - 1
    for fi in range(short):
        det.update(fi, (300.0, 400.0))
    for fi in range(short, short + RallyDetector.FAULT_FRAMES):
        det.update(fi, None)
    assert det.get_rallies() == []


def test_ball_reappearance_resets_missing_counter():
    """Ball detected before FAULT_FRAMES → rally stays active."""
    det = RallyDetector(fps=30)
    det.update(0, (300.0, 400.0))
    for fi in range(1, RallyDetector.FAULT_FRAMES - 1):
        det.update(fi, None)   # missing, but not enough
    det.update(RallyDetector.FAULT_FRAMES, (300.0, 400.0))  # ball back
    assert det.state == RallyDetector.ACTIVE


# --- Full cycle ---

def test_full_rally_recorded_with_correct_fields():
    """Ball present for long enough then gone → RallyRecord with correct fields."""
    det = RallyDetector(fps=30)
    start = 10
    end_ball = 10 + RallyDetector.MIN_RALLY_FRAMES + 20   # well above minimum
    for fi in range(start, end_ball):
        det.update(fi, (300.0, 400.0))
    for fi in range(end_ball, end_ball + RallyDetector.FAULT_FRAMES):
        det.update(fi, None)
    rallies = det.get_rallies()
    assert len(rallies) == 1
    r = rallies[0]
    assert r.rally_num   == 1
    assert r.start_frame == start
    assert r.end_reason  == "fault"


def test_two_rallies_recorded_in_sequence():
    det = RallyDetector(fps=30)
    length = RallyDetector.MIN_RALLY_FRAMES + 10
    gap    = RallyDetector.FAULT_FRAMES

    # First rally
    for fi in range(length):
        det.update(fi, (300.0, 400.0))
    for fi in range(length, length + gap):
        det.update(fi, None)

    # Second rally
    start2 = length + gap + 10
    for fi in range(start2, start2 + length):
        det.update(fi, (300.0, 400.0))
    for fi in range(start2 + length, start2 + length + gap):
        det.update(fi, None)

    rallies = det.get_rallies()
    assert len(rallies) == 2
    assert rallies[0].rally_num == 1
    assert rallies[1].rally_num == 2
