import pytest
from rally_detector import RallyDetector, RallyRecord

def test_rally_record_fields():
    r = RallyRecord(
        rally_num=1,
        start_frame=10,
        end_frame=50,
        fps=30,
        net_crossings=4,
        end_reason="out",
        two_bounce_complete=True,
    )
    assert r.start_sec == pytest.approx(10 / 30)
    assert r.end_sec == pytest.approx(50 / 30)
    assert r.duration_sec == pytest.approx((50 - 10) / 30)

def test_detector_starts_idle():
    det = RallyDetector(fps=30)
    assert det.state == RallyDetector.IDLE
    assert det.get_rallies() == []


# --- Task 2: Court bounds sanity check ---

def test_court_bounds_accepted_on_first_call():
    det = RallyDetector(fps=30)
    bounds = (0.0, 0.0, 100.0, 200.0)   # area = 100*200 = 20000
    det._validate_and_set_court_bounds(bounds)
    assert det._court_bounds == bounds
    assert det._reference_court_area == pytest.approx(20000.0)

def test_court_bounds_rejected_when_too_large():
    det = RallyDetector(fps=30)
    small = (0.0, 0.0, 100.0, 200.0)    # area 20000
    big   = (0.0, 0.0, 300.0, 200.0)    # area 60000 — 3× reference
    det._validate_and_set_court_bounds(small)
    det._validate_and_set_court_bounds(big)
    assert det._court_bounds == small    # rejected — held previous

def test_court_bounds_accepted_within_ratio():
    det = RallyDetector(fps=30)
    first  = (0.0, 0.0, 100.0, 200.0)  # area 20000
    second = (0.0, 0.0, 110.0, 200.0)  # area 22000 — 1.1× OK
    det._validate_and_set_court_bounds(first)
    det._validate_and_set_court_bounds(second)
    assert det._court_bounds == second


# --- Task 3: Net crossing and ball-in-bounds detection ---

def _make_det_with_court(net_y=500.0):
    det = RallyDetector(fps=30)
    det._validate_and_set_court_bounds((0.0, 0.0, 400.0, 1000.0))
    det._net_y = net_y
    return det

def test_ball_in_bounds_inside():
    det = _make_det_with_court()
    assert det._ball_in_bounds((200.0, 500.0)) is True

def test_ball_in_bounds_outside():
    det = _make_det_with_court()
    assert det._ball_in_bounds((500.0, 500.0)) is False  # x > xmax=400

def test_ball_in_bounds_none():
    det = _make_det_with_court()
    assert det._ball_in_bounds(None) is False

def test_net_crossing_detected():
    det = _make_det_with_court(net_y=500.0)
    det.state = det.OPEN_PLAY
    det._ball_last_side = "near"
    det._update_net_crossing((200.0, 300.0))   # y=300 < net_y=500 → "far" side
    assert det._net_crossings == 1

def test_no_crossing_same_side():
    det = _make_det_with_court(net_y=500.0)
    det.state = det.OPEN_PLAY
    det._ball_last_side = "near"
    det._update_net_crossing((200.0, 700.0))   # still near side (y > net_y)
    assert det._net_crossings == 0


# --- Task 4: Bounce detection with net-crossing fallback ---

def test_bounce_detected_on_y2_reversal():
    """Ball y2 rises then falls → local maximum = bounce."""
    det = RallyDetector(fps=30)
    # Feed increasing y2 (ball falling) then decreasing (ball rising after bounce)
    for y2 in [300.0, 320.0, 340.0, 355.0, 360.0]:   # falling
        det._update_bounce_detection(y2)
    for y2 in [350.0, 330.0]:                          # rising — bounce happened
        det._update_bounce_detection(y2)
    assert det._bounce_count >= 1

def test_no_bounce_on_monotone_fall():
    det = RallyDetector(fps=30)
    for y2 in [200.0, 250.0, 300.0, 350.0, 400.0]:
        det._update_bounce_detection(y2)
    assert det._bounce_count == 0

def test_two_bounce_satisfied_via_bounces():
    det = RallyDetector(fps=30)
    det._bounce_count = 2
    det._net_crossings = 1
    assert det._two_bounce_satisfied() is True

def test_two_bounce_satisfied_via_net_crossings_fallback():
    """If bounce count is low but net crossings >= 2, use crossings as proxy."""
    det = RallyDetector(fps=30)
    det._bounce_count = 0    # noisy tracking — no bounces detected
    det._net_crossings = 2   # but two crossings seen → implies two bounces occurred
    assert det._two_bounce_satisfied() is True

def test_two_bounce_not_satisfied():
    det = RallyDetector(fps=30)
    det._bounce_count = 0
    det._net_crossings = 1
    assert det._two_bounce_satisfied() is False
