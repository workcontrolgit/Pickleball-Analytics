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
