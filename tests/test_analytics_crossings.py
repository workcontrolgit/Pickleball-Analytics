import pytest
from analytics import Analytics


def make_analytics():
    a = Analytics({"player_heatmap": True, "ball_heatmap": True})
    a.set_canvas_size(400, 900)
    a.set_video_context(total_frames=900, fps=30)
    # Manually set kitchen bounds so net_y is defined
    a._kitchen_y_min = 300.0
    a._kitchen_y_max = 600.0
    a._net_y = 450.0
    a._court_bounds = (0.0, 0.0, 400.0, 900.0)
    return a


def test_initial_crossing_state():
    a = Analytics({"player_heatmap": True, "ball_heatmap": True})
    assert a._net_y is None
    assert a._ball_last_side is None
    assert a._rally_net_crossings == 0
    assert a._rally_start_frame is None
    assert a._long_rallies == []


def test_net_y_set_from_kitchen_bounds():
    a = Analytics({"player_heatmap": True, "ball_heatmap": True})
    a.set_canvas_size(400, 900)
    a._kitchen_y_min = 300.0
    a._kitchen_y_max = 600.0
    import numpy as np
    kpts = np.array([
        [50, 100], [200, 100], [350, 100],
        [50, 300], [200, 300], [350, 300],
        [50, 600], [200, 600], [350, 600],
        [50, 800], [200, 800], [350, 800],
    ], dtype=np.float32)
    a.update_kitchen_from_keypoints(kpts)
    assert a._net_y is not None
    assert abs(a._net_y - (a._kitchen_y_min + a._kitchen_y_max) / 2.0) < 1.0


def test_no_crossing_ball_stays_above():
    a = make_analytics()
    for i in range(10):
        a.update_counters(i, [], (200.0, 100.0))  # y=100 < net_y=450 → always above
    assert a._rally_net_crossings == 0


def test_single_crossing_detected():
    a = make_analytics()
    a.update_counters(0, [], (200.0, 100.0))  # above
    a.update_counters(1, [], (200.0, 800.0))  # below → 1 crossing
    assert a._rally_net_crossings == 1


def test_five_crossings_recorded_as_long_rally():
    a = make_analytics()
    # Alternate sides to produce 5 crossings
    sides = [100.0, 800.0, 100.0, 800.0, 100.0, 800.0]  # 5 crossings
    for i, y in enumerate(sides):
        a.update_counters(i, [], (200.0, y))
    # End rally with gap
    for i in range(len(sides), len(sides) + 20):
        a.update_counters(i, [], None)
    assert len(a._long_rallies) == 1
    start, end, crossings = a._long_rallies[0]
    assert crossings == 5
    assert start == 0


def test_four_crossings_not_recorded():
    a = make_analytics()
    sides = [100.0, 800.0, 100.0, 800.0, 100.0]  # 4 crossings
    for i, y in enumerate(sides):
        a.update_counters(i, [], (200.0, y))
    for i in range(len(sides), len(sides) + 20):
        a.update_counters(i, [], None)
    assert len(a._long_rallies) == 0


def test_missing_ball_detection_holds_last_side():
    a = make_analytics()
    a.update_counters(0, [], (200.0, 100.0))   # above
    a.update_counters(1, [], None)              # no ball — side held
    a.update_counters(2, [], (200.0, 100.0))   # above again — no crossing
    assert a._rally_net_crossings == 0


def test_rally_start_frame_recorded():
    a = make_analytics()
    a.update_counters(5, [], (200.0, 100.0))   # rally starts at frame 5
    assert a._rally_start_frame == 5


def test_video_ends_mid_rally_with_5_crossings():
    a = make_analytics()
    sides = [100.0, 800.0, 100.0, 800.0, 100.0, 800.0]  # 5 crossings
    for i, y in enumerate(sides):
        a.update_counters(i, [], (200.0, y))
    # No gap — rally still active; force finalize
    a._finalize_current_rally(frame_idx=len(sides))
    assert len(a._long_rallies) == 1
