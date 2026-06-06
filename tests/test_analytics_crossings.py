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
