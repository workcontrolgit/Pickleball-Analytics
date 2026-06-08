import numpy as np
import pytest
from serve_analyzer import ServeResult
from analytics import Analytics


def make_result(player_id, score, fault_field="stance", fault_val="poor"):
    kwargs = dict(
        frame_idx=100, timestamp_sec=3.3, player_id=player_id,
        is_serve=True, score=score, stance="good", ball_toss="good",
        contact_point="good", follow_through="good",
        landing_zone="deep", coaching_tip="Good."
    )
    kwargs[fault_field] = fault_val
    return ServeResult(**kwargs)


def test_panel_serve_summary_returns_correct_size():
    analytics = Analytics(filters={})
    results = [make_result(1, 8), make_result(1, 6), make_result(2, 7)]
    panel = analytics.panel_serve_summary(results, (300, 300))
    assert panel.shape == (300, 300, 3)


def test_panel_serve_summary_empty_results():
    analytics = Analytics(filters={})
    panel = analytics.panel_serve_summary([], (300, 300))
    assert panel.shape == (300, 300, 3)
