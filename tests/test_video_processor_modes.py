"""Tests for process_video mode constants and VideoProcessor mode guard."""
from process_video import (
    MODE_VIDEO_ANALYSIS,
    MODE_SPLIT_RALLIES,
    MODE_DETECT_SERVE,
)


def test_mode_constants_are_distinct_strings():
    modes = [MODE_VIDEO_ANALYSIS, MODE_SPLIT_RALLIES, MODE_DETECT_SERVE]
    assert len(set(modes)) == 3
    assert all(isinstance(m, str) for m in modes)


def test_mode_constants_have_expected_values():
    assert MODE_VIDEO_ANALYSIS == "video_analysis"
    assert MODE_SPLIT_RALLIES  == "split_rallies"
    assert MODE_DETECT_SERVE   == "detect_serve"
