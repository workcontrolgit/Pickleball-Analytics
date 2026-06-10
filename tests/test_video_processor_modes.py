"""Tests for process_video mode constants and VideoProcessor mode guard."""
from process_video import (
    MODE_VIDEO_ANALYSIS,
    MODE_SPLIT_RALLIES,
    MODE_DETECT_SERVE,
)
from unittest.mock import patch


def test_mode_constants_are_distinct_strings():
    modes = [MODE_VIDEO_ANALYSIS, MODE_SPLIT_RALLIES, MODE_DETECT_SERVE]
    assert len(set(modes)) == 3
    assert all(isinstance(m, str) for m in modes)


def test_mode_constants_have_expected_values():
    assert MODE_VIDEO_ANALYSIS == "video_analysis"
    assert MODE_SPLIT_RALLIES  == "split_rallies"
    assert MODE_DETECT_SERVE   == "detect_serve"


def test_video_analysis_has_analytics_no_serve_detector():
    """VIDEO_ANALYSIS mode: analytics present, serve objects absent."""
    with patch("process_video.BallTracker"), \
         patch("process_video.PlayerTracker"), \
         patch("process_video.CourtDetector"), \
         patch("process_video.Analytics") as MockAnalytics, \
         patch("process_video.ServeDetector") as MockServeDetector, \
         patch("process_video.OllamaServeAnalyzer") as MockAnalyzer, \
         patch("process_video.VideoProcessor._make_output_dir", return_value="/tmp/run"), \
         patch("process_video.setup_logger"):
        from process_video import VideoProcessor, MODE_VIDEO_ANALYSIS
        p = VideoProcessor("fake.mp4", {}, mode=MODE_VIDEO_ANALYSIS)
        MockAnalytics.assert_called_once()
        MockServeDetector.assert_not_called()
        MockAnalyzer.assert_not_called()
        assert hasattr(p, "analytics")
        assert not hasattr(p, "serve_detector")
        assert not hasattr(p, "serve_analyzer")


def test_split_rallies_has_analytics_no_serve_detector():
    """SPLIT_RALLIES mode: analytics present, serve objects absent."""
    with patch("process_video.BallTracker"), \
         patch("process_video.PlayerTracker"), \
         patch("process_video.CourtDetector"), \
         patch("process_video.Analytics") as MockAnalytics, \
         patch("process_video.ServeDetector") as MockServeDetector, \
         patch("process_video.OllamaServeAnalyzer") as MockAnalyzer, \
         patch("process_video.VideoProcessor._make_output_dir", return_value="/tmp/run"), \
         patch("process_video.setup_logger"):
        from process_video import VideoProcessor, MODE_SPLIT_RALLIES
        p = VideoProcessor("fake.mp4", {}, mode=MODE_SPLIT_RALLIES)
        MockAnalytics.assert_called_once()
        MockServeDetector.assert_not_called()
        MockAnalyzer.assert_not_called()
        assert hasattr(p, "analytics")
        assert not hasattr(p, "serve_detector")
        assert not hasattr(p, "serve_analyzer")


def test_detect_serve_has_serve_detector_no_analytics():
    """DETECT_SERVE mode: serve objects present, analytics absent."""
    with patch("process_video.BallTracker"), \
         patch("process_video.PlayerTracker"), \
         patch("process_video.CourtDetector"), \
         patch("process_video.Analytics") as MockAnalytics, \
         patch("process_video.ServeDetector") as MockServeDetector, \
         patch("process_video.OllamaServeAnalyzer") as MockAnalyzer, \
         patch("process_video.VideoProcessor._make_output_dir", return_value="/tmp/run"), \
         patch("process_video.setup_logger"):
        from process_video import VideoProcessor, MODE_DETECT_SERVE
        p = VideoProcessor("fake.mp4", {}, mode=MODE_DETECT_SERVE)
        MockAnalytics.assert_not_called()
        MockServeDetector.assert_called_once()
        MockAnalyzer.assert_called_once()
        assert not hasattr(p, "analytics")
        assert hasattr(p, "serve_detector")
        assert hasattr(p, "serve_analyzer")


# --- Task 7: MODE_DETECT_RALLIES ---

def test_mode_detect_rallies_constant():
    from process_video import MODE_DETECT_RALLIES
    assert MODE_DETECT_RALLIES == "detect_rallies"

def test_detect_rallies_has_rally_detector_no_analytics():
    """DETECT_RALLIES mode: RallyDetector present, Analytics absent."""
    with patch("process_video.BallTracker"), \
         patch("process_video.PlayerTracker"), \
         patch("process_video.CourtDetector"), \
         patch("process_video.Analytics") as MockAnalytics, \
         patch("process_video.ServeDetector"), \
         patch("process_video.OllamaServeAnalyzer"), \
         patch("process_video.RallyDetector") as MockRallyDetector, \
         patch("process_video.VideoProcessor._make_output_dir", return_value="/tmp/run"), \
         patch("process_video.setup_logger"):
        from process_video import VideoProcessor, MODE_DETECT_RALLIES
        p = VideoProcessor("fake.mp4", {}, mode=MODE_DETECT_RALLIES)
        MockAnalytics.assert_not_called()
        MockRallyDetector.assert_called_once()
        assert hasattr(p, "rally_detector")
        assert not hasattr(p, "analytics")


# --- Task 8: MODE_CLIP_FROM_REPORT ---

def test_mode_clip_from_report_constant():
    from process_video import MODE_CLIP_FROM_REPORT
    assert MODE_CLIP_FROM_REPORT == "clip_from_report"

def test_clip_from_report_init_no_detectors():
    """CLIP_FROM_REPORT mode: no detectors or analytics instantiated."""
    with patch("process_video.BallTracker") as MockBT, \
         patch("process_video.PlayerTracker") as MockPT, \
         patch("process_video.CourtDetector") as MockCD, \
         patch("process_video.Analytics") as MockAnalytics, \
         patch("process_video.ServeDetector") as MockSD, \
         patch("process_video.OllamaServeAnalyzer") as MockAnalyzer, \
         patch("process_video.RallyDetector") as MockRD, \
         patch("process_video.VideoProcessor._make_output_dir", return_value="/tmp/run"), \
         patch("process_video.setup_logger"):
        from process_video import VideoProcessor, MODE_CLIP_FROM_REPORT
        p = VideoProcessor("fake.mp4", {}, mode=MODE_CLIP_FROM_REPORT,
                           rally_report_path="/tmp/rally_report.json")
        MockAnalytics.assert_not_called()
        MockBT.assert_not_called()
        MockPT.assert_not_called()
        MockCD.assert_not_called()
        MockSD.assert_not_called()
        MockRD.assert_not_called()
        assert p.rally_report_path == "/tmp/rally_report.json"
