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


from unittest.mock import patch, MagicMock


def _make_processor(mode):
    """Create a VideoProcessor with all heavy dependencies mocked out."""
    with patch("process_video.BallTracker"), \
         patch("process_video.PlayerTracker"), \
         patch("process_video.CourtDetector"), \
         patch("process_video.Analytics"), \
         patch("process_video.ServeDetector"), \
         patch("process_video.OllamaServeAnalyzer"), \
         patch("process_video.VideoProcessor._make_output_dir", return_value="/tmp/run"), \
         patch("process_video.setup_logger"):
        from process_video import VideoProcessor
        return VideoProcessor.__new__(VideoProcessor)


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
