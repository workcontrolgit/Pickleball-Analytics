import os
import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
from process_video import VideoProcessor


def make_processor(tmp_path, mode='full'):
    filters = {"player_heatmap": True, "ball_heatmap": True,
                "kitchen_detection": True, "court_zone": True}
    with patch.object(VideoProcessor, '__init__', lambda self, vp, f, m: None):
        p = VideoProcessor.__new__(VideoProcessor)
    p.video_path = str(tmp_path / "fake.mp4")
    p.filters = filters
    p.mode = mode
    p.output_dir = str(tmp_path)
    p.analytics = MagicMock()
    p.analytics._long_rallies = []
    return p


def make_fake_video(path, num_frames=60, fps=30, w=64, h=64):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for _ in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_no_clips_when_no_long_rallies(tmp_path):
    p = make_processor(tmp_path)
    p.analytics._long_rallies = []
    p._clip_long_rallies(total_frames=60, fps=30)
    clips = list(tmp_path.glob("rally_*.mp4"))
    assert len(clips) == 0


def test_one_clip_written_for_one_long_rally(tmp_path):
    video_path = tmp_path / "fake.mp4"
    make_fake_video(video_path, num_frames=120, fps=30)
    p = make_processor(tmp_path)
    p.video_path = str(video_path)
    p.analytics._long_rallies = [(10, 50, 6)]
    p._clip_long_rallies(total_frames=120, fps=30)
    clips = list(tmp_path.glob("rally_*.mp4"))
    assert len(clips) == 1
    assert (tmp_path / "rally_01.mp4").exists()


def test_two_clips_written_for_two_long_rallies(tmp_path):
    video_path = tmp_path / "fake.mp4"
    make_fake_video(video_path, num_frames=300, fps=30)
    p = make_processor(tmp_path)
    p.video_path = str(video_path)
    p.analytics._long_rallies = [(10, 40, 6), (100, 150, 8)]
    p._clip_long_rallies(total_frames=300, fps=30)
    assert (tmp_path / "rally_01.mp4").exists()
    assert (tmp_path / "rally_02.mp4").exists()


def test_buffer_clamped_to_zero(tmp_path):
    video_path = tmp_path / "fake.mp4"
    make_fake_video(video_path, num_frames=60, fps=30)
    p = make_processor(tmp_path)
    p.video_path = str(video_path)
    # Rally starts at frame 5 — buffer of 60 frames would go negative
    p.analytics._long_rallies = [(5, 30, 6)]
    # Should not raise
    p._clip_long_rallies(total_frames=60, fps=30)
    assert (tmp_path / "rally_01.mp4").exists()
