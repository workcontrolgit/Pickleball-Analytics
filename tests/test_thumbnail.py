"""Tests for the _extract_thumbnail module-level helper."""
import numpy as np
from unittest.mock import patch, MagicMock


def _import_extract():
    """Import after patching so we don't need a real display."""
    import sys
    for mod in ("customtkinter", "tkinter", "tkinter.filedialog"):
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()
    import main
    return main._extract_thumbnail


def test_returns_pil_image_for_valid_video():
    """Should return a PIL Image when OpenCV can read a frame."""
    from PIL import Image
    fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    with patch("main.cv2") as mock_cv2:
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, fake_frame)
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.cvtColor.return_value = fake_frame
        mock_cv2.COLOR_BGR2RGB = 4
        mock_cv2.resize.return_value = np.zeros((146, 260, 3), dtype=np.uint8)
        mock_cv2.INTER_AREA = 3

        fn = _import_extract()
        result = fn("fake.mp4", thumb_w=260, thumb_h=146)

    assert isinstance(result, Image.Image)


def test_returns_none_for_unreadable_video():
    """Should return None (not raise) when OpenCV cannot open the file."""
    with patch("main.cv2") as mock_cv2:
        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = cap

        fn = _import_extract()
        result = fn("bad.mp4", thumb_w=260, thumb_h=146)

    assert result is None
