import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from serve_detector import ServeCandidate
from serve_analyzer import OllamaServeAnalyzer, ServeResult


def make_ollama_response(content: str) -> MagicMock:
    """Build a mock matching the ollama ChatResponse object (attribute-based, not dict)."""
    msg = MagicMock()
    msg.content = content
    resp = MagicMock()
    resp.message = msg
    return resp


def make_candidate(player_id=1, frame_idx=100):
    return ServeCandidate(
        frame_idx=frame_idx,
        timestamp_sec=3.33,
        player_id=player_id,
        ball_pos=(150.0, 200.0),
        frame_small=np.zeros((720, 1280, 3), dtype=np.uint8),
    )


VALID_RESPONSE = """{
  "is_serve": true,
  "score": 8,
  "stance": "good",
  "ball_toss": "good",
  "contact_point": "good",
  "follow_through": "good",
  "landing_zone": "deep",
  "coaching_tip": "Great serve, keep it up."
}"""

REJECTED_RESPONSE = '{"is_serve": false, "score": 0, "stance": "unknown", "ball_toss": "unknown", "contact_point": "unknown", "follow_through": "unknown", "landing_zone": "unknown", "coaching_tip": ""}'


@patch("serve_analyzer.ollama")
def test_valid_serve_stored(mock_ollama):
    mock_ollama.chat.return_value = make_ollama_response(VALID_RESPONSE)
    analyzer = OllamaServeAnalyzer(model="qwen2.5vl:7b", workers=1)
    analyzer.submit(make_candidate(player_id=1))
    analyzer.shutdown()
    results = analyzer.get_results()
    assert len(results) == 1
    assert results[0].score == 8
    assert results[0].player_id == 1
    assert results[0].is_serve is True


@patch("serve_analyzer.ollama")
def test_rejected_serve_discarded(mock_ollama):
    mock_ollama.chat.return_value = make_ollama_response(REJECTED_RESPONSE)
    analyzer = OllamaServeAnalyzer(model="qwen2.5vl:7b", workers=1)
    analyzer.submit(make_candidate(player_id=2))
    analyzer.shutdown()
    results = analyzer.get_results()
    assert len(results) == 0


@patch("serve_analyzer.ollama")
def test_malformed_json_discarded(mock_ollama):
    mock_ollama.chat.return_value = make_ollama_response("not json at all")
    analyzer = OllamaServeAnalyzer(model="qwen2.5vl:7b", workers=1)
    analyzer.submit(make_candidate())
    analyzer.shutdown()
    assert len(analyzer.get_results()) == 0
