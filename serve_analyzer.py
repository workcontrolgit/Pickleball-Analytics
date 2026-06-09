"""
Ollama Serve Analyzer

Purpose
-------
Scores serve candidates using a local Ollama vision LLM in a background
thread pool. Never blocks the video processing pipeline.

Usage
-----
    analyzer = OllamaServeAnalyzer()
    analyzer.submit(candidate)         # non-blocking
    ...
    analyzer.shutdown()                # wait for all pending calls
    results = analyzer.get_results()   # list[ServeResult]
"""

from __future__ import annotations

import base64
import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from loguru import logger

import cv2

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore

from serve_detector import ServeCandidate


PROMPT = """You are a pickleball coach analyzing a serve frame.
Respond in valid JSON only — no explanation, no markdown, no code block.

{
  "is_serve": true or false,
  "score": integer 1-10,
  "stance": "good" or "poor" or "unknown",
  "ball_toss": "good" or "too_low" or "too_far" or "unknown",
  "contact_point": "good" or "too_low" or "too_far" or "unknown",
  "follow_through": "good" or "poor" or "unknown",
  "landing_zone": "deep" or "short" or "wide" or "unknown",
  "coaching_tip": "one short sentence"
}"""


@dataclass
class ServeResult:
    frame_idx: int
    timestamp_sec: float
    player_id: Optional[int]
    is_serve: bool
    score: int
    stance: str
    ball_toss: str
    contact_point: str
    follow_through: str
    landing_zone: str
    coaching_tip: str


class OllamaServeAnalyzer:
    TIMEOUT_SEC = 30

    def __init__(self, model: str = "qwen2.5vl:7b", workers: int = 2):
        self._model = model
        self._results: list[ServeResult] = []
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=workers)

    def submit(self, candidate: ServeCandidate) -> None:
        """Submit a candidate for async analysis. Non-blocking."""
        logger.bind(frame_idx=candidate.frame_idx, player_id=candidate.player_id).info("ollama_submit")
        self._executor.submit(self._analyze, candidate)

    def get_results(self) -> list[ServeResult]:
        """Return all confirmed serve results accumulated so far."""
        with self._lock:
            return list(self._results)

    def shutdown(self) -> None:
        """Wait for all pending analyses to complete."""
        self._executor.shutdown(wait=True)

    def _analyze(self, candidate: ServeCandidate) -> None:
        if ollama is None:
            logger.warning("ollama_not_installed")
            return
        log = logger.bind(frame_idx=candidate.frame_idx, player_id=candidate.player_id)
        log.info("ollama_call_start")
        try:
            _, buf = cv2.imencode(".jpg", candidate.frame_small)
            image_b64 = base64.b64encode(buf).decode()

            response = ollama.chat(
                model=self._model,
                messages=[{
                    "role": "user",
                    "content": PROMPT,
                    "images": [image_b64],
                }],
                options={"num_predict": 200},
            )
            content = response.message.content
            # Strip markdown code fences if present
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            data = json.loads(content)
            log.bind(is_serve=data.get("is_serve"), score=data.get("score")).info("ollama_response")
            if not data.get("is_serve", False):
                return

            result = ServeResult(
                frame_idx=candidate.frame_idx,
                timestamp_sec=candidate.timestamp_sec,
                player_id=candidate.player_id,
                is_serve=True,
                score=int(data.get("score", 5)),
                stance=data.get("stance", "unknown"),
                ball_toss=data.get("ball_toss", "unknown"),
                contact_point=data.get("contact_point", "unknown"),
                follow_through=data.get("follow_through", "unknown"),
                landing_zone=data.get("landing_zone", "unknown"),
                coaching_tip=data.get("coaching_tip", ""),
            )
            with self._lock:
                self._results.append(result)
            log.bind(score=result.score).info("serve_result_stored")

        except Exception:
            log.bind(error=traceback.format_exc()).warning("ollama_call_failed")
