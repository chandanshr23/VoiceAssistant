"""
Model cascade — route each segment to the smallest model that can handle it.

This is the primary CPU/GPU load reduction strategy. Instead of running
every audio segment through Whisper base/large, we classify the segment
first and pick the cheapest model that meets the accuracy bar.

Decision tree:
    SNR >= 12dB AND duration < 5s AND no_speech_prob historically low
        → tiny  (10x faster than base, ~2-3% WER increase on clean audio)
    SNR 8-12dB OR duration 5-10s
        → base  (default, good balance)
    SNR < 8dB OR duration > 10s OR previous segment was garbled
        → small (heavier, better on noise and long-form)

Real-time factor reference (CPU, INT8):
    tiny:   ~0.15x   (1s audio in 150ms)
    base:   ~0.35x   (1s audio in 350ms)
    small:  ~0.7x    (1s audio in 700ms)
    medium: ~1.5x    (1s audio in 1500ms — not realtime on CPU)
    large:  ~3x      (GPU required for realtime)

Source: faster-whisper benchmarks, CTranslate2 team.

Interview relevance: when the candidate said "p95 < 800ms", this cascade
is how you hit it. Without it, even base on CPU is ~350ms * realtime_factor
which means a 3s segment takes 1050ms → p95 > 800ms. With cascade routing
clean segments to tiny, you get ~450ms for the same 3s segment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from asr.whisper_engine import ModelSize, WhisperEngine, InferenceRequest, TranscriptSegment, ComputeType

logger = logging.getLogger(__name__)


@dataclass
class CascadeConfig:
    # SNR thresholds (dB)
    snr_high: float = 12.0          # above this → tiny
    snr_low: float = 8.0            # below this → small

    # Duration thresholds (ms)
    duration_short: float = 5000    # below this → tiny eligible
    duration_long: float = 10000    # above this → small

    # Quality signal from previous inference
    avg_log_prob_threshold: float = -0.5    # below this → previous was bad, upgrade model

    # Minimum confidence to trust a tiny-model result; re-run with base if below
    retry_confidence_threshold: float = 0.7


class ModelCascade:
    """
    Routes inference requests to the right model size.

    Maintains three WhisperEngine instances (lazy-loaded) at different sizes.
    Each engine has its own worker thread and queue.

    Memory layout:
    - tiny:  ~74MB RAM
    - base:  ~145MB RAM
    - small: ~487MB RAM
    Total: ~706MB — fits in 1GB RAM, reasonable for a production server.

    On GPU (A10G/T4):
    - Load all three into VRAM simultaneously
    - Swap is zero-cost (all already on device)
    - On CPU: load lazily, accept cold-start latency on first use
    """

    def __init__(
        self,
        device: str = "cpu",
        config: Optional[CascadeConfig] = None,
        cpu_threads: int = 4,
    ):
        self.device = device
        self.config = config or CascadeConfig()
        self.cpu_threads = cpu_threads

        self._engines: dict[ModelSize, Optional[WhisperEngine]] = {
            ModelSize.TINY: None,
            ModelSize.BASE: None,
            ModelSize.SMALL: None,
        }
        self._last_segment_quality: float = 0.0     # rolling quality signal

    def start(self, preload: list[ModelSize] = None):
        """
        Start engines. Pass preload=[ModelSize.BASE] to only eagerly load
        one model (others lazy). On resource-constrained machines, preloading
        all three is wasteful if most traffic hits base anyway.
        """
        preload = preload or [ModelSize.BASE]
        for size in preload:
            self._get_or_create_engine(size)

    def stop(self):
        for engine in self._engines.values():
            if engine is not None:
                engine.stop()

    def transcribe(
        self,
        audio_float: np.ndarray,
        snr_db: float,
        start_ms: float = 0.0,
        segment_id: int = 0,
        language: Optional[str] = None,
        word_timestamps: bool = True,
        allow_retry: bool = True,
    ) -> TranscriptSegment:
        """
        Select model → transcribe → optionally retry with larger model.
        """
        duration_ms = len(audio_float) / 16.0
        selected = self._select_model(snr_db, duration_ms)

        logger.debug(
            f"Segment {segment_id}: SNR={snr_db:.1f}dB, "
            f"duration={duration_ms:.0f}ms → {selected.value}"
        )

        engine = self._get_or_create_engine(selected)
        request = InferenceRequest(
            audio_float=audio_float,
            start_ms=start_ms,
            segment_id=segment_id,
            model_size=selected,
            language=language,
            word_timestamps=word_timestamps,
        )
        result = engine.transcribe_sync(request)

        # Retry with larger model if quality is low
        if allow_retry and self._should_retry(result, selected):
            larger = self._next_model(selected)
            if larger:
                logger.info(
                    f"Segment {segment_id}: retrying with {larger.value} "
                    f"(avg_log_prob={result.avg_log_prob:.3f})"
                )
                engine2 = self._get_or_create_engine(larger)
                request.model_size = larger
                result = engine2.transcribe_sync(request)

        self._last_segment_quality = result.avg_log_prob
        return result

    def get_all_stats(self) -> dict:
        return {
            size.value: engine.get_stats()
            for size, engine in self._engines.items()
            if engine is not None
        }

    # ── Private ──────────────────────────────────────────────────────────────

    def _select_model(self, snr_db: float, duration_ms: float) -> ModelSize:
        cfg = self.config

        # Degrade model if previous segment was low quality
        quality_penalty = self._last_segment_quality < cfg.avg_log_prob_threshold

        if (
            snr_db >= cfg.snr_high
            and duration_ms <= cfg.duration_short
            and not quality_penalty
        ):
            return ModelSize.TINY

        if snr_db < cfg.snr_low or duration_ms > cfg.duration_long or quality_penalty:
            return ModelSize.SMALL

        return ModelSize.BASE

    def _should_retry(self, result: TranscriptSegment, current_model: ModelSize) -> bool:
        if current_model == ModelSize.SMALL:
            return False    # already at largest we support
        # Retry if: hallucination likely (no_speech_prob high) or very low log prob
        if result.no_speech_prob > 0.8:
            return False    # probably silence, not worth retrying
        return result.avg_log_prob < self.config.avg_log_prob_threshold

    def _next_model(self, current: ModelSize) -> Optional[ModelSize]:
        order = [ModelSize.TINY, ModelSize.BASE, ModelSize.SMALL]
        idx = order.index(current)
        return order[idx + 1] if idx + 1 < len(order) else None

    def _get_or_create_engine(self, size: ModelSize) -> WhisperEngine:
        if self._engines[size] is None:
            compute = (
                ComputeType.FLOAT16 if self.device == "cuda" else ComputeType.INT8
            )
            engine = WhisperEngine(
                model_size=size,
                device=self.device,
                compute_type=compute,
                cpu_threads=self.cpu_threads,
            )
            engine.start()
            self._engines[size] = engine
        return self._engines[size]
