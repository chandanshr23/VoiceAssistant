"""
Silero VAD — DNN-based VAD, fallback for noisy audio.

Why two VADs? WebRTC GMM is extremely fast (microseconds per frame, CPU-only)
but degrades on noisy audio (SNR < 10dB, music background, HVAC noise).
Silero uses a small LSTM trained on diverse noisy datasets — significantly
more accurate on hard audio, at the cost of ~5ms/frame on CPU vs ~0.1ms for WebRTC.

Architecture of Silero VAD:
- Input: 512-sample windows (32ms at 16kHz), 64-sample step (4ms hop)
- Model: LSTM → FC → sigmoid
- Output: per-window probability of speech [0, 1]
- Size: ~1.8MB (ONNX) — fits in L2 cache, inference is fast

Cascade strategy (used in model_cascade.py):
- SNR >= 12dB  → WebRTC (fast, good enough)
- SNR 5-12dB   → Silero (noisy, need DNN)
- SNR < 5dB    → Silero + denoising preprocessing (very noisy)

This mirrors what the Sarvam candidate described as "CPU/GPU load reduction":
you don't run the expensive model on every call, only when needed.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SileroSegment:
    start_ms: float
    end_ms: float
    confidence: float           # mean speech probability in segment
    audio_float: np.ndarray     # float32 in [-1, 1]

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


class SileroVAD:
    """
    Wrapper around Silero VAD with the same segment-extraction API as WebRTCVAD.
    Loaded lazily so the import doesn't fail if torch isn't installed.

    Usage:
        vad = SileroVAD()
        vad.load()   # downloads ~1.8MB ONNX model on first run, cached after
        segments = vad.extract_segments(audio_float, sample_rate=16000)

    The model is stateful (LSTM hidden state) — call reset_state() between
    independent audio clips if processing in a loop, otherwise state leaks
    between clips and confidence scores drift.
    """

    MODEL_REPO = "snakers4/silero-vad"
    MODEL_NAME = "silero_vad"

    def __init__(
        self,
        threshold: float = 0.5,          # speech probability threshold
        min_speech_ms: int = 250,        # discard shorter segments
        min_silence_ms: int = 100,       # merge segments with shorter gaps
        speech_pad_ms: int = 30,         # padding around each segment
        sample_rate: int = 16000,
    ):
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms
        self.sample_rate = sample_rate
        self._model = None
        self._utils = None

    def load(self):
        """Download and cache Silero VAD. Requires torch."""
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir=self.MODEL_REPO,
                model=self.MODEL_NAME,
                force_reload=False,
                onnx=False,         # use PyTorch model, not ONNX
                verbose=False,
            )
            self._model = model
            self._utils = utils
            logger.info("Silero VAD loaded successfully")
        except ImportError:
            raise RuntimeError(
                "torch not installed. Install with: pip install torch --index-url https://download.pytorch.org/whl/cpu"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Silero VAD: {e}")

    def _ensure_loaded(self):
        if self._model is None:
            self.load()

    def reset_state(self):
        """Reset LSTM hidden state. Call between independent audio clips."""
        if self._model is not None:
            self._model.reset_states()

    def get_speech_timestamps(self, audio_float: np.ndarray) -> List[dict]:
        """
        Returns list of {start: int, end: int} in samples.
        This is Silero's native output format — we convert to ms in extract_segments.
        """
        self._ensure_loaded()
        import torch

        get_speech_timestamps = self._utils[0]
        audio_tensor = torch.from_numpy(audio_float).float()

        timestamps = get_speech_timestamps(
            audio_tensor,
            self._model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.speech_pad_ms,
        )
        return timestamps

    def extract_segments(self, audio_float: np.ndarray) -> List[SileroSegment]:
        """
        Main entry point. Returns SileroSegment list, parallel API to WebRTCVAD.
        """
        self.reset_state()
        timestamps = self.get_speech_timestamps(audio_float)
        segments = []
        for ts in timestamps:
            start_sample = ts["start"]
            end_sample = ts["end"]
            chunk = audio_float[start_sample:end_sample]

            # Compute mean speech probability over the segment
            # (Silero doesn't expose per-frame probs in this API, so we re-score)
            confidence = self._score_chunk(chunk)

            segments.append(SileroSegment(
                start_ms=start_sample / self.sample_rate * 1000,
                end_ms=end_sample / self.sample_rate * 1000,
                confidence=confidence,
                audio_float=chunk,
            ))
        return segments

    def _score_chunk(self, chunk: np.ndarray) -> float:
        """
        Run Silero on a chunk and return mean speech probability.
        Used for confidence scoring after segment extraction.
        """
        if self._model is None or len(chunk) < 512:
            return 1.0
        try:
            import torch
            tensor = torch.from_numpy(chunk).float().unsqueeze(0)
            with torch.no_grad():
                prob = self._model(tensor, self.sample_rate).item()
            return float(prob)
        except Exception:
            return 1.0


# ── Denoising preprocessing (for SNR < 5dB) ───────────────────────────────

def spectral_subtraction_denoise(
    audio_float: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 128,
    noise_estimation_frames: int = 10,
    alpha: float = 2.0,
    beta: float = 0.01,
) -> np.ndarray:
    """
    Spectral subtraction — classic signal-processing denoiser.
    No ML required. Works for stationary noise (HVAC, fan hum, white noise).
    Fails for non-stationary noise (music, babble).

    Algorithm:
    1. Estimate noise spectrum from first N frames (assumed to be silence/noise)
    2. STFT → subtract alpha * noise spectrum from magnitude
    3. Floor negative values to beta (prevents musical noise)
    4. ISTFT back to time domain

    Interview context: this is what "CPU/GPU load reduction" partially means.
    A fast denoiser on CPU → cleaner audio → better VAD accuracy → fewer
    false-positive segments → less Whisper invocations → lower GPU load.
    """
    # STFT
    window = np.hanning(n_fft)
    num_frames = (len(audio_float) - n_fft) // hop_length + 1

    if num_frames < noise_estimation_frames + 1:
        return audio_float  # too short to denoise

    # Build STFT frame by frame
    stft = np.array([
        np.fft.rfft(audio_float[i * hop_length: i * hop_length + n_fft] * window)
        for i in range(num_frames)
    ])  # shape: (num_frames, n_fft//2 + 1)

    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Estimate noise from first N frames
    noise_mag = np.mean(magnitude[:noise_estimation_frames], axis=0)

    # Spectral subtraction with flooring
    enhanced_mag = magnitude - alpha * noise_mag
    enhanced_mag = np.maximum(enhanced_mag, beta * magnitude)

    # Reconstruct
    enhanced_stft = enhanced_mag * np.exp(1j * phase)

    # ISTFT with overlap-add
    output = np.zeros(len(audio_float))
    for i, frame in enumerate(enhanced_stft):
        time_frame = np.fft.irfft(frame) * window
        start = i * hop_length
        output[start: start + n_fft] += time_frame

    # Normalize
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val * np.max(np.abs(audio_float))

    return output.astype(np.float32)
