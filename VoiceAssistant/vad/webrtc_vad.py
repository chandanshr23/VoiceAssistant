"""
WebRTC VAD module — GMM-based frame-level speech detector.

WebRTC's VAD (originally from Google) internally uses a GMM trained on
speech-vs-noise. It operates on fixed-size frames (10/20/30ms) of 16-bit
16kHz PCM. No GPU needed — this is the CPU-efficient path.

Interview-ready talking points:
- Why WebRTC over energy-based: energy threshold fails on noise; GMM learned
  speech-specific spectral features (zero-crossing rate, energy distribution)
  so it's noise-robust without being a full DNN.
- Aggressiveness 0-3: internally maps to different GMM decision boundaries.
  0 = less filtering (keeps borderline frames), 3 = aggressive noise rejection.
- Why 30ms frames: trades latency for accuracy. 10ms has lower latency but
  more false positives; 30ms is the sweet spot for telephony-grade audio.
"""

import collections
import wave
import contextlib
import numpy as np
import webrtcvad
from dataclasses import dataclass, field
from typing import Generator, List, Tuple


VALID_SAMPLE_RATES = {8000, 16000, 32000, 48000}
VALID_FRAME_MS = {10, 20, 30}


@dataclass
class VADFrame:
    index: int          # frame index in audio
    start_ms: float     # start time in ms
    end_ms: float       # end time in ms
    is_speech: bool
    raw_bytes: bytes


@dataclass
class SpeechSegment:
    start_ms: float
    end_ms: float
    audio_bytes: bytes
    frame_count: int = 0

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


class RingBuffer:
    """
    Fixed-size circular buffer used for pre-roll padding.

    Why you need this: when VAD detects start-of-speech, the actual speech
    onset was a few frames earlier (VAD has reaction delay). The ring buffer
    holds the last N frames so we can backfill them into the segment,
    recovering the attack of the word. This directly impacts WER — missing
    word-initial phonemes degrades accuracy significantly.
    """

    def __init__(self, maxlen: int):
        self._buf: collections.deque = collections.deque(maxlen=maxlen)

    def append(self, frame: VADFrame):
        self._buf.append(frame)

    def drain(self) -> List[VADFrame]:
        frames = list(self._buf)
        self._buf.clear()
        return frames

    def __len__(self):
        return len(self._buf)


class WebRTCVAD:
    """
    Production-grade WebRTC VAD with smoothing and segment extraction.

    Key design decisions:
    1. Frame-level classification → segment-level output (not raw frame flags)
    2. Pre-roll padding: capture N frames before detected speech onset
    3. Post-roll padding: keep N frames after detected speech end
       (handles "umm" pauses mid-sentence — don't split them into two segments)
    4. Min segment duration: discard very short "segments" (mouth noise, clicks)
    5. Triggered state machine: avoids rapid on/off flickering

    This is standard in production ASR pipelines. The raw frame output from
    webrtcvad is too noisy to feed directly to Whisper.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        aggressiveness: int = 2,
        padding_ms: int = 300,       # post-speech padding before segment close
        pre_roll_ms: int = 150,      # frames to backfill before speech onset
        min_speech_ms: int = 250,    # discard segments shorter than this
        speech_ratio_trigger: float = 0.75,   # fraction of ring that must be speech to open
        speech_ratio_release: float = 0.25,   # fraction below which segment closes
    ):
        if sample_rate not in VALID_SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {VALID_SAMPLE_RATES}, got {sample_rate}")
        if frame_ms not in VALID_FRAME_MS:
            raise ValueError(f"frame_ms must be one of {VALID_FRAME_MS}, got {frame_ms}")
        if not (0 <= aggressiveness <= 3):
            raise ValueError("aggressiveness must be 0-3")

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.aggressiveness = aggressiveness
        self.padding_ms = padding_ms
        self.pre_roll_ms = pre_roll_ms
        self.min_speech_ms = min_speech_ms
        self.speech_ratio_trigger = speech_ratio_trigger
        self.speech_ratio_release = speech_ratio_release

        self._vad = webrtcvad.Vad(aggressiveness)
        self._frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # 16-bit = 2 bytes/sample

        # Ring buffer size = padding_ms // frame_ms
        padding_frames = padding_ms // frame_ms
        self._ring: RingBuffer = RingBuffer(maxlen=padding_frames)
        self._pre_roll_frames = pre_roll_ms // frame_ms

        self._triggered = False
        self._voiced_frames: List[VADFrame] = []

    @property
    def frame_bytes(self) -> int:
        return self._frame_bytes

    def _frame_generator(self, audio_bytes: bytes) -> Generator[VADFrame, None, None]:
        """Chop raw PCM bytes into fixed-size frames."""
        n = len(audio_bytes)
        frame_idx = 0
        offset = 0
        while offset + self._frame_bytes <= n:
            chunk = audio_bytes[offset: offset + self._frame_bytes]
            start_ms = frame_idx * self.frame_ms
            yield VADFrame(
                index=frame_idx,
                start_ms=start_ms,
                end_ms=start_ms + self.frame_ms,
                is_speech=False,    # filled below
                raw_bytes=chunk,
            )
            offset += self._frame_bytes
            frame_idx += 1

    def classify_frames(self, audio_bytes: bytes) -> List[VADFrame]:
        """
        Run GMM classifier on every frame. Returns list of VADFrame with
        is_speech filled in. No smoothing applied yet — raw frame labels.
        """
        frames = []
        for frame in self._frame_generator(audio_bytes):
            try:
                frame.is_speech = self._vad.is_speech(frame.raw_bytes, self.sample_rate)
            except Exception:
                frame.is_speech = False
            frames.append(frame)
        return frames

    def extract_segments(self, audio_bytes: bytes) -> List[SpeechSegment]:
        """
        Full pipeline: frame classify → state machine smoothing → segment list.

        State machine logic:
        - UNTRIGGERED: accumulate frames in ring buffer. If speech fraction in
          ring > speech_ratio_trigger → switch to TRIGGERED, dump ring as
          start of segment (pre-roll).
        - TRIGGERED: accumulate voiced frames. Keep ring updated too.
          If speech fraction in ring < speech_ratio_release → close segment
          (but include the ring frames as post-roll padding).

        This is the same logic used in WebRTC's own voice activity detection
        for RTP stream handling. The ratio-based hysteresis prevents chatter.
        """
        frames = self.classify_frames(audio_bytes)
        segments: List[SpeechSegment] = []
        ring = RingBuffer(maxlen=self.padding_ms // self.frame_ms)
        voiced_frames: List[VADFrame] = []
        triggered = False

        for frame in frames:
            if not triggered:
                ring.append(frame)
                # Check ratio of speech frames in the ring
                ring_frames = list(ring._buf)
                num_voiced = sum(1 for f in ring_frames if f.is_speech)
                ratio = num_voiced / max(len(ring_frames), 1)

                if ratio >= self.speech_ratio_trigger:
                    triggered = True
                    # Backfill ring as pre-roll
                    voiced_frames = ring.drain()
            else:
                voiced_frames.append(frame)
                ring.append(frame)
                ring_frames = list(ring._buf)
                num_voiced = sum(1 for f in ring_frames if f.is_speech)
                ratio = num_voiced / max(len(ring_frames), 1)

                if ratio <= self.speech_ratio_release:
                    # End of speech — include ring as post-roll already in voiced_frames
                    triggered = False
                    seg = self._build_segment(voiced_frames)
                    if seg and seg.duration_ms >= self.min_speech_ms:
                        segments.append(seg)
                    voiced_frames = []
                    ring.drain()

        # Flush any remaining open segment (audio ended while speaking)
        if triggered and voiced_frames:
            seg = self._build_segment(voiced_frames)
            if seg and seg.duration_ms >= self.min_speech_ms:
                segments.append(seg)

        return segments

    def _build_segment(self, frames: List[VADFrame]) -> SpeechSegment | None:
        if not frames:
            return None
        return SpeechSegment(
            start_ms=frames[0].start_ms,
            end_ms=frames[-1].end_ms,
            audio_bytes=b"".join(f.raw_bytes for f in frames),
            frame_count=len(frames),
        )


# ── Utilities ──────────────────────────────────────────────────────────────

def read_wav_pcm(path: str) -> Tuple[bytes, int]:
    """
    Read a WAV file and return (raw_pcm_bytes, sample_rate).
    Enforces 16kHz mono 16-bit — WebRTC's hard requirements.
    Use librosa/ffmpeg upstream to convert arbitrary audio.
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if channels != 1:
        raise ValueError(f"WAV must be mono, got {channels} channels. Convert with: ffmpeg -i input.wav -ac 1 -ar 16000 -acodec pcm_s16le output.wav")
    if sampwidth != 2:
        raise ValueError(f"WAV must be 16-bit (sampwidth=2), got {sampwidth}")
    if rate not in VALID_SAMPLE_RATES:
        raise ValueError(f"Sample rate must be one of {VALID_SAMPLE_RATES}, got {rate}")

    return frames, rate


def pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """
    Convert 16-bit PCM bytes to float32 array in [-1, 1].
    This is what Whisper expects as input (after its mel spectrogram).
    """
    audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    return audio_int16.astype(np.float32) / 32768.0


def compute_snr_estimate(audio_float: np.ndarray, frame_ms: int = 30, sample_rate: int = 16000) -> float:
    """
    Rough SNR estimate using energy variance across frames.
    
    Used for model cascade selection: if SNR < threshold, route to Silero
    (DNN-based VAD) which handles noisy audio better than WebRTC GMM.

    Real SNR requires a noise reference signal. This is a heuristic —
    compute the 10th percentile frame energy as noise floor, mean as signal.
    Good enough for cascade routing, not for reporting.
    """
    frame_samples = int(sample_rate * frame_ms / 1000)
    frames = [
        audio_float[i: i + frame_samples]
        for i in range(0, len(audio_float) - frame_samples, frame_samples)
    ]
    if not frames:
        return 0.0

    energies = np.array([np.mean(f ** 2) for f in frames])
    noise_floor = np.percentile(energies, 10) + 1e-10
    signal_power = np.mean(energies)
    snr_db = 10 * np.log10(signal_power / noise_floor)
    return float(snr_db)
