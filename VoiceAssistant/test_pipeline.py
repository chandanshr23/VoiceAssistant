"""
python test_pipeline.py

End-to-end test using synthetic audio — no real .wav file needed.
Generates a fake speech-like signal, runs it through VAD, then transcribes
with Whisper. Verifies the full pipeline is wired up correctly.

Expected output:
    VAD found N segments
    Segment 0: 0ms → Xms (Xms)
    Transcribing with Whisper base on cuda ...
    [some text or empty — synthetic audio won't produce real words]
    Pipeline OK. Latency: Xms
"""

import time
import wave
import struct
import math
import tempfile
import os

import numpy as np

def generate_test_audio(duration_s=3.0, sample_rate=16000):
    """
    Generate synthetic audio: 0.5s silence → 2s sine-wave 'speech' → 0.5s silence.
    Real speech has harmonics — we use a 200Hz fundamental + harmonics.
    This is enough for VAD to detect as speech (energy + spectral shape).
    """
    n_samples = int(duration_s * sample_rate)
    audio = np.zeros(n_samples, dtype=np.float32)

    # Add harmonics at 200Hz (voiced speech fundamental range)
    speech_start = int(0.5 * sample_rate)
    speech_end = int(2.5 * sample_rate)
    t = np.arange(speech_end - speech_start) / sample_rate
    for harmonic in [1, 2, 3, 4, 5]:
        freq = 200 * harmonic
        amplitude = 0.3 / harmonic
        audio[speech_start:speech_end] += amplitude * np.sin(2 * np.pi * freq * t)

    # Add a little noise throughout (realistic)
    audio += np.random.randn(n_samples).astype(np.float32) * 0.005

    return audio, sample_rate


def audio_to_pcm_bytes(audio_float, sample_rate):
    """Convert float32 audio to 16-bit PCM bytes and save as WAV."""
    audio_int16 = (audio_float * 32767).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    with wave.open(tmp.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return tmp.name, audio_int16.tobytes()


def main():
    print("Generating synthetic test audio...")
    audio_float, sr = generate_test_audio(duration_s=3.0)
    wav_path, pcm_bytes = audio_to_pcm_bytes(audio_float, sr)
    print(f"  Audio: 3s, {sr}Hz, mono, 16-bit PCM")
    print(f"  Saved to: {wav_path}")

    # ── VAD ──────────────────────────────────────────────────────────────
    print("\nRunning WebRTC VAD...")
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from vad.webrtc_vad import WebRTCVAD
    vad = WebRTCVAD(sample_rate=sr, aggressiveness=2)
    t0 = time.perf_counter()
    segments = vad.extract_segments(pcm_bytes)
    vad_ms = (time.perf_counter() - t0) * 1000

    print(f"  VAD latency: {vad_ms:.1f}ms")
    print(f"  Found {len(segments)} segment(s)")
    for i, seg in enumerate(segments):
        print(f"  Segment {i}: {seg.start_ms:.0f}ms → {seg.end_ms:.0f}ms ({seg.duration_ms:.0f}ms)")

    if not segments:
        print("\n  No segments found — try lowering aggressiveness to 1")
        print("  (synthetic audio may not perfectly mimic speech)")
        return

    # ── Whisper ──────────────────────────────────────────────────────────
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    print(f"\nTranscribing with Whisper base on {device} ({compute})...")

    from faster_whisper import WhisperModel
    t1 = time.perf_counter()
    model = WhisperModel("base", device=device, compute_type=compute)
    load_ms = (time.perf_counter() - t1) * 1000
    print(f"  Model load: {load_ms:.0f}ms (cached after first run)")

    seg = segments[0]
    seg_float = np.frombuffer(seg.audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    t2 = time.perf_counter()
    result_iter, info = model.transcribe(
        seg_float,
        language="en",
        beam_size=5,
        vad_filter=False,
    )
    text = " ".join(s.text.strip() for s in result_iter)
    asr_ms = (time.perf_counter() - t2) * 1000

    print(f"  Transcript: '{text}' (synthetic audio — may be empty or garbled)")
    print(f"  ASR latency: {asr_ms:.1f}ms")
    print(f"  Detected language: {info.language} (prob={info.language_probability:.2f})")

    # ── Summary ──────────────────────────────────────────────────────────
    total_ms = vad_ms + asr_ms
    audio_ms = 3000
    rtf = total_ms / audio_ms
    print(f"\n{'='*40}")
    print(f"  VAD:    {vad_ms:.1f}ms")
    print(f"  ASR:    {asr_ms:.1f}ms")
    print(f"  Total:  {total_ms:.1f}ms for 3s audio")
    print(f"  RTF:    {rtf:.3f}x {'(real-time capable)' if rtf < 1.0 else '(not real-time)'}")
    print(f"  Status: Pipeline OK ✓")

    os.unlink(wav_path)


if __name__ == "__main__":
    main()
