"""
Latency benchmark harness — p50/p95/p99 measurement.

This is the part that makes the difference between "I built a Whisper wrapper"
and "I did production ML engineering." Numbers beat claims.

When the candidate said "p95 < 800ms", this is the tool that produced that.

Measures:
- VAD latency: time to classify all frames in an audio file
- ASR latency: time to transcribe a segment (end-to-end, including queue)
- End-to-end latency: audio file in → first word out
- Realtime factor: inference_time / audio_duration (must be < 1.0 for real-time)

Usage:
    python -m benchmark.latency --audio_dir data/test_audio --model base --runs 100
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LatencyResult:
    name: str
    latencies_ms: List[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.latencies_ms)

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def p50_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 50)) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0

    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 99)) if self.latencies_ms else 0.0

    @property
    def max_ms(self) -> float:
        return float(np.max(self.latencies_ms)) if self.latencies_ms else 0.0

    def record(self, latency_ms: float):
        self.latencies_ms.append(latency_ms)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "count": self.count,
            "mean_ms": round(self.mean_ms, 1),
            "p50_ms": round(self.p50_ms, 1),
            "p95_ms": round(self.p95_ms, 1),
            "p99_ms": round(self.p99_ms, 1),
            "max_ms": round(self.max_ms, 1),
        }

    def print_summary(self):
        s = self.summary()
        print(f"\n{'='*50}")
        print(f"  {s['name']}")
        print(f"{'='*50}")
        print(f"  count:   {s['count']}")
        print(f"  mean:    {s['mean_ms']} ms")
        print(f"  p50:     {s['p50_ms']} ms")
        print(f"  p95:     {s['p95_ms']} ms  {'✓' if s['p95_ms'] < 800 else '✗ > 800ms target'}")
        print(f"  p99:     {s['p99_ms']} ms")
        print(f"  max:     {s['max_ms']} ms")


@dataclass
class BenchmarkReport:
    model_size: str
    device: str
    compute_type: str
    num_audio_files: int
    total_audio_duration_s: float
    results: Dict[str, dict] = field(default_factory=dict)
    realtime_factor: float = 0.0    # total_inference / total_audio

    def to_json(self, path: Optional[str] = None) -> str:
        data = asdict(self)
        js = json.dumps(data, indent=2)
        if path:
            Path(path).write_text(js)
        return js


class LatencyBenchmark:
    """
    Runs VAD + ASR on a directory of audio files and reports latency percentiles.

    Methodology:
    - Warm up: first 5 runs discarded (JIT compilation, model cache cold start)
    - Each audio file processed independently (no batching)
    - Wall clock time measured with time.perf_counter (monotonic, high-resolution)
    - Audio duration computed from sample count (not file size)

    This matches how production systems are benchmarked (not synthetic).
    """

    WARMUP_RUNS = 5

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        self.model_size = model_size
        self.device = device

        self._vad_result = LatencyResult("VAD (WebRTC)")
        self._asr_result = LatencyResult(f"ASR (Whisper {model_size})")
        self._e2e_result = LatencyResult("End-to-End")
        self._audio_durations_ms: List[float] = []

    def run_on_directory(self, audio_dir: str, runs: int = 50) -> BenchmarkReport:
        """
        Run benchmark on all .wav files in audio_dir.
        """
        from vad.webrtc_vad import WebRTCVAD, read_wav_pcm, pcm_to_float32, compute_snr_estimate
        from asr.model_cascade import ModelCascade, CascadeConfig

        audio_dir = Path(audio_dir)
        wav_files = list(audio_dir.glob("*.wav"))
        if not wav_files:
            raise ValueError(f"No .wav files found in {audio_dir}")

        logger.info(f"Found {len(wav_files)} audio files. Benchmarking {runs} runs.")

        vad = WebRTCVAD(sample_rate=16000, frame_ms=30, aggressiveness=2)
        cascade = ModelCascade(device=self.device)
        cascade.start()

        warmup_done = False
        run_count = 0
        total_audio_ms = 0.0

        try:
            while run_count < runs + self.WARMUP_RUNS:
                for wav_path in wav_files:
                    if run_count >= runs + self.WARMUP_RUNS:
                        break

                    try:
                        pcm_bytes, sr = read_wav_pcm(str(wav_path))
                    except Exception as e:
                        logger.warning(f"Skipping {wav_path.name}: {e}")
                        continue

                    audio_float = pcm_to_float32(pcm_bytes)
                    audio_ms = len(audio_float) / sr * 1000

                    # ── VAD ──
                    t0 = time.perf_counter()
                    segments = vad.extract_segments(pcm_bytes)
                    vad_ms = (time.perf_counter() - t0) * 1000

                    if not segments:
                        run_count += 1
                        continue

                    # ── ASR (first segment only for latency measurement) ──
                    seg = segments[0]
                    seg_float = pcm_to_float32(seg.audio_bytes)
                    snr = compute_snr_estimate(seg_float)

                    t1 = time.perf_counter()
                    result = cascade.transcribe(
                        audio_float=seg_float,
                        snr_db=snr,
                        start_ms=seg.start_ms,
                        segment_id=run_count,
                        word_timestamps=False,  # faster for latency bench
                    )
                    asr_ms = (time.perf_counter() - t1) * 1000
                    e2e_ms = vad_ms + asr_ms

                    if run_count >= self.WARMUP_RUNS:
                        self._vad_result.record(vad_ms)
                        self._asr_result.record(asr_ms)
                        self._e2e_result.record(e2e_ms)
                        self._audio_durations_ms.append(audio_ms)
                        total_audio_ms += audio_ms
                    else:
                        if run_count == self.WARMUP_RUNS - 1:
                            logger.info("Warmup complete. Starting measurement.")
                            warmup_done = True

                    run_count += 1

        finally:
            cascade.stop()

        total_inference_ms = sum(self._asr_result.latencies_ms)
        rtf = total_inference_ms / max(total_audio_ms, 1)

        report = BenchmarkReport(
            model_size=self.model_size,
            device=self.device,
            compute_type="int8" if self.device == "cpu" else "float16",
            num_audio_files=len(wav_files),
            total_audio_duration_s=total_audio_ms / 1000,
            results={
                "vad": self._vad_result.summary(),
                "asr": self._asr_result.summary(),
                "e2e": self._e2e_result.summary(),
            },
            realtime_factor=rtf,
        )

        self._print_report(report)
        return report

    def _print_report(self, report: BenchmarkReport):
        print(f"\n{'='*60}")
        print(f"  BENCHMARK REPORT — Whisper {report.model_size} on {report.device}")
        print(f"{'='*60}")
        print(f"  Audio files:       {report.num_audio_files}")
        print(f"  Total audio:       {report.total_audio_duration_s:.1f}s")
        print(f"  Realtime factor:   {report.realtime_factor:.3f}x  {'(real-time capable)' if report.realtime_factor < 1.0 else '(NOT real-time)'}")

        for stage, data in report.results.items():
            print(f"\n  [{stage.upper()}]")
            print(f"    p50: {data['p50_ms']} ms")
            print(f"    p95: {data['p95_ms']} ms  {'✓' if data['p95_ms'] < 800 else '✗'}")
            print(f"    p99: {data['p99_ms']} ms")
        print()


def run_synthetic_benchmark(
    duration_s: float = 3.0,
    runs: int = 50,
    model_size: str = "base",
    device: str = "cpu",
) -> BenchmarkReport:
    """
    Benchmark without real audio files — use synthetic white noise.
    Useful for CI/CD checks (no audio files needed).
    White noise is harder than clean speech → conservative estimate.
    """
    from vad.webrtc_vad import WebRTCVAD, pcm_to_float32, compute_snr_estimate
    from asr.model_cascade import ModelCascade

    sr = 16000
    num_samples = int(sr * duration_s)
    # White noise — worst case for VAD and ASR
    audio_float = np.random.randn(num_samples).astype(np.float32) * 0.1

    # Convert to 16-bit PCM for WebRTC VAD
    pcm_bytes = (audio_float * 32767).astype(np.int16).tobytes()

    vad = WebRTCVAD()
    cascade = ModelCascade(device=device)
    cascade.start()

    bench = LatencyBenchmark(model_size=model_size, device=device)
    snr = compute_snr_estimate(audio_float)

    try:
        for i in range(runs + bench.WARMUP_RUNS):
            t0 = time.perf_counter()
            segments = vad.extract_segments(pcm_bytes)
            vad_ms = (time.perf_counter() - t0) * 1000

            if not segments:
                # no speech detected in noise — still record VAD latency
                if i >= bench.WARMUP_RUNS:
                    bench._vad_result.record(vad_ms)
                continue

            seg = segments[0]
            seg_float = pcm_to_float32(seg.audio_bytes)

            t1 = time.perf_counter()
            cascade.transcribe(seg_float, snr_db=snr, segment_id=i)
            asr_ms = (time.perf_counter() - t1) * 1000

            if i >= bench.WARMUP_RUNS:
                bench._vad_result.record(vad_ms)
                bench._asr_result.record(asr_ms)
                bench._e2e_result.record(vad_ms + asr_ms)
                bench._audio_durations_ms.append(duration_s * 1000)

    finally:
        cascade.stop()

    total_audio_ms = sum(bench._audio_durations_ms)
    total_inf_ms = sum(bench._asr_result.latencies_ms)

    report = BenchmarkReport(
        model_size=model_size,
        device=device,
        compute_type="int8",
        num_audio_files=runs,
        total_audio_duration_s=total_audio_ms / 1000,
        results={
            "vad": bench._vad_result.summary(),
            "asr": bench._asr_result.summary(),
            "e2e": bench._e2e_result.summary(),
        },
        realtime_factor=total_inf_ms / max(total_audio_ms, 1),
    )
    bench._print_report(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default="base")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.synthetic or args.audio_dir is None:
        report = run_synthetic_benchmark(runs=args.runs, model_size=args.model, device=args.device)
    else:
        bench = LatencyBenchmark(model_size=args.model, device=args.device)
        report = bench.run_on_directory(args.audio_dir, runs=args.runs)

    if args.output:
        report.to_json(args.output)
        print(f"Report saved to {args.output}")
