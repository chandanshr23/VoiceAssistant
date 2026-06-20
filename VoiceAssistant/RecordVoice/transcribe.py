# transcribe.py
from faster_whisper import WhisperModel
import torch

model = WhisperModel("base", device="cuda", compute_type="float16")
segments, info = model.transcribe("test.wav")

print(f"Language: {info.language}")
for s in segments:
    print(f"[{s.start:.1f}s → {s.end:.1f}s] {s.text.strip()}")