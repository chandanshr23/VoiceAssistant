import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
BLOCK_SIZE = 4800       # 300ms chunks fed into queue
SILENCE_THRESHOLD = 500 # RMS below this = silence

model = WhisperModel("base", device="cuda", compute_type="float16")
audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

def transcribe_loop():
    buffer = np.array([], dtype=np.int16)
    silence_count = 0

    while True:
        chunk = audio_queue.get()
        chunk_int16 = (chunk[:, 0] * 32767).astype(np.int16)
        buffer = np.concatenate([buffer, chunk_int16])
        rms = np.sqrt(np.mean(chunk_int16.astype(np.float32)**2))

        if rms < SILENCE_THRESHOLD:
            silence_count += 1
        else:
            silence_count = 0

        # After 0.6s of silence following speech, transcribe what we have
        if silence_count >= 2 and len(buffer) > SAMPLE_RATE * 0.5:
            audio_float = buffer.astype(np.float32) / 32768.0
            segments, _ = model.transcribe(audio_float, language="en")
            text = " ".join(s.text.strip() for s in segments)
            if text:
                print(f">> {text}")
            buffer = np.array([], dtype=np.int16)
            silence_count = 0

print("Listening... (Ctrl+C to stop)")
t = threading.Thread(target=transcribe_loop, daemon=True)
t.start()

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                    blocksize=BLOCK_SIZE, callback=callback):
    while True:
        sd.sleep(100)