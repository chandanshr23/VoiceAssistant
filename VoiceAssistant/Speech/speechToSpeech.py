# speech_to_speech_cpu.py
import sounddevice as sd
import numpy as np
import queue
import threading
import time
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
import requests
import json

SAMPLE_RATE = 16000
BLOCK_SIZE = 4800
SILENCE_THRESHOLD = 400    # slightly more aggressive silence detection
SILENCE_FRAMES = 2         # 0.6s of silence triggers transcription

print("Loading Whisper tiny (CPU)...")
t0 = time.perf_counter()
stt_model = WhisperModel(
    "tiny",                 # tiny = 39MB, ~600ms on CPU vs base = 145MB, ~2s
    device="cpu",
    compute_type="int8",    # int8 = 2x faster than float32 on CPU, minimal quality loss
    cpu_threads=8,          # use all your cores
    num_workers=1,
)
print(f"  Loaded in {(time.perf_counter()-t0)*1000:.0f}ms")

print("Loading Kokoro TTS (CPU)...")
t0 = time.perf_counter()
tts = Kokoro("kokoro-v1.0.onnx", "voices.bin")
print(f"  Loaded in {(time.perf_counter()-t0)*1000:.0f}ms")

# Local LLM API config (LM Studio)
LOCAL_LLM_URL = "http://127.0.0.1:1234/v1/completions"  # LM Studio endpoint
LOCAL_LLM_MODEL = "local-model"  # LM Studio auto-detects loaded model


audio_queue = queue.Queue()
is_playing = threading.Event()   # block mic while TTS is playing

def callback(indata, frames, time_info, status):
    if not is_playing.is_set():  # ignore mic input while speaking
        audio_queue.put(indata.copy())

def get_reply(text: str) -> str:
    """
    Local LLM model via LM Studio API (OpenAI-compatible)
    System prompt keeps responses short = less TTS time = lower latency.
    """
    system_prompt = (
        "You are Donna, a voice assistant. "
        "Reply in 1-2 short sentences only. "
        "Never use lists or bullet points."
    )
    
    prompt = f"{system_prompt}\n\nUser: {text}\n\nAssistant:"
    
    try:
        response = requests.post(
            LOCAL_LLM_URL,
            json={
                "model": LOCAL_LLM_MODEL,
                "prompt": prompt,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 60,  # max tokens for short replies
            },
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        reply = result.get("choices", [{}])[0].get("text", "").strip()
        return reply if reply else "I didn't understand that."
    except Exception as e:
        print(f"Error calling LM Studio: {e}")
        return "Sorry, I'm having trouble thinking right now."

def speak(text: str):
    is_playing.set()
    samples, sr = tts.create(text, voice="af_heart", speed=1.1)  # slightly faster speech
    sd.play(samples, sr)
    sd.wait()
    is_playing.clear()

def transcribe_loop():
    buffer = np.array([], dtype=np.int16)
    silence_count = 0
    has_speech = False

    while True:
        chunk = audio_queue.get()
        chunk_int16 = (chunk[:, 0] * 32767).astype(np.int16)
        rms = np.sqrt(np.mean(chunk_int16.astype(np.float32) ** 2))

        if rms > SILENCE_THRESHOLD:
            has_speech = True
            silence_count = 0
            buffer = np.concatenate([buffer, chunk_int16])
        elif has_speech:
            silence_count += 1
            buffer = np.concatenate([buffer, chunk_int16])

            if silence_count >= SILENCE_FRAMES:
                # ── measure each stage ──────────────────────────
                audio_float = buffer.astype(np.float32) / 32768.0
                audio_duration_ms = len(audio_float) / SAMPLE_RATE * 1000

                t0 = time.perf_counter()
                segments, _ = stt_model.transcribe(
                    audio_float,
                    language="en",
                    beam_size=1,        # greedy decoding = faster, slight accuracy drop
                    best_of=1,
                    temperature=0.0,
                    vad_filter=True,    # whisper's built-in VAD trims extra silence
                    vad_parameters={"min_silence_duration_ms": 300},
                )
                text = " ".join(s.text.strip() for s in segments)
                stt_ms = (time.perf_counter() - t0) * 1000

                if text:
                    print(f"\nYou:      {text}")
                    print(f"          [STT: {stt_ms:.0f}ms for {audio_duration_ms:.0f}ms audio]")

                    t1 = time.perf_counter()
                    reply = get_reply(text)
                    llm_ms = (time.perf_counter() - t1) * 1000

                    print(f"Donna:    {reply}")
                    print(f"          [LLM: {llm_ms:.0f}ms]")

                    t2 = time.perf_counter()
                    speak(reply)
                    tts_ms = (time.perf_counter() - t2) * 1000

                    total_ms = stt_ms + llm_ms + tts_ms
                    print(f"          [TTS: {tts_ms:.0f}ms | Total: {total_ms:.0f}ms]")

                buffer = np.array([], dtype=np.int16)
                silence_count = 0
                has_speech = False

print("\nListening... (Ctrl+C to stop)\n")
t = threading.Thread(target=transcribe_loop, daemon=True)
t.start()

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='float32',
    blocksize=BLOCK_SIZE,
    callback=callback
):
    while True:
        sd.sleep(100)