# record.py
import sounddevice as sd
import soundfile as sf
import numpy as np

DURATION = 5      # seconds
SAMPLE_RATE = 16000

print("Recording in 3...")
import time; time.sleep(1)
print("Recording in 2...")
time.sleep(1)
print("Recording in 1...")
time.sleep(1)
print("Speak now!")

audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
sd.wait()

sf.write('test.wav', audio, SAMPLE_RATE)
print("Saved to test.wav")