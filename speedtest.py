from matplotlib import pyplot as plt
import numpy as np
import librosa
import librosa.display
from matplotlib.animation import FuncAnimation
from FourierTransform import FastFourierTransform, RealFastFourierTransform
import time

def frame_audio(audio, FFT_size=2048, fps=24, sample_rate=22050, window='hann'):
    hop_size = int(sample_rate / fps)
    num_frames = 1 + int((len(audio) - FFT_size) / hop_size)
    frames = np.zeros((num_frames, FFT_size))
    window_func = np.hanning(FFT_size) if window == 'hann' else np.ones(FFT_size)
    for i in range(num_frames):
        frames[i] = audio[i * hop_size:i * hop_size + FFT_size] * window_func
    return frames

audio, sr = librosa.load('GoodMorningAlarm.mp3', sr=None)
frames = frame_audio(audio, FFT_size=2048, fps=24, sample_rate=sr)

# Start the timer
start_time = time.time()

dft_results = np.array([FastFourierTransform(frame) for frame in frames])

# Stop the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(str(elapsed_time) + " s")

# Start the timer
start_time = time.time()

dft_results = np.array([RealFastFourierTransform(frame) for frame in frames])

# Stop the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(str(elapsed_time) + " s")






