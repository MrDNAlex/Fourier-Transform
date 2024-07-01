import matplotlib.pyplot as plt
import numpy as np
import librosa
from matplotlib.animation import FuncAnimation
import subprocess


# Frame the audio with a window function
def frame_audio(audio, FFT_size=2048, fps=24, sample_rate=22050, window='hann'):
    hop_size = int(sample_rate / fps)
    num_frames = 1 + int((len(audio) - FFT_size) / hop_size)
    frames = np.zeros((num_frames, FFT_size))
    #window_func = np.hanning(FFT_size) if window == 'hann' else np.ones(FFT_size)
    for i in range(num_frames):
        frames[i] = audio[i * hop_size:i * hop_size + FFT_size] #* window_func
    return frames

# Calculate magnitudes from DFT results
def get_magnitudes(frame):
    return np.abs(np.fft.rfft(frame))

# Aggregate magnitudes into frequency bands
def aggregate_frequencies(magnitudes, num_bands, sample_rate, FFT_size):
    freq_bins = np.fft.rfftfreq(FFT_size, 1/sample_rate)
    bands = np.zeros(num_bands)
    log_freqs = np.logspace(np.log10(freq_bins[1]), np.log10(freq_bins[-1]), num_bands + 1)
    for i in range(num_bands):
        start_idx = np.searchsorted(freq_bins, log_freqs[i])
        end_idx = np.searchsorted(freq_bins, log_freqs[i + 1], side='right')
        if end_idx > start_idx:
            bands[i] = np.mean(magnitudes[start_idx:end_idx])
    return bands / np.max(bands)  # Normalize the bands

# Smooth data for visual effect
def smooth_data(data, window_len=15):
    return np.convolve(data, np.ones(window_len)/window_len, mode='same')

# Load audio
audio, sr = librosa.load('Overkill.mp3', sr=None)
frames = frame_audio(audio, sample_rate=sr)
frames = [frames[i] for i in range(2500)]
num_bands = 64
dft_results = np.array([aggregate_frequencies(get_magnitudes(frame), num_bands, sr, 2048) for frame in frames])

# Apply smoothing for better visual effect
dft_results_smoothed = np.array([smooth_data(bands) for bands in dft_results])

# Set up the plot
fig, ax = plt.subplots()
bars = ax.bar(range(num_bands), dft_results_smoothed[0])

# Animation update function
def update(frame):
    for bar, h in zip(bars, dft_results_smoothed[frame]):
        bar.set_height(h)
    return bars

print("Saving Animation")

# Create animation
ani = FuncAnimation(fig, update, frames=len(dft_results_smoothed), blit=True)
ani.save('animation.mp4', writer='ffmpeg', fps=24)

command = [
    "C:\\FFmpeg\\bin\\ffmpeg.exe",
    "-i", "animation.mp4",
    "-i", "Overkill.mp3",
    "-c:v", "copy",
    "-c:a", "aac",
    "-strict", "experimental",
    "audio_visualize.mp4"
]

print("Merging Video and Audio")

result = subprocess.run(command, capture_output=True, text=True)


plt.show()
