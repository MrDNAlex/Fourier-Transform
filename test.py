from matplotlib import pyplot as plt
import numpy as np
import librosa
from matplotlib.animation import FuncAnimation

def frame_audio(audio, FFT_size=2048, fps=24, sample_rate=22050, window='hann'):
    hop_size = int(sample_rate / fps)
    num_frames = 1 + int((len(audio) - FFT_size) / hop_size)
    frames = np.zeros((num_frames, FFT_size))
    if window == 'hann':
        window_func = np.hanning(FFT_size)
    else:
        window_func = np.ones(FFT_size)
    for n in range(num_frames):
        frames[n] = audio[n*hop_size:n*hop_size+FFT_size] * window_func
    return frames

def get_magnitudes(frame):
    dft_results = np.fft.rfft(frame)
    magnitude_spectrum = np.abs(dft_results)
    return magnitude_spectrum

def aggregate_frequencies(magnitudes, num_bands, sample_rate, FFT_size):
    freq_bins = np.fft.rfftfreq(FFT_size, 1/sample_rate)
    bands = np.zeros(num_bands)
    
    # Logarithmic scaling
    log_freqs = np.logspace(np.log10(freq_bins[1]), np.log10(freq_bins[-1]), num_bands)
    
    for i in range(num_bands - 1):
        start_bin = np.where(freq_bins >= log_freqs[i])[0]
        end_bin = np.where(freq_bins < log_freqs[i + 1])[0] if i + 1 < num_bands else np.arange(len(freq_bins))

        if len(start_bin) > 0 and len(end_bin) > 0:
            start_idx = start_bin[0]
            end_idx = end_bin[-1] if i + 1 < num_bands else len(freq_bins) - 1
            if end_idx > start_idx:
                bands[i] = np.mean(magnitudes[start_idx:end_idx])
    
    # Normalize bands
    if np.max(bands) > 0:
        bands = bands / np.max(bands)
    
    return bands

def smooth_data(data, window_len=5):
    smoothed_data = np.convolve(data, np.ones(window_len) / window_len, mode='same')
    return smoothed_data

audio, sr = librosa.load('Overkill.mp3', sr=None)
frames = frame_audio(audio, FFT_size=2048, fps=24, sample_rate=sr)
frames = [frames[i] for i in range(480)]
fps = 24
hop_length = int(sr / fps)

num_bands = 64
dft_results = np.array([aggregate_frequencies(get_magnitudes(frame), num_bands, sr, 2048) for frame in frames])
dft_results_smoothed = np.array([smooth_data(frame) for frame in dft_results])

fig, ax = plt.subplots()
line, = ax.plot(dft_results_smoothed[0])
ax.set_ylim(0, 1)  # Since we normalized the bands

def update(frame):
    line.set_ydata(dft_results_smoothed[frame])
    return line,

ani = FuncAnimation(fig, update, frames=len(dft_results_smoothed), blit=True)
ani.save('audio_visualizer2.gif', writer='pillow', fps=24)
plt.show()
