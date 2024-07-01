import subprocess
from matplotlib import pyplot as plt
import numpy as np
import librosa
import librosa.display
from matplotlib.animation import FuncAnimation
from FourierTransform import FastFourierTransform, RealFastFourierTransform, RealFastFourierTransformFrequency

def FrameAudio(audio, FFT_size=2048, fps=24, sample_rate=22050, window='hann'):
    hop_size = int(sample_rate / fps)
    num_frames = 1 + int((len(audio) - FFT_size) / hop_size)
    frames = np.zeros((num_frames, FFT_size))
    window_func = np.hanning(FFT_size) if window == 'hann' else np.ones(FFT_size)
    for i in range(num_frames):
        frames[i] = audio[i * hop_size:i * hop_size + FFT_size] * window_func
    return frames

def GetMagnitudes (frame):
    dft_results = FastFourierTransform(frame)

    dft_results = dft_results[:len(dft_results)//2 +1:]
    magnitude_spectrum = np.array([np.sqrt(real**2 + imag**2) for real, imag in dft_results])
    return magnitude_spectrum

def GetMagnitudes2 (frame):
    return np.abs(np.fft.rfft(frame))
    # dft_results = FastFourierTransform(frame)
    # magnitude_spectrum = np.array([np.sqrt(real**2 + imag**2) for real, imag in dft_results])
    # return magnitude_spectrum

# Aggregate magnitudes into frequency bands
def AgreggateFrequencies(magnitudes, num_bands, sample_rate, FFT_size):
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
def smooth_data(data, window_len=20):
    return np.convolve(data, np.ones(window_len)/window_len, mode='same')

fps = 24
FFT_size = 2048

# Load audio
audio, sr = librosa.load('Overkill.mp3', sr=None)

hop_size = int(sr / fps)

print(hop_size)

frames = FrameAudio(audio, sample_rate=sr, fps=fps, FFT_size=FFT_size)
frames = [frames[i] for i in range(100)]
num_bands = 64
dft_results = np.array([GetMagnitudes(frame) for frame in frames])

print(len(dft_results[0]))

dft_results2 = np.array([GetMagnitudes2(frame) for frame in frames])

print(len(dft_results2[0]))

dif = dft_results - dft_results2

print(dif)

print(np.sum(dif))
