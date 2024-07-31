import subprocess
import time
from matplotlib import pyplot as plt
import numpy as np
import librosa
from scipy.signal import convolve2d
import librosa.display
from matplotlib.animation import FuncAnimation
import concurrent.futures
from FourierTransform import FastFourierTransform,  manual_rfftfreq

# Consider 2d Smotthing Data before aggrgating?
# Try to get max of overall, and then for each box, find the max over time and do max / max for box, and then scale that box to that factor so that everything eventually reaches Max 

fps = 30
FFT_size = 2048*4
num_bands = 64

# Start the timer
start_time = time.time()

# Load audio
audio, sr = librosa.load('WatchOut.mp3', sr=None)

print(sr)

def FrameAudio(audio, FFT_size=2048, fps=24, sample_rate=22050, window='hann'):
    hop_size = int(sample_rate / fps)
    num_frames = 1 + int((len(audio) - FFT_size) / hop_size)
    frames = np.zeros((num_frames, FFT_size))
    window_func = np.hanning(FFT_size) if window == 'hann' else np.ones(FFT_size)
    for i in range(num_frames):
        frames[i] = audio[i * hop_size:i * hop_size + FFT_size] * window_func
    return frames

def GetMagnitudes (frame):
    # return np.abs(np.fft.rfft(frame))
    dft_results = FastFourierTransform(frame)

    dft_results = dft_results[:len(dft_results)//2 +1:]
    magnitude_spectrum = np.array([np.sqrt(real**2 + imag**2) for real, imag in dft_results])
    return magnitude_spectrum

# Aggregate magnitudes into frequency bands
def AgreggateFrequencies(magnitudes, num_bands, sample_rate, FFT_size):
    freq_bins = manual_rfftfreq(FFT_size, 1/sample_rate)
    bands = np.zeros(num_bands)
    log_freqs = np.logspace(np.log10(freq_bins[1]), np.log10(freq_bins[-1]), num_bands + 1)
    for i in range(num_bands):
        start_idx = np.searchsorted(freq_bins, log_freqs[i])
        end_idx = np.searchsorted(freq_bins, log_freqs[i + 1], side='right')
        if end_idx > start_idx:
            bands[i] = np.mean(magnitudes[start_idx:end_idx])
    return bands / np.max(bands)  # Normalize the bands

# Smooth data for visual effect
def SmoothData(data, window_len=15):
    return np.convolve(data, np.ones(window_len)/window_len, mode='same')

def SmoothData2D(data, window_len=5):
    kernel = np.ones((window_len, window_len)) / (window_len)
    return convolve2d(data, kernel, mode='same', boundary='wrap')

def Parallel (frame):
    return np.array(AgreggateFrequencies(GetMagnitudes(frame), num_bands, sr, FFT_size))

if __name__ == '__main__':
    
    hop_size = int(sr / fps)

    print(hop_size)

    frames = FrameAudio(audio, sample_rate=sr, fps=fps, FFT_size=FFT_size)
    # frames = [frames[i] for i in range(2500)]
    print("Length" + str(len(frames[0])))

    dft_results = 0

    with concurrent.futures.ProcessPoolExecutor() as executor:
        dft_results = np.array(list(executor.map(Parallel, frames)))

    dft_results_smoothed = SmoothData2D(dft_results)

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
    ani.save('animation.mp4', writer='ffmpeg', fps=fps)

    command = [
        "C:\\FFmpeg\\App\\bin\\ffmpeg.exe",
        "-i", "animation.mp4",
        "-i", "WatchOut.mp3",
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        "audio_visualize.mp4"
    ]

    print("Merging Video and Audio")

    result = subprocess.run(command, capture_output=True, text=True)

    # Stop the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(str(elapsed_time) + " s")

    plt.show()