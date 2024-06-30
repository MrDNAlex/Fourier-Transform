from matplotlib import pyplot as plt
import numpy as np
import librosa
import librosa.display
from matplotlib.animation import FuncAnimation
from FourierTransform import FastFourierTransform

# def frame_audio(audio, FFT_size=2048, hop_size=512, sample_rate=22050):
#     num_frames = 1 + int((len(audio) - FFT_size) / hop_size)
#     frames = np.zeros((num_frames, FFT_size))
#     for n in range(num_frames):
#         frames[n] = audio[n*hop_size:n*hop_size+FFT_size]
#     return frames

def frame_audio(audio, FFT_size=2048, fps=24, sample_rate=22050, window='hann'):
    # Calculate hop size based on the desired fps
    hop_size = int(sample_rate / fps)

    print(hop_size)
    
    # Calculate the number of frames needed based on the hop size
    num_frames = 1 + int((len(audio) - FFT_size) / hop_size)
    frames = np.zeros((num_frames, FFT_size))
    
    # Select the window function
    if window == 'hann':
        window_func = np.hanning(FFT_size)
    else:
        window_func = np.ones(FFT_size)  # Default to a rectangular window if none specified

    # Apply window function and frame the audio
    for n in range(num_frames):
        frames[n] = audio[n*hop_size:n*hop_size+FFT_size] * window_func

    return frames

# def frame_audio(audio, FFT_size=2048, hop_size=512, sample_rate=22050, window='hann'):
#     num_frames = 1 + int((len(audio) - FFT_size) / hop_size)
#     frames = np.zeros((num_frames, FFT_size))
#     window_func = np.hanning(FFT_size)  # Using Hann window
#     for n in range(num_frames):
#         frames[n] = audio[n*hop_size:n*hop_size+FFT_size] * window_func
#     return frames

def plot_spectrogram(D, sr, hop_length):
    # Convert DFT output to magnitude
    D_magnitude = np.sqrt(D[:, 0]**2 + D[:, 1]**2).T
    plt.figure(figsize=(12, 6))

    x = [i for i in range(len(D_magnitude))]

    plt.plot(x,D_magnitude)
    # librosa.display.specshow(librosa.amplitude_to_db(D_magnitude, ref=np.max),
    #                          sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.title('Power Spectrogram')
    #plt.colorbar(format='%+2.0f dB')
    #plt.tight_layout()
    plt.show()

def GetMagnitudes (frame):

    dft_results = FastFourierTransform(frame)

    magnitude_spectrum = np.array([np.sqrt(real**2 + imag**2) for real, imag in dft_results])

    return magnitude_spectrum

def aggregate_frequencies(magnitudes, num_bands):
    bands = np.zeros(num_bands)
    for i in range(num_bands):
        start = int(np.floor((i / num_bands) * len(magnitudes)))
        end = int(np.floor(((i + 1) / num_bands) * len(magnitudes)))
        bands[i] = np.mean(magnitudes[start:end])
    return bands

def smooth_data(data, window_len=10):
    # Apply a simple moving average for smoothing
    smoothed_data = np.convolve(data, np.ones(window_len) / window_len, mode='same')
    return smoothed_data

audio, sr = librosa.load('Overkill.mp3', sr=None)
frames = frame_audio(audio, FFT_size=2048, fps=24, sample_rate=sr)

frames = [frames[i] for i in range(1000)]

fps = 24
hop_length = int(sr / fps)  # Calculate hop length

print(len(frames))

dft_results = np.array([GetMagnitudes(frame) for frame in frames])

print("Finished Trasnforming")

# Initialize the plot
fig, ax = plt.subplots()
line, = ax.plot(dft_results[0])  # Start with the first frame's magnitude spectrum
ax.set_ylim(0, np.max(dft_results))  # Set the y-axis limit

print("Animating")

# Update function for the animation
def update(frame):
    line.set_ydata(dft_results[frame])  # Update the plot with the new data for the current frame
    return line,

print("Finished Animating, Saving File")


num_bands = 64
dft_results = np.array([aggregate_frequencies(GetMagnitudes(frame), num_bands) for frame in frames])
dft_results_smoothed = np.array([smooth_data(frame) for frame in dft_results])

fig, ax = plt.subplots()
line, = ax.plot(dft_results_smoothed[0])
ax.set_ylim(0, np.max(dft_results_smoothed))

def update(frame):
    line.set_ydata(dft_results_smoothed[frame])
    return line,

ani = FuncAnimation(fig, update, frames=len(dft_results_smoothed), blit=True)
ani.save('audio_visualizer.gif', writer='pillow', fps=24)
plt.show()