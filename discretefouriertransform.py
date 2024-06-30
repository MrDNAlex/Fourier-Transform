from matplotlib import pyplot as plt
import numpy as np
import librosa
import librosa.display

pi = np.pi

def frame_audio(audio, FFT_size=2048, hop_size=512, sample_rate=22050):
    num_frames = 1 + int((len(audio) - FFT_size) / hop_size)
    frames = np.zeros((num_frames, FFT_size))
    for n in range(num_frames):
        frames[n] = audio[n*hop_size:n*hop_size+FFT_size]
    return frames


def ReconstructSignal (N, amplitudes, phases) -> list[float]:

    signal = []

    for n in range(N):

        sum = 0
        for k in range(N):

            sum += amplitudes[k] * np.cos(2*pi*k*(n/N) + phases[k])

        signal.append(sum)

    return signal


def fft_simple(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]


def fft(x):
    N = len(x)
    if N <= 1:
        return x  # Base case: Returns the array directly if it has 1 or fewer elements.
    
    # Recursively apply FFT to even and odd indexed parts of the array
    even = fft(x[0::2])
    odd = fft(x[1::2])
    
    # Separate results for even and odd into real and imaginary components
    if isinstance(even[0], tuple):
        # If the elements are tuples, unpack them
        even_real, even_imag = zip(*even)
        odd_real, odd_imag = zip(*odd)
    else:
        # For the first recursion where elements are still pure real
        even_real, even_imag = even, [0] * len(even)
        odd_real, odd_imag = odd, [0] * len(odd)

    # Prepare lists to hold the real and imaginary parts of the output
    real_part = [0] * N
    imaginary_part = [0] * N

    for k in range(N // 2):
        # Calculate the twiddle factors
        cos_val = np.cos(2 * np.pi * k / N)
        sin_val = np.sin(2 * np.pi * k / N)

        # Compute the terms to be added and subtracted, using real and imaginary parts
        t_real = cos_val * odd_real[k] + sin_val * odd_imag[k]
        t_imag = -sin_val * odd_real[k] + cos_val * odd_imag[k]

        # Combine the even and odd parts
        real_part[k] = even_real[k] + t_real
        imaginary_part[k] = even_imag[k] + t_imag

        real_part[k + N//2] = even_real[k] - t_real
        imaginary_part[k + N//2] = even_imag[k] - t_imag

    # Zip the real and imaginary parts together for the next level of recursion
    return list(zip(real_part, imaginary_part))


# Signal must be comprised of 
def Signal(x:float) -> float:
    return 3*np.cos(x*2*pi*20) + 2*np.cos(x*2*pi* 40)

# Amplitude = real^2 + imag^2 all sqrt

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


# x = np.linspace(0, 10, 100)

# signal = [Signal(i) for i in x]

# print("Processing Signal")

# output = DiscreteFourierTransform(signal)

# print("Signal Processed")

# amplitudes = [np.sqrt(re**2 + im**2) for re, im in output]

# phase = [np.angle(re,im) for re, im in output]

# plt.subplot(211)
# plt.plot(x, signal, label="Signal")

# new_signal = ReconstructSignal(len(output),amplitudes, phase)

# plt.subplot(212)
# plt.plot(x,new_signal , label="New Signal")

# plt.show()



# Load audio
audio, sr = librosa.load('GoodMorningAlarm.mp3', sr=None)
frames = frame_audio(audio, FFT_size=2048, hop_size=512, sample_rate=sr)

print(len(frames))

dft_results = np.array([fft(frame) for frame in frames])

# Plot the spectrogram
plot_spectrogram(dft_results, sr, hop_length=512)





