import numpy as np

def DiscreteFourierTransform (input_signal: list[float]):

    output_signal = []

    N = len(input_signal)

    for k in range(N):

        real_sum = 0
        imaginary_sum = 0

        for n in range(N):

            angle = 2*np.pi * k * (n / N)

            real = input_signal[n] * np.cos(angle)

            imaginary = input_signal[n] * np.sin(angle)

            real_sum += real
            imaginary_sum += imaginary
        
        output_signal.append([real_sum, imaginary_sum])

    return output_signal

def FastFourierTransform (signal):

    N = len(signal)

    # if (N <= 1):
    #     return signal
    
    if N <= 1:
        return [(signal[0], 0)]
    
    # Compute Even and Odd Terms of the Fourier Transform
    even = FastFourierTransform(signal[0::2])
    odd = FastFourierTransform(signal[1::2])

    # Check if the first index of Even are a Tupple
    if isinstance(even[0], tuple):
        # Unpack the Tupples if they are Tupples
        even_real, even_imag = zip(*even)
        odd_real, odd_imag = zip(*odd)
    else:
        # Create a Tuple with the second part being an Array
        even_real, even_imag = even, [0] * len(even)
        odd_real, odd_imag = odd, [0] * len(odd)

    # Create Arrays for what will be returned
    real_part = [0] * N
    imaginary_part = [0] * N

    for k in range(N//2):
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

    return list(zip(real_part, imaginary_part))


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

def RealFastFourierTransform(x):

    # N = len(signal)

    # if (N <= 1):
    #     return signal

    # # Compute Even and Odd Terms of the Fourier Transform
    # even = FastFourierTransform(signal[0::2])
    # odd = FastFourierTransform(signal[1::2])

    # # Check if the first index of Even are a Tupple
    # if isinstance(even[0], tuple):
    #     # Unpack the Tupples if they are Tupples
    #     even_real, even_imag = zip(*even)
    #     odd_real, odd_imag = zip(*odd)
    # else:
    #     # Create a Tuple with the second part being an Array
    #     even_real, even_imag = even, [0] * len(even)
    #     odd_real, odd_imag = odd, [0] * len(odd)

    # # Create Arrays for what will be returned
    # real_part = [0] * (N//2 +1)
    # imaginary_part = [0] * (N//2 +1)

    # for k in range(N//2):
    #     # Calculate the twiddle factors
    #     cos_val = np.cos(2 * np.pi * k / N)
    #     sin_val = np.sin(2 * np.pi * k / N)

    #     # Compute the terms to be added and subtracted, using real and imaginary parts
    #     t_real = cos_val * odd_real[k] + sin_val * odd_imag[k]
    #     t_imag = -sin_val * odd_real[k] + cos_val * odd_imag[k]

    #     # Combine the even and odd parts
    #     real_part[k] = even_real[k] + t_real
    #     imaginary_part[k] = even_imag[k] + t_imag

    #     # real_part[k + N//2] = even_real[k] - t_real
    #     # imaginary_part[k + N//2] = even_imag[k] - t_imag

    # return list(zip(real_part, imaginary_part))

    N = len(x)
    # Initialize lists to store real and imaginary parts
    real_part = [0.0] * (N // 2 + 1)
    imaginary_part = [0.0] * (N // 2 + 1)

    # Compute the DFT for each frequency that's needed for the real FFT
    for k in range(N // 2 + 1):
        sum_real = 0.0
        sum_imag = 0.0
        for n in range(N):
            angle = 2 * np.pi * k * n / N
            sum_real += x[n] * np.cos(angle)
            sum_imag -= x[n] * np.sin(angle)  # Note the negative sign for the imaginary part
        real_part[k] = sum_real
        imaginary_part[k] = sum_imag

    # Zip the real and imaginary parts together to match the output format
    return list(zip(real_part, imaginary_part))


def manual_rfftfreq(FFT_size, sample_rate):
    """
    Manually calculate the frequency bins for the FFT of a real-valued signal.
    
    Parameters:
    - FFT_size (int): Number of points in the FFT.
    - sample_rate (float): Rate at which the original signal was sampled.
    
    Returns:
    - np.array: Array of frequency values corresponding to each FFT bin.
    """
    # Calculate the frequencies for each bin
    freqs = np.arange(0, FFT_size // 2 + 1) * (sample_rate / FFT_size)
    return freqs

# def RealFastFourierTransformFrequency (n, d=1.0):
#     if not isinstance(n, (int, np.integer)):
#         raise ValueError("n should be an integer")
#     val = 1.0/(n*d)
#     N = n//2 + 1
#     results = np.arange(0, N, dtype=int)
#     return results * val