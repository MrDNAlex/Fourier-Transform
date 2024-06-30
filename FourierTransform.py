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

    if (N <= 1):
        return signal
    
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
