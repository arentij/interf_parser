from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


# Function to create a bandpass Butterworth filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


# Function to apply the bandpass filter to a signal
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Generate a sample signal for demonstration purposes
def generate_sample_signal(fs, duration):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # Combine two sine waves (one at 40 MHz and another at 10 MHz)
    signal = np.sin(2 * np.pi * 40e6 * t) + 0.5 * np.sin(2 * np.pi * 10e6 * t)
    return t, signal


# Function to find the strongest frequency in a sampled signal
def find_strongest_frequency(signal, fs):
    # Compute the FFT of the signal
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / fs)[:N // 2]

    # Find the index of the peak in the FFT magnitude
    idx_peak = np.argmax(np.abs(yf[:N // 2]))
    strongest_freq = xf[idx_peak]

    return strongest_freq, np.abs(yf[idx_peak])

# # Parameters
# fs = 25e6  # Sampling frequency (25 MHz)
# duration = 1e-6  # Duration of the signal in seconds
#
# # Generate the sample signal
# t, signal = generate_sample_signal(fs, duration)
#
# # Find the strongest frequency
# strongest_freq, magnitude = find_strongest_frequency(signal, fs)
#
# # Print the strongest frequency and its magnitude
# print(f"The strongest frequency is {strongest_freq} Hz with a magnitude of {magnitude}")
#
# # Plot the signal and its FFT
# plt.figure(figsize=(12, 6))
#
# # Plot the time-domain signal
# plt.subplot(2, 1, 1)
# plt.plot(t, signal)
# plt.title('Time-Domain Signal')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
#
# # Plot the frequency spectrum
# plt.subplot(2, 1, 2)
# N = len(signal)
# yf = fft(signal)
# xf = fftfreq(N, 1 / fs)[:N // 2]
# plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
# plt.title('Frequency Spectrum')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Magnitude')
# plt.axvline(x=strongest_freq, color='r', linestyle='--', label=f'Strongest Frequency: {strongest_freq} Hz')
# plt.legend()
#
# plt.tight_layout()
# plt.show()


def complex_demodulate(signal, fs, fc):
    """
    Perform complex demodulation on a signal.

    Parameters:
    - signal: The input signal (real-valued)
    - fs: Sampling frequency
    - fc: Carrier frequency

    Returns:
    - demodulated_signal: The baseband complex signal
    """
    t = np.arange(len(signal)) / fs
    complex_exponential = np.exp(-1j * 2 * np.pi * fc * t)
    mixed_signal = signal * complex_exponential

    # Design a low-pass filter to remove high-frequency components
    nyquist_rate = fs / 2.0
    cutoff_freq = fc / 2.0 / nyquist_rate  # Half of the carrier frequency
    b, a = butter(5, cutoff_freq, btype='low')

    # Apply the filter
    demodulated_signal = filtfilt(b, a, mixed_signal)

    return demodulated_signal


def calculate_phase_difference(signal1, signal2, fs, fc):
    """
    Calculate the phase difference between two signals after complex demodulation.

    Parameters:
    - signal1: First input signal (real-valued)
    - signal2: Second input signal (real-valued)
    - fs: Sampling frequency
    - fc: Carrier frequency

    Returns:
    - phase_difference: The phase difference as a function of time (in radians)
    """
    demodulated_signal1 = complex_demodulate(signal1, fs, fc)
    demodulated_signal2 = complex_demodulate(signal2, fs, fc)

    phase1 = np.angle(demodulated_signal1)
    phase2 = np.angle(demodulated_signal2)

    # Unwrap the phase to prevent discontinuities
    unwrapped_phase1 = np.unwrap(phase1)
    unwrapped_phase2 = np.unwrap(phase2)

    # Calculate the phase difference
    phase_difference = unwrapped_phase1 - unwrapped_phase2

    return phase_difference


if __name__ == "__main__":
    # Example usage
    fs = 25e6  # Sampling frequency
    window_w = 2e6
    fc = 40e6 + 1e5*np.random.uniform(-1, 1)  # Carrier frequency
    t = np.arange(0, 1e-3, 1 / fs)
    driver = np.cos(2 * np.pi * fc * t)  # Example signal 1
    phase_modulation = np.pi * np.sin(6100 * 2 * np.pi * t) # this is the actual signal that is modulated
    noise_signal = np.random.normal(scale=np.sqrt(0.1), size=len(phase_modulation))
    phase_modulation = phase_modulation + noise_signal

    signal2 = np.cos(2 * np.pi * fc * t + phase_modulation)  # Example signal 2 with time-varying phase
    signal2 = signal2 + 10.5*np.cos(2*np.pi*fc*2*t + 81)

    noise = np.random.normal(scale=np.sqrt(0.1), size=len(signal2))
    signal2 = signal2 + noise

    fc1, ampl = find_strongest_frequency(driver, fs)
    print(fc1)

    filtered_signal = bandpass_filter(signal2, fc1 - window_w, fc1 + window_w, fs, order=6)

    phase_diff = calculate_phase_difference(driver, filtered_signal, fs, fc1)

    # Plotting the phase difference


    plt.plot(t, -phase_modulation)
    plt.plot(t, phase_diff)
    plt.xlabel('Time (s)')
    plt.ylabel('Phase Difference (radians)')
    plt.title('Phase Difference vs. Time')
    plt.legend(['original', 'deconstructed'])
    plt.show()
