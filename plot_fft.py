import pyarrow.parquet as pq
import numpy as np
from scipy.fft import fft, fftfreq
import pandas as pd
import matplotlib.pyplot as plt


def guess_fs(driver):
    first_peak_freq, period_driver, psd_diff = find_first_non_zero_peak(driver)
    print(f'PSD4800 - PSD1200 = {psd_diff}')
    fs_rec = reconstruct_fs(first_peak_freq, period_driver, psd_diff)
    # print(f'first peak {first_peak_freq}, period {period_driver}')
    return fs_rec


def find_closest_value(values, f):
    """
    Finds the closest value in the list to the given value f.

    Parameters:
    values (list): A list of numerical values.
    f (float): The value to compare against the list elements.

    Returns:
    float: The value from the list that is closest to f.
    """
    return min(values, key=lambda x: abs(x - f))


def reconstruct_fs(first_peak, period, psd_diff):
    period_list = [5, 2.5, 3.125, 6.25, 12.5, 25, 2.7778, 50, 100, 200]
    fs_list =     [12.5, 25, 125, 250, 500, 1000, 6.25, 2000, 4000, 8000]
    closest_period = find_closest_value(period_list, period)
    if closest_period == 5:
        if psd_diff < -0.025:
            return 50e6
        elif psd_diff > 0.025:
            return 12.5e6
        # if that didn't work
        closest_f = find_closest_value([4800, 1200], first_peak)
        if closest_f == 1200:
            return 50e6
        else:
            return 12.5e6
    else:
        return fs_list[period_list.index(closest_period)]*1e6


def find_first_non_zero_peak(signal, threshold=0.020):
    """
    Finds the frequency of the first non-zero peak in the FFT of a signal
    with a magnitude above a specified threshold.

    Parameters:
    signal (array-like): The input signal (1D array or pandas Series).
    fs (float): The sampling frequency in Hz.
    threshold (float): The minimum magnitude of the peak to consider.

    Returns:
    float: The frequency of the first non-zero peak in Hz above the threshold.
    """
    # Ensure the signal is a numpy array
    fs = 1e9
    signal = np.asarray(signal)

    # Number of samples in the signal
    N = len(signal)

    # Check if the signal is empty or too short
    if N == 0:
        raise ValueError("The signal is empty.")
    if N == 1:
        raise ValueError("The signal is too short to compute FFT.")

    # Compute the FFT
    fft_values = np.fft.fft(signal)

    # Compute the corresponding frequencies
    frequencies = np.fft.fftfreq(N, 1/fs)

    # Compute the Power Spectral Density (PSD)
    psd = np.abs(fft_values)**2 / N

    # Ignore the first zero frequency component (DC component)

    # psd[1] = psd[2]
    psd[0] = 0

    # Find indices where PSD values exceed the threshold
    peak_indices = np.where(psd > threshold)[0]
    if len(peak_indices) == 0:
        raise ValueError("No peaks above the specified threshold found in the signal.")

    # Get the frequency of the first non-zero peak above the threshold
    first_peak_index = peak_indices[0]
    first_peak_frequency = frequencies[first_peak_index]
    #
    f_2 = 4800
    f_1 = 1200
    ind_f_1 = 0
    ind_f_2 = 0
    for ind_f in range(N):
        if frequencies[ind_f] >= f_1:
            ind_f_1 = ind_f
            break
    for ind_f in range(ind_f_1,N):
        if frequencies[ind_f] >= f_2:
            ind_f_2 = ind_f
            break


    # Find the peak frequency
    peak_freq = np.abs(fft_values[:N // 2]).argmax()

    # Convert frequency to period
    period = fs / frequencies[peak_freq]
    fft_chunk = [[], []]
    # plotting
    # if False:
    #     plt.plot(frequencies[0:ind_f_cut], psd[0:ind_f_cut])
    #     plt.show()
    psd4800_psd1200 = psd[ind_f_2] - psd[ind_f_1]
    return first_peak_frequency, period, psd4800_psd1200

def find_period_fft(signal):
    # Compute the FFT of the signal
    N = len(signal)
    fft_vals = fft(signal)
    fft_freqs = fftfreq(N)

    # Find the peak frequency
    peak_freq = np.abs(fft_vals[:N // 2]).argmax()

    # Convert frequency to period
    period = 1 / fft_freqs[peak_freq]
    return period


if __name__ == "__main__":

    # Path to the Parquet file
    file_path = '40ghz_samples/CMFX_00060_scope_3.parquet'
    file_path = 'CMFX_00101_scope.parquet'
    # file_path = '/Users/arturperevalov/Documents/MATLAB/interferometer/may02/CMFX_00072_scope.parquet'

    df = pd.read_parquet(file_path)

    driver_co2 = df['INT02 Driver (V)'].values
    #
    # plt.plot(driver_co2[0:100])
    # plt.show()
    # plt.close()
    fs_g = guess_fs(driver_co2)
    print(f"Guessed Fs is {fs_g/1e6} MHz")
    #
    fs = 1000e6
    #
    #

    first_peak_freq, period_driver, psd_diff = find_first_non_zero_peak(driver_co2)
    print(f"The frequency of the first non-zero peakis: {first_peak_freq} ")

    # period_driver = find_period_fft(driver_co2)
    print(f"Period of the main signal is {period_driver}")
    #
    # factors_list = [1, 2, 4, 8, 20, 40, 80]
    # f_s_reconstruct = 1e9/find_closest_value(factors_list, first_peak_freq/60)
    # print(f"Expected Fs={f_s_reconstruct}")

    fs_rec = reconstruct_fs(first_peak_freq, period_driver, psd_diff)
    print(f"Reconstruct fs {fs_rec/1e6} MHz")


    N = len(driver_co2)
    # print(N)
    # fft_vals = fft(driver_co2)
    # fft_freqs = fftfreq(N)
    # N = len(signal)
    # hanning_window = np.hanning(N)
    fft_values = np.fft.fft(driver_co2)
    frequencies = np.fft.fftfreq(N, 1/fs)
    psd = np.abs(fft_values)**2 / N

    # print(f"FFT resolution is {frequencies[1]-frequencies[0]}")
    half_N = N // 2
    frequencies = frequencies[:half_N]
    psd = psd[:half_N]

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, psd)
    plt.title('Power Spectral Density using FFT')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (V^2/Hz)')
    plt.grid(True)
    plt.show()
