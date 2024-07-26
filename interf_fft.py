import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from main import calculate_phase_difference
import os
from plot_fft import guess_fs


def phase2distance(phase_co2, phase_red, N_av=10000):
    lambda_co2 = 10.56e-6
    lambda_red = 0.639e-6
    distance_co2 = np.unwrap(phase_co2) / 2 / np.pi * lambda_co2

    distance_co2 -= sum(distance_co2[0:N_av]) / N_av
    distance_red = np.unwrap(phase_red) / 2 / np.pi * lambda_red
    distance_red -= sum(distance_red[0:N_av]) / N_av
    return distance_co2, distance_red


def estimate_main_frequency(data, sampling_rate):
    # Perform FFT
    fft_result = fft(data)
    freqs = np.fft.fftfreq(len(data), d=1 / sampling_rate)

    # Find the main frequency peak
    magnitude = np.abs(fft_result)
    peaks, _ = find_peaks(magnitude)

    if len(peaks) == 0:
        raise ValueError("No peaks found in the FFT result.")

    main_peak_index = peaks[np.argmax(magnitude[peaks])]
    main_frequency = abs(freqs[main_peak_index])

    return main_frequency


def compute_window_times(signal_length, fs, t_off=0, window_size=100, overlap=0.5):
    # Calculate the total duration of the signal
    total_time = signal_length / fs

    # Generate the time array
    time_array = np.linspace(-total_time/2+t_off, total_time/2 + t_off, signal_length, endpoint=False)

    # Calculate the step size based on the overlap
    step_size = int(window_size * (1 - overlap))

    # Initialize an empty list to store the average time values
    avg_time_values = []

    # Iterate over the signal with the given window size and overlap
    for start in range(0, signal_length - window_size + 1, step_size):
        end = start + window_size
        window_times = time_array[start:end]

        # Calculate the average time value for the current window
        avg_time = np.mean(window_times)
        avg_time_values.append(avg_time)

    return avg_time_values


def estimate_phase_difference(signal, driver, sampling_rate, main_frequency, segment_length, overlap=0.5):
    step_size = int(segment_length * (1 - overlap))
    num_segments = (len(signal) - segment_length) // step_size + 1

    phase_differences = []

    for i in range(num_segments):
        start = i * step_size
        end = start + segment_length

        segment1 = signal[start:end]
        segment3 = driver[start:end]

        # FFT for each segment
        fft_segment1 = fft(segment1)
        fft_segment3 = fft(segment3)

        freqs = np.fft.fftfreq(segment_length, d=1 / sampling_rate)

        # Find the index of the main frequency
        main_freq_index = np.argmin(np.abs(freqs - main_frequency))

        # Calculate phases
        phase1 = np.angle(fft_segment1[main_freq_index])
        phase3 = np.angle(fft_segment3[main_freq_index])

        # Calculate phase difference
        phase_difference = phase1 - phase3
        phase_differences.append(phase_difference)

    return phase_differences


def process_parquet_file(filename, folder_in='CMFX_RAW', folder_out='CMFX', f_s=(50e6+1), t_bias=0):
    print(f"Working on {filename}")
    df = pd.read_parquet(filename)

    output_name = filename.replace(folder_in, folder_out)

    output1 = pd.DataFrame({'Opened file': [True]})
    result_fft_filename = output_name.replace('.parquet', '_results_fft.parquet')
    output1.to_parquet(result_fft_filename)


    signal_co2 = df['INT02 (V)'].values
    driver_co2 = df['INT02 Driver (V)'].values

    signal_red = df['INT01 (V)'].values
    driver_red = df['INT01 Driver (V)'].values

    # fs_guessed = True
    t_off = 0
    if True:
        # checking if the external Fs was provided if not -let's read the file
        result_fft_filename = filename.replace('.parquet', '_param.csv')
        param_file_opened = False
        if os.path.exists(result_fft_filename):
            try:
                df_param = pd.read_csv(result_fft_filename, quotechar='"', skipinitialspace=True, delimiter=',',
                                       engine='python')
                fs_value = float(df_param['fs'][0])
                t_off = float(df_param['t_off'][0])
                param_file_opened = True
                if f_s == 50e6+1:  # if for some goddamn reason an external f_s was provided it will keep it, otherwise let's take the proper one
                    f_s = fs_value

            except Exception as e:
                print(f"An error occurred while reading the param file: {e}")

        if not param_file_opened:
            # print('f_s was not provided, trying to guess')
            fs_guessed = guess_fs(driver_co2)
            print(f"Guessed Fs is {fs_guessed/1e6} MHz")
            f_s = fs_guessed
            # t_off = 0

    print(f'Fs= {f_s/1e6} MHz ')

    # return False
    # INT01 (V)  INT02 (V)  INT01 Driver (V)  INT02 Driver (V)


    # Take first 10000 points from the third channel
    driver_co2_segment = driver_co2[:10000]
    driver_red_segment = driver_red[:10000]
    # Sampling rate is 50 MHz
    sampling_rate = f_s

    # Estimate main frequency
    main_frequency_co2 = estimate_main_frequency(driver_co2_segment, sampling_rate)
    main_frequency_red = estimate_main_frequency(driver_red_segment, sampling_rate)
    print(f"Estimated main frequency: {main_frequency_co2} Hz")

    # Estimate phase differences
    segment_length = 100
    phase_differences_co2_fft = estimate_phase_difference(signal_co2, driver_co2, sampling_rate, main_frequency_co2, segment_length)
    phase_differences_red_fft = estimate_phase_difference(signal_red, driver_red, sampling_rate, main_frequency_red, segment_length)

    t_fft = compute_window_times(len(driver_co2), sampling_rate, t_off, segment_length)

    # Save the results as a DataFrame
    result_fft_df = pd.DataFrame({'phase_differences_co2_fft': phase_differences_co2_fft,
                                  'phase_differences_red_fft': phase_differences_red_fft,
                                  't_fft': t_fft,
                                  'f_s': f_s,
                                  't_off': t_off,
                                  'segment_length': segment_length,
                                  'External Fs and t_off provided': param_file_opened})

    # Save to a new Parquet file
    result_fft_filename = output_name.replace('.parquet', '_results_fft.parquet')
    result_fft_df.to_parquet(result_fft_filename)
    # print(f"Results saved to: {result_filename}")

    distance_co2_fft, distance_red_fft = phase2distance(phase_differences_co2_fft, phase_differences_red_fft)

    # lambda_co2 = 10.56e-6
    # lambda_red = 0.639e-6
    # distance_co2_fft = np.unwrap(phase_differences_co2_fft)/2/np.pi*lambda_co2
    # N_av = 1000
    # distance_co2_fft -= sum(distance_co2_fft[0:N_av])/N_av
    # distance_red_fft = np.unwrap(phase_differences_red_fft)/2/np.pi*lambda_red
    # distance_red_fft -= sum(distance_red_fft[0:N_av])/N_av
    plt.plot(t_fft, +distance_co2_fft)
    plt.plot(t_fft, -distance_red_fft)
    plt.plot(t_fft, -distance_red_fft - distance_co2_fft)
    # plt.plot(distance_red_av * 1e6)
    plt.legend(['CO2 distance FFT', 'Red distance FFT', 'difference FFT'])
    plt.xlabel('T, s')
    plt.ylabel('nL, m')

    base_filename = os.path.splitext(os.path.basename(filename))[0]

    if param_file_opened:
        plt.title(f"{base_filename}\nProvided Fs={f_s/1e6} MHz, Segment N={segment_length}")
    else:
        plt.title(f"{base_filename}\nGuessed Fs={f_s/1e6} MHz, Segment N={segment_length}")

    # Construct the destination file path with the new extension
    # output_name
    destination_path = f'{os.path.dirname(output_name)}/images/{base_filename}_fft.jpg'


    plt.savefig(destination_path)
    # plt.show()
    plt.close()

    # now lets do the dcd
    doing_dcd = False
    if doing_dcd:
        phase_diff_co2_dcd = calculate_phase_difference(driver_co2, signal_co2, sampling_rate, main_frequency_co2)
        phase_diff_red_dcd = calculate_phase_difference(driver_red, signal_red, sampling_rate, main_frequency_co2)

        result_dcd_df = pd.DataFrame({'phase_differences_co2_dcd': phase_diff_co2_dcd,
                                      'phase_differences_red_dcd': phase_diff_red_dcd})

        # Save to a new Parquet file
        result_dcd_filename = output_name.replace('.parquet', '_results_dcd.parquet')
        result_dcd_df.to_parquet(result_dcd_filename)

        distance_co2_dcd, distance_red_dcd = phase2distance(phase_diff_co2_dcd, phase_diff_red_dcd)

        total_time_dcd = len(distance_co2_dcd) / sampling_rate
        # Generate the time array
        time_array_dcd = np.linspace(0, total_time_dcd, len(distance_co2_dcd), endpoint=False)
        # return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        window_size = 100
        distance_co2_dcd_lp = np.convolve(distance_co2_dcd, np.ones(window_size)/window_size, mode='valid')
        distance_red_dcd_lp = np.convolve(distance_red_dcd, np.ones(window_size)/window_size, mode='valid')
        time_array_dcd_lp   = np.convolve(time_array_dcd, np.ones(window_size)/window_size, mode='valid')

        distance_co2_dcd_lp = distance_co2_dcd_lp[::window_size//2]
        distance_red_dcd_lp = distance_red_dcd_lp[::window_size//2]
        time_array_dcd_lp = time_array_dcd_lp[::window_size//2]

        plt.plot(time_array_dcd_lp, +distance_co2_dcd_lp)
        plt.plot(time_array_dcd_lp, -distance_red_dcd_lp)
        # plt.plot(-distance_co2_dcd - distance_red_dcd)
        # plt.plot(distance_red_av * 1e6)
        plt.legend(['CO2 distance DCD', 'Red distance DCD', 'difference DCD'])

        base_filename = os.path.splitext(os.path.basename(filename))[0]
        # Construct the destination file path with the new extension
        destination_path = f'{os.path.dirname(output_name)}/images/{base_filename}_dcd.jpg'
        plt.xlabel('T, s')
        plt.ylabel('nL, m')
        if param_file_opened:

            plt.title(f"{base_filename}\nProvided Fs={f_s / 1e6} MHz")
        else:
            plt.title(f"{base_filename}\nGuessed Fs={f_s / 1e6} MHz ")

        plt.savefig(destination_path)
        # plt.show()
        plt.close()


if __name__ == "__main__":

    # Example usage
    # filename = 'CMFX_00101_scope.parquet'
    # filename = '/Users/arturperevalov/Documents/MATLAB/interferometer/may02/CMFX_00950_scope.parquet'
    # filename = '/Users/arturperevalov/Documents/MATLAB/interferometer/jun20/CMFX_01084_scope.parquet'
    # filename = '40ghz_samples/CMFX_00060_scope_7.parquet'
    # fs = 25e6
    filename = 'CMFX_RAW/tests/interferometer/CMFX_00056_scope.parquet'
    # process_parquet_file(filename)
