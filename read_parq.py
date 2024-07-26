import pandas as pd
from main import find_strongest_frequency
import matplotlib.pyplot as plt
import numpy as np
from main import calculate_phase_difference


def moving_average(arr, window_size):
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


# df = pd.read_parquet('~/Documents/MATLAB/interferometer/jun20/CMFX_01086_scope.parquet')
df = pd.read_parquet('CMFX_00100_scope.parquet')
# df = pd.read_parquet('CMFX_00036_scope_5.parquet')

# df = pd.read_parquet('/Users/arturperevalov/Documents/MATLAB/interferometer/jul10/CMFX_00036_scope_16.parquet')
# df = pd.read_parquet('/Users/arturperevalov/Documents/MATLAB/interferometer/jun20/CMFX_01086_scope.parquet')
fs = 12.5e6
print('Parquet reading done')

print(df.head())
driver_co2 = df.iloc[:, 3]
signal_co2 = df.iloc[:, 1]
driver_co2_chunk = df.iloc[1:10000, 3]
signal_co2_chunk = df.iloc[1:10000, 1]
driver_co2_chunk = np.asarray(driver_co2_chunk)

fc_co2, ampl = find_strongest_frequency(driver_co2_chunk, fs)
print(fc_co2)

phase_diff_co2 = calculate_phase_difference(driver_co2, signal_co2, fs, fc_co2)
phase_diff_co2 = np.unwrap(phase_diff_co2)
#
driver_red = df.iloc[:, 2]
signal_red = df.iloc[:, 0]
driver_red_chunk = np.asarray(df.iloc[1:10000, 2])
fc_red, ampl = find_strongest_frequency(driver_red_chunk, fs)
print(fc_red)
phase_diff_red = calculate_phase_difference(driver_red, signal_red, fs, fc_red)
phase_diff_red = np.unwrap(phase_diff_red)
#
lambda_co2 = 10.56e-6
lambda_red = 0.639e-6

distance_red = -phase_diff_red*lambda_red/2/np.pi
# distance_red = distance_red - np.mean(distance_red[len(distance_red) // 3:len(distance_red)*2//3])
distance_co2 = phase_diff_co2*lambda_co2/2/np.pi
# distance_co2 = distance_co2 - np.mean(distance_co2[len(distance_co2) // 3:len(distance_co2)*2//3])
# plt.plot(driver_co2_chunk)
# plt.plot(signal_co2_chunk)
window_size = 1000
print('Downsampling')
distance_red_av = moving_average(distance_red, window_size)
distance_red_av = distance_red_av[::window_size//2]
distance_red_av = distance_red_av - np.mean(distance_red_av)
distance_co2_av = moving_average(distance_co2, window_size)
distance_co2_av = distance_co2_av[::window_size//2]
distance_co2_av = distance_co2_av - np.mean(distance_co2_av)


# plt.plot(t, phase_diff)
# plt.xlabel('Time (s)')
# plt.ylabel('Phase Difference (radians)')
# plt.title('Phase Difference vs. Time')
# plt.legend(['original', 'deconstructed'])
plt.plot(distance_co2_av*1e6)
plt.plot(distance_red_av*1e6)
plt.legend(['CO2 distance', 'Red distance'])
plt.show()

