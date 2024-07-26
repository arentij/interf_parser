import pyarrow.parquet as pq
import numpy as np
from scipy.fft import fft, fftfreq


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


# Path to the Parquet file
file_path = '/Users/arturperevalov/Documents/MATLAB/interferometer/jul10/CMFX_00036_scope_16.parquet'
file_path  = 'CMFX_00101_scope.parquet'

# Open the Parquet file
parquet_file = pq.ParquetFile(file_path)

# Get the total number of rows
total_rows = parquet_file.metadata.num_rows

# Read the first 1 million rows
first_million_rows = parquet_file.read_row_group(0).to_pandas()[:1000]

driver_co2 = first_million_rows.iloc[:, 2]
print(f'Total number of points (rows) in the file: {total_rows}')
# print(driver_co2)
print(find_period_fft(driver_co2.values))
