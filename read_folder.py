import os
import pyarrow.parquet as pq
import pandas as pd
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


# Directory containing Parquet files
parquet_dir = '.'
output_csv = 'output.csv'

# Initialize a list to store the results
results = []

# Iterate over all files in the directory
for filename in os.listdir(parquet_dir):
    if filename.endswith('.parquet'):
        file_path = os.path.join(parquet_dir, filename)

        # Read the Parquet file
        # parquet_file = pq.ParquetFile(filepath)

        # Initialize a DataFrame to hold all rows from the file
        # all_data = pd.DataFrame()

        # Read each row group and append to all_data DataFrame
        # for i in range(parquet_file.num_row_groups):
        #     row_group = parquet_file.read_row_group(i).to_pandas()
        #     all_data = pd.concat([all_data, row_group], ignore_index=True)
        parquet_file = pq.ParquetFile(file_path)

        # Get the total number of rows
        total_rows = parquet_file.metadata.num_rows

        # Read the first 1 million rows
        first_rows = parquet_file.read_row_group(0).to_pandas()[:1000]

        driver_co2 = first_rows.iloc[:, 2]
        # Apply the function foo on the data
        result = find_period_fft(driver_co2.values)

        # Get the number of rows in the original file
        # num_rows = len(all_data)

        # Append the results to the list
        results.append({'file': filename, 'result': result, 'num_rows': total_rows})

# Convert the results list to a DataFrame and write to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)

print(f'Results written to {output_csv}')
