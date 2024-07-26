import os
import time
import pyarrow.parquet as pq
import pandas as pd
from interf_fft import process_parquet_file

def wait_for_file_to_stabilize(filepath, check_interval=1, stabilization_time=5):
    """
    Wait until the file at 'filepath' stops changing in size.

    Parameters:
    - filepath (str): The path to the file.
    - check_interval (int): The interval (in seconds) between size checks.
    - stabilization_time (int): The time (in seconds) the file size must remain constant before considering it stabilized.

    Returns:
    - bool: True if the file size has stabilized, False if the file does not exist or an error occurs.
    """
    # print(f"Checking size of {filepath}")
    if not os.path.exists(filepath):
        # print(f"File {filepath} does not exist.")
        return False

    previous_size = os.path.getsize(filepath)
    unchanged_time = 0

    while True:
        time.sleep(check_interval)
        current_size = os.path.getsize(filepath)

        if current_size == previous_size:
            unchanged_time += check_interval
            if unchanged_time >= stabilization_time:
                return True
        else:
            unchanged_time = 0  # Reset the unchanged time counter
            previous_size = current_size


def process_parquet_file_draft(filename, output_folder='.'):
    # Replace this with your actual processing function
    print(f"Processing file: {filename}")
    # Example: read and print some basic info
    # parquet_file = pq.ParquetFile(filename)
    df = pd.read_parquet(filename)

    base_name = os.path.basename(filename)
    result_name = f"{output_folder}/{os.path.splitext(base_name)[0]}_result_fft.parquet"
    # print(result_name)
    # out_file =  output_folder
    df.head().to_parquet(result_name)

    # num_rows = parquet_file.metadata.num_rows
    # print(num_rows)
    # print(f"Number of rows in {filename}: {num_rows}")
    # Example result, replace with actual result processing
    # result = num_rows
    # print(num_rows)
    # result_data = pd.DataFrame({'n_rows:': num_rows})
    # result_fft_filename = filename.replace('.parquet', '_results.parquet')


    # result_data.to_parquet()

    return True


def result_file_exists(filename, result_folder):
    base_name = os.path.basename(filename)
    result_name = f"{os.path.splitext(base_name)[0]}_results_fft.parquet"
    result_path = os.path.join(result_folder, result_name)
    # print(f"Checking if this exists {result_path}")
    # print(f"It exists = {os.path.exists(result_path)}")
    #
    # directory_path = os.path.dirname(result_path)
    #
    # # List all files in the directory
    # files = os.listdir(directory_path)
    #
    # # Iterate over the list and print each file
    # for file_name in files:
    #     # Check if it's a file (not a directory)
    #     if os.path.isfile(os.path.join(directory_path, file_name)):
    #         print(file_name)
    return os.path.exists(result_path)


def process_files_in_folder(folder_in, folder_out):
    while True:
        # List all parquet files in folder1 that corresponds to the experimental run results
        folder1 = folder_in + 'interferometer'
        files = [f for f in os.listdir(folder1) if f.endswith('.parquet')]
        folder2 = folder_out + 'interferometer'

        for file_name in files:
            file_path = os.path.join(folder1, file_name)

            if not result_file_exists(file_path, folder2):
                # print(file_path)
                wait_for_file_to_stabilize(file_path)

                try:
                    process_parquet_file(file_path, folder_in, folder_out)
                except Exception as e:
                    print(e)
                    continue
                #
                # # Write result as needed or handle post-processing
                # result_file_name = f"{os.path.splitext(file_name)[0]}_result_fft.parquet"
                # result_file_path = os.path.join(folder2, result_file_name)
                # # Here you could save `result` to result_file_path if necessary

        # List all parquet files in folder2 that corresponds to the test run results
        folder1 = folder_in + 'tests/interferometer'
        files = [f for f in os.listdir(folder1) if f.endswith('.parquet')]
        folder2 = folder_out + 'tests/interferometer'

        for file_name in files:
            file_path = os.path.join(folder1, file_name)

            if not result_file_exists(file_path, folder2):
                # print(file_path)
                wait_for_file_to_stabilize(file_path)
                try:
                    process_parquet_file(file_path, folder_in, folder_out)
                except Exception as e:
                    print(e)
                    continue

        time.sleep(10)  # Wait before checking the folder again
        print('Here we go again')


if __name__ == "__main__":
    folder1 = 'CMFX_RAW/'  # Replace with your folder path
    folder2 = 'CMFX/'  # Replace with your result folder path
    process_files_in_folder(folder1, folder2)
