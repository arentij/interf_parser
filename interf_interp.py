import os
import time
import pandas as pd
import signal


# Define the foo() function here
def foo(data):
    # Example function implementation
    # Replace this with your actual function logic
    return data


def get_parquet_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.parquet')]


def check_file_growth(filepath, check_interval=1, retries=3):
    previous_size = -1
    for _ in range(retries):
        current_size = os.path.getsize(filepath)
        if current_size == previous_size:
            return True
        previous_size = current_size
        time.sleep(check_interval)
    return False


def process_new_files(data_dirs, results_dirs):
    processed_files = {data_dir: set() for data_dir in data_dirs}

    while True:
        for data_dir, results_dir in zip(data_dirs, results_dirs):
            all_files = get_parquet_files(data_dir)
            new_files = [f for f in all_files if f not in processed_files[data_dir]]

            for file in new_files:
                filepath = os.path.join(data_dir, file)

                # Modify the filename to include _results before the file extension
                base, ext = os.path.splitext(file)
                result_filename = f"{base}_results{ext}"
                result_filepath = os.path.join(results_dir, result_filename)

                # Check if the result file already exists
                if os.path.exists(result_filepath):
                    print(f"Result file {result_filename} already exists in {results_dir}. Skipping processing.")
                    processed_files[data_dir].add(file)
                    continue

                if check_file_growth(filepath):
                    try:
                        data = pd.read_parquet(filepath)
                        result = foo(data)

                        result.to_parquet(result_filepath)

                        processed_files[data_dir].add(file)
                        print(f"Processed and saved: {file} as {result_filename} in {results_dir}")
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
                else:
                    print(f"File {file} is still growing in {data_dir}. Skipping for now.")

        time.sleep(5)  # Check for new files every 5 seconds


def signal_handler(sig, frame):
    print('Gracefully stopping...')
    exit(0)


if __name__ == "__main__":
    data_directories = [".", "data"]
    results_directories = ["results", "results2"]

    for results_directory in results_directories:
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

    # Register the signal handler for graceful termination
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    process_new_files(data_directories, results_directories)
