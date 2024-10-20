import os
import shutil
import argparse
import re

def filter_and_copy_csv_files(input_dir, output_dir, start_id=None, end_id=None):
    """
    Filters CSV files in the input directory and copies them to the output directory.
    Optionally, it copies only the files whose IDs are within the specified range.

    Args:
        input_dir (str): Directory containing the input CSV files.
        output_dir (str): Directory where the filtered CSV files will be copied.
        start_id (int, optional): Start of the range of file IDs to include. Defaults to None.
        end_id (int, optional): End of the range of file IDs to include. Defaults to None.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Pattern to match files with IDs at the end of their names like MathewXX.csv
    file_pattern = re.compile(r'(\D+)(\d+)\.csv')

    # Iterate over all files in the input directory
    for file_name in os.listdir(input_dir):
        # Check if the file is a CSV
        if file_name.endswith('.csv'):
            # Use regex to match the name and extract the ID
            match = file_pattern.match(file_name)
            if match:
                try:
                    # Extract the numeric part as the ID
                    file_id = int(match.group(2))
                except ValueError:
                    print(f"Warning: Couldn't extract ID from file {file_name}. Skipping.")
                    continue

                # Check if the file ID is within the specified range, if provided
                if start_id is not None and file_id < start_id:
                    continue
                if end_id is not None and file_id > end_id:
                    continue

                # Copy the file to the output directory
                source_path = os.path.join(input_dir, file_name)
                destination_path = os.path.join(output_dir, file_name)
                shutil.copy(source_path, destination_path)
                print(f"Copied {file_name} to {output_dir}")
            else:
                print(f"Warning: {file_name} does not match the expected pattern. Skipping.")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Filter and copy CSV files based on file ID range.")
    parser.add_argument(
        "input_dir", type=str, help="Directory containing the input CSV files (e.g., data/raw/recording_data/)."
    )
    parser.add_argument(
        "output_dir", type=str, help="Directory where the filtered CSV files will be copied (e.g., data/raw/sequences/)."
    )
    parser.add_argument(
        "--start_id", type=int, help="Start of the file ID range to include (optional).", default=None
    )
    parser.add_argument(
        "--end_id", type=int, help="End of the file ID range to include (optional).", default=None
    )
    
    args = parser.parse_args()

    # Run the filtering and copying process
    filter_and_copy_csv_files(args.input_dir, args.output_dir, args.start_id, args.end_id)
