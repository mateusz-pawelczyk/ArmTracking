import pandas as pd
import os
import argparse

def safe_numeric_conversion(df):
    """
    Safely convert applicable columns to numeric while ignoring errors in non-numeric columns.
    """
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass  # Ignore columns that can't be converted to numeric
    return df

def parse_and_merge_tables(file_paths):
    """
    Parses the tables from the given list of files and merges tables of the same kind.

    Args:
        file_paths (list): List of paths to the files containing the tables.

    Returns:
        dict: A dictionary where the keys are table names, and the values are merged DataFrames.
    """
    # Initialize a dictionary to store DataFrames with table names as keys
    merged_tables = {}
    to_delete = []

    for recording_idx, file_path in enumerate(file_paths):
        with open(file_path, 'r') as file:
            # Read the entire content of the file
            content = file.read()

        # Split the content by double newlines to separate the tables
        tables = content.strip().split('\n\n')

        for table in tables:
            # Split the table content by lines
            lines = table.strip().split('\n')

            # Extract the table name from the first line
            table_name = lines[0].strip()

            # Extract the Hz information (metadata) from the second line
            hz = lines[1].strip()

            # Parse the column headers and units from the next three lines
            group_headers = lines[2].split(',')  # Group headers (first row)
            column_headers = lines[3].split(',')  # Column names (second row)
            units = lines[4].split(',')  # Units (third row)

            # Create a list to hold the full column names
            columns = []
            current_group = ""

            # Construct the full column names using group headers, column names, and units
            for group_header, column_header, unit in zip(group_headers, column_headers, units):
                if group_header.strip():  # Update the current group if it's not empty
                    current_group = group_header.strip()
                    if ":" in current_group:
                        current_group = current_group.split(":")[1]
                full_column_name = ""
                if current_group:
                    full_column_name = f"{current_group}:"
                full_column_name += f"{column_header.strip()}"

                columns.append(full_column_name)

            # Parse the data rows starting from the 6th line onward
            data = [line.split(',') for line in lines[5:]]

            # Create a DataFrame for the current table
            df = pd.DataFrame(data, columns=columns)

            # Convert numeric columns to appropriate types where possible
            df = safe_numeric_conversion(df)

            # Attach metadata (Hz information) to the DataFrame as an attribute
            df.attrs['hz'] = hz

            # Check for duplicate column names and handle them by renaming
            if len(df.columns) != len(set(df.columns)):
                # Automatically renames duplicate columns
                df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
                to_delete.append(file_path)

            if table_name == "Trajectories":
                wanted_columns = ["Frame", "shoulder", "elbow", "wrist", "ThumbTip"]
                columns = [col for col in df.columns if (":" not in col and col in wanted_columns) or (":" in col and col.split(":")[0] in wanted_columns)]
                df = df.loc[:, columns]

            # Add 'Sequence' column to track the file source
            df['Sequence'] = recording_idx

            # If the table has already been seen, merge the new data into the existing DataFrame
            if table_name in merged_tables:
                existing_df = merged_tables[table_name]

                # Align columns before concatenation
                merged_tables[table_name] = pd.concat([existing_df, df], ignore_index=True)
            else:
                merged_tables[table_name] = df

    # Optionally, remove files with duplicate column names
    for csv in to_delete:
        os.remove(csv)

    return merged_tables

def save_merged_tables(merged_tables, output_dir):
    """
    Saves merged tables to CSV files in the specified output directory.

    Args:
        merged_tables (dict): Dictionary of merged DataFrames.
        output_dir (str): Directory to save the CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    for table_name, df in merged_tables.items():
        output_file = os.path.join(output_dir, f"{table_name}.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved {table_name} table to {output_file}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Concatenate tables from multiple sequence files.")
    parser.add_argument(
        "input_dir", type=str, help="Directory containing the input sequence files."
    )
    parser.add_argument(
        "output_dir", type=str, help="Directory to save the concatenated output files."
    )
    args = parser.parse_args()

    # Get all file paths from the input directory
    file_paths = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]

    # Parse and merge the tables from all files
    merged_tables = parse_and_merge_tables(file_paths)

    # Save the merged tables to CSV files
    save_merged_tables(merged_tables, args.output_dir)
