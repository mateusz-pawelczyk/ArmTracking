#!/usr/bin/env python
# coding: utf-8

# In[405]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/home/tm_ba/Desktop/Bachelorarbeit_code')
print(os.get_exec_path())


# In[406]:


import pandas as pd
import os

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
                full_column_name = ""
                if current_group:
                    full_column_name += f"{current_group}:"
                full_column_name += f"{column_header.strip()}"
                if unit.strip():
                    full_column_name += f" ({unit.strip()})"
                columns.append(full_column_name)

            # Parse the data rows starting from the 6th line onward
            data = [line.split(',') for line in lines[5:]]

            # Create a DataFrame for the current table
            df = pd.DataFrame(data, columns=columns)

            # Convert numeric columns to appropriate types where possible
            df = df.apply(pd.to_numeric, errors='ignore')

            # Attach metadata (Hz information) to the DataFrame as an attribute
            df.attrs['hz'] = hz

            # Check for duplicate column names and handle them by renaming
            if len(df.columns) != len(set(df.columns)):
                # Automatically renames duplicate columns
                df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
                to_delete.append(file_path)

            # Add 'Recording' column to track the file source
            df['Recording'] = recording_idx

            
            df.drop(columns=[col for col in df.columns if "Triangle1" in col], inplace=True)

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


# In[407]:


data_directory  = "Mathew" 
data_paths = [os.path.join(data_directory, data_path) for data_path in os.listdir(data_directory)]
tables_dict = parse_and_merge_tables(data_paths)


# In[408]:


# Display the tables in the dictionary
for table_name, df in tables_dict.items():
    print(f"Table Name: {table_name}")
    print(f"Hz: {df.attrs['hz']}")
    print(f"Columns: {df.columns}")
    os.makedirs('csv_new', exist_ok=True)
    df.to_csv(f"csv_new/{table_name}.csv", index=False)  # Save the DataFrame to a CSV file
    print("\n" + "-"*50 + "\n")  # Separator between tables


# In[409]:


trajectories = tables_dict["Trajectories"]
trajectories


# In[410]:


trajectories["Recording"].value_counts()


# In[411]:


df["Recording"].value_counts()


# In[412]:


num_landmarks = (trajectories.shape[1] - 2) // 3 


# In[413]:


speeds = []
for i in range(1, trajectories.shape[0]):
    speed_frame = []
    for j in range(2, num_landmarks * 3 + 2, 3):
        dx = trajectories.iloc[i, j] - trajectories.iloc[i-1, j]
        dy = trajectories.iloc[i, j + 1] - trajectories.iloc[i-1, j + 1]
        dz = trajectories.iloc[i, j + 2] - trajectories.iloc[i-1, j + 2]
        speed = np.sqrt(dx**2 + dy**2 + dz**2)
        speed_frame.append(speed)
    speed_frame.append(trajectories.iloc[i]["Recording"])
    speeds.append(speed_frame)

speeds_df = pd.DataFrame(speeds, columns=[f'{trajectories.columns[2 + i*3].split(":")[1]}' for i in range(num_landmarks)] + ["Recording"])
speeds_df.to_csv('./csv_new/Speeds.csv', index=False)


# In[414]:


speeds_df["Recording"].value_counts()


# In[415]:


recording_idx = 23
visualized_speeds_df = speeds_df[speeds_df["Recording"] == recording_idx].drop("Recording", axis=1)
visualized_speeds_df


# In[416]:


plt.figure(figsize=(10, 6))
for column in visualized_speeds_df.columns:
    plt.plot(visualized_speeds_df.index, visualized_speeds_df[column], label=column, alpha=0.5)

plt.title('Landmark Speeds Over Time')
plt.xlabel('Frame')
plt.ylabel('Speed')
plt.legend(title='Landmarks', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[417]:


# Schritt 1: Berechnen des IQR und Ausreißer-Grenzen festlegen
Q1 = speeds_df.quantile(0.25)
Q3 = speeds_df.quantile(0.75)
IQR = Q3 - Q1

# Festlegung der Grenzen, um Ausreißer zu entfernen (1.5-facher IQR)
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Schritt 2: Entfernen der Ausreißer
filtered_speeds_df = speeds_df[~((speeds_df < lower_bound) | (speeds_df > upper_bound)).any(axis=1)]


# In[418]:


def moving_average(data, window_size):
    return data.rolling(window=window_size, min_periods=1, center=True).mean()

# Apply the moving average to each column in the DataFrame
window_size = 10  # Adjust the window size as needed
smoothed_speeds_df = filtered_speeds_df.apply(moving_average, window_size=window_size)


# In[419]:


visualized_speeds_df = smoothed_speeds_df[smoothed_speeds_df["Recording"] == recording_idx].drop("Recording", axis=1)
visualized_speeds_df


# In[420]:


# Plotting smoothed speeds with transparency
plt.figure(figsize=(10, 6))
for column in visualized_speeds_df.columns:
    plt.plot(visualized_speeds_df.index, visualized_speeds_df[column], label=column, alpha=0.7)  # Adjust alpha for transparency

plt.title('Smoothed Landmark Speeds Over Time')
plt.xlabel('Frame')
plt.ylabel('Speed')
plt.legend(title='Landmarks', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[421]:


trajectories.isna().sum().sum()


# In[434]:


import pandas as pd
import numpy as np

def read_and_process_data(file_path):
    # Load data
    data = pd.read_csv(file_path)

    # Calculate differences between consecutive frames for all columns except 'Frame' and 'Sub Frame'
    position_columns = [col for col in data.columns if col not in ['Frame', 'Sub Frame']]
    differences = data[position_columns].diff().abs()

    # Identify outliers using the IQR method
    Q1 = differences.quantile(0.25)
    Q3 = differences.quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR

    # Create a mask for outliers
    outliers = (differences > outlier_threshold)

    # Interpolate outliers and handle NaNs
    for col in position_columns:
        # Mark outliers as NaN
        data.loc[outliers[col], col] = np.nan
        
        # Remove only the outliers at the beginning of the dataset (consecutive NaNs at the start)
        first_valid_index = data[col].first_valid_index()  # Find the first valid (non-NaN) index
        if first_valid_index is not None:
            data = data.loc[first_valid_index:].reset_index(drop=True)  # Drop rows before first valid index

        # Interpolate NaN values (including consecutive NaNs) using linear interpolation
        data[col] = data[col].interpolate(method='linear', limit_direction='both')
        
        # Optional: Fill any remaining NaNs at the start or end of the series using forward and backward fill
        data[col] = data[col].ffill().bfill()

    return data

# Usage
processed_data = read_and_process_data('csv_new/Trajectories.csv')
processed_data.to_csv('csv_new/processed_Trajectories.csv', index=False)


# In[435]:


speeds = []
for i in range(1, processed_data.shape[0]):
    speed_frame = []
    for j in range(2, num_landmarks * 3 + 2, 3):
        dx = processed_data.iloc[i, j] - processed_data.iloc[i-1, j]
        dy = processed_data.iloc[i, j + 1] - processed_data.iloc[i-1, j + 1]
        dz = processed_data.iloc[i, j + 2] - processed_data.iloc[i-1, j + 2]
        speed = np.sqrt(dx**2 + dy**2 + dz**2)
        speed_frame.append(speed)
    speed_frame.append(processed_data.iloc[i]["Recording"])
    speeds.append(speed_frame)

processed_speeds_df = pd.DataFrame(speeds, columns=[f'{processed_data.columns[2 + i*3].split(":")[1]}' for i in range(num_landmarks)] + ["Recording"])


# In[436]:


visualized_speeds_df = processed_speeds_df[processed_speeds_df["Recording"] == recording_idx].drop("Recording", axis=1)
visualized_speeds_df


# In[437]:


# Plotting smoothed speeds with transparency
plt.figure(figsize=(10, 6))
for column in visualized_speeds_df.columns:
    plt.plot(visualized_speeds_df.index[1:], visualized_speeds_df[column][1:], label=column, alpha=0.7)  # Adjust alpha for transparency

plt.title('Smoothed Landmark Speeds Over Time')
plt.xlabel('Frame')
plt.ylabel('Speed')
plt.legend(title='Landmarks', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[445]:


get_ipython().system('jupyter nbconvert --to script read_data.ipynb')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




