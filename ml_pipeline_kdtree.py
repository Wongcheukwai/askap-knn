import pandas as pd
pd.options.display.max_rows = None
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
import matplotlib.dates as mdates
import glob
import os
import time
import pickle
import itertools
import sys
sys.setrecursionlimit(10000)

pd.options.display.max_rows = None

# Assuming the rerun directory has five subdirectories, each corresponding to a different subsystem,
# and you want to create a list where each element is a DataFrame containing the anomalies for that subsystem.

# First, let's define the path to the rerun directory and the subdirectories we're interested in.
rerun_path = '/Users/wan404/Documents/bmf.raw/data/rerun/'
subsystems = ['chiller_all']

# Initialize a dictionary where keys are subsystem names and values will be dataframes of anomalies.
subsystem_anomalies = {subsystem: pd.DataFrame() for subsystem in subsystems}

# Create a directory to save the figures
figures_directory = '/Users/wan404/Documents/bmf.raw/data/figures2_14/'
os.makedirs(figures_directory, exist_ok=True)

# Process files for each subsystem
for subsystem in subsystems:
    csv_files = glob.glob(os.path.join(rerun_path, subsystem, '*.csv'))
    anomalies_list = []

    for file in csv_files:
        print('begin', file)
        df1 = pd.read_csv(file)
        field_1 = pd.DataFrame(df1, columns=['time', 'value'])
        field_1['time'] = pd.to_datetime(field_1['time'], errors='coerce')
        field_1 = field_1.set_index('time')

        field_1.index = pd.to_datetime(field_1.index)
        field_1['value'] = field_1['value'].astype(float)

        time_stamp = df1['time']
        time_list = time_stamp.values

        ########### this is kdtree ###########
        # Specify the start and end dates for the reference set
        reference_start = '2021-02-13 08:00:00'
        reference_end = '2021-3-18 23:59:59'

        # Specify the start and end dates for the query set
        query_start = '2022-02-13 00:00:00'
        query_end = '2022-02-15 09:00:00'

        # Split the data into reference and query sets
        reference_set = field_1[(field_1.index >= reference_start) & (field_1.index <= reference_end)]
        query_set = field_1[(field_1.index >= query_start) & (field_1.index <= query_end)]

        # Create a KDTree from the reference set
        reference_values = reference_set.values.reshape(-1, 1)
        kd_tree = KDTree(reference_values, leafsize=30)

        # Query the KDTree with the query set
        query_values = query_set.values.reshape(-1, 1)
        distances, indices = kd_tree.query(query_values, k=3)

        mean_distances = distances.mean(axis=1)

        if subsystem == 'chiller_all':
            para = 3200
            # para = 5
        elif subsystem == 'drx_all' or subsystem == 'bmf_all':
            para = 30
            # para = 1
        elif subsystem == 'paf_indoor' or subsystem == 'paf_all':
            para = 30
            # para = 1

        threshold = para * np.mean(mean_distances)

        anomaly_mask = mean_distances > threshold

        anomalies = pd.Series(anomaly_mask, index=query_set.index)

        anomalies_with_scores = pd.DataFrame({'Timestamp': query_set.index, 'Severity': mean_distances})
        anomalies_with_scores = anomalies_with_scores[anomalies_with_scores['Severity'] > threshold]

        # Add neighboring points information to the anomalies DataFrame
        anomalies_with_scores['Neighboring_Points'] = [query_set.iloc[idx].values.tolist() for idx in
                                                       indices[anomaly_mask]]

        top_anomalies = anomalies_with_scores.sort_values(by='Severity', ascending=False).head(30)

        top_anomalies['Timestamp'] = pd.to_datetime(top_anomalies['Timestamp'])
        top_anomalies['Second'] = top_anomalies['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Print the anomalies and their neighboring points
        print("Anomalies and their neighboring points:")
        for index, row in top_anomalies.iterrows():
            print(f"Anomaly: {row['Second']}")
            print(f"Neighboring Points: {row['Neighboring_Points']}")
            print("---")

        secondly_anomaly_counts = top_anomalies.groupby('Second').size().sort_values(ascending=False)

        print("anomalies per second:")

        file_name_with_extension = os.path.basename(file)
        file_name, _ = os.path.splitext(file_name_with_extension)

        df_anomalies = secondly_anomaly_counts.to_frame(name=file_name)
        anomalies_list.append(df_anomalies)
        ###############################

        # Plot the time series data only if query_values.size >= 3 and there are anomalies
        if query_values.size >= 3 and len(anomalies[anomalies]) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(query_set.index, query_set.values, marker='o', linestyle='', label='Data', markersize=3)

            # Mark the anomalies on the x-axis
            anomaly_indices = anomalies[anomalies].index
            for i, anomaly_index in enumerate(anomaly_indices):
                plt.axvline(anomaly_index, color='red', linestyle='dashed', alpha=0.5, ymin=0, ymax=0.95)

                # Plot the neighboring points for each anomaly with different colors and markers
                for j in range(3):  # 对于每个异常点的三个最近邻
                    neighboring_point = query_set.iloc[indices[field_1.index == anomaly_index][0][j]]
                    plt.plot(anomaly_index, neighboring_point, marker=markers[j], color=colors[j],
                             linestyle='', markersize=5, label=f'Neighboring Point {j + 1}')

            # Configure the x-axis date format and tick frequency
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

            plt.xticks(rotation=45)
            plt.legend()

            plt.title(f"{file.split('/')[-1]}")
            plt.tight_layout()

            # Save the figure to a file
            figure_path = os.path.join(figures_directory, f"{file_name}.png")
            plt.savefig(figure_path)
            plt.close()

            print(f"Figure saved: {figure_path}")

        print('anomaly done')

    # Concatenate all dataframes in the list to create one dataframe per subsystem
    if anomalies_list:
        subsystem_anomalies[subsystem] = pd.concat(anomalies_list, axis=1)
        subsystem_anomalies[subsystem].fillna(0, inplace=True)
        subsystem_anomalies[subsystem] = subsystem_anomalies[subsystem]