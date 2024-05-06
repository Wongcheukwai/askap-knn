import pandas as pd

pd.options.display.max_rows = None
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.dates as mdates
import glob
import os
import time
import pickle
import itertools

# 设置Pandas和Matplotlib的选项
pd.options.display.max_rows = None

colors = ['b', 'g', 'r']  # 定义三种颜色
markers = ['o', 's', '^']  # 定义三种标记

# Assuming the rerun directory has five subdirectories, each corresponding to a different subsystem,
# and you want to create a list where each element is a DataFrame containing the anomalies for that subsystem.

# First, let's define the path to the rerun directory and the subdirectories we're interested in.
rerun_path = '/Users/wan404/Documents/bmf.raw/data/rerun/'
subsystems = ['chiller_all', 'bmf_all', 'drx_all', 'paf_indoor', 'paf_all']

# Initialize a dictionary where keys are subsystem names and values will be dataframes of anomalies.
subsystem_anomalies = {subsystem: pd.DataFrame() for subsystem in subsystems}

start_period = '2021-01-13 08:00:00'
end_period = '2022-02-25 09:00:00'

# Create a directory to save the figures
figures_directory = '/Users/wan404/Documents/bmf.raw/data/figures2_14/'
os.makedirs(figures_directory, exist_ok=True)

# Define a list of colors and markers for neighboring points
colors = ['b', 'g', 'c', 'm', 'y']
markers = ['o', 's', '^', 'D', 'v']

# Process files for each subsystem
for subsystem in subsystems:
    csv_files = glob.glob(os.path.join(rerun_path, subsystem, '*.csv'))
    anomalies_list = []

    for file in csv_files:
        print('begin', file)
        df1 = pd.read_csv(file)
        field_1 = pd.DataFrame(df1, columns=['time', 'value'])
        field_1['time'] = pd.to_datetime(field_1['time'], errors='coerce')
        field_1 = field_1[(field_1['time'] >= start_period) & (field_1['time'] <= end_period)]

        field_1 = field_1.set_index('time')
        field_1.index = pd.to_datetime(field_1.index)
        field_1['value'] = field_1['value'].astype(float)

        time_stamp = df1['time']
        time_list = time_stamp.values

        ########### this is knn ###########
        knn = NearestNeighbors(n_neighbors=3)

        values_reshaped = field_1.values.reshape(-1, 1)

        if values_reshaped.size < 3:
            print(f"Skipping empty file: {file}")
            continue

        knn.fit(values_reshaped) # reference set  leaf_size 10 to 40

        distances, indices = knn.kneighbors(values_reshaped) # query neighbour

        mean_distances = np.mean(distances, axis=1)

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

        anomalies = pd.Series(anomaly_mask, index=field_1.index)

        anomalies_with_scores = pd.DataFrame({'Timestamp': field_1.index, 'Severity': mean_distances})
        anomalies_with_scores = anomalies_with_scores[anomalies_with_scores['Severity'] > threshold]

        # Add neighboring points information to the anomalies DataFrame
        anomalies_with_scores['Neighboring_Points'] = [field_1.iloc[idx].values.tolist() for idx in
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

        # Plot the time series data only if values_reshaped.size >= 3 and there are anomalies
        if values_reshaped.size >= 3 and len(anomalies[anomalies]) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(field_1.index, field_1.values, marker='o', linestyle='', label='Data', markersize=3)

            # Mark the anomalies on the x-axis
            anomaly_indices = anomalies[anomalies].index
            for i, anomaly_index in enumerate(anomaly_indices):
                plt.axvline(anomaly_index, color='red', linestyle='dashed', alpha=0.5, ymin=0, ymax=0.95)

                # Plot the neighboring points for each anomaly with different colors and markers
                for j in range(3):  # 对于每个异常点的三个最近邻
                    neighboring_point = field_1.iloc[indices[field_1.index == anomaly_index][0][j]]
                    plt.plot(anomaly_index, neighboring_point, marker=markers[j], color=colors[j],
                             linestyle='', markersize=5, label=f'Neighboring Point {j + 1}')

            # Set the x-axis limits to start_period and end_period
            plt.xlim(pd.Timestamp(start_period), pd.Timestamp(end_period))

            # Configure the x-axis date format and tick frequency
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

            # Divide the x-axis into 10 equal timeslots
            num_timeslots = 10
            start_time = pd.Timestamp(start_period)
            end_time = pd.Timestamp(end_period)
            timeslot_length = (end_time - start_time) / num_timeslots
            timeslot_ticks = [start_time + i * timeslot_length for i in range(num_timeslots + 1)]
            plt.gca().set_xticks(timeslot_ticks)

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
        subsystem_anomalies[subsystem] = subsystem_anomalies[subsystem].astype(int)
        subsystem_anomalies[subsystem].index.name = 'second'
        subsystem_anomalies[subsystem].reset_index(inplace=True)

# Now, convert the dictionary of dataframes into a list of dataframes
summary_df_list = list(subsystem_anomalies.values())

# Check the number of anomalies detected in each subsystem by printing the shape of each DataFrame
for i, df in enumerate(summary_df_list):
    print(f"Subsystem {subsystems[i]} anomalies shape: {df.shape}")

# Save the list of DataFrames to a pickle file
with open('/Users/wan404/Documents/bmf.raw/data/summary_df_list.pkl', 'wb') as f:
    pickle.dump(summary_df_list, f)

print("The list of summary DataFrames has been saved to 'summary_df_list.pkl'.")

print("Results gathering finished for each subsystem.")