import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from functools import reduce
import glob
import os

# Load data
bmf_path = '/your path/bmf_3.csv'  # Update with your file path
drx_path = '/your path/drx.csv'  # Update with your file path
chiller_path = '/your path/chiller.csv'  # Update with your file path
paf_outdoor_path = '/your path/pafoutdoor.csv'  # Update with your file path
paf_indoor_path = '/your path/pafindoor.csv'  # Update with your file path

# Read CSV files
bmf = pd.read_csv(bmf_path)
chiller = pd.read_csv(chiller_path)
drx = pd.read_csv(drx_path)
paf_out = pd.read_csv(paf_outdoor_path)
paf_in = pd.read_csv(paf_indoor_path)

# Preprocess the time column
dfs = [chiller, bmf, drx, paf_in, paf_out]
for df in dfs:
    df['second'] = pd.to_datetime(df['second'], errors='coerce')

# Set time range
start_period = '2022-02-10 06:00:00'
end_period = '2022-02-10 07:00:00'

# Filter data within the time range
dfs_filtered = [df[(df['second'] >= start_period) & (df['second'] <= end_period)] for df in dfs]

a = dfs_filtered[0]
b = dfs_filtered[1]
c = dfs_filtered[2]
d = dfs_filtered[3]
e = dfs_filtered[4]

alist = a.columns[(a != 0).any()].tolist()
blist = b.columns[(b != 0).any()].tolist()
clist = c.columns[(c != 0).any()].tolist()
dlist = d.columns[(d != 0).any()].tolist()
elist = e.columns[(e != 0).any()].tolist()

# Create time intervals
time_bins = pd.date_range(start=start_period, end=end_period, periods=11)  # Create 10 bins

# Calculate the event count for each subsystem in each time interval
histograms = {}
for df, name in zip(dfs_filtered, ['chiller', 'bmf', 'drx', 'paf_in', 'paf_out']):
    hist, bin_edges = np.histogram(df['second'], bins=time_bins)
    histograms[name] = hist

# Initialize a zero matrix for the heatmap
histogram_matrix = np.zeros((len(histograms), len(time_bins) - 1))

# Fill the matrix
for idx, (name, hist) in enumerate(histograms.items()):
    histogram_matrix[idx, :] = hist

# Create labels for the time intervals
bin_labels = [f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')}"
              for start, end in zip(time_bins[:-1], time_bins[1:])]

# Convert to DataFrame
histogram_df = pd.DataFrame(histogram_matrix, index=histograms.keys(), columns=bin_labels)

# Plot the heatmap
plt.figure(figsize=(20, 5))
ax = sns.heatmap(histogram_df, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': 'Event Count'})
plt.title(f'Event Count in Each Subsystem from {start_period[2: -3]} to {end_period[2: -3]}')
plt.xlabel('Time Intervals')
plt.ylabel('Subsystems')
time_labels = [t.split(' - ')[0] for t in bin_labels]  # Display only the start time
ax.set_xticklabels(time_labels)
plt.yticks(rotation=0)  # Set the y-axis label rotation
plt.show()