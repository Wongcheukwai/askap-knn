import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from functools import reduce

# Define file paths, these need to be modified according to your file system
bmf_path = '/your path/bmf_3.csv'  # Update with your file path
drx_path = '/your path/drx.csv'  # Update with your file path
chiller_path = '/your path/chiller.csv'  # Update with your file path
paf_outdoor_path = '/your path/pafoutdoor.csv'  # Update with your file path
paf_indoor_path = '/your path/pafindoor.csv'  # Update with your file path

# Load data
bmf = pd.read_csv(bmf_path)
chiller = pd.read_csv(chiller_path)
drx = pd.read_csv(drx_path)
paf_out = pd.read_csv(paf_outdoor_path)
paf_in = pd.read_csv(paf_indoor_path)

# Convert time column to datetime format
dfs = [chiller, bmf, drx, paf_in, paf_out]
for df in dfs:
    df['second'] = pd.to_datetime(df['second'], errors='coerce')

# Filter data within the specific time range
start_period = pd.to_datetime('2022-02-10 06:00:00')
end_period = pd.to_datetime('2022-02-10 07:00:00')
dfs_filtered = [df[(df['second'] >= start_period) & (df['second'] <= end_period)] for df in dfs]

# Calculate time intervals
time_bins = pd.date_range(start=start_period, end=end_period, periods=11)

# Calculate cumulative values for each time interval
histograms = {}
for df, name in zip(dfs_filtered, ['chiller', 'bmf', 'drx', 'paf_in', 'paf_out']):
    hist, bin_edges = np.histogram(df['second'], bins=time_bins)
    histograms[name] = hist

# Plot line chart
plt.figure(figsize=(20, 5))

# Set different markers for each subsystem
markers = ['o', 's', '^', 'D', 'v']  # Circle, Square, Triangle, Diamond, Inverted Triangle

# Convert bin edges to numeric and calculate bin centers
bin_centers_numeric = mdates.date2num(bin_edges[:-1]) + np.diff(mdates.date2num(bin_edges)) / 2

# Plot line charts for each subsystem using different markers
for (name, hist), marker in zip(histograms.items(), markers):
    plt.plot_date(mdates.num2date(bin_centers_numeric), hist, marker=marker, linestyle='-', label=name)

# Set legend
plt.legend()

# Format datetime axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=9))  # One tick every 9 minutes
plt.gcf().autofmt_xdate()  # Automatically rotate date labels

# Set title and axis labels
plt.title(f'Cumulative Events in Each Subsystem')
plt.xlabel('Time Intervals')
plt.ylabel('Cumulative Events')

# Display chart here
plt.show()
