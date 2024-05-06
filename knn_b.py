import pandas as pd
pd.options.display.max_rows = None
from adtk.data import validate_series
from adtk.visualization import plot
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.dates as mdates
import pickle
from adtk.detector import QuantileAD
from adtk.detector import VolatilityShiftAD
from sklearn.ensemble import IsolationForest
from sklearn.cluster import OPTICS
from sklearn.neighbors import LocalOutlierFactor
import time

start_time = time.time()
# this demo uses the chiller data csv file, I can send it to you.

# read the csv file for 2021
'''
dfs = []
for i in range(1, 13):
    month_str = f"{i:02d}"
    df = pd.read_csv('/Users/wan404/Documents/bmf.raw/ade.bmf.temps-2021-{}.raw.csv'.format(month_str), usecols=[2, 3, 4, 5, 8])
    dfs.append(df)
df2022 = pd.concat(dfs, ignore_index=True)
df2022.to_csv('/Users/wan404/Documents/bmf.raw/data2023.csv', index=False)
print('done')
'''

df0 = pd.read_csv('/Users/wan404/Documents/bmf.raw/ade.bmf.temps-2021-03.raw.csv', usecols=[2, 3, 4, 5, 8])
df0.drop(df0.head(3).index,inplace=True)
df0.columns = ['table', 'time', 'value', 'field', 'card']
df0 = df0.reset_index(drop = True)
pd.set_option('display.max_columns', None)

# df1 = pd.read_csv('/Users/wan404/Documents/bmf.raw/data2022.csv')
# df1.drop(df1.head(3).index,inplace=True)
# df1.columns = ['table', 'time', 'value', 'field', 'card']
# df1 = df1.reset_index(drop = True)
# pd.set_option('display.max_columns', None)

# frame = [df0, df1]
# df = pd.concat(frame)

df = df0

card_name = 'c01'
filtered_data = df[(df['table'] == '1') & (df['card'] == card_name)]
#filtered_data = df[(df['table'] == '5')] #paf
field_1 = pd.DataFrame(filtered_data, columns=['time', 'value'])
#field_1 = field_1[field_1['time'].str.contains('2021-06-14|2021-06-15|2021-06-16', na=False)]

field_1 = field_1.set_index('time')
field_1.index = pd.to_datetime(field_1.index)
field_1['value'] = field_1['value'].astype(float)
print(field_1.head(20))

from adtk.data import validate_series
field_1 = validate_series(field_1)

# Extract a single column from the DataFrame
time_stamp = df['time']

# Convert the column to a list using the tolist() method
time_list = time_stamp.values

########### this is knn ###########
#'''
# Initialize k-NN with specified parameters
knn = NearestNeighbors(n_neighbors=2)

# Reshape the data to fit the k-NN model

values_reshaped = field_1.values.reshape(-1, 1)

# Fit the model to the data
knn.fit(values_reshaped)

# Find the distances and indices of the k nearest neighbors
distances, indices = knn.kneighbors(values_reshaped)

# Calculate the mean distance to the k nearest neighbors
mean_distances = np.mean(distances, axis=1)

# Define a threshold for anomalies based on the mean distance
threshold = 100 * np.mean(mean_distances)

# Create a boolean mask for anomalies
anomaly_mask = mean_distances > threshold

# Create a boolean series with the same index as the original series
anomalies = pd.Series(anomaly_mask, index=field_1.index)

##############################
end_time = time.time()  # End time measurement
total_time = end_time - start_time
print(f"Total time to run the code: {total_time} seconds")

###############################
# Plot the time series data
fig, ax = plt.subplots()
ax.plot(field_1.index, field_1.values, marker='o', linestyle='', label='Data', markersize=1)

# Mark the anomaly on the x-axis
anomaly_indices = anomalies[anomalies].index
for anomaly_index in anomaly_indices:
    ax.axvline(anomaly_index, color='red', linestyle='dashed', alpha=0.5, ymin=0, ymax=0.95)

# Configure the x-axis date format and tick frequency
date_format = mdates.DateFormatter('%Y-%m-%d')
locator = mdates.AutoDateLocator(minticks=10, maxticks=20)

ax.xaxis.set_major_formatter(date_format)
ax.xaxis.set_major_locator(locator)

fig.autofmt_xdate()
plt.show()
###############################

with open("/Users/wan404/Documents/bmf.raw/bmf_anomaly_tempFan_{}.pkl".format(card_name), "wb") as f:
     pickle.dump(anomalies, f)
with open("/Users/wan404/Documents/bmf.raw/bmf_field_tempFan_{}.pkl".format(card_name), "wb") as f:
     pickle.dump(field_1, f)

#'''
# Plot the results
#plot(field_1, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=1, figsize=(10, 5))

###############################
# ax = plot(field_1, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=1, figsize=(10, 5))

#plt.show()

