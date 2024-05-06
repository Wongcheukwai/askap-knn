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

df = pd.read_csv('/Users/wan404/Documents/chiller.raw/bms.chiller-2021-chill02.raw.csv')
df.drop(df.head(3).index,inplace=True)
#df.drop(df.tail(8).index,inplace=True)
df.drop(labels=['#datatype', 'string'], axis = 1, inplace=True)
df.columns = ['table', 'time', 'value', 'field', 'measurement', 'station', 'unit']
df = df.reset_index(drop = True)
pd.set_option('display.max_columns', None)
#print(df.head(10))
#print(df.tail(30))

'''
df1 = pd.read_csv('/Users/wan404/Documents/chiller.raw/bms.chiller-2022-chill01.raw.csv')
df1.drop(df.head(3).index,inplace=True)
#df.drop(df.tail(8).index,inplace=True)
df1.drop(labels=['#datatype', 'string'], axis = 1, inplace=True)
df1.columns = ['table', 'time', 'value', 'field', 'measurement', 'station', 'unit']
df1 = df1.reset_index(drop = True)
#pd.set_option('display.max_columns', None)

frame = [df, df1]
df = pd.concat(frame)
'''

unique_values = df['field'].unique()
# index_del = [4, 5, 6, 7] # 2023需要删掉的index
index_del = [6, 7, 8, 9] # 2021 2022需要删掉的index
unique_values = np.delete(unique_values, index_del)
print('field', unique_values)
unique_no = 1 # binary： 0,3,4,14, Noraml: 1,2,（12）,13,（15） seasonal: 5,9, on and off: 6, 8, 10, 11 fault or normal：7 for chiller 1 and 2
#for i in range(0, unique_values.size):
field_1 = df.loc[df['field']==unique_values[unique_no]] # seasonal 16 13 11 10 OFF: 15 14 12 9 8 7 5 4 2 normal 6 3 1 0 for chiller 3
field_1 = pd.DataFrame(field_1, columns=['time', 'value'])
field_1 = field_1.set_index('time')
field_1.index = pd.to_datetime(field_1.index)
field_1['value'] = field_1['value'].astype(float)
#field_1['value'] = field_1['value'].map({'Normal': 0, 'FAULT': 1})

# print('field', field_1)
from adtk.data import validate_series
field_1 = validate_series(field_1)

# Extract a single column from the DataFrame
time_stamp = df['time']

# Convert the column to a list using the tolist() method
time_list = time_stamp.values

########### this is knn ###########
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
threshold = 3 * np.mean(mean_distances)

# Create a boolean mask for anomalies
anomaly_mask = mean_distances > threshold

# Create a boolean series with the same index as the original series
anomalies = pd.Series(anomaly_mask, index=field_1.index)

##############################
anomaly_times = anomalies[anomalies].index
anomaly_times_df = pd.DataFrame(anomaly_times)

# Save to CSV file
anomaly_times_df.to_csv('anomaly_timestamps.csv', index=False, header=False)

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


# Plot the results
#plot(field_1, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=1, figsize=(10, 5))

###############################
# ax = plot(field_1, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=1, figsize=(10, 5))
# #
# ax[0].set_title(unique_values[unique_no])
# # print('done')
# #
# plt.show()

