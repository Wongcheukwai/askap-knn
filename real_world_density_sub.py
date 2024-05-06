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

df = pd.read_csv('/Users/wan404/Documents/chiller.raw/bms.chiller-2021-chill01.raw.csv')
df.drop(df.head(3).index,inplace=True)
df.drop(labels=['#datatype', 'string'], axis = 1, inplace=True)
df.columns = ['table', 'time', 'value', 'field', 'measurement', 'station', 'unit']
df = df.reset_index(drop = True)
pd.set_option('display.max_columns', None)

unique_values = df['field'].unique()
index_del = [6, 7, 8, 9] # 2021 2022需要删掉的index
unique_values = np.delete(unique_values, index_del)
print('field', unique_values)
#unique_no = 9 # binary： 0,3,4,14, Noraml: 1,2,（12）,13,（15） seasonal: 5,9, on and off: 6, 8, 10, 11 fault or normal：7
#field_1 = df.loc[df['field']==unique_values[unique_no]] # seasonal 16 13 11 10 OFF: 15 14 12 9 8 7 5 4 2 normal 6 3 1 0 for chiller 3
fields = [unique_values[i] for i in [5, 9]]  # Select three fields for demonstration. You can adjust this.
multi_df = df[df['field'].isin(fields)]

pivot_df = multi_df.pivot(index='time', columns='field', values='value').dropna()
#print('field1', pivot_df.head(100))
pivot_df.index = pd.to_datetime(pivot_df.index)
pivot_df = pivot_df.astype(float)

# field_1 = pd.DataFrame(field_1, columns=['time', 'value'])
# field_1 = field_1.set_index('time').head(100)
# field_1.index = pd.to_datetime(field_1.index)
# field_1['value'] = field_1['value'].astype(float)

df0 = pd.read_csv('/Users/wan404/Documents/bmf.raw/ade.bmf.temps-2021-06.raw.csv', usecols=[2, 3, 4, 5, 8])
df0.drop(df0.head(3).index,inplace=True)
df0.columns = ['table', 'time', 'value', 'field', 'card']
df0 = df0.reset_index(drop = True)
pd.set_option('display.max_columns', None)

df = df0
card_name = 'c01'
unique_values = df['field'].unique()
print('field', unique_values)
fields = ['bul_fpga_temp', 'bul_tempATX_temp', 'bul_tempEthernet_temp', 'bul_tempFan_temp', 'bul_tempFpga_temp']
filtered_data = df[(df['field'].isin(fields)) & (df['card'] == card_name)]
#filtered_data = df[(df['table'] == '5')] #paf
pivot_df1 = filtered_data.pivot(index='time', columns='field', values='value').dropna()
#print('field1', pivot_df.head(100))
pivot_df1.index = pd.to_datetime(pivot_df.index)
pivot_df = pivot_df.astype(float)#field_1 = field_1[field_1['time'].str.contains('2021-06-14|2021-06-15|2021-06-16', na=False)]



from adtk.data import validate_series
#field_1 = validate_series(field_1)
pivot_df = validate_series(pivot_df)

########### this is knn ###########
# Initialize k-NN with specified parameters
knn = NearestNeighbors(n_neighbors=2)

# Reshape the data to fit the k-NN model

#values_reshaped = field_1.values.reshape(-1, 1)

# Fit the model to the data
knn.fit(pivot_df)

# Find the distances and indices of the k nearest neighbors
distances, indices = knn.kneighbors(pivot_df)

# Calculate the mean distance to the k nearest neighbors
mean_distances = np.mean(distances, axis=1)

print(mean_distances)
# Define a threshold for anomalies based on the mean distance
threshold = 300 * np.mean(mean_distances)
# Create a boolean mask for anomalies
anomaly_mask = mean_distances > threshold

# Create a boolean series with the same index as the original series
anomaly_times = pivot_df.index[anomaly_mask]

fig, axes = plt.subplots(nrows=len(pivot_df.columns), figsize=(10, 7))

for col, ax in zip(pivot_df.columns, axes):
    ax.plot(pivot_df.index, pivot_df[col], label=col, marker='o', linestyle='', markersize=1)
    ax.set_ylabel(col)
    for anomaly_time in anomaly_times:
        ax.axvline(anomaly_time, color='red', linestyle='dashed', alpha=0.5)

    # Configure the x-axis date format and tick frequency
    date_format = mdates.DateFormatter('%Y-%m-%d')
    locator = mdates.AutoDateLocator(minticks=10, maxticks=20)

    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis.set_major_locator(locator)

plt.tight_layout()
plt.show()



