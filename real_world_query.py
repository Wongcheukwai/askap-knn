import pandas as pd
import argparse
pd.options.display.max_rows = None
from adtk.data import validate_series
from adtk.visualization import plot
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.dates as mdates
import pickle

import random
from datetime import timedelta

parser = argparse.ArgumentParser(description='Process and analyze time-series data.')
parser.add_argument('--dataset', type=str, default='chiller',help='name of the dataset')
parser.add_argument('--path', type=str, help='path to the CSV file')
parser.add_argument('--interval', type=str, default='default', help='the average value of an interval of time series')
parser.add_argument('--start', type=str, default='2021-11-22 08:00:00', help='start point of your query')
parser.add_argument('--end', type=str, default='2021-12-28 10:00:00', help='end point of your query')
parser.add_argument('--table', type=str, default='chiller_ChilledWaterLeavingTemp', help='which table in the dataset you would like to read')
parser.add_argument('--method', type=str, default='knn', choices=['knn', 'dbscan'], help='method of anomaly detection')
parser.add_argument('--num_anomaly', type=int, default='40', help='number of anomaly you would like to display')
parser.add_argument('--subsystem', type=str, help='the subsystem u would like to make query on')
parser.add_argument('--subtable', type=str, help='the subtable u would like to make query on')

args = parser.parse_args()
print("-" * 40)
print('Experiment: %s %s' % (args.dataset, args.table))
print('Time: from %s to %s' % (args.start, args.end))
print('Interval: %s' % (args.interval))
print('Method: %s' % args.method)
print('Number of anomaly: %d' % args.num_anomaly)
print('Subsystem and table: %s % s' % (args.subsystem, args.subtable))

print("-" * 40)

df = pd.read_csv('/Users/wan404/Documents/chiller.raw/bms.chiller-2021-chill03.raw.csv')
df.drop(df.head(3).index,inplace=True)
#df.drop(df.tail(8).index,inplace=True)
df.drop(labels=['#datatype', 'string'], axis = 1, inplace=True)
df.columns = ['table', 'time', 'value', 'field', 'measurement', 'station', 'unit']
df = df.reset_index(drop = True)
#pd.set_option('display.max_columns', None)

df1 = pd.read_csv('/Users/wan404/Documents/chiller.raw/bms.chiller-2022-chill01.raw.csv')
df1.drop(df1.head(3).index,inplace=True)
#df1.drop(df.tail(8).index,inplace=True)
df1.drop(labels=['#datatype', 'string'], axis = 1, inplace=True)
df1.columns = ['table', 'time', 'value', 'field', 'measurement', 'station', 'unit']
df1 = df1.reset_index(drop = True)
#pd.set_option('display.max_columns', None)

'''
df2 = pd.read_csv('/Users/wan404/Documents/chiller.raw/bms.chiller-2023-chill01.raw.csv')
df2.drop(df2.head(3).index,inplace=True)
#df.drop(df.tail(8).index,inplace=True)
df2.drop(labels=['#datatype', 'string'], axis = 1, inplace=True)
df2.columns = ['table', 'time', 'value', 'field', 'measurement', 'station', 'unit']
df2 = df2.reset_index(drop = True)
#pd.set_option('display.max_columns', None)
'''

frame = [df, df1]
df = pd.concat(frame)

unique_values = df['field'].unique()
# index_del = [4, 5, 6, 7] # 2023需要删掉的index
index_del = [6, 7, 8, 9] # 2021 2022需要删掉的index
unique_values = np.delete(unique_values, index_del)
#print('field', unique_values)
#unique_no = 9 # binary： 0,3,4,14, Noraml: 1,2,（12）,13,（15） seasonal: 5,9, on and off: 6, 8, 10, 11 fault or normal：7

field_1 = df.loc[df['field']==args.table] # seasonal 16 13 11 10 OFF: 15 14 12 9 8 7 5 4 2 normal 6 3 1 0 for chiller 3
field_1 = pd.DataFrame(field_1, columns=['time', 'value'])
#field_1_1 = field_1[field_1['time'].str.contains('2021-03-22', na=False)]
#print(field_1_1)
field_1['time'] = pd.to_datetime(field_1['time'], errors='coerce')

print(field_1)
field_1 = field_1[(field_1['time'].dt.date >= pd.to_datetime(args.start).date()) &
                           (field_1['time'].dt.date <= pd.to_datetime(args.end).date())]

field_1 = field_1.set_index('time')
field_1.index = pd.to_datetime(field_1.index)
field_1['value'] = field_1['value'].astype(float)
#field_1['value'] = field_1['value'].map({'Normal': 0, 'FAULT': 1})

num_values = len(field_1)
# Calculate the standard deviation
std_deviation = field_1['value'].std()
print('Num of points:', num_values)
print('Standard deviation:', std_deviation)

from adtk.data import validate_series
field_1 = validate_series(field_1)

# Extract a single column from the DataFrame
time_stamp = df['time']

# Convert the column to a list using the tolist() method
time_list = time_stamp.values

####################### autoregressionad ###########################
# from adtk.detector import AutoregressionAD
# #autoregression_ad = AutoregressionAD(n_steps=7*2, step_size=5, c=12.0)
# autoregression_ad = AutoregressionAD(n_steps=7*2, step_size=10, c=50)
# anomalies_auto = autoregression_ad.fit_detect(field_1)
# anomaly_value_channel = anomalies_auto.values
# anomaly_index_channel = (anomaly_value_channel > 0.5).reshape(-1)
# threshold = 0.5
# anomalies = pd.Series(anomaly_index_channel, index=field_1.index)


# ########### this is dbscan ###########
# Initialize DBSCAN with specified parameters

# values_reshaped = field_1.values.reshape(-1, 1)
#
# dbscan = DBSCAN(eps=0.5, min_samples=5)
#
# # Fit the model to the data
# dbscan.fit(values_reshaped)
#
# # Create a boolean mask for anomalies (label -1 represents anomalies in DBSCAN)
# anomaly_mask = dbscan.labels_ == -1
#
# anomalies = pd.Series(anomaly_mask, index=field_1.index)

#####################################
# Apply OPTICS for anomaly detection
# model = LocalOutlierFactor(n_neighbors=100, contamination=0.005)
# anomaly_scores = model.fit_predict(field_1.values.reshape(-1, 1))
# anomalies = pd.Series(anomaly_scores, index=field_1.index) == -1
# print('done')
########### this is knn ###########
# Initialize k-NN with specified parameters
knn = NearestNeighbors(n_neighbors=10)

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

# Create a DataFrame to store anomalies with their severity scores
anomalies_with_scores = pd.DataFrame({'Timestamp': field_1.index, 'Severity': mean_distances})
anomalies_with_scores = anomalies_with_scores[anomalies_with_scores['Severity'] > threshold]


#############################这是画图的#############################
'''
# Plot 2: Time Series Data with Anomalies Highlighted
plt.figure(figsize=(12, 6))
plt.plot(field_1.index, field_1['value'], label='Data', color='blue')
for idx, row in anomalies_with_scores.iterrows():
    plt.scatter(row['Timestamp'], field_1.loc[row['Timestamp'], 'value'],
                s=row['Severity'] * 100,  # Adjust the scaling factor as needed
                color='red', label='Anomaly' if idx == 0 else "")

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Anomalies and Severity Scores %s' % args.table)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
'''

# Sort the anomalies by severity in descending order
top_anomalies = anomalies_with_scores.sort_values(by='Severity', ascending=False).head(args.num_anomaly)
sub_anomalies = anomalies_with_scores.sort_values(by='Severity', ascending=False).head(args.num_anomaly // 2)

# Pretty print the top ten anomalies
print("Top %d Most Severe Anomalies in Chiller:" % args.num_anomaly)
print("-" * 40)
for idx, row in top_anomalies.iterrows():
    print(f"Timestamp: {row['Timestamp']}, Severity: {row['Severity']:.2f}")
print("-" * 40)

print('Related Anomalies in %s:' % args.subsystem)
print("-" * 40)
for idx, row in sub_anomalies.iterrows():
    print(f"Timestamp: {row['Timestamp']}, Severity: {row['Severity']:.2f}")
print("-" * 40)

###############################
# Plot the time series data
'''
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


with open("/Users/wan404/Documents/knn_2_%s.pkl" % args.table, "wb") as f:
     pickle.dump(anomalies, f)
with open("/Users/wan404/Documents/lknn_2_%s.pkl" % args.table, "wb") as f:
     pickle.dump(field_1, f)
'''

