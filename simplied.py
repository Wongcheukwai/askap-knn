import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from adtk.data import validate_series
from adtk.visualization import plot
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

s_train = pd.read_csv("/Users/wan404/Documents/ASKAP_2.csv", index_col="channel", parse_dates=True, squeeze=True)
e = s_train.columns.values.tolist()
s_train = s_train[e[401]]
dic = {}
#s_train = pd.read_csv("/Users/wan404/Documents/nyc_taxi.csv", index_col="timestamp", parse_dates=True, squeeze=True)
a = s_train.index
#print(s_train)
s_train.index = pd.to_datetime(s_train.index)
s_train = validate_series(s_train)
print('s_train', s_train)

# Reshape the data to fit the k-NN model
values_reshaped = s_train.values.reshape(-1, 1)

########### this is dbscan ###########
# Initialize DBSCAN with specified parameters
dbscan = DBSCAN(eps=3, min_samples=2)

# Fit the model to the data
dbscan.fit(values_reshaped)

# Create a boolean mask for anomalies (label -1 represents anomalies in DBSCAN)
anomaly_mask = dbscan.labels_ == -1

########### this is knn ###########
# Initialize k-NN with specified parameters
# knn = NearestNeighbors(n_neighbors=2)
#
# # Fit the model to the data
# knn.fit(values_reshaped)
#
# # Find the distances and indices of the k nearest neighbors
# distances, indices = knn.kneighbors(values_reshaped)
#
# # Calculate the mean distance to the k nearest neighbors
# mean_distances = np.mean(distances, axis=1)
#
# # Define a threshold for anomalies based on the mean distance
# threshold = 3 * np.mean(mean_distances)
#
# # Create a boolean mask for anomalies
# anomaly_mask = mean_distances > threshold

# Create a boolean series with the same index as the original series
anomalies = pd.Series(anomaly_mask, index=s_train.index)

# Plot the results
plot(s_train, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=1, figsize=(10, 5))

plt.show()

###############################kmeans#########################
# import numpy as np
# import pandas as pd
# from sklearn.cluster import KMeans
# from adtk.data import validate_series
# from adtk.visualization import plot
# import matplotlib.pyplot as plt
#
# # Generate synthetic time series data
# np.random.seed(0)
# n = 100
# timestamps = pd.date_range("2020-01-01", periods=n, freq="D")
# values = np.random.randn(n)
# series = pd.Series(values, index=timestamps)
#
# # Validate the series format
# series = validate_series(series)
#
# # Introduce artificial anomalies
# series["2020-01-15"] += 10
# series["2020-01-25"] += -10
#
# # Reshape the data to fit the clustering model
# values_reshaped = series.values.reshape(-1, 1)
#
# # Initialize KMeans with 2 clusters (normal and anomaly)
# kmeans = KMeans(n_clusters=2, random_state=0)
#
# # Fit the model to the data
# kmeans.fit(values_reshaped)
#
# # Predict the cluster labels
# labels = kmeans.predict(values_reshaped)
#
# # Create a boolean mask for anomalies (assuming the smaller cluster represents anomalies)
# anomaly_cluster = np.argmin(np.bincount(labels))
# anomaly_mask = labels == anomaly_cluster
#
# # Create a boolean series with the same index as the original series
# anomalies = pd.Series(anomaly_mask, index=series.index)
#
# # Plot the results
# plot(series, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=0.3, figsize=(10, 5))
#
# plt.show()