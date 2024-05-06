import pandas as pd
pd.options.display.max_rows = None
from adtk.data import validate_series
from adtk.visualization import plot
import matplotlib.pyplot as plt
import numpy as np
from adtk.detector import AutoregressionAD
import csv
from sklearn.cluster import KMeans

import datetime
starttime = datetime.datetime.now()

s_train = pd.read_csv("/Users/wan404/Documents/ASKAP_2.csv", index_col="channel", parse_dates=True, squeeze=True)
e = s_train.columns.values.tolist()
dic_channel= {}
dic_value = {}
#for i in range(len(e)):
for i in range(1):
    each_beam = s_train[e[i]]
    each_beam.index = pd.to_datetime(each_beam.index)
    each_beam = validate_series(each_beam)
    autoregression_ad = AutoregressionAD(n_steps=2, step_size=10, c=6.0)
    anomalies = autoregression_ad.fit_detect(each_beam)
    index_channel = each_beam.index
    amplitude_value_channel = each_beam.values
    anomaly_value_channel = anomalies.values
    threshold = 0.5
    anomaly_index_channel = list(np.where(anomaly_value_channel == 1))
    anomaly_value_channel = amplitude_value_channel[anomaly_index_channel]
    dic_channel[e[i]] = anomaly_index_channel
    dic_value[e[i]] = anomaly_value_channel
    #plot(each_beam, anomaly=anomalies, ts_markersize=1, anomaly_color='red', anomaly_tag="marker", anomaly_markersize=2)

    print('done')

value_1 = np.array(list(dic_value.values()))
value_1 = np.concatenate(value_1, axis=0)

print(value_1)
#np.save('/Users/wan404/Documents/askap2.npy', value_1)

data_1 = np.load('/Users/wan404/Documents/askap1.npy')
# Fit the k-means model
model = KMeans(n_clusters=4)
model.fit(data_1.reshape(-1, 1))

# Plot the data
plt.scatter(data_1, np.zeros_like(data_1), c=model.labels_, cmap='viridis')
plt.show()
# endtime = datetime.datetime.now()
# print((endtime - starttime).seconds)
#
# with open('/Users/wan404/Documents/dic_channel.csv', 'w', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=dic_channel.keys())
#     writer.writeheader()
#     writer.writerow(dic_channel)
#
# with open('/Users/wan404/Documents/dic_channel.csv', 'a', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=dic_value.keys())
#     writer.writerow(dic_value)
#
# import pandas as pd
#
# df = pd.read_csv('/Users/wan404/Documents/dic_channel.csv')
# data = df.values
# index1 = list(df.keys())
# data = list(map(list, zip(*data)))
# data = pd.DataFrame(data, index=index1)
# data.to_csv('/Users/wan404/Documents/dic_channel.csv', header=0)

print('done')


#plot(s_train, anomaly=anomalies, ts_markersize=1, anomaly_color='red', anomaly_tag="marker", anomaly_markersize=2)


#plot(s_train, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker")
#plt.show()
