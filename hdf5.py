import pandas as pd
pd.options.display.max_rows = None
from adtk.data import validate_series
from adtk.visualization import plot
import matplotlib.pyplot as plt
import numpy as np
import h5py
import csv
# Open the HDF5 file in read-only mode

import h5py

import datetime
starttime = datetime.datetime.now()

with h5py.File('example.h5', 'r') as file:

    # Access the "my_dataset" dataset in the "B35" group
    dataset = file['B00/amplitude/ampchan_median']

    # Do something with the data
    print(dataset[:].shape)
    print(dataset[:])

    df = pd.DataFrame(dataset)
    df = df.T
print(df)

from adtk.detector import AutoregressionAD

e = df.columns.values.tolist()
dic_channel = {}
dic_value = {}
for i in range(len(e)):
    each_beam = df[e[i]]
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

endtime = datetime.datetime.now()
print((endtime - starttime).seconds)

with open('/Users/wan404/Documents/h5_channel.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=dic_channel.keys())
    writer.writeheader()
    writer.writerow(dic_channel)

with open('/Users/wan404/Documents/h5_channel.csv', 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=dic_value.keys())
    writer.writerow(dic_value)

import pandas as pd

df = pd.read_csv('/Users/wan404/Documents/h5_channel.csv')
data = df.values
index1 = list(df.keys())
data = list(map(list, zip(*data)))
data = pd.DataFrame(data, index=index1)
data.to_csv('/Users/wan404/Documents/h5_channel.csv', header=0)

print('done')


#df = pd.read_csv('/Users/wan404/Documents/chiller.raw/bms.chiller-2023-chill01.raw.csv')


# df.drop(df.head(3).index,inplace=True)
# df.drop(df.tail(8).index,inplace=True)
# df.drop(labels=['#datatype', 'string'], axis = 1, inplace=True)
# df.columns = ['table', 'time', 'value', 'field', 'measurement', 'station', 'unit']
# df= df.reset_index(drop = True)
# pd.set_option('display.max_columns', None)
# #print(df.head(10))
# unique_values = df['field'].unique()
# index_del = [4,5,6,7]
# unique_values = np.delete(unique_values, index_del)
# print('field', unique_values)
#
# #for i in range(0, unique_values.size):
# field_1 = df.loc[df['field']==unique_values[16]] #   seasonal 16 13 11 10 OFF: 15 14 12 9 8 7 5 4 2 normal 6 3 1 0
# field_1 = pd.DataFrame(field_1, columns=['time', 'value'])
# field_1 = field_1.set_index('time')
# field_1.index = pd.to_datetime(field_1.index)
# field_1['value'] = field_1['value'].astype(float)
#
# # freq = pd.infer_freq(field_1.index)
# # if freq is None:
# #     freq = '20min'
# # field_1 = field_1.resample(freq).mean()
# #
# # field_1 = field_1.fillna(method='ffill')
#
# from adtk.data import validate_series
# field_1 = validate_series(field_1)
#
#
# plot(field_1, ts_linewidth=1, ts_markersize=0.4)
#
#
# from adtk.detector import AutoregressionAD
# autoregression_ad = AutoregressionAD(n_steps=7*2, step_size=24, c=3.0)
# anomalies = autoregression_ad.fit_detect(field_1)
# ax = plot(field_1, anomaly=anomalies, ts_markersize=1, anomaly_color='red', anomaly_tag="marker", anomaly_markersize=2)
# ax[0].set_title(unique_values[1])
# print('done')

# from adtk.detector import SeasonalAD
# seasonal_ad = SeasonalAD(c=3.0, side="both")
# anomalies = seasonal_ad.fit_detect(field_1)
# ax = plot(field_1, anomaly=anomalies, ts_markersize=1, anomaly_color='red', anomaly_tag="marker", anomaly_markersize=2)
# ax[0].set_title(unique_values[1])
# print('done')

# from adtk.transformer import ClassicSeasonalDecomposition
# s_transformed = ClassicSeasonalDecomposition().fit_transform(field_1).rename("Seasonal decomposition residual")
# ax = plot(pd.concat([field_1, s_transformed], axis=1), ts_markersize=1)
# ax[0].set_title(unique_values[1])
plt.show()

# field_1 =
# from adtk.data import validate_series
# s = validate_series(s)


#plot(s_train, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker")
