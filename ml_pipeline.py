import pandas as pd
pd.options.display.max_rows = None
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.dates as mdates
import glob
import os
import time
# 设置Pandas和Matplotlib的选项
pd.options.display.max_rows = None

# 获取所有CSV文件的路
#csv_files = glob.glob('/Users/wan404/Documents/bmf.raw/data/data_all/paf_indoor/*.csv')
csv_files = glob.glob('/Users/wan404/Documents/bmf.raw/data/rerun/**/*.csv', recursive=True)
# print(csv_files)
# df1 = pd.read_csv(csv_files[0])
# df2 = pd.read_csv(csv_files[1])
# df1 = pd.concat([df1, df2], ignore_index=True)

summary_df = pd.DataFrame()

# 确定子图的排列
num_files = len(csv_files)
cols = 1  # 选择每行显示的子图数量
rows = num_files // cols + (num_files % cols > 0)

# 创建一个足够大的图表来显示所有子图
#plt.figure(figsize=(15, 5 * rows))

start_period = '2021-06-15  01:30:00'
end_period = '2021-06-15  04:00:00'

for i, file in enumerate(csv_files, start=1):
    print('begin', file)
    start_time = time.time()  # 记录开始时间
    df1 = pd.read_csv(file)
    field_1 = pd.DataFrame(df1, columns=['time', 'value'])
    field_1['time'] = pd.to_datetime(field_1['time'], errors='coerce')
    field_1 = field_1[(field_1['time'] >= start_period) & (field_1['time'] <= end_period)]
    # field_1_filtered = field_1[(field_1['time'].dt.date >= pd.to_datetime(start_period).date()) &
    #                   (field_1['time'].dt.date <= pd.to_datetime(end_period).date())]

    field_1 = field_1.set_index('time')
    field_1.index = pd.to_datetime(field_1.index)
    field_1['value'] = field_1['value'].astype(float)
    #field_1['value'] = field_1['value'].map({'Normal': 0, 'FAULT': 1})

    # Extract a single column from the DataFrame
    time_stamp = df1['time']

    # Convert the column to a list using the tolist() method
    time_list = time_stamp.values

    ########### this is knn ###########
    # Initialize k-NN with specified parameters
    knn = NearestNeighbors(n_neighbors=3)

    # Reshape the data to fit the k-NN model

    values_reshaped = field_1.values.reshape(-1, 1)

    if values_reshaped.size < 3:
        print(f"Skipping empty file: {file}")
        continue  # 跳过后续操作，继续下一个文件

    # Fit the model to the data
    knn.fit(values_reshaped)

    # Find the distances and indices of the k nearest neighbors
    distances, indices = knn.kneighbors(values_reshaped)

    # Calculate the mean distance to the k nearest neighbors
    mean_distances = np.mean(distances, axis=1)

    # Define a threshold for anomalies based on the mean distance
    threshold = 3 * np.mean(mean_distances) # 3 for drx bmf, 3000 for paf, 3200 for chiller

    # Create a boolean mask for anomalies
    anomaly_mask = mean_distances > threshold

    # Create a boolean series with the same index as the original series
    anomalies = pd.Series(anomaly_mask, index=field_1.index)

    # Create a DataFrame to store anomalies with their severity scores
    anomalies_with_scores = pd.DataFrame({'Timestamp': field_1.index, 'Severity': mean_distances})
    anomalies_with_scores = anomalies_with_scores[anomalies_with_scores['Severity'] > threshold]

    # Sort the anomalies by severity in descending order
    top_anomalies = anomalies_with_scores.sort_values(by='Severity', ascending=False).head(30)

    # 确保Timestamp列是datetime类型
    top_anomalies['Timestamp'] = pd.to_datetime(top_anomalies['Timestamp'])

    # 创建一个新列，仅包含日期和小时
    #top_anomalies['Hour'] = top_anomalies['Timestamp'].dt.strftime('%Y-%m-%d %H:00:00')
    top_anomalies['Second'] = top_anomalies['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    #top_anomalies['HalfHour'] = top_anomalies['Timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:') + ('00:00' if x.minute < 30 else '30:00'))

    # 计算每个小时的异常数量
    #hourly_anomaly_counts = top_anomalies.groupby('Hour').size().sort_values(ascending=False)
    secondly_anomaly_counts = top_anomalies.groupby('Second').size().sort_values(ascending=False)
    #half_hourly_anomaly_counts = top_anomalies.groupby('HalfHour').size().sort_values(ascending=False)

    # 打印每个小时的异常数量
    print("anomalies per second:")

    # 使用os.path.basename获取文件名和扩展名（例如：'bmf_fpga_c01_2021.csv'）
    file_name_with_extension = os.path.basename(file)

    # 使用os.path.splitext分离文件名和扩展名
    file_name, _ = os.path.splitext(file_name_with_extension)

    #summary_df = pd.concat([summary_df, hourly_anomaly_counts.rename(file_name)], axis=1)
    summary_df = pd.concat([summary_df, secondly_anomaly_counts.rename(file_name)], axis=1)

    # # Pretty print the top ten anomalies
    # print("Top 5 Most Severe Anomalies in Chiller:")
    # print("-" * 40)
    # for idx, row in top_anomalies.iterrows():
    #     print(f"Timestamp: {row['Timestamp']}, Severity: {row['Severity']:.2f}")
    # print("-" * 40)

    ###############################

    print('anomaly done')

    end_time = time.time()
    processing_time_seconds = end_time - start_time
    processing_time_minutes = processing_time_seconds // 60  # 计算完整分钟数
    processing_time_seconds = processing_time_seconds % 60  # 计算剩余的秒数

    # 打印每个文件的处理时间，格式化为分钟和秒
    print(f'{os.path.basename(file)} takes {int(processing_time_minutes)} minutes and {processing_time_seconds:.2f} seconds')

'''
###############################
# Plot the time series data
    ax = plt.subplot(rows, cols, i)
    ax.plot(field_1.index, field_1.values, marker='o', linestyle='', label='Data', markersize=1)

# Mark the anomaly on the x-axis
    anomaly_indices = anomalies[anomalies].index
    for anomaly_index in anomaly_indices:
        ax.axvline(anomaly_index, color='red', linestyle='dashed', alpha=0.5, ymin=0, ymax=0.95)

    # Configure the x-axis date format and tick frequency
    # 设置x轴的日期格式和刻度为每月
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    ax.tick_params(axis='x', rotation=45)
    ax.set_title(f"{file.split('/')[-1]}")
    print('figure done\n')
'''

# 调整汇总DataFrame的格式：填充缺失值，转换为整数等
summary_df.fillna(0, inplace=True)
summary_df = summary_df.astype(int)

# 将每个文件名作为汇总DataFrame的行索引
#summary_df.index.name = 'Hour'
summary_df.index.name = 'second'

summary_df.reset_index(inplace=True)

# 保存汇总结果到新的CSV文件
summary_df.to_csv('/Users/wan404/Documents/bmf.raw/data/test1.csv', index=False)

print("results gathering finished and saved to'chiller_summary_of_anomalies.csv'")

#plt.tight_layout()

#plt.savefig('/Users/wan404/Documents/bmf.raw/data/chiller_my_figure.png')

#plt.show()
