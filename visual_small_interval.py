import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
# Use reduce to perform consecutive merges on the 'Hour' column
from functools import reduce
import glob
import os
import pickle

with open('/Users/wan404/Documents/bmf.raw/data/summary_df_list.pkl', 'rb') as f:
    summary_df_list = pickle.load(f)

# 读取CSV文件
bmf = summary_df_list[1]
chiller = summary_df_list[0]
drx = summary_df_list[2]
paf_out = summary_df_list[4]
paf_in = summary_df_list[3]

# 预处理时间列
dfs = [chiller, bmf, drx, paf_in, paf_out]
for df in dfs:
    df['second'] = pd.to_datetime(df['second'], errors='coerce')

# 设置时间范围
start_period = '2022-02-14 08:00:00'
end_period = '2022-02-14 09:00:00'

# Create a directory to save the figures
figures_directory = '/Users/wan404/Documents/bmf.raw/data/figures/'
os.makedirs(figures_directory, exist_ok=True)
# 筛选时间范围内的数据
dfs_filtered = [df[(df['second'] >= start_period) & (df['second'] <= end_period)] for df in dfs]

a = dfs_filtered[0]
b = dfs_filtered[1]
c = dfs_filtered[2]
d = dfs_filtered[3]
e = dfs_filtered[4]

# 创建时间间隔
time_bins = pd.date_range(start=start_period, end=end_period, periods=11)  # 创建10个bins

# 计算每个子系统在每个时间间隔的事件计数
histograms = {}
for df, name in zip(dfs_filtered, ['chiller', 'bmf', 'drx', 'paf_in', 'paf_out']):
    hist, bin_edges = np.histogram(df['second'], bins=time_bins)
    histograms[name] = hist

# 初始化一个全零矩阵用于热力图
histogram_matrix = np.zeros((len(histograms), len(time_bins)-1))

# 填充矩阵
for idx, (name, hist) in enumerate(histograms.items()):
    histogram_matrix[idx, :] = hist

# 创建时间间隔的标签
bin_labels = [f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')}"
              for start, end in zip(time_bins[:-1], time_bins[1:])]

# 转换为DataFrame
histogram_df = pd.DataFrame(histogram_matrix, index=histograms.keys(), columns=bin_labels)

# 绘制热力图
plt.figure(figsize=(20, 5))
ax = sns.heatmap(histogram_df, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': 'Event Count'})
plt.title(f'Event Count in Each Subsystem from {start_period[2: -3]} to {end_period[2: -3]}')
plt.xlabel('Time Intervals')
plt.ylabel('Subsystems')
time_labels = [t.split(' - ')[0] for t in bin_labels]  # 只显示开始时间
ax.set_xticklabels(time_labels)
plt.yticks(rotation=0)  # 设置y轴标签旋转度
plt.show()
