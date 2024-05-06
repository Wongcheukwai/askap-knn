import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
# Use reduce to perform consecutive merges on the 'Hour' column
from functools import reduce
import glob
import os

# 加载数据
bmf_path = '/Users/wan404/Documents/bmf.raw/data/data_all/bmf_3.csv'  # 更新为你的文件路径
drx_path = '/Users/wan404/Documents/bmf.raw/data/data_all/drx.csv'  # 更新为你的文件路径
chiller_path = '/Users/wan404/Documents/bmf.raw/data/data_all/chiller.csv'  # 更新为你的文件路径
paf_outdoor_path = '/Users/wan404/Documents/bmf.raw/data/data_all/pafoutdoor.csv'  # 更新为你的文件路径
paf_indoor_path = '/Users/wan404/Documents/bmf.raw/data/data_all/pafindoor.csv'  # 更新为你的文件路径

# 读取CSV文件
bmf = pd.read_csv(bmf_path)
chiller = pd.read_csv(chiller_path)
drx = pd.read_csv(drx_path)
paf_out = pd.read_csv(paf_outdoor_path)
paf_in = pd.read_csv(paf_indoor_path)

# 预处理时间列
dfs = [chiller, bmf, drx, paf_in, paf_out]
for df in dfs:
    df['second'] = pd.to_datetime(df['second'], errors='coerce')

# 设置时间范围
start_period = '2022-02-10 06:00:00'
end_period = '2022-02-10 07:00:00'

# 筛选时间范围内的数据
dfs_filtered = [df[(df['second'] >= start_period) & (df['second'] <= end_period)] for df in dfs]

a = dfs_filtered[0]
b = dfs_filtered[1]
c = dfs_filtered[2]
d = dfs_filtered[3]
e = dfs_filtered[4]

alist = a.columns[(a != 0).any()].tolist()
blist = b.columns[(b != 0).any()].tolist()
clist = c.columns[(c != 0).any()].tolist()
dlist = d.columns[(d != 0).any()].tolist()
elist = e.columns[(e != 0).any()].tolist()

print(alist)
print(blist)
print(clist)
print(dlist)
print(elist)


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
