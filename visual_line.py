import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from functools import reduce

# 定义文件路径，这些需要根据您的文件系统进行修改
bmf_path = '/Users/wan404/Documents/bmf.raw/data/data_all/bmf_3.csv'  # 更新为你的文件路径
drx_path = '/Users/wan404/Documents/bmf.raw/data/data_all/drx.csv'  # 更新为你的文件路径
chiller_path = '/Users/wan404/Documents/bmf.raw/data/data_all/chiller.csv'  # 更新为你的文件路径
paf_outdoor_path = '/Users/wan404/Documents/bmf.raw/data/data_all/pafoutdoor.csv'  # 更新为你的文件路径
paf_indoor_path = '/Users/wan404/Documents/bmf.raw/data/data_all/pafindoor.csv'  # 更新为你的文件路径

# 加载数据
bmf = pd.read_csv(bmf_path)
chiller = pd.read_csv(chiller_path)
drx = pd.read_csv(drx_path)
paf_out = pd.read_csv(paf_outdoor_path)
paf_in = pd.read_csv(paf_indoor_path)

# 转换时间列格式
dfs = [chiller, bmf, drx, paf_in, paf_out]
for df in dfs:
    df['second'] = pd.to_datetime(df['second'], errors='coerce')

# 筛选特定时间范围内的数据
start_period = pd.to_datetime('2022-02-10 06:00:00')
end_period = pd.to_datetime('2022-02-10 07:00:00')
dfs_filtered = [df[(df['second'] >= start_period) & (df['second'] <= end_period)] for df in dfs]

# 计算时间间隔
time_bins = pd.date_range(start=start_period, end=end_period, periods=11)

# 计算每个时间间隔内的累积值
histograms = {}
for df, name in zip(dfs_filtered, ['chiller', 'bmf', 'drx', 'paf_in', 'paf_out']):
    hist, bin_edges = np.histogram(df['second'], bins=time_bins)
    histograms[name] = hist

# 绘制折线图
plt.figure(figsize=(20, 5))

# 为每个子系统设置不同的标记
markers = ['o', 's', '^', 'D', 'v']  # 圆形，正方形，三角形，菱形，倒三角形

# 将时间间隔的边界转换为数值型，以便计算中点
bin_centers_numeric = mdates.date2num(bin_edges[:-1]) + np.diff(mdates.date2num(bin_edges))/2

# 为每个子系统绘制折线图，使用不同的标记
for (name, hist), marker in zip(histograms.items(), markers):
    plt.plot_date(mdates.num2date(bin_centers_numeric), hist, marker=marker, linestyle='-', label=name)

# 设置图例
plt.legend()

# 格式化日期时间格式
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=9))  # 每9分钟一个时间标记
plt.gcf().autofmt_xdate()  # 自动旋转日期标记

# 设置标题和轴标签
plt.title(f'Cumulative Events in Each Subsystem')
plt.xlabel('Time Intervals')
plt.ylabel('Cumulative Events')


# 显示图表
plt.show()
