import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
# Use reduce to perform consecutive merges on the 'Hour' column
from functools import reduce

# 加载数据
bmf_path = '/Users/wan404/Documents/bmf.raw/data/bmf_summary_of_anomalies.csv'  # 更新为你的文件路径
drx_path = '/Users/wan404/Documents/bmf.raw/data/drx_summary_of_anomalies.csv'  # 更新为你的文件路径
chiller_path = '/Users/wan404/Documents/bmf.raw/data/chiller_summary_of_anomalies.csv'  # 更新为你的文件路径
paf_path = '/Users/wan404/Documents/bmf.raw/data/paf2000_summary_of_anomalies.csv'  # 更新为你的文件路径

bmf = pd.read_csv(bmf_path)
chiller = pd.read_csv(chiller_path)
drx = pd.read_csv(drx_path)
paf = pd.read_csv(paf_path)

# List of dataframes for merging
dfs = [bmf, drx, chiller, paf]

# Perform the merge
merged_df_all = reduce(lambda left, right: pd.merge(left, right, on='Hour', how='outer'), dfs)

# Fill NaN values with 0 for non-intersecting time points
merged_df_all_filled = merged_df_all.fillna(0)

# Verify the result
merged_df_all_filled.head(), merged_df_all_filled.shape

df = merged_df_all_filled

# 查看数据前几行以了解其结构
print(df.head())

# Sort the DataFrame by 'Hour'
df.sort_values('Hour', inplace=True)

# Check if any row sums up to 1, and if so, drop those rows

df['row_sum'] = df.drop('Hour', axis=1).sum(axis=1)

df_filtered = df[df['row_sum'] > 1].drop('row_sum', axis=1)

# Set 'Hour' as index and transpose for the heatmap
heatmap_data = df_filtered.set_index('Hour').T

plt.figure(figsize=(70, 40))  # 根据需要调整图形大小
ax = sns.heatmap(df_filtered.set_index('Hour').T, cmap='viridis', annot=True, annot_kws={"size": 30})

xticks = ax.get_xticks()
xticklabels = [item.get_text()[2:13] for item in ax.get_xticklabels()]  # 保留到小时级别
ax.set_xticklabels(xticklabels)

plt.title('Anomaly Counts Across Different Subsystems', fontsize=50)
plt.xlabel('Time')
plt.xticks(rotation=75)  # 根据需要调整标签旋转角度
plt.ylabel('CSV Files', size=50)  # 调整y轴标签的字体大小为15
plt.xticks(size=30)  # 调整y轴标签的字体大小
plt.yticks(size=30)  # 调整y轴标签的字体大小

#plt.savefig('/Users/wan404/Documents/bmf.raw/data/all_heatmap.png')
plt.show()


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
#
# # 加载数据
# file_path = '/Users/wan404/Documents/bmf.raw/data/bmf_summary_of_anomalies.csv'  # 更新为你的文件路径
# df = pd.read_csv(file_path)
#
# # 确保'Hour'列是datetime类型
# #df['Hour'] = pd.to_datetime(df['Hour'])
# df['Hour'] = pd.to_datetime(df['Hour'], format='%Y-%m-%d %H:%M')  # 根据你的时间字符串格式调整
#
#
#
#
# # 设置索引并转置DataFrame以适应heatmap的需求
# df = df.set_index('Hour').T
#
# # 绘制热力图
# plt.figure(figsize=(50, 30))  # 根据需要调整图形大小
# ax = sns.heatmap(df, cmap='viridis', annot=True, annot_kws={"size": 30})
# plt.title('Anomaly Counts Across Different CSV Files Over Time')
# plt.xlabel('Time', size=20)  # 调整x轴标签的字体大小为15
# plt.ylabel('CSV Files', size=20)  # 调整y轴标签的字体大小为15
# plt.xticks(rotation=50)  # 根据需要调整标签旋转角度
# plt.yticks(size=20)  # 调整y轴标签的字体大小
# plt.xticks(size=20)  # 调整y轴标签的字体大小
#
#
# # 保存图像
# #plt.savefig('/Users/wan404/Documents/bmf.raw/data/bmf_heatmap_time_sorted.png')
#
# # 显示图形
# plt.show()

