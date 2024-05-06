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
bmf_path = '/Users/wan404/Documents/bmf.raw/data/test1.csv'  # 更新为你的文件路径

time_stamp = 'Around 21-03-23 11:00:00'
merged_df_all_filled = pd.read_csv(bmf_path)

###############################################
csv_bmf = glob.glob('/Users/wan404/Documents/bmf.raw/data/data_all/bmf_all/*.csv')
file_bmf = [os.path.basename(f).rstrip('.csv') for f in csv_bmf]

csv_drx = glob.glob('/Users/wan404/Documents/bmf.raw/data/data_all/drx_all/*.csv')
file_drx = [os.path.basename(f).rstrip('.csv') for f in csv_drx]

csv_chiller = glob.glob('/Users/wan404/Documents/bmf.raw/data/data_all/chiller_all/*.csv')
file_chiller = [os.path.basename(f).rstrip('.csv') for f in csv_chiller]

csv_paf = glob.glob('/Users/wan404/Documents/bmf.raw/data/data_all/paf_all/*.csv')
file_paf = [os.path.basename(f).rstrip('.csv') for f in csv_paf]
#file_paf = [name[4:] if name.startswith('paf_') else name for name in file_paf]

csv_indoor = glob.glob('/Users/wan404/Documents/bmf.raw/data/data_all/paf_indoor/*.csv')
file_indoor = [os.path.basename(f).rstrip('.csv') for f in csv_indoor]
#file_indoor = [name[4:] if name.startswith('paf_') else name for name in file_indoor]

###############################################


###############################################
# 初始化求和列为0
merged_df_all_filled['Beamformer'] = 0
merged_df_all_filled['Digitiser'] = 0
merged_df_all_filled['Chiller'] = 0
merged_df_all_filled['PAF_Outdoor'] = 0
merged_df_all_filled['PAF_Indoor'] = 0

# 每个数据源的列进行求和，如果列存在
for col in file_bmf:
    if col in merged_df_all_filled.columns:
        merged_df_all_filled['Beamformer'] += merged_df_all_filled[col]

for col in file_drx:
    if col in merged_df_all_filled.columns:
        merged_df_all_filled['Digitiser'] += merged_df_all_filled[col]

for col in file_chiller:
    if col in merged_df_all_filled.columns:
        merged_df_all_filled['Chiller'] += merged_df_all_filled[col]

for col in file_paf:
    if col in merged_df_all_filled.columns:
        merged_df_all_filled['PAF_Outdoor'] += merged_df_all_filled[col]

for col in file_indoor:
    if col in merged_df_all_filled.columns:
        merged_df_all_filled['PAF_Indoor'] += merged_df_all_filled[col]

summed_df = merged_df_all_filled[['second', 'Beamformer', 'Digitiser', 'Chiller', 'PAF_Outdoor', 'PAF_Indoor']]

##############################################

df = summed_df

# 查看数据前几行以了解其结构
print(df.head())
#df['second'] = pd.to_datetime(df['second'], errors='coerce')

# Sort the DataFrame by 'Hour'
df = df.sort_values(by='second')
print(df.head())

# Set 'Hour' as index and transpose for the heatmap
heatmap_data = df.set_index('second').T

plt.figure(figsize=(80, 40))  # 根据需要调整图形大小
ax = sns.heatmap(df.set_index('second').T, cmap='viridis', annot=True, annot_kws={"size": 30})
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
xticks = ax.get_xticks()
xticklabels = [item.get_text()[5:] for item in ax.get_xticklabels()]  # 保留到小时级别
ax.set_xticklabels(xticklabels)

plt.title(f'Anomaly Across Different Subsystems Around {time_stamp}', fontsize=50)
plt.xlabel('Time')
plt.xticks(rotation=85)  # 根据需要调整标签旋转角度
plt.ylabel('CSV Files', size=50)  # 调整y轴标签的字体大小为15
plt.xticks(size=35)  # 调整y轴标签的字体大小
plt.yticks(size=35)  # 调整y轴标签的字体大小

plt.savefig(f'/Users/wan404/Documents/bmf.raw/data/data_all/{time_stamp}.png')
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

