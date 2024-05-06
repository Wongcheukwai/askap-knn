import pandas as pd
pd.options.display.max_rows = None
from adtk.data import validate_series
from adtk.visualization import plot
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.dates as mdates
import pickle
from adtk.detector import QuantileAD
from adtk.detector import VolatilityShiftAD
from sklearn.ensemble import IsolationForest
from sklearn.cluster import OPTICS
from sklearn.neighbors import LocalOutlierFactor
import time
import json
import pickle



## 这是读取paf里只有一个成card的csv
# with open('list.pkl', 'rb') as file:
#     unique_values = pickle.load(file)
'''

for u in unique_values:
    df = pd.read_csv('/Users/wan404/Documents/paf.raw/ade.paf.temps-2021-01.raw.csv', usecols=[5, 8])
    df.drop(df.head(3).index, inplace=True)
    df.columns = ['field', 'card']
    df = df.reset_index(drop=True)
    filter = df[(df['field'] == u)]
    only_C = filter['card'].eq('C').all()
    if not only_C:
        print('not only C', u)

    if only_C:
        dfs = []
        for i in range(1, 13):
            month_str = f"{i:02d}"
            df0 = pd.read_csv('/Users/wan404/Documents/paf.raw/ade.paf.temps-2022-{}.raw.csv'.format(month_str), usecols=[3, 4, 5, 8])
            df0.drop(df0.head(3).index, inplace=True)
            df0.columns = ['time', 'value', 'field', 'card']
            filtered_data = df0[(df0['field'] == u)]
            final_value = pd.DataFrame(filtered_data, columns=['time', 'value'])
            dfs.append(final_value)

        df2022 = pd.concat(dfs, ignore_index=True)
        df2022.to_csv('/Users/wan404/Documents/bmf.raw/data/paf/{}_2022.csv'.format(u), index=False)
print('done')
'''

# 这是针对drx和bmf的
# dfs = []
# for i in range(1, 13):
#     month_str = f"{i:02d}"
#     df0 = pd.read_csv('/Users/wan404/Documents/paf.raw/ade.paf.temps-2022-{}.raw.csv'.format(month_str),
#                       usecols=[3, 4, 5, 8])
#     df0.drop(df0.head(3).index, inplace=True)
#     df0.columns = ['time', 'value', 'field', 'card']
#     filtered_data = df0[(df0['field'] == 'bul_tempLocal_temp') & (df0['card'] =='tempLocal2')]
#     final_value = pd.DataFrame(filtered_data, columns=['time', 'value'])
#     dfs.append(final_value)
#
# df2022 = pd.concat(dfs, ignore_index=True)
# df2022.to_csv('/Users/wan404/Documents/bmf.raw/data/paf/bul_tempLocal_temp_local2_2022.csv', index=False)

# 这是针对chiller
dfs = []
table = 'chiller_CondensorWaterHeaderReturnTemp'
chiller = 'chill03'
df0 = pd.read_csv('/Users/wan404/Documents/chiller.raw/bms.chiller-2021-{}.raw.csv'.format(chiller), usecols=[3, 4, 5])
df0.drop(df0.head(3).index, inplace=True)
df0.columns = ['time', 'value', 'field']
filtered_data = df0[(df0['field'] == table)]
final_value = pd.DataFrame(filtered_data, columns=['time', 'value'])
dfs.append(final_value)

df1 = pd.read_csv('/Users/wan404/Documents/chiller.raw/bms.chiller-2022-{}.raw.csv'.format(chiller), usecols=[3, 4, 5])
df1.drop(df1.head(3).index, inplace=True)
df1.columns = ['time', 'value', 'field']
filtered_data = df1[(df1['field'] == table)]
final_value = pd.DataFrame(filtered_data, columns=['time', 'value'])
dfs.append(final_value)

df2022 = pd.concat(dfs, ignore_index=True)
df2022.to_csv('/Users/wan404/Documents/bmf.raw/data/chiller_all/{}_{}_2021-2022.csv'.format(table, chiller), index=False)


##### 这是为了整合2021和2022的csv
# import os
# import pandas as pd
# import glob
#
# # 获取所有CSV文件的路径
# csv_files = glob.glob('/Users/wan404/Documents/bmf.raw/data/paf/*.csv')
#
# # 将文件按前缀分组
# files_grouped_by_prefix = {}
# for file_path in csv_files:
#     # 提取文件名和扩展名
#     base_name = os.path.basename(file_path)
#     # 假设年份前有一个下划线，并且年份是4位数字
#     prefix, _ = base_name.rsplit('_', 1)
#     # 去除年份和扩展名
#     #prefix = prefix.rsplit('_', 1)
#     # 按前缀分组文件
#     if prefix in files_grouped_by_prefix:
#         files_grouped_by_prefix[prefix].append(file_path)
#     else:
#         files_grouped_by_prefix[prefix] = [file_path]
#
# # 读取、合并和保存每一组数据
# for prefix, files in files_grouped_by_prefix.items():
#     combined_df = pd.DataFrame()
#     for file in files:
#         # 读取CSV文件
#         df = pd.read_csv(file)
#         # 合并数据
#         combined_df = pd.concat([combined_df, df], ignore_index=True)
#     # 保存合并后的数据到新CSV文件
#     combined_df.to_csv(f'/Users/wan404/Documents/bmf.raw/data/paf_all/{prefix}_2021-2023.csv', index=False)
#
# print("all files have been combined")
