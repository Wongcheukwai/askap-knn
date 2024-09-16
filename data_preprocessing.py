import pandas as pd
import argparse
import os
import glob

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


# Argument parser to select the mode (paf, drx, bmf, chiller)
parser = argparse.ArgumentParser(description="Process different modes: paf, drx, bmf, chiller.")
parser.add_argument('--mode', type=str, required=True, help='Choose the mode: paf, drx, bmf, chiller, or combine')
args = parser.parse_args()


if args.mode == 'paf':
    # This is for reading CSV in paf where only one card exists
    with open('list.pkl', 'rb') as file:
        unique_values = pickle.load(file)

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
                df0 = pd.read_csv('/Users/wan404/Documents/paf.raw/ade.paf.temps-2022-{}.raw.csv'.format(month_str),
                                  usecols=[3, 4, 5, 8])
                df0.drop(df0.head(3).index, inplace=True)
                df0.columns = ['time', 'value', 'field', 'card']
                filtered_data = df0[(df0['field'] == u)]
                final_value = pd.DataFrame(filtered_data, columns=['time', 'value'])
                dfs.append(final_value)

            df2022 = pd.concat(dfs, ignore_index=True)
            df2022.to_csv('/Users/wan404/Documents/bmf.raw/data/paf/{}_2022.csv'.format(u), index=False)
    print('done')

elif args.mode == 'drx' or args.mode == 'bmf':
    # This is for drx and bmf
    dfs = []
    for i in range(1, 13):
        month_str = f"{i:02d}"
        df0 = pd.read_csv('/Users/wan404/Documents/paf.raw/ade.paf.temps-2022-{}.raw.csv'.format(month_str),
                          usecols=[3, 4, 5, 8])
        df0.drop(df0.head(3).index, inplace=True)
        df0.columns = ['time', 'value', 'field', 'card']
        filtered_data = df0[(df0['field'] == 'bul_tempLocal_temp') & (df0['card'] == 'tempLocal2')]
        final_value = pd.DataFrame(filtered_data, columns=['time', 'value'])
        dfs.append(final_value)

    df2022 = pd.concat(dfs, ignore_index=True)
    df2022.to_csv('/Users/wan404/Documents/bmf.raw/data/paf/bul_tempLocal_temp_local2_2022.csv', index=False)

elif args.mode == 'chiller':
    # This is for the chiller data
    dfs = []
    table = 'chiller_CondensorWaterHeaderReturnTemp'
    chiller = 'chill03'
    
    # Read 2021 data
    df0 = pd.read_csv('/Users/wan404/Documents/chiller.raw/bms.chiller-2021-{}.raw.csv'.format(chiller), usecols=[3, 4, 5])
    df0.drop(df0.head(3).index, inplace=True)
    df0.columns = ['time', 'value', 'field']
    filtered_data = df0[(df0['field'] == table)]
    final_value = pd.DataFrame(filtered_data, columns=['time', 'value'])
    dfs.append(final_value)

    # Read 2022 data
    df1 = pd.read_csv('/Users/wan404/Documents/chiller.raw/bms.chiller-2022-{}.raw.csv'.format(chiller), usecols=[3, 4, 5])
    df1.drop(df1.head(3).index, inplace=True)
    df1.columns = ['time', 'value', 'field']
    filtered_data = df1[(df1['field'] == table)]
    final_value = pd.DataFrame(filtered_data, columns=['time', 'value'])
    dfs.append(final_value)

    # Combine and save
    df2022 = pd.concat(dfs, ignore_index=True)
    df2022.to_csv('/Users/wan404/Documents/bmf.raw/data/chiller_all/{}_{}_2021-2022.csv'.format(table, chiller), index=False)

elif args.mode == 'combine':
    # This is for combining 2021 and 2022 CSV files
    csv_files = glob.glob('/Users/wan404/Documents/bmf.raw/data/paf/*.csv')

    # Group files by prefix
    files_grouped_by_prefix = {}
    for file_path in csv_files:
        base_name = os.path.basename(file_path)
        prefix, _ = base_name.rsplit('_', 1)
        if prefix in files_grouped_by_prefix:
            files_grouped_by_prefix[prefix].append(file_path)
        else:
            files_grouped_by_prefix[prefix] = [file_path]

    # Combine files by prefix
    for prefix, files in files_grouped_by_prefix.items():
        combined_df = pd.DataFrame()
        for file in files:
            df = pd.read_csv(file)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        combined_df.to_csv(f'/Users/wan404/Documents/bmf.raw/data/paf_all/{prefix}_2021-2023.csv', index=False)

    print("All files have been combined.")

else:
    print("Invalid mode! Please choose from: paf, drx, bmf, chiller, or combine.")