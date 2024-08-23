import numpy as np
import pandas as pd
from hurst import compute_Hc
from tqdm import tqdm
import os

import matplotlib
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression
import numpy as np

matplotlib.use('TkAgg')

# filepath='hurst_ag2409_5000.csv'
# data=pd.read_csv(filepath)
#
# data['index'] = range(len(data))
# data['Time'] = data['TradingDay'].astype(str) + '.' +data['UpdateTime'].astype(str).str.strip()+ '.' + data['UpdateMillisec'].astype(str)
# data.reset_index(drop=True, inplace=True)
# time_format = '%Y%m%d.%H:%M:%S.%f'
# data['Timestamp'] = pd.to_datetime(data['Time'], format=time_format)
#
#
#
# plt.plot(data['index'], data['LastPrice'], label='LastPrice', color='grey', alpha=0.3)
#
# data['hurst_trend_plot'] = data.apply(lambda row: row['LastPrice'] if row['Hurst'] >=0.5 else None, axis=1)
# data['hurst_fluctuation_plot'] = data.apply(lambda row: row['LastPrice'] if row['Hurst'] <0.5 else None, axis=1)  #fu 0.45
#
# # plt.plot(data['index'], data['hurst_trend_plot'], color='blue', label='hurst_trend_plot')
# plt.plot(data['index'], data['hurst_fluctuation_plot'], color='red', label='hurst_fluctuation_plot')
#
# # 自定义 X 轴标签的显示
# def format_func(value, tick_number):
#     # 根据当前索引值找到对应的时间戳
#     index = int(value)
#     if index >= 0 and index < len(data):
#         return data['Timestamp'].iloc[index].strftime('%Y-%m-%d %H:%M')
#     else:
#         return ''
#
# # 应用自定义的格式化函数
# plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
#
# plt.xlabel('Time')
# plt.ylabel('LastPrice')
# plt.title('Last Price Over Time with Missing Data Ignored')
# plt.legend()
#
# plt.show()

# 读取CSV文件
# files = ['hurst_ag2409_300.csv','hurst_ag2409_500.csv', 'hurst_ag2409_1000.csv', 'hurst_ag2409_2000.csv', 'hurst_ag2409_5000.csv']
# files = ['hurst_fu2409_500.csv', 'hurst_fu2409_1000.csv', 'hurst_fu2409_2000.csv', 'hurst_fu2409_4000.csv']
files=['hurst_m2409_500.csv', 'hurst_m2409_1000.csv', 'hurst_m2409_2000.csv']
dataframes = [pd.read_csv(file) for file in files]
data = dataframes[0].copy()
data['index'] = range(len(data))
data['Time'] = data['TradingDay'].astype(str) + '.' + data['UpdateTime'].astype(str).str.strip() + '.' + data[
    'UpdateMillisec'].astype(str)
data.reset_index(drop=True, inplace=True)
time_format = '%Y%m%d.%H:%M:%S.%f'
data['Timestamp'] = pd.to_datetime(data['Time'], format=time_format)

param_values = []
for i, df in enumerate(dataframes):
    param_value = files[i].split('_')[-1].split('.')[0]
    data[f'Hurst_{param_value}'] = df['Hurst']
    param_values.append(param_value)

# 设定不同的阈值
thresholds = {
    '300': 0.0,
    '500': 0.5,
    '1000': 0.0,
    '2000': 0.5,
    '4000': 0.0,
    '5000': 0.0,
}

# threshold = thresholds.get(i, 0.5)  # 默认阈值为0.5

data['hurst_fluctuation_plot'] = data.apply(
    lambda row: row['LastPrice'] if any(row[f'Hurst_{i}'] < thresholds.get(i, 0.5) for i in param_values) else None,
    axis=1)

plt.plot(data['index'], data['LastPrice'], label='LastPrice', color='grey', alpha=0.3)
plt.plot(data['index'], data['hurst_fluctuation_plot'], color='red', label='hurst_fluctuation_plot')


# 自定义 X 轴标签的显示
def format_func(value, tick_number):
    # 根据当前索引值找到对应的时间戳
    index = int(value)
    if index >= 0 and index < len(data):
        return data['Timestamp'].iloc[index].strftime('%Y-%m-%d %H:%M')
    else:
        return ''


# 应用自定义的格式化函数
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))

plt.xlabel('Time')
plt.ylabel('LastPrice')
plt.title('Last Price Over Time with Missing Data Ignored')
plt.legend()

plt.show()
