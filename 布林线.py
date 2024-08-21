import os

import matplotlib
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression
import numpy as np
matplotlib.use('TkAgg')

ins='ag2409'


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

filepath = 'marketdata20240801.csv'
data = pd.read_csv(filepath)
column_lists = data.columns
data.index.names = column_lists[:2]
data.index = data.index.set_levels([data.index.levels[0].astype(str), data.index.levels[1].str.strip()])
data = data.iloc[:, :-2]
data.columns = column_lists[2:]
data['TradingDay'] = data.index.get_level_values('TradingDay')
data = data.droplevel('TradingDay')
# data= data[['LastPrice', 'TradingDay','UpdateTime', 'UpdateMillisec']]
filtered_data_main = data.loc[ins]


filtered_data_main['Time'] = filtered_data_main['TradingDay'].astype(str) + '.' +filtered_data_main['UpdateTime'].astype(str).str.strip()+ '.' + filtered_data_main['UpdateMillisec'].astype(str)



filtered_data_main.reset_index(drop=True, inplace=True)
time_format = '%Y%m%d.%H:%M:%S.%f'
filtered_data_main['Timestamp'] = pd.to_datetime(filtered_data_main['Time'], format=time_format)

# import seaborn as sns
# filtered_data_main['Timestamp'] = filtered_data_main['Timestamp'].astype(str)
# sns.lineplot(data=filtered_data_main, x='Timestamp', y='LastPrice')
# plt.show()

data=filtered_data_main[['LastPrice','Timestamp']]

# data.set_index('Timestamp', inplace=True)

bollinger_period = 800
std_dev_multiplier = 3.5# 可以设置为动态的值

data['Middle Band'] = data['LastPrice'].rolling(window=bollinger_period).mean()
data['Std Dev'] = data['LastPrice'].rolling(window=bollinger_period).std()

# 动态倍数，例如基于市场波动调整
data['Upper Band'] = data['Middle Band'] + (data['Std Dev'] * std_dev_multiplier)
data['Lower Band'] = data['Middle Band'] - (data['Std Dev'] * std_dev_multiplier)

# 计算布林带宽度 (Band Width) 和布林带百分比
data['Band Width'] = (data['Upper Band'] - data['Lower Band']) / data['Middle Band']
data['Band Percentage'] = (data['LastPrice'] - data['Lower Band']) / (data['Upper Band'] - data['Lower Band'])

# # 找出超出布林线的点
#
data['upper_band_exceed'] = data.apply(lambda row: row['LastPrice'] if row['LastPrice'] > row['Upper Band'] else None, axis=1)
data['lower_band_exceed'] = data.apply(lambda row: row['LastPrice'] if row['LastPrice'] < row['Lower Band'] else None, axis=1)

std_dev_multiplier1=1
# 初始化一个标志来跟踪条件
flag = False

# 遍历 DataFrame
for i in range(len(data)):
    if pd.notna(data.at[i, 'upper_band_exceed']):
        flag = True
    elif flag and data.at[i, 'LastPrice'] > data.at[i, 'Middle Band']-data.at[i, 'Std Dev']*std_dev_multiplier1:
        data.at[i, 'upper_band_exceed'] = data.at[i, 'LastPrice']
    elif flag and data.at[i, 'LastPrice'] <= data.at[i, 'Middle Band']-data.at[i, 'Std Dev']*std_dev_multiplier1:
        flag = False

# 初始化一个标志来跟踪条件
flag = False

# 遍历 DataFrame
for i in range(len(data)):
    if pd.notna(data.at[i, 'lower_band_exceed']):
        flag = True
    elif flag and data.at[i, 'LastPrice'] < data.at[i, 'Middle Band']+data.at[i, 'Std Dev']*std_dev_multiplier1:
        data.at[i, 'lower_band_exceed'] = data.at[i, 'LastPrice']
    elif flag and data.at[i, 'LastPrice'] >= data.at[i, 'Middle Band']+data.at[i, 'Std Dev']*std_dev_multiplier1:
        flag = False
print(data['upper_band_exceed'])

# 计算动态阈值
# 例如使用布林带宽度的历史均值的标准差作为动态阈值
width_mean = data['Band Width'].rolling(window=bollinger_period).mean()
width_std = data['Band Width'].rolling(window=bollinger_period).std()
dynamic_bandwidth_threshold = width_mean + width_std




# # Extract the Timestamp column
# timestamps = data['Timestamp']
#
# # Save the Timestamp column to a CSV file
# timestamps.to_csv('timestamps.csv', index=False)
# # data['Timestamp'] = data['Timestamp'].astype(str)
# # plt.plot(data['Timestamp'], data['LastPrice'], label='LastPrice', color='grey', alpha=0.3)
# # plt.plot(data['Timestamp'], data['Upper Band'], label='Upper Band', color='blue')
# # plt.plot(data['Timestamp'], data['Lower Band'], label='Lower Band', color='blue')
#
# # Convert columns to NumPy arrays
# timestamps = data['Timestamp'].to_numpy()
# last_price = data['LastPrice'].to_numpy()
# upper_band = data['Upper Band'].to_numpy()
# lower_band = data['Lower Band'].to_numpy()
#
# start_index = len(timestamps) // 2
#
# # Slice the data arrays to include only the last 50%
# timestamps_50 = timestamps[start_index:]
# last_price_50 = last_price[start_index:]
# upper_band_50 = upper_band[start_index:]
# lower_band_50 = lower_band[start_index:]
#
# # Plot using NumPy arrays
# plt.plot(timestamps_50, last_price_50, label='LastPrice', color='grey', alpha=0.3)
# # plt.plot(timestamps, upper_band, label='Upper Band', color='blue')
# # plt.plot(timestamps, lower_band, label='Lower Band', color='blue')
#
# plt.legend()
# plt.tight_layout()
# plt.show()

from matplotlib.ticker import FuncFormatter


# 创建一个新的整数索引，用于表示连续的时间点
data['index'] = range(len(data))

# 绘制数据，使用新的索引
plt.plot(data['index'], data['LastPrice'], label='LastPrice', color='grey', alpha=0.3)

plt.plot(data['index'], data['Middle Band'], label='Middle Band', color='blue', alpha=0.3)
plt.plot(data['index'], data['Upper Band'], label='Upper Band', color='blue', alpha=0.3)
plt.plot(data['index'], data['Lower Band'], label='Lower Band', color='blue', alpha=0.3)

plt.plot(data['index'], data['upper_band_exceed'], color='green', label='upper_band_exceed')
plt.plot(data['index'], data['lower_band_exceed'], color='red', label='lower_band_exceed')




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
