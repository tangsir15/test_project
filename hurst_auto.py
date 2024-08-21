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
ins='au2409'


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

filepath = 'C:/Users/admin/Desktop/斜率/marketdata20240801.csv'
data = pd.read_csv(filepath)
column_lists = data.columns
data.index.names = column_lists[:2]
data.index = data.index.set_levels([data.index.levels[0].astype(str), data.index.levels[1].str.strip()])
data = data.iloc[:, :-2]
data.columns = column_lists[2:]
data['TradingDay'] = data.index.get_level_values('TradingDay')
data = data.droplevel('TradingDay')

data = data.loc[ins]

data['Time'] = data['TradingDay'].astype(str) + '.' +data['UpdateTime'].astype(str).str.strip()+ '.' + data['UpdateMillisec'].astype(str)



data.reset_index(drop=True, inplace=True)
time_format = '%Y%m%d.%H:%M:%S.%f'
data['Timestamp'] = pd.to_datetime(data['Time'], format=time_format)


# 定义计算Hurst指数的函数
def hurst_exponent(ts):
    H, c, data_rescaled_range = compute_Hc(ts, kind='price', simplified=True)
    return H

# 设定t为过去观察的数量
t = 2000

# 初始化进度条
total_windows = len(data['LastPrice']) - t + 1
tqdm.pandas(desc="Calculating Hurst", total=total_windows, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}] {percentage:3.0f}%")

# 使用rolling和apply方法计算Hurst指数，并显示进度条
data['Hurst'] = data['LastPrice'].rolling(window=t).progress_apply(lambda x: hurst_exponent(x), raw=False)

data.to_csv(f'hurst_{ins}_{t}.csv')

print(data['Hurst'].describe())

# 创建一个新的整数索引，用于表示连续的时间点
data['index'] = range(len(data))

plt.plot(data['index'], data['LastPrice'], label='LastPrice', color='grey', alpha=0.3)

data['hurst_trend_plot'] = data.apply(lambda row: row['LastPrice'] if row['Hurst'] >=0.5 else None, axis=1)
data['hurst_fluctuation_plot'] = data.apply(lambda row: row['LastPrice'] if row['Hurst'] <0.5 else None, axis=1)

plt.plot(data['index'], data['hurst_trend_plot'], color='blue', label='hurst_trend_plot')
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
