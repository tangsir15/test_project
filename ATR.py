import os
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression
import numpy as np


ins='fu2409'


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

data.set_index('Timestamp', inplace=True)

# 生成分钟级别的OHLC数据
ohlc_data = data['LastPrice'].resample('1T').ohlc()
# 计算True Range
ohlc_data['High-Low'] = ohlc_data['high'] - ohlc_data['low']
ohlc_data['High-Close'] = (ohlc_data['high'] - ohlc_data['close'].shift(1)).abs()
ohlc_data['Low-Close'] = (ohlc_data['low'] - ohlc_data['close'].shift(1)).abs()
ohlc_data['TR'] = ohlc_data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
# 计算ATR
ohlc_data['ATR'] = ohlc_data['TR'].rolling(window=60, min_periods=1).mean()

# 结果
print(ohlc_data[['ATR']].head(50))

ohlc_data = ohlc_data[['ATR']].reset_index().sort_values('Timestamp')
tick_data_sorted = data.reset_index().sort_values('Timestamp')


# 计算ATR的98百分位数
atr_98_percentile = ohlc_data['ATR'].quantile(0.95)

# 标记ATR值大于98百分位数的行
# ohlc_data['ATR_Above_98_Percentile'] = ohlc_data['ATR'] > atr_98_percentile
ohlc_data['ATR_Above_98_Percentile'] = ohlc_data['ATR'] > 4.6
# 合并数据
tick_data_with_atr = pd.merge_asof(tick_data_sorted, ohlc_data, on='Timestamp')


tick_data_with_atr['Timestamp'] = tick_data_with_atr['Timestamp'].astype(str)
# 绘图
plt.figure(figsize=(12, 6))
plt.plot(tick_data_with_atr['Timestamp'], tick_data_with_atr['LastPrice'], label='LastPrice', color='grey', alpha=0.3)


# 标记ATR值大于98百分位数的点

atr_above_98 = tick_data_with_atr[tick_data_with_atr['ATR_Above_98_Percentile']]
plt.scatter(atr_above_98['Timestamp'], atr_above_98['LastPrice'], color='red', label='ATR > 3.5')

plt.legend()
plt.tight_layout()
plt.show()


# bollinger_period = 20
# std_dev_multiplier = 2  # 可以设置为动态的值
#
# data['Middle Band'] = data['LastPrice'].rolling(window=bollinger_period).mean()
# data['Std Dev'] = data['LastPrice'].rolling(window=bollinger_period).std()
#
# # 动态倍数，例如基于市场波动调整
# data['Upper Band'] = data['Middle Band'] + (data['Std Dev'] * std_dev_multiplier)
# data['Lower Band'] = data['Middle Band'] - (data['Std Dev'] * std_dev_multiplier)
#
# # 计算布林带宽度 (Band Width) 和布林带百分比
# data['Band Width'] = (data['Upper Band'] - data['Lower Band']) / data['Middle Band']
# data['Band Percentage'] = (data['LastPrice'] - data['Lower Band']) / (data['Upper Band'] - data['Lower Band'])
#
# # 计算动态阈值
# # 例如使用布林带宽度的历史均值的标准差作为动态阈值
# width_mean = data['Band Width'].rolling(window=bollinger_period).mean()
# width_std = data['Band Width'].rolling(window=bollinger_period).std()
# dynamic_bandwidth_threshold = width_mean + width_std



