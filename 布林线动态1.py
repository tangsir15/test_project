import os

import matplotlib
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression
import numpy as np
matplotlib.use('TkAgg')

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

# data.set_index('Timestamp', inplace=True)

from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

bollinger_period = 800  # Bollinger 带的周期
slope_period = 200  # 用于计算斜率的周期
base_multiplier = 3.0  # 基础倍数

# 计算移动平均线和标准差
data['Middle Band'] = data['LastPrice'].rolling(window=bollinger_period).mean()
data['Std Dev'] = data['LastPrice'].rolling(window=bollinger_period).std()
print(data['Std Dev'].describe())
# 计算斜率
def calculate_slope(series):
    if len(series) < slope_period:
        return np.nan
    X = np.arange(slope_period).reshape(-1, 1)  # 时间步长 (0, 1, 2, ..., 99)
    y = series.values[-slope_period:]  # 最后 100 个值
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_[0]  # 返回斜率

# 计算每个时间点的斜率
data['Slope'] = data['Middle Band'].rolling(window=slope_period).apply(calculate_slope, raw=False)
print(data['Slope'].describe())
# 根据市场趋势动态调整 std_dev_multiplier
def dynamic_multiplier(slope, base_multiplier):
    if np.isnan(slope):
        return base_multiplier
    elif slope > 0.002:  # 上升趋势，增大倍数
        return base_multiplier + 1.5
    elif slope < -0.002:  # 下降趋势，减小倍数
        return base_multiplier - 1.0
    else:  # 横盘趋势，保持基础倍数
        return base_multiplier

# 应用动态倍数
data['std_dev_multiplier'] = data['Slope'].apply(lambda x: dynamic_multiplier(x, base_multiplier))

# 计算动态调整后的 Bollinger 带
data['Upper Band'] = data['Middle Band'] + (data['Std Dev'] * data['std_dev_multiplier'])
data['Lower Band'] = data['Middle Band'] - (data['Std Dev'] * data['std_dev_multiplier'])



# # 找出超出布林线的点
#
data['upper_band_exceed'] = data.apply(lambda row: row['LastPrice'] if row['LastPrice'] > row['Upper Band'] else None, axis=1)
data['lower_band_exceed'] = data.apply(lambda row: row['LastPrice'] if row['LastPrice'] < row['Lower Band'] else None, axis=1)

std_dev_multiplier1=0.5
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
