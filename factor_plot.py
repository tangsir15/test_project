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

data=pd.read_csv('y2409_factor_era1.csv')#  y2409_factor_era.csv  y2409_factor_era_pre1.csv y2409_factor_era1.csv
print(data)
data['index'] = range(len(data))
data['upper'] = data.apply(
    lambda row: row['LastPrice'] if row['factor'] > 0 else None, axis=1)
data['down'] = data.apply(
    lambda row: row['LastPrice'] if row['factor'] < 0else None, axis=1)
# 绘制数据，使用新的索引
plt.plot(data['index'], data['LastPrice'], label='LastPrice', color='grey', alpha=0.3)

plt.plot(data['index'], data['upper'], color='green', label='down')
plt.plot(data['index'], data['down'], color='red', label='upper')

# # 自定义 X 轴标签的显示
# def format_func(value, tick_number):
#     # 根据当前索引值找到对应的时间戳
#     index = int(value)
#     if index >= 0 and index < len(data):
#         return data['Timestamp'].iloc[index].strftime('%Y-%m-%d %H:%M')
#     else:
#         return ''
# # 应用自定义的格式化函数
# plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
plt.xlabel('Time')
plt.ylabel('LastPrice')

plt.legend()

plt.show()