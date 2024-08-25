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

ins='m2409'


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

filepath = 'marketdata20240801.csv'
filepath1='marketdata20240802.csv'
filepath2='marketdata20240805.csv'
data1 = pd.read_csv(filepath)
data2 = pd.read_csv(filepath1)
data3 = pd.read_csv(filepath2)

# 合并数据集
data = pd.concat([data1, data2,data3])
column_lists = data.columns
data.index.names = column_lists[:2]
data.index = data.index.set_levels([data.index.levels[0].astype(str), data.index.levels[1].str.strip()])
data = data.iloc[:, :-2]
data.columns = column_lists[2:]
data['TradingDay'] = data.index.get_level_values('TradingDay')
data = data.droplevel('TradingDay')
# data= data[['LastPrice', 'TradingDay','UpdateTime', 'UpdateMillisec']]
data = data.loc[ins]


data['Time'] = data['TradingDay'].astype(str) + '.' +data['UpdateTime'].astype(str).str.strip()+ '.' + data['UpdateMillisec'].astype(str)



data.reset_index(drop=True, inplace=True)
time_format = '%Y%m%d.%H:%M:%S.%f'
data['Timestamp'] = pd.to_datetime(data['Time'], format=time_format)



data=data[['LastPrice','Timestamp']]

def era(x):
    if x.LastPrice >= x.MA4:
        # 上涨
        if x.LastPrice >= x.MA3 and x.MA3 >= x.MA4:
            if x.LastPrice >= x.MA2 and x.MA2 >= x.MA3:
                if x.LastPrice >= x.MA1 and x.MA1 >= x.MA2:
                    # MA1+
                    return 4
                else:
                    # MA2+
                    return 3
            else:
                # MA3+
                return 2
        else:
            # MA4+
            return 1
    else:
        # 下跌
        if x.LastPrice <= x.MA3 and x.MA3 <= x.MA4:
            if x.LastPrice <= x.MA2 and x.MA2 <= x.MA3:
                if x.LastPrice <= x.MA1 and x.MA1 <= x.MA2:
                    # MA1-
                    return -4
                else:
                    # MA2-
                    return -3
            else:
                # MA3-
                return -2
        else:
            # MA4-
            return -1

def calculate_slope(series, window):
    slopes = [np.nan] * len(series)  # 初始化为全NaN
    x = np.arange(window).reshape(-1, 1)  # 提前创建好x的值
    for i in range(window, len(series)):
        if not np.isnan(series[i-window:i]).any():  # 确保窗口内没有NaN
            y = series[i-window:i].values
            model = LinearRegression().fit(x, y)
            slopes[i] = model.coef_[0]  # 只有在没有NaN时才计算斜率
    return slopes



window=500
# 计算均线
data['MA1'] = data['LastPrice'].rolling(window=window).mean()
data['MA2'] = data['LastPrice'].rolling(window=window*2).mean()
data['MA3'] = data['LastPrice'].rolling(window=window*4).mean()
data['MA4'] = data['LastPrice'].rolling(window=window*12).mean()

data['MA1_slope'] =  calculate_slope(data['MA1'] ,window*3)


data['MA5']=data['LastPrice'].rolling(window=window*24).mean()
data['era'] = data.apply(lambda r:era(r), axis=1)

# 标记斜率由正变负的点
data['MA1_Turning_Point_Pos_to_Neg'] = (
    (data['MA1_slope'].shift(1) > 0) &
    (data['MA1_slope'] < 0)
)

# 标记斜率由负变正的点
data['MA1_Turning_Point_Neg_to_Pos'] = (
    (data['MA1_slope'].shift(1) < 0) &
    (data['MA1_slope'] > 0)
)

print(data['era'].value_counts())
print(data['era'].tail(100))
# 计算均线之间的差值
data['diff1'] = data['MA2'] - data['MA1']
data['diff2'] = data['MA3'] - data['MA2']

# 检查均线是否满足聚合到发散的形态，并记录信号点
data['signal'] = (data['MA1'] > data['MA2']) & (data['MA2'] > data['MA3']) & \
                 (data['diff1'].shift(1) > data['diff1']) & (data['diff2'].shift(1) > data['diff2']) & \
                 (data['diff1'] < data['diff1'].shift(-1)) & (data['diff2'] < data['diff2'].shift(-1))

data['signal1']=(data['MA1'] > data['MA2']) & (data['MA2'] > data['MA3'])& (data['MA3'] > data['MA4'])
data['first_signal'] = data['signal1'] & (~data['signal1'].shift(1).fillna(False))

# 创建一个新的整数索引，用于表示连续的时间点
data['index'] = range(len(data))
# # 绘制数据
# plt.plot(data['index'], data['LastPrice'], label='LastPrice', color='grey', alpha=0.3)
plt.plot(data['index'], data['MA1'], label='MA1', color='black', alpha=0.3)
plt.plot(data['index'], data['MA2'], label='MA2', color='blue', alpha=0.3)
plt.plot(data['index'], data['MA3'], label='MA3', color='red', alpha=0.3)
plt.plot(data['index'], data['MA4'], label='MA4', color='green', alpha=0.3)

plt.plot(data['index'], data['MA5'], label='MA5', color='yellow', alpha=0.3)

# # 标记信号点
# plt.scatter(data['index'][data['signal']], data['LastPrice'][data['signal']], color='red', label='Signal', marker=',')
# plt.scatter(data['index'][data['signal1']], data['LastPrice'][data['signal1']], color='blue', label='Signal1', marker=',')
# plt.scatter(data['index'][data['first_signal']], data['LastPrice'][data['first_signal']], color='green', label='First Signal', marker='o')

# plt.plot(data['index'][data['era'] == 4], data['LastPrice'][data['era'] == 4], color='red', label='era4')


data['era_4_data'] = data.apply(lambda row: row['LastPrice'] if row['era'] >= 3 else None, axis=1)
data['era_down'] = data.apply(lambda row: row['LastPrice'] if row['era'] <= -3 else None, axis=1)
plt.plot(data['index'], data['LastPrice'], label='LastPrice', color='grey', alpha=0.3)
plt.plot(data['index'], data['era_4_data'], color='blue', label='era4')
plt.plot(data['index'], data['era_down'], color='red', label='era_down')

plt.scatter(data['index'][data['MA1_Turning_Point_Pos_to_Neg']],
            data['MA1'][data['MA1_Turning_Point_Pos_to_Neg']],
            color='red', label='MA1_Turning_Point_Pos_to_Neg', marker='v')
plt.scatter(data['index'][data['MA1_Turning_Point_Neg_to_Pos']],
            data['MA1'][data['MA1_Turning_Point_Neg_to_Pos']],
            color='green', label='MA1_Turning_Point_Neg_to_Pos', marker='^')

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


