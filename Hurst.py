import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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

filepath = 'C:/Users/admin/Desktop/斜率/marketdata20240801.csv'
data = pd.read_csv(filepath)
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

# import seaborn as sns
# filtered_data_main['Timestamp'] = filtered_data_main['Timestamp'].astype(str)
# sns.lineplot(data=filtered_data_main, x='Timestamp', y='LastPrice')
# plt.show()

data=data[['LastPrice','Timestamp']]

def hurst_exponent(ts,log_t=2):
    """计算Hurst指数，基于过去数据点"""
    if len(ts) < 2:
        return np.nan

    # 计算对数收益率
    log_returns = np.log(ts / ts.shift(log_t)).dropna()
    N = len(log_returns)

    # 储存各个子序列的重标极差
    R_S = []

    # 从长度2到N/2进行循环
    for n in range(2, N // 2 + 1):

        # 划分子区间
        subseries = [log_returns[i:i + n] for i in range(0, N - n + 1, n)]

        # 计算每个子区间的标准差
        stds = [np.std(series) for series in subseries]

        # 计算每个子区间的极差R
        Ra = [np.max(series) - np.min(series) for series in subseries]

        # 计算每个子区间的重标极差Ra/Sa
        R_S_values = np.array(Ra) / np.array(stds)

        # 计算子区间的平均重标极差
        avg_R_S = np.mean(R_S_values)
        R_S.append((n, avg_R_S))


    # 转换为numpy数组以便线性回归
    R_S = np.array(R_S)

    # 移除包含NaN值的行
    R_S = R_S[~np.isnan(R_S).any(axis=1)]

    # 取对数
    ln_n = np.log(R_S[:, 0])
    ln_R_S = np.log(R_S[:, 1])

    # 线性回归以计算斜率
    model = LinearRegression()
    model.fit(ln_n.reshape(-1, 1), ln_R_S)
    hurst_value = model.coef_[0]  # 获取斜率即为Hurst指数

    return hurst_value


def dynamic_hurst(ts, t):
    """动态计算Hurst指数"""
    if len(ts) < t:
        return [np.nan] * len(ts)  # 如果数据长度小于t，返回NaN

    hurst_values = []

    # 动态计算Hurst指数
    for i in range(t, len(ts) + 1):
        current_window = ts[i - t:i]  # 选取过去t个数据点
        hurst_value = hurst_exponent(current_window)
        hurst_values.append(hurst_value)

    # 用NaN填充开始的部分，保证与原数据长度一致
    hurst_values = [np.nan] * (t - 1) + hurst_values
    return hurst_values


# 设定t为过去观察的数量
t = 2000  # 以100为过去数据点的长度
hurst_values = dynamic_hurst(data['LastPrice'], t)

# 将输出结果转为DataFrame或Series方便查看
data['Hurst'] = hurst_values
print(data[['LastPrice', 'Hurst']].dropna())  # 打印含Hurst指数的结果

