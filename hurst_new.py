

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from tqdm import tqdm
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



def hurst_exponent(ts, log_t=2):
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
        subseries = np.array([log_returns[i:i + n] for i in range(0, N - n + 1, n)])

        # 计算每个子区间的标准差和极差
        stds = np.std(subseries, axis=1)
        Ra = np.ptp(subseries, axis=1)  # 极差

        # 计算每个子区间的重标极差Ra/Sa
        R_S_values = Ra / stds

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

# 设定t为过去观察的数量
t = 2000  # 以2000为过去数据点的长度
# 初始化进度条
# tqdm.pandas()
# 初始化进度条
total_windows = len(data['LastPrice']) - t + 1
tqdm.pandas(desc="Calculating Hurst", total=total_windows, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}] {percentage:3.0f}%")



# 使用rolling和apply方法计算Hurst指数，并显示进度条
data['Hurst'] = data['LastPrice'].rolling(window=t).progress_apply(lambda x: hurst_exponent(x), raw=False)

print(data[['LastPrice', 'Hurst']].dropna())  # 打印含Hurst指数的结果