import os
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
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


filtered_data_main['Time'] = filtered_data_main['TradingDay'].astype(str) + '.' +filtered_data_main['UpdateTime'].astype(str) + '.' + filtered_data_main['UpdateMillisec'].astype(str)
filtered_data_main.reset_index(drop=True, inplace=True)
filtered_data_main.set_index('Time', inplace=True)
filtered_data_main['MA100'] = filtered_data_main['LastPrice'].rolling(window=100).mean()
filtered_data_main['MA300'] = filtered_data_main['LastPrice'].rolling(window=300).mean()
filtered_data_main['MA1000'] = filtered_data_main['LastPrice'].rolling(window=1000).mean()



def calculate_slope(series, window):
    slopes = [np.nan] * len(series)  # 初始化为全NaN
    x = np.arange(window).reshape(-1, 1)  # 提前创建好x的值
    for i in range(window, len(series)):
        if not np.isnan(series[i-window:i]).any():  # 确保窗口内没有NaN
            y = series[i-window:i].values
            model = LinearRegression().fit(x, y)
            slopes[i] = model.coef_[0]  # 只有在没有NaN时才计算斜率
    return slopes

# 使用150周期窗口计算MA300的斜率，忽略前300个值
filtered_data_main['Slope_MA300'] = calculate_slope(filtered_data_main['MA300'], 150)



# 标记斜率由正变负的点
filtered_data_main['Turning_Point_Pos_to_Neg'] = (
    (filtered_data_main['Slope_MA300'].shift(1) > 0) &
    (filtered_data_main['Slope_MA300'] < 0)
)

# 标记斜率由负变正的点
filtered_data_main['Turning_Point_Neg_to_Pos'] = (
    (filtered_data_main['Slope_MA300'].shift(1) < 0) &
    (filtered_data_main['Slope_MA300'] > 0)
)





# # 创建新列并初始化为NaN
# filtered_data_main['Turning_Point'] = np.nan
# # 设置Turning_Point_Pos_to_Neg为True的位置为0
# filtered_data_main.loc[filtered_data_main['Turning_Point_Pos_to_Neg'], 'Turning_Point'] = 0
# # 设置Turning_Point_Neg_to_Pos为True的位置为1
# filtered_data_main.loc[filtered_data_main['Turning_Point_Neg_to_Pos'], 'Turning_Point'] = 1
# count = filtered_data_main['Turning_Point'].sum()
# print(f"Turning_Point count: {count}")
# print(filtered_data_main['Turning_Point'].head(200))






# # 计算Turning_Point_Pos_to_Neg中为True的数量
# count_pos_to_neg = filtered_data_main['Turning_Point_Pos_to_Neg'].sum()
# # 计算Turning_Point_Neg_to_Pos中为True的数量
# count_neg_to_pos = filtered_data_main['Turning_Point_Neg_to_Pos'].sum()
# print(f"Turning_Point_Pos_to_Neg count: {count_pos_to_neg}")
# print(f"Turning_Point_Neg_to_Pos count: {count_neg_to_pos}")
#
#
#
# 绘图
plt.figure(figsize=(12, 6))
plt.plot(filtered_data_main.index, filtered_data_main['LastPrice'], label='LastPrice',color='grey',alpha=0.3)
plt.plot(filtered_data_main.index, filtered_data_main['MA300'], label='MA300', color='orange')
plt.plot(filtered_data_main.index, filtered_data_main['MA1000'], label='MA1000', color='blue')
plt.scatter(filtered_data_main[filtered_data_main['Turning_Point_Pos_to_Neg']].index,
            filtered_data_main['LastPrice'][filtered_data_main['Turning_Point_Pos_to_Neg']],
            color='red', label='Turning Point Pos to Neg', marker='v')
plt.scatter(filtered_data_main[filtered_data_main['Turning_Point_Neg_to_Pos']].index,
            filtered_data_main['LastPrice'][filtered_data_main['Turning_Point_Neg_to_Pos']],
            color='green', label='Turning Point Neg to Pos', marker='^')

plt.legend()
plt.tight_layout()
plt.show()










