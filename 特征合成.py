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


data['Time'] = data['TradingDay'].astype(str) + '.' +data['UpdateTime'].astype(str) + '.' + data['UpdateMillisec'].astype(str)
data.reset_index(drop=True, inplace=True)
data.set_index('Time', inplace=True)
data['Volume'] = data['Volume'].diff()

data['Bid_Ask_Volume_Ratio'] = data[['BidVolume1','BidVolume2', 'BidVolume3', 'BidVolume4', 'BidVolume5']].sum(axis=1)/  data[['AskVolume1','AskVolume2', 'AskVolume3', 'AskVolume4', 'AskVolume5']].sum(axis=1)

data['Bid_Ask_Volume_Ratio_diff']=data['Bid_Ask_Volume_Ratio'].diff()
data['LastPrice_Return'] = data['LastPrice'].pct_change()




def calculate_and_normalize_momentum(data, column, window,normalized_window):
    """
    计算动量（与前n个tick的差值）并对其进行滑动窗口标准化处理。

    :param data: 包含tick数据的DataFrame
    :param column: 需要计算动量的价格列名（如 'lastprice'）
    :param window: 滑动窗口的大小（即回溯的n个tick）
    :return: 包含标准化价格动量的新列
    """
    # 计算价格动量（当前价格与n个tick前价格的差值）
    temp = abs(data[column] - data[column].shift(window))

    normalized_momentum = []

    for i in range(len(data)):
        if i < normalized_window * 2 - 1:
            normalized_momentum.append(None)  # 数据不足以形成一个完整的窗口时
        else:
            # 获取当前窗口的动量数据
            normalized_window_data =temp.iloc[i - normalized_window + 1:i + 1]
            min_val = normalized_window_data.min()
            max_val = normalized_window_data.max()

            # 对动量进行标准化处理
            if max_val - min_val != 0:
                normalized_value = (temp.iloc[i] - min_val) / (max_val - min_val)
            else:
                normalized_value = 0.0  # 如果最大值和最小值相等，标准化值设为0

            normalized_momentum.append(normalized_value)

    # 将标准化动量结果加入DataFrame
    data[f'Normalized_{column}_Momentum'] = normalized_momentum
    return data

# def calculate_and_normalize_moving_average(data, column, window, normalized_window):
#     """
#     计算指定窗口大小的移动平均值并进行标准化处理。
#
#     :param data: 包含tick数据的DataFrame
#     :param column: 需要计算均值的列名
#     :param window: 滑动窗口的大小
#     :param normalized_window: 标准化窗口的大小
#     :return: 包含标准化移动平均值的新列
#     """
#     # 计算移动平均值
#     moving_average = data[column].rolling(window=window).mean()
#
#     normalized_moving_average = []
#
#     for i in range(len(data)):
#         if i < normalized_window * 2 - 1:
#             normalized_moving_average.append(None)  # 数据不足以形成一个完整的窗口时
#         else:
#             # 获取当前窗口的移动平均值数据
#             normalized_window_data = moving_average.iloc[i - normalized_window + 1:i + 1]
#             min_val = normalized_window_data.min()
#             max_val = normalized_window_data.max()
#
#             # 对移动平均值进行标准化处理
#             if max_val - min_val != 0:
#                 normalized_value = (moving_average.iloc[i] - min_val) / (max_val - min_val)
#             else:
#                 normalized_value = 0.0  # 如果最大值和最小值相等，标准化值设为0.5
#
#             normalized_moving_average.append(normalized_value)
#
#     # 将标准化移动平均值结果加入DataFrame
#     data[f'Normalized_Moving_Average_{column}_{window}'] = normalized_moving_average
#     return data

window = 40
normalized_window=2000
data = calculate_and_normalize_momentum(data, 'LastPrice_Return', window,normalized_window)

data = calculate_and_normalize_momentum(data, 'Volume', window,normalized_window)

data = calculate_and_normalize_momentum(data, 'Bid_Ask_Volume_Ratio_diff', window,normalized_window)

data = calculate_and_normalize_momentum(data, 'OpenInterest', window,normalized_window)

print(data[['LastPrice_Return', 'Normalized_LastPrice_Return_Momentum', 'Volume', 'Normalized_Volume_Momentum', 'Bid_Ask_Volume_Ratio_diff', 'Normalized_Bid_Ask_Volume_Ratio_diff_Momentum','OpenInterest','Normalized_OpenInterest_Momentum']].tail())




data['Composite_Momentum']=data['Normalized_LastPrice_Return_Momentum']*0+data['Normalized_Volume_Momentum']*0.3+data['Normalized_Bid_Ask_Volume_Ratio_diff_Momentum']*0.3+data['Normalized_OpenInterest_Momentum']*0.4
print(data[['LastPrice_Return', 'Composite_Momentum']].tail())

threshold = data['Composite_Momentum'].quantile(0.999)  # 95%的百分位数
data['Momentum_Signal'] = (data['Composite_Momentum'] > threshold).astype(int)

plt.figure(figsize=(12, 8))



# plt.plot(data.index, data['MA300'], label='MA300', color='orange')

# # 标注Momentum_Signal为1的点
# momentum_signal_points = data[data['Momentum_Signal'] == 1]
# plt.scatter(momentum_signal_points.index, momentum_signal_points['LastPrice'], color='blue', label='Momentum Signal', marker='o')


# 创建图形和轴
fig, ax1 = plt.subplots(figsize=(12, 8))

# 绘制LastPrice
ax1.plot(data.index, data['LastPrice'], label='LastPrice', color='grey')
ax1.set_xlabel('Time')
ax1.set_ylabel('LastPrice', color='grey')
ax1.tick_params(axis='y', labelcolor='grey')
ax1.tick_params(axis='x', rotation=35)
ax1.xaxis.set_major_locator(MaxNLocator(nbins=20))

# 创建共享x轴的第二个y轴
ax2 = ax1.twinx()
ax2.bar(data.index, data['Composite_Momentum'], color='blue', alpha=0.3, label='Composite Momentum')
ax2.set_ylabel('Composite Momentum', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.set_ylim(data['Composite_Momentum'].min(), data['Composite_Momentum'].max()*1.1)
# 添加图例
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))

# 显示图形
plt.show()



# # 1. 价格动量
# filtered_data_main['Price_Momentum'] = filtered_data_main['LastPrice'].diff()
#
# # 2. 成交量动量
# filtered_data_main['Volume_Momentum'] = filtered_data_main['Volume'].diff()
#
# # 3. 盘口力量对比
# filtered_data_main['Total_BidVolume'] = filtered_data_main[['BidVolume2', 'BidVolume3', 'BidVolume4', 'BidVolume5']].sum(axis=1)
# filtered_data_main['Total_AskVolume'] = filtered_data_main[['AskVolume2', 'AskVolume3', 'AskVolume4', 'AskVolume5']].sum(axis=1)
# filtered_data_main['Bid_Ask_Volume_Ratio'] = filtered_data_main['Total_BidVolume'] / filtered_data_main['Total_AskVolume']
#
# # 4. 未平仓合约量变化
# filtered_data_main['OpenInterest_Change'] = filtered_data_main['OpenInterest'].diff()
#
# # 5. 成交额动量
# filtered_data_main['Turnover_Momentum'] = filtered_data_main['Turnover'].diff()
#
# # 6. 综合动量指标
# filtered_data_main['Composite_Momentum'] = (
#     filtered_data_main['Price_Momentum'].abs() * 0.4 +
#     filtered_data_main['Volume_Momentum'].abs() * 0.005 +
#     filtered_data_main['Bid_Ask_Volume_Ratio'].diff().abs() +
#     filtered_data_main['OpenInterest_Change'].abs() * 0.03
# )



# # 7. 设置动量信号阈值
# threshold = filtered_data_main['Composite_Momentum'].std()
# filtered_data_main['Momentum_Signal'] = (filtered_data_main['Composite_Momentum'] > threshold).astype(int)


# threshold = filtered_data_main['Composite_Momentum'].quantile(0.99)  # 95%的百分位数
# filtered_data_main['Momentum_Signal'] = (filtered_data_main['Composite_Momentum'] > threshold).astype(int)


# # 计算过去一段时间（如100个tick）的最大值作为阈值
# window = 200
# filtered_data_main['Max_Momentum'] = filtered_data_main['Composite_Momentum'].rolling(window=window).max()
# filtered_data_main['Momentum_Signal'] = (filtered_data_main['Composite_Momentum'] > filtered_data_main['Max_Momentum']).astype(int)


# # 计算布林带
# window = 100
# filtered_data_main['BB_Middle'] = filtered_data_main['Composite_Momentum'].rolling(window=window).mean()
# filtered_data_main['BB_Upper'] = filtered_data_main['BB_Middle'] + 2 * filtered_data_main['Composite_Momentum'].rolling(window=window).std()
#
# # 动量信号
# filtered_data_main['Momentum_Signal'] = (filtered_data_main['Composite_Momentum'] > filtered_data_main['BB_Upper']).astype(int)
#
#
# # 查看结果
# print(filtered_data_main[['Price_Momentum', 'Volume_Momentum', 'Bid_Ask_Volume_Ratio', 'OpenInterest_Change', 'Composite_Momentum', 'Momentum_Signal']].head(100))
#





# plt.figure(figsize=(12, 6))
#
# # 绘制LastPrice和MA300
# plt.plot(filtered_data_main.index, filtered_data_main['LastPrice'], label='LastPrice', color='grey')
# plt.plot(filtered_data_main.index, filtered_data_main['MA300'], label='MA300', color='orange')
#
# # 标注Momentum_Signal为1的点
# momentum_signal_points = filtered_data_main[filtered_data_main['Momentum_Signal'] == 1]
# plt.scatter(momentum_signal_points.index, momentum_signal_points['LastPrice'], color='blue', label='Momentum Signal', marker='o')
#
# plt.legend()
# plt.show()





# # 绘图
# plt.figure(figsize=(12, 6))
#
#
# plt.plot(filtered_data_main.index, filtered_data_main['LastPrice'], label='LastPrice',color='grey')
# plt.plot(filtered_data_main.index, filtered_data_main['MA300'], label='MA300', color='orange')
#
#
# plt.scatter(filtered_data_main[filtered_data_main['Turning_Point_Pos_to_Neg']].index,
#             filtered_data_main['MA300'][filtered_data_main['Turning_Point_Pos_to_Neg']],
#             color='red', label='Turning Point Pos to Neg', marker='v')
# plt.scatter(filtered_data_main[filtered_data_main['Turning_Point_Neg_to_Pos']].index,
#             filtered_data_main['MA300'][filtered_data_main['Turning_Point_Neg_to_Pos']],
#             color='green', label='Turning Point Neg to Pos', marker='^')
# plt.legend()
# plt.show()







