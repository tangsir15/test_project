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

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

def read_data(start_date, end_date):
    '''
    读取日期范围内的所有文件，如果没有该日期则跳过。

    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: 合并后的数据
    '''
    date_range = pd.date_range(start=start_date, end=end_date)
    data_list = []

    for single_date in date_range:
        file_path = f'marketdata{single_date.strftime("%Y%m%d")}.csv'
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            data_list.append(data)
        else:
            print(f"File {file_path} does not exist, skipping.")

    if data_list:
        combined_data = pd.concat(data_list)
        return combined_data
    else:
        return pd.DataFrame()  # 返回空的 DataFrame

def data_preprocess(data, ins):
    '''
    数据处理

    :param data: 输入数据
    :param ins:需要返回的合约代码
    :return: 需要的合约LastPrice和Timestamp
    '''
    column_lists = data.columns
    data.index.names = column_lists[:2]
    data.index = data.index.set_levels([data.index.levels[0].astype(str), data.index.levels[1].str.strip()])
    data = data.iloc[:, :-2]
    data.columns = column_lists[2:]
    data['TradingDay'] = data.index.get_level_values('TradingDay')
    data = data.droplevel('TradingDay')
    # data= data[['LastPrice', 'TradingDay','UpdateTime', 'UpdateMillisec']]
    data = data.loc[ins]
    data['Time'] = data['TradingDay'].astype(str) + '.' + data[
        'UpdateTime'].astype(str).str.strip() + '.' + data['UpdateMillisec'].astype(str)

    data.reset_index(drop=True, inplace=True)
    time_format = '%Y%m%d.%H:%M:%S.%f'
    data['Timestamp'] = pd.to_datetime(data['Time'], format=time_format)
    data = data[['LastPrice', 'Timestamp']]
    return data


# 根据标准差动态调整 base_multiplier
def dynamic_multiplier(std, base_multiplier, std_threshold,dynamic_multiple=5):
    if std < std_threshold:  # 标准差较小，增大倍数
        return base_multiplier + (std_threshold - std) / std * dynamic_multiple
    else:  # 标准差较大，减小倍数
        return base_multiplier - (std - std_threshold) / std * 2.0


def Bolllinger(data, bollinger_period, base_multiplier, std_threshold_para=0.7,quit_dev_multiplier1=1,dynamic_multiple=5):
    '''

    :param data:
    :param bollinger_period:
    :param base_multiplier: 基础倍数
    :param std_threshold_para: 标准差阈值参数
    :param quit_dev_multiplier1: 退出倍数参数
    :param dynamic_multiple: 动态调整参数
    :return:
    '''
    # 计算移动平均线和标准差
    data['Middle Band'] = data['LastPrice'].rolling(window=bollinger_period).mean()
    data['Std Dev'] = data['LastPrice'].rolling(window=bollinger_period).std()

    std_threshold = data['Std Dev'].quantile(std_threshold_para)  # 标准差的阈值
    print(std_threshold)
    # 应用动态倍数
    data['std_dev_multiplier'] = data['Std Dev'].apply(lambda x: dynamic_multiplier(x, base_multiplier, std_threshold,dynamic_multiple))

    # 计算动态调整后的布林带
    data['Upper Band'] = data['Middle Band'] + (data['Std Dev'] * data['std_dev_multiplier'])
    data['Lower Band'] = data['Middle Band'] - (data['Std Dev'] * data['std_dev_multiplier'])

    # # 找出超出布林线的点
    data['upper_band_exceed'] = data.apply(
        lambda row: row['LastPrice'] if row['LastPrice'] > row['Upper Band'] else None, axis=1)
    data['lower_band_exceed'] = data.apply(
        lambda row: row['LastPrice'] if row['LastPrice'] < row['Lower Band'] else None, axis=1)


    # 初始化一个标志来跟踪条件
    flag = False
    # 遍历 DataFrame
    for i in range(len(data)):
        if pd.notna(data.at[i, 'upper_band_exceed']):
            flag = True
        elif flag and data.at[i, 'LastPrice'] > data.at[i, 'Middle Band'] - data.at[i, 'Std Dev'] * quit_dev_multiplier1:
            data.at[i, 'upper_band_exceed'] = data.at[i, 'LastPrice']
        elif flag and data.at[i, 'LastPrice'] <= data.at[i, 'Middle Band'] - data.at[
            i, 'Std Dev'] * quit_dev_multiplier1:
            flag = False

    # 初始化一个标志来跟踪条件
    flag = False
    # 遍历 DataFrame
    for i in range(len(data)):
        if pd.notna(data.at[i, 'lower_band_exceed']):
            flag = True
        elif flag and data.at[i, 'LastPrice'] < data.at[i, 'Middle Band'] + data.at[i, 'Std Dev'] * quit_dev_multiplier1:
            data.at[i, 'lower_band_exceed'] = data.at[i, 'LastPrice']
        elif flag and data.at[i, 'LastPrice'] >= data.at[i, 'Middle Band'] + data.at[
            i, 'Std Dev'] * quit_dev_multiplier1:
            flag = False
    return data



def plot_bollinger(data):
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
    plt.title(f'{start_date}-{end_date}:{contract_code}')
    plt.legend()

    plt.show()

def plot_bollinger_for_contract(contract_code,bollinger_params):
    if contract_code in bollinger_params:
        params = bollinger_params[contract_code]
        data = read_data(start_date, end_date)
        data = data_preprocess(data, contract_code)
        data = Bolllinger(data, *params)
        plot_bollinger(data)
    else:
        print(f"Contract code {contract_code} not found in parameters dictionary.")


if __name__ == '__main__':
    start_date = '2024-08-01'
    end_date = '2024-08-02'
    bollinger_params = {
        'm2409': (1000, 3, 0.7, 0.5, 5),  #(1000, 3, 0.7, 0.5, 5)
        'y2409': (1000, 3, 0.5, 1.5, 5),  # (1000, 3, 0.5, 1.5, 5)
        # 'ag2409': (1000, 3, 0.7, 1, 4),
        # 'i2409': (1000, 3, 0.6, 1, 3),
    }

    contract_code = 'm2409'
    plot_bollinger_for_contract(contract_code,bollinger_params)

    # #手动调参
    # ins='y2409'
    # data = read_data(start_date, end_date)
    # data = data_preprocess(data,ins)
    # data = Bolllinger(data, 1000, 3, 0.7, 1,5)
    # plot_bollinger(data)