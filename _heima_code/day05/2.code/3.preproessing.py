# coding:utf-8

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def minmax_demo():
    """
    归一化演示
    :return:None
    """
    data = pd.read_csv("./data/dating.txt")
    print(data)

    # 1.实例化
    transfer = MinMaxScaler(feature_range=(3,5))

    # 2.进行转换, 调用fit_transform
    ret_data = transfer.fit_transform(data[["milage", "Liters", "Consumtime"]])
    print("归一化之后的数据为:\n",ret_data)


def stand_demo():
    """
    标准化演示
    :return:None
    """
    data = pd.read_csv("./data/dating.txt")
    print(data)

    # 1.实例化
    transfer = StandardScaler()

    # 2.进行转换, 调用fit_transform
    ret_data = transfer.fit_transform(data[["milage", "Liters", "Consumtime"]])
    print("标准化之后的数据为:\n",ret_data)
    print("每一列的方差为:\n", transfer.var_)
    print("每一列的平均值为:\n", transfer.mean_)


# minmax_demo()
stand_demo()
