import torch
import numpy as np
import math

def cal_Rsqure(real_data, preout_data):
    # real_data:真实数据
    # preout_data:预测数据
    # 计算R方
    real_data = real_data.flatten()
    preout_data = preout_data.flatten()
    Rsqure = 1-(torch.sum((real_data-preout_data)**2) / torch.sum((real_data-torch.mean(real_data))**2))
    return Rsqure

def cal_MAE(realdata, preoutdata):
    realdata = realdata.flatten()
    preoutdata = preoutdata.flatten()
    mae = torch.mean(abs(realdata-preoutdata))
    return mae

def cal_RMSE(realdata, preoutdata):
    realdata = realdata.flatten()
    preoutdata = preoutdata.flatten()
    mse = torch.mean((realdata - preoutdata)**2)  # 计算均方误差
    rmse = torch.sqrt(mse)  # 计算均方根误差
    return rmse

def calcMean(x, y):
    """
    作用：计算特征和类的平均值
    参数：
        x：类的数据
        y：特征的数据
    返回值：特征和类的平均值
    """
    x_sum = sum(x)
    y_sum = sum(y)
    n = len(x)
    x_mean = float(x_sum) / n
    y_mean = float(y_sum) / n
    return x_mean, y_mean  # 返回均值


def calcPearson(x, y):
    x_mean, y_mean = calcMean(x, y)  # 调用上面的函数返回均值
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0

    # 计算协方差
    for i in range(n):
        sumTop += (x[i] - x_mean) * (y[i] - y_mean)
        # 计算标准差
    for i in range(n):
        x_pow += math.pow(x[i] - x_mean, 2)
    for i in range(n):
        y_pow += math.pow(y[i] - y_mean, 2)
    sumBottom = np.sqrt(x_pow * y_pow)
    p = sumTop / sumBottom  # 协方差 / 标准差
    return p