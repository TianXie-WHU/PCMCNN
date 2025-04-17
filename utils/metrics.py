import torch
import numpy as np
import math

def cal_Rsqure(real_data, preout_data):
    """Calculate the coefficient of determination (R-squared) between real and predicted data.

       Args:
           real_data (torch.Tensor): Ground truth values
           preout_data (torch.Tensor): Predicted values

       Returns:
           float: R-squared value in range [0, 1]. Higher values indicate better fit.

       Formula:
           R² = 1 - (SS_res / SS_tot)
           Where SS_res = sum((y_real - y_pred)^2), SS_tot = sum((y_real - mean(y_real))^2)
       """
    real_data = real_data.flatten()
    preout_data = preout_data.flatten()
    Rsqure = 1-(torch.sum((real_data-preout_data)**2) / torch.sum((real_data-torch.mean(real_data))**2))
    return Rsqure

def cal_MAE(realdata, preoutdata):
    """Calculate Mean Absolute Error (MAE) between real and predicted values.

    Args:
        realdata (torch.Tensor): Ground truth values
        preoutdata (torch.Tensor): Model predictions

    Returns:
        float: MAE value ≥ 0, where lower values indicate better performance
    """
    realdata = realdata.flatten()
    preoutdata = preoutdata.flatten()
    mae = torch.mean(abs(realdata-preoutdata))
    return mae

def cal_RMSE(realdata, preoutdata):
    """Calculate Root Mean Square Error (RMSE) between real and predicted values.

    Args:
        realdata (torch.Tensor): Ground truth measurements
        preoutdata (torch.Tensor): Model outputs

    Returns:
        float: RMSE value ≥ 0, sensitive to large errors
    """
    realdata = realdata.flatten()
    preoutdata = preoutdata.flatten()
    mse = torch.mean((realdata - preoutdata)**2)  # Calculate the mean square error
    rmse = torch.sqrt(mse)  # Calculate the root mean square error
    return rmse

def calcMean(x, y):
    """Calculate mean values for two paired data vectors.

    Args:
        x (list): First data vector
        y (list): Second data vector (same length as x)
    """
    x_sum = sum(x)
    y_sum = sum(y)
    n = len(x)
    x_mean = float(x_sum) / n
    y_mean = float(y_sum) / n
    return x_mean, y_mean  # Returns the mean value


def calcPearson(x, y):
    """Compute Pearson product-moment correlation coefficient (PPMCC).

   Args:
       x (list): First variable measurements
       y (list): Second variable measurements (same length as x)

   Returns:
       float: Pearson correlation coefficient in [-1, 1]

   Formula:
       r = Σ[(x_i - x̄)(y_i - ȳ)] / √[Σ(x_i - x̄)^2 * Σ(y_i - ȳ)^2]
   """
    x_mean, y_mean = calcMean(x, y)  # Calling the above function returns the mean value
    n = len(x)
    sumTop = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i] - x_mean) * (y[i] - y_mean)
    for i in range(n):
        x_pow += math.pow(x[i] - x_mean, 2)
    for i in range(n):
        y_pow += math.pow(y[i] - y_mean, 2)
    sumBottom = np.sqrt(x_pow * y_pow)
    p = sumTop / sumBottom
    return p