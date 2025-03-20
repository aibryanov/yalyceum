import numpy as np


def MSE(y_true, y_pred):
    n = len(y_true)
    mse = sum((y_pred - y_true) ** 2) / n
    return mse


def MAE(y_true, y_pred):
    n = len(y_true)
    mae = sum(np.abs(y_pred - y_true)) / n
    return mae


def RMSE(y_true, y_pred):
    n = len(y_true)
    rmse = np.sqrt(sum((y_pred - y_true) ** 2) / n)
    return rmse


if __name__ == "__main__":
    y_true = np.fromiter(map(float, input().split()), dtype=float)
    y_pred = np.fromiter(map(float, input().split()), dtype=float)

    print("MSE: {:.2f}".format(MSE(y_true, y_pred)))
    print("MAE: {:.2f}".format(MAE(y_true, y_pred)))
    print("RMSE: {:.2f}".format(RMSE(y_true, y_pred)))
