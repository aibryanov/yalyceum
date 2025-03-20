import numpy as np


def R_score(y_true, y_pred):
    y_mean = np.mean(y_true)

    r_score = 1 - (sum((y_true - y_pred) ** 2) / sum((y_mean - y_true) ** 2))

    return r_score


if __name__ == "__main__":
    y_true = np.fromiter(map(float, input().split()), dtype=float)
    y_pred = np.fromiter(map(float, input().split()), dtype=float)

    print("R2: {:.2f}".format(R_score(y_true, y_pred)))
