import numpy as np


def calculate(produdcts: np.ndarray, cook: list):
    result = np.matmul(cook, produdcts)
    print("Молоко, литры:", int(np.ceil(result[0])))
    print("Яйца, штуки:", int(np.ceil(result[1])))
    print("Мука, кг:", int(np.ceil(result[2])))
    