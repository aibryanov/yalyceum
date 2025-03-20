import numpy as np
import pandas as pd


def norma2(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.linalg.norm(vec1 - vec2)


def k_nearest(data: np.ndarray, n: int, k: int) -> np.ndarray:
    target_row = data[n]
    distances = np.linalg.norm(data - target_row, axis=1)
    distances[n] = np.inf
    nearest_inds = np.argsort(distances)[:k]
    nearest_neighs = data[nearest_inds]
    
    return nearest_neighs


if __name__ == "__main__":
    df = pd.read_csv("penguins.csv")
    df.dropna(inplace=True)
    data = df[["bill_length_mm", "bill_depth_mm"]].to_numpy()
    n = int(input())
    k = int(input())

    result = k_nearest(data, n, k)

    for i in range(k):
        print(result[i])