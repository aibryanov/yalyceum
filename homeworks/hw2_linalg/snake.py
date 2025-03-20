import numpy as np


def snake(m: int, n: int) -> np.ndarray:
    matr = np.zeros((m, n), dtype=int)
    count = 1
    for i in range(m):
        for j in range(n):
            matr[i, j] = count
            count += 1
        if i % 2 == 1:
            matr[i, :] = matr[i, ::-1]   
    return matr