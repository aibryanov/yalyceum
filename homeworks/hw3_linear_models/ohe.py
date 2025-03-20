import numpy as np


def onehot_encoding(x: np.ndarray) -> np.ndarray:
    sorted_types = sorted(list(set(x)))
    hot = []
    
    for i in x:
        row = []
        for j in sorted_types:
            if j == i:
                row.append(1)
            else:
                row.append(0)
        hot.append(row)
        
    return np.array(hot)


