import numpy as np


def grad(func, x):
    delta_x = 10**(-9)
    der = (func(x + delta_x) - func(x)) / delta_x
    return der


def gradient_descent(func, start_point, gamma, epsilon, steps):
    history = [[round(start_point, 3)]]
    curr_point = start_point
    
    if steps != 0: 
        for _ in range(steps):
            next_point = curr_point - gamma * grad(func, curr_point)
            history.append([round(next_point, 3)])
            curr_point = next_point
    else:  # Условие остановки по epsilon
        next_point = curr_point - gamma * grad(func, curr_point)
        while abs(func(next_point) - func(curr_point)) >= epsilon:
            history.append([round(next_point, 3)])
            curr_point = next_point
            next_point = curr_point - gamma * grad(func, curr_point)
        history.append([round(next_point, 3)])
    
    return np.array(history)
