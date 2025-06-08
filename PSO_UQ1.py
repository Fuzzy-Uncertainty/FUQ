import numpy as np
                          
import pyswarms as ps

def PSO_UQ(y_pred, y_true, iters=200):
    n = len(y_true)

    def coverage_components(a, b):
        c1_list = []
        c2_list = []
        for j in range(len(a)):
            c1 = 0
            c2 = 0
            for i in range(n):
                if y_pred[i] - a[j] < y_true[i] <= y_pred[i]:
                    c1 += 1 - (y_pred[i] - y_true[i]) / a[j]
                elif y_true[i] <= y_pred[i] - a[j]:
                    c1 += 0
                if y_pred[i] < y_true[i] <= y_pred[i] + b[j]:
                    c2 += 1 - (y_true[i] - y_pred[i]) / b[j]
                elif y_true[i] > y_pred[i] + b[j]:
                    c2 += 0
            c1_list.append(c1 / n)
            c2_list.append(c2 / n)
        return np.array(c1_list), np.array(c2_list)

    def specificity(a, b):
        return 1 / (a + b) * (1 - np.exp(-(a + b)))

    def func(solution):
        a = solution[:, 0]
        b = solution[:, 1]
        spec = specificity(a, b)
        cov_l, cov_r = coverage_components(a, b)
        return -spec * cov_l * cov_r  # negative for minimization

    # --- PSO setup ---
    max_bound = 20 * np.ones(2)
    min_bound = 0.5 * np.ones(2)  # avoid collapse to 0
    bounds = (min_bound, max_bound)

    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2,
                                        options=options, bounds=bounds)
    cost, pos = optimizer.optimize(func, 200)
    cost_history = optimizer.cost_history
    fitness = -np.array(cost_history)  # Flip sign to show actual fitness
    return pos, cost, optimizer, fitness
