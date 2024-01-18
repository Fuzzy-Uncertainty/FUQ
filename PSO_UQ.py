import numpy as np
from sklearn.metrics import mean_absolute_error
import pyswarms as ps

#from pyswarms.utils.functions import single_obj as fx
# Set-up hyperparameters


def PSO_UQ(y_pred,y_true,iters=1000):
    n=len(y_true)
    def coverage(a,b):
        cost=[]
        for j in range(len(a)):
            c=0
            for i in range(len(y_true)):
                if y_true[i]<=y_pred[i]-a[j]:
                    c+=0
                elif y_true[i]<=y_pred[i]:
                    c+= 1-(y_pred[i]-y_true[i])/a[j]
                elif y_true[i]<=y_pred[i]+b[j]:
                    c+= 1-(y_true[i]-y_pred[i])/b[j]
                else:
                    c+=0
                    if c<0: 
                        print("this is a warning")
            cost.append(c/n)
        return cost

    def func(solution):
        a=solution[:, 0]
        b=solution[:,1]
        specificty=1/(a+b)*(1-np.exp(-(a+b)))
        return -specificty*coverage(a,b)
    
    e=mean_absolute_error(y_true, y_pred)
    max_bound = e*100 * np.ones(2)
    min_bound = np.zeros(2)
    bounds = (min_bound, max_bound)

    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    # Perform optimization
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)
    cost, pos = optimizer.optimize(func, iters)
    return pos,cost,optimizer

