import numpy as np
from sklearn.metrics import mean_absolute_error
import pygad



def GA_UQ(yp,y):

    def coverage(a,b):
        c=0
        n=len(y)
        for i in range(len(y)):
            ypn=float(yp[i])
            if y[i]<=ypn-a:
                c+=0
            elif y[i]<=ypn:
                c+= 1-(ypn-y[i])/a
            elif y[i]<=ypn+b:
                c+= 1-(y[i]-ypn)/b
            else:
                c+=0
        return c/n

    def fitness(ga_instance,solution, solution_idx):
        a=solution[0]
        b=solution[1]
        specificty=1/(a+b)*(1-np.exp(-(a+b)))
        return specificty*coverage(a,b)


    fitness_function = fitness

    num_generations = 200
    num_parents_mating = 12

    sol_per_pop = 16
    num_genes = 2

    init_range_low = 0
    init_range_high = 100

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = 10
    gene_space = [{'low': 0, 'high': 20}, {'low': 0, 'high': 20}]

    ga_instance = pygad.GA(num_generations=num_generations,
                        save_solutions=True,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        init_range_low=init_range_low,
                        init_range_high=init_range_high,
                        parent_selection_type=parent_selection_type,
                        keep_parents=keep_parents,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type,
                        mutation_percent_genes=mutation_percent_genes,
                        gene_space=gene_space)
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    return solution,solution_fitness, ga_instance