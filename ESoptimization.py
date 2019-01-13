""" Fabio D'Isidoro - ETH Zurich - September 2017

    Tests on self-adaptive evolutionary strategy optimizer.

"""

import numpy as np
from test_functions import generate_3dlandscape, rastring_cost_function, test_function2D, rosenbrock_cost_function, ackley_function,  plot_optimization_result2D
from evolutionary_strategy2_with_plots_sigma_variation_restart_option import EvolutionaryStrategy

# choose cost function from the imported ones
cost_function = ackley_function

# choose optimizer parameters for the chosen cost function
if cost_function == test_function2D:
    mu = 2
    domain = np.array([[-40.0,40.0],[-40.0,40.0]])
    accuracies = [0.01, 0.01]
    max_numb_generations = 100
    with_plots = True
    with_sigma_mutation = True
    self_adaptivity = 1.0

elif cost_function == rastring_cost_function:
    ndim = 6
    mu = 3
    domain = np.array([[-5.12,5.12],]*ndim)
    accuracies = [0.01,]*ndim
    max_numb_generations = 100
    with_plots = True
    with_sigma_mutation = True
    self_adaptivity = 1.0

elif cost_function == rosenbrock_cost_function:
    ndim = 2
    mu = 3
    domain = np.array([[-2.048,2.048],]*ndim)
    accuracies = [0.01,]*ndim
    max_numb_generations = 200
    with_plots = True
    with_sigma_mutation = True
    self_adaptivity = 1.0
    with_restart = True

elif cost_function == ackley_function:
    ndim = 6
    mu = 3
    domain = np.array([[-32.768,32.768],]*ndim)
    accuracies = [0.01,]*ndim
    max_numb_generations = 200
    with_plots = True
    with_sigma_mutation = True
    self_adaptivity = 0.5
    with_restart = True

# plot landscape
dim1 = 1
dim2 = 2
npoints = 50
landscape = generate_3dlandscape(cost_function, domain, dim1, dim2, npoints)

# optimizer
optimizer = EvolutionaryStrategy(cost_function, domain, mu, accuracies, var = 1.0, rho = 2, max_numb_generations = max_numb_generations, 
                                 with_plots = with_plots, with_sigma_mutation = with_sigma_mutation, self_adaptivity = self_adaptivity, with_restart = with_restart)    

# run
optimizer.run()

# print
print(" \n Found optimal parameters are ")
print(optimizer.solution.par)
print(" Best fitness found is ")
print(optimizer.solution.fit)
print(" The best fitness was found at generation ")
print(optimizer.solution_generation)

# plot results
plot_optimization_result2D(landscape, domain, optimizer.solution.par, dim1, dim2)