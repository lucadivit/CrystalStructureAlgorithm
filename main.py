import numpy as np
from CryStAl import CryStAl

# This is the BIRD function --> https://www.indusmic.com/post/bird-function
function = lambda x: np.sin(x[0])*(np.exp(1-np.cos(x[1]))**2)+np.cos(x[1])*(np.exp(1-np.sin(x[0]))**2)+(x[0]-x[1])**2

crystal = CryStAl(function_to_optimize=function, problem_dimension=2, approach="min", lower_bound=-2 * np.pi, upper_bound=np.pi)

best_fitness, Cr_b, history = crystal.start_crystals_construction()
print(f"Best Crystal Is {Cr_b} with fitness {best_fitness}")
print(history.tail())