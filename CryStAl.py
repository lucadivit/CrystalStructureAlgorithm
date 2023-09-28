import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class CryStAl:

    def __init__(self, function_to_optimize: object, problem_dimension: int, approach: str, lower_bound: float = -1e4,
                 upper_bound: float = 1e4, num_crystals: int = 50, num_iterations: int = 20):
        self.__DIM = problem_dimension
        self.__APPROACH = approach.lower()
        self.__LB = lower_bound
        self.__UB = upper_bound
        self.__NUM_CRYSTALS = num_crystals
        self.__NUM_ITERATIONS = num_iterations
        self.__FUNCTION = function_to_optimize
        if self.__APPROACH not in ["min", "max"]:
            raise Exception("'approach' must be 'min' or 'max'")

    def __create_crystals(self) -> np.array:
        crystals = np.random.uniform(low=self.__LB, high=self.__UB, size=(self.__NUM_CRYSTALS, self.__DIM))
        return crystals

    def __compute_fitnesses(self, crystals: np.array, eval_function: object) -> (np.array, float, int):
        fitnesses = np.apply_along_axis(eval_function, 1, crystals)
        if self.__APPROACH == "min":
            best_fitness_index = np.argmin(fitnesses)
        elif self.__APPROACH == "max":
            best_fitness_index = np.argmax(fitnesses)
        best_fitness_value = fitnesses[best_fitness_index]
        fitnesses = fitnesses.reshape(-1, 1)
        return fitnesses, best_fitness_value, best_fitness_index

    def __compute_r_values(self, n: int = 2) -> (float, float, float):
        rnd_values = n * np.random.rand(4)
        rnd_values = np.round_(rnd_values, decimals=5)
        r = round(rnd_values[0] - 1, 5)
        r_1 = round(rnd_values[1] - 1, 5)
        r_2 = round(rnd_values[2] - 1, 5)
        r_3 = round(rnd_values[3] - 1, 5)
        return r, r_1, r_2, r_3

    def __take_random_crystals(self, crystals: np.array, nb_random_crystals_to_take: int = 0) -> np.array:
        if nb_random_crystals_to_take <= 0:
            nb_random_crystals_to_take = np.random.randint(low=1, high=self.__NUM_CRYSTALS + 1)
        indexes_of_random_crystal = np.random.choice(range(0, self.__NUM_CRYSTALS), nb_random_crystals_to_take, replace=False)
        return crystals[indexes_of_random_crystal]

    def __compute_simple_cubicle(self, Cr_old, Cr_main, r) -> np.array:
        Cr_new = Cr_old + r * Cr_main
        return Cr_new

    def __compute_cubicle_with_best_crystals(self, Cr_old, Cr_main, Cr_b, r_1, r_2) -> np.array:
        Cr_new = Cr_old + r_1 * Cr_main + r_2 * Cr_b
        return Cr_new

    def __compute_cubicle_with_mean_crystals(self, Cr_old, Cr_main, Fc, r_1, r_2) -> np.array:
        Cr_new = Cr_old + r_1 * Cr_main + r_2 * Fc
        return Cr_new

    def __compute_cubicle_with_best_and_mean_crystals(self, Cr_old, Cr_main, Cr_b, Fc, r_1, r_2, r_3) -> np.array:
        Cr_new = Cr_old + r_1 * Cr_main + r_2 * Cr_b + r_3 * Fc
        return Cr_new

    def __is_new_fitness_better(self, old_crystal_fitness, new_crystal_fitness) -> bool:
        if self.__APPROACH == "min":
            if new_crystal_fitness < old_crystal_fitness:
                res = True
            else:
                res = False
        elif self.__APPROACH == "max":
            if new_crystal_fitness > old_crystal_fitness:
                res = True
            else:
                res = False
        return res

    def start_crystals_construction(self, save_history: bool = False, verbose: bool = False, task_name: str = "crystal") -> (float, np.array, pd.DataFrame):
        crystals = self.__create_crystals()
        fitnesses, best_fitness, best_index = self.__compute_fitnesses(crystals=crystals, eval_function=self.__FUNCTION)
        Cr_b = crystals[best_index]
        historical_loss = [best_fitness]
        historical_crystal = [list(Cr_b)]
        if verbose:
            print("-------------------")
            print(f"Current Best Crystal Is {Cr_b} With Fitness {best_fitness}")
        for _ in range(0, self.__NUM_ITERATIONS):
            for crystal_idx in range(0, self.__NUM_CRYSTALS):
                new_crystals = np.array([])
                Cr_main = self.__take_random_crystals(crystals=crystals, nb_random_crystals_to_take=1).flatten()
                Cr_old = crystals[crystal_idx]
                Fc = self.__take_random_crystals(crystals=crystals).mean(axis=0)
                r, r_1, r_2, r_3 = self.__compute_r_values()
                Cr_new = self.__compute_simple_cubicle(Cr_old=Cr_old, Cr_main=Cr_main, r=r)
                new_crystals = np.hstack((new_crystals, Cr_new))
                Cr_new = self.__compute_cubicle_with_best_crystals(Cr_old=Cr_old, Cr_main=Cr_main, Cr_b=Cr_b, r_1=r_1, r_2=r_2)
                new_crystals = np.vstack((new_crystals, Cr_new))
                Cr_new = self.__compute_cubicle_with_mean_crystals(Cr_old=Cr_old, Cr_main=Cr_main, Fc=Fc, r_1=r_1, r_2=r_2)
                new_crystals = np.vstack((new_crystals, Cr_new))
                Cr_new = self.__compute_cubicle_with_best_and_mean_crystals(Cr_old=Cr_old, Cr_main=Cr_main, Cr_b=Cr_b, Fc=Fc, r_1=r_1, r_2=r_2, r_3=r_3)
                new_crystals = np.vstack((new_crystals, Cr_new))
                new_crystals = np.clip(new_crystals, a_min=self.__LB, a_max=self.__UB)
                new_crystal_fitnesses, new_crystal_best_fitness, new_crystal_best_index = self.__compute_fitnesses(crystals=new_crystals, eval_function=self.__FUNCTION)
                current_crystal_fitness = fitnesses[crystal_idx][0]
                if self.__is_new_fitness_better(old_crystal_fitness=current_crystal_fitness, new_crystal_fitness=new_crystal_best_fitness):
                    crystals[crystal_idx] = new_crystals[new_crystal_best_index]
            fitnesses, best_fitness, best_index = self.__compute_fitnesses(crystals=crystals, eval_function=self.__FUNCTION)
            Cr_b = crystals[best_index]
            historical_loss.append(best_fitness)
            historical_crystal.append(list(Cr_b))
            if verbose:
                print("-------------------")
                print(f"Current Best Crystal Is {Cr_b} With Fitness {best_fitness}")
        hist_len = len(historical_loss)
        df = pd.DataFrame({"Function Value": historical_loss, "Iteration": list(range(0, hist_len)), "Best Crystal": historical_crystal})
        if save_history:
            sns.lineplot(x="Iteration", y="Function Value", data=df).set_title("Function Value Over Iterations")
            plt.savefig(f"{task_name}.png")
        return best_fitness, Cr_b, df
