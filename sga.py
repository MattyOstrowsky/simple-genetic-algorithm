from random import randint
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt


def fitness(values: list) -> list:
    """ function

    Args:
        values (list): x

    Returns:
        list: f(x)
    """
    return [(np.e ** x * np.sin(10.0 * np.pi * x) + 1.0) / (x + 5.0) for x in values]


class GSA:
    def __init__(self, interval: list, fitness_function, population_size: int, n_generations: int, mutation_probability: float):
        """ Genetic algorithm for finding maxima of function.
        Args:
            interval (list): function test interval
            fitness_function ([type]): optimization function
            population_size (int): constant population size
            n_generations (int): number of tested generations after which the algorithm stops
            mutation_probability (float): the likelihood of a mutation occurring in a chromosome
        """
        self.interval = interval
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_probability = mutation_probability
        self.history = []
        # check how many genes need
        sampling = (self.interval[1] - self.interval[0]) * (10 ** 6)
        self.n_genes = self.check_power(sampling)
        # init chromosome
        self.chromosome = self.init_population()
        print('initial chromosomes:')
        print(tabulate(self.chromosome, tablefmt='grid'))
        for _ in range(self.n_generations):
            self.chromosome = self.epoch()
        plt.show()

    def init_population(self):
        initial_chromosome = [[randint(0, 1) for _ in range(
            self.n_genes)] for _ in range(self.population_size)]
        return initial_chromosome

    def epoch(self):
        plt.clf()
        x = np.linspace(start=0.5, stop=2.5, num=200)
        plt.plot(x, self.fitness_function(x))
        # decode binary
        parents = self.decode(self.chromosome, self.interval)
        fit = self.fitness_function(parents)
        plt.scatter(parents, fit, marker='x')
        plt.pause(0.0001)
        # sort for selection
        sorted_parents = self.sort_fittest_parents(
            parents, self.population_size, fit)
        self.history.append(list(sorted_parents[0]))
        # crossover and mutation
        chromosomes = self.crossover(sorted_parents)
        return chromosomes

    @staticmethod
    def check_power(val: float) -> int:
        powered = 2
        pow = 1
        while val > powered:
            pow += 1
            powered = 2 ** pow
        return pow

    @staticmethod
    def decode(population: list, interval: list) -> list:
        result = [
            interval[0] + (int("".join(str(i)
                           for i in individual), 2) / 10 ** 6)
            for individual in population
        ]
        return [0.5 if x > interval[1] or x < interval[0] else x for x in result]

    @staticmethod
    def sort_fittest_parents(chromos_value: list, population_size: int, fit: list) -> list:
        idx = [x for x in range(population_size)]
        parents = list(zip(idx, chromos_value, fit))
        parents.sort(key=lambda x: x[2], reverse=True)
        return parents

    def crossover(self, parents: list) -> list:
        children = []
        for _ in range(int(self.population_size / 2)):
            parent_x, parent_y = np.random.randint(
                0, self.population_size / 2, 2)
            chromosome_p_x = self.chromosome[parents[parent_x][0]]
            chromosome_p_y = self.chromosome[parents[parent_y][0]]
            spliter = np.random.randint(0, self.n_genes)
            children.append(
                self.mutate(
                    chromosome_p_x[:spliter] + chromosome_p_y[spliter:])
            )
            children.append(
                self.mutate(
                    chromosome_p_y[:spliter] + chromosome_p_x[spliter:])
            )
        return children

    def mutate(self, child: list):
        if np.random.choice([False, True], p=[1 - self.mutation_probability, self.mutation_probability]):
            child[np.random.randint(0, self.n_genes)] ^= 1
        return child
