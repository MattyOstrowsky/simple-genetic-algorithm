
import numpy as np
from sga import GSA
from matplotlib import pyplot as plt
from tabulate import tabulate


def fitness(values: list) -> list:
    """ function

    Args:
        values (list): x

    Returns:
        list: f(x)
    """
    return [(np.e ** x * np.sin(10.0 * np.pi * x) + 1.0) / x + 5.0 for x in values]


if '__main__' == __name__:

    ga = GSA(interval=[0.5, 2.5], fitness_function=fitness,
             population_size=50, n_generations=10, mutation_probability=0.9)

    x = np.linspace(start=0.5, stop=2.5, num=200)

    plt.plot(x, fitness(x))
    plt.scatter([col[1] for col in ga.history], [col[2]
                for col in ga.history], marker='.', c='b', s=250, label='Winners of each generation')
    plt.scatter(ga.history[-1][1], ga.history[-1][2], marker='.',
                c='r', s=300, label='Winner')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    print(tabulate(ga.history, headers=[
          'n_chromosome', 'encoded_value', 'fitness'], showindex=True, tablefmt='grid'))
