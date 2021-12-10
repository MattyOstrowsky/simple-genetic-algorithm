
import numpy as np
from sga import GSA, fitness
from matplotlib import pyplot as plt


if '__main__' == __name__:

    ga = GSA(interval=[0.5, 2.5], fitness_function=fitness,
             population_size=50, n_generations=100, mutation_probability=0.5)

    x = np.linspace(start=0.5, stop=2.5, num=200)

    plt.plot(x, fitness(x))
    plt.scatter([col[1] for col in ga.history], [col[2]
                for col in ga.history], marker='.', c='b', s=250)
    plt.scatter(ga.history[-1][1], ga.history[-1][2], marker='.', c='r', s=300)
    plt.show()
