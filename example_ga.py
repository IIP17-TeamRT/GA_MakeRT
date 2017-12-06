import random
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

"""
this script is copied from
http://darden.hatenablog.com/entry/2017/04/18/225459#遺伝子を作成する関数
"""


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

n_gene = 100
min_ind = numpy.ones(n_gene) * -1.0
max_ind = numpy.ones(n_gene) * 1.0


def create_ind_uniform(min_ind, max_ind):
    ind = []
    for min, max in zip(min_ind, max_ind):
        ind.append(random.uniform(min, max))
    return ind


toolbox.register("create_ind", create_ind_uniform, min_ind, max_ind)
toolbox.register(
    "individual", tools.initIterate, creator.Individual, toolbox.create_ind
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalOneMax(individual):
    return sum(individual),


def cxTwoPointCopy(ind1, ind2):
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = (
        ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
    )

    return ind1, ind2


def mutUniformDbl(individual, min_ind, max_ind, indpb):
    size = len(individual)
    for i, min, max in zip(range(size), min_ind, max_ind):
        if random.random() < indpb:
            individual[i] = random.uniform(min, max)
    return individual,


toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register(
    "mutate", mutUniformDbl, min_ind=min_ind, max_ind=max_ind, indpb=0.05
)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    random.seed(64)

    pop = toolbox.population(n=300)

    hof = tools.HallOfFame(1, similar=numpy.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    algorithms.eaSimple(
        pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=1000,
        stats=stats, halloffame=hof
    )

    return pop, stats, hof


if __name__ == "__main__":
    main()
