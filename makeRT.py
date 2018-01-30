import numpy as np
import matplotlib.pyplot as plt

import random
from deap import base
from deap import creator
from deap import tools


def evalOneMax(individual):
    arrayind = np.array(individual)
    hitpoint = 0
    for i in range(40*30):
        hitpoint = hitpoint+(1-abs(img_ans[i]-arrayind[i]))
    return hitpoint,


def doGT(whichcx):
    toolbox.register("mate", whichcx)
    random.seed(64)
    pop = toolbox.population(n=300)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 200

    print("Start of evolution")

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind,  fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print(" Evaluate %i individuals" % len(pop))
    avgs = np.zeros(NGEN)
    for g in range(NGEN):
        print("--Generation %i --" % g)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invaild_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invaild_ind)
        for ind,  fit in zip(invaild_ind,  fitnesses):
            ind.fitness.values = fit

        print(" Evaluated % i individuals" % len(invaild_ind))

        pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits)/length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2/length-mean**2)**0.5

        print(" Min %s" % min(fits))
        print(" Max %s" % max(fits))
        print(" Avg %s" % mean)
        print(" std %s" % std)
        avgs[g] = mean

    print("-- End of (successful) evolution --")
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s %s" % (best_ind,  best_ind.fitness.values))

    best_ind_img = np.array(best_ind)
    best_ind_img = best_ind_img.reshape((40, 30))
    plt.plot(avgs)


def main():
    cxs = [tools.cxTwoPoint, tools.cxOnePoint]
    [doGT(cx) for cx in cxs]
    plt.show()


if __name__  == "__main__":
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    img_ans = plt.imread('TAsample2.png')
    img_ans = img_ans[..., 1]
    img_ans = img_ans.reshape(40*30, )
    img_ans = img_ans*255

    for i in range(40*30):
        if img_ans[i] < 128:
            img_ans[i] = 0
        else:
            img_ans[i] = 1

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual", tools.initRepeat, creator.Individual,
        toolbox.attr_bool, 40*30
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    main()
