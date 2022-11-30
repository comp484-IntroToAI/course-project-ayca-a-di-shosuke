import random
import pickle
import statistics as stats
from deap import creator, base, tools, algorithms
from Rouge import Rouge


def evaluate(individual):
    return stats.mean(individual)


def evaluate_pop():
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit


def select_next_gen():
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = map(toolbox.clone, offspring)

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        crossover(child1, child2)
    for mutant in offspring:
        mutation(mutant)
    invalid_fitness(offspring)
    pop[:] = offspring


def crossover(child1, child2):
    if random.random() < CXPB:
        toolbox.mate(child1, child2)
        del child1.fitness.values
        del child2.fitness.values


def mutation(mutant):
    if random.random() < MUTPB:
        toolbox.mutate(mutant)
        del mutant.fitness.values


def invalid_fitness(offspring):
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit


def run_algorithm():
    for g in range(NGEN):
        select_next_gen()


def read_list(file_name):
    with open(file_name, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list


def init_individual(icls, content):
    return icls(content)


def init_population(pcls, ind_init):
    contents = read_list("TrainingSentenceWeights")

    return pcls(ind_init(c) for c in contents)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("individual", init_individual, creator.Individual)
toolbox.register("population", init_population, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

pop = toolbox.population()
CXPB, MUTPB, NGEN = 0.5, 0.2, 1

# Evaluate the entire population
fitnesses = map(toolbox.evaluate, pop)
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

for g in range(NGEN):
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = map(toolbox.clone, offspring)

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # The population is entirely replaced by the offspring
    pop[:] = offspring
    
# print(pop[:10])

