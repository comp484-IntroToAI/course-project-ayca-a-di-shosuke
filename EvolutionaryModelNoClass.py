import random
import pickle
import heapq
import operator
import statistics as stats
from deap import creator, base, tools, algorithms
from Rouge import Rouge


def read_list(file_name):
    with open(file_name, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list


def init_individual(icls, content):
    return icls(content)


def init_population(pcls, ind_init):
    contents = read_list("TrainingSentenceWeights")
    return pcls(ind_init(c) for c in contents)


def evaluate_pop():
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (fit,)


def evaluate(individual):
    return stats.mean(individual)


def select_next_gen():
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

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
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = (fit,)


def get_best_inds(limit=6):
    lst = [ind.fitness.values[0] for ind in pop]
    print(lst)
    # lst = [153,124,1243,2345,4654,4654,4654,234,46,345]
    lst_i = [(lst[i], i) for i in range(len(lst))]
    print(lst_i)
    
    best = heapq.nlargest(limit, lst_i)
    inds = sorted([best[i][1] for i in range(len(best))])

    print(best)
    print(inds)

    sentences = read_list("trainingSentences")
    summary = [sentences[i].capitalize() for i in inds]
    print(' '.join(summary))


def run_algorithm():
    for g in range(NGEN):
        select_next_gen()

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
CXPB, MUTPB, NGEN = 0.5, 0.2, 15



# Evaluate the entire population
evaluate_pop()
run_algorithm()

get_best_inds()

