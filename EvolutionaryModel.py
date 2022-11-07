import random
from deap import creator, base, tools, algorithms

class GeneticAlgorithm():
    def __init__(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_bool, n=100)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.evaluate)
        
        self.pop = self.toolbox.population(n=50)
        self.CXPB, self.MUTPB, self.NGEN = 0.5, 0.2, 40
        
        
    def evaluate(self, individual):
        pass
    
    def evaluate_pop(self):
        fitnesses = map(self.toolbox.evaluate, self.pop)
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit
    
    def select_next_gen(self):
        # Select the next generation individuals
        offspring = self.toolbox.select(self.pop, len(self.pop))
        # Clone the selected individuals
        offspring = map(self.toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            self.crossover(child1, child2)
        for mutant in offspring:
            self.mutation(mutant)
        self.invalid_fitness(offspring)
        self.pop[:] = offspring
    
    
    def crossover(self, child1, child2):
        if random.random() < self.CXPB:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
    
    def mutation(self, mutant):
        if random.random() < self.MUTPB:
            self.toolbox.mutate(mutant)
            del mutant.fitness.values
    
    def invalid_fitness(self, offspring):
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
    
    def run_algorithm(self):
        for g in range(self.NGEN):
            self.select_next_gen()