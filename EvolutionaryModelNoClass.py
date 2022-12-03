import random
import pickle
import heapq
import operator
import statistics as stats
from sacremoses import MosesTokenizer, MosesDetokenizer
from deap import creator, base, tools, algorithms
from rouge_score import rouge_scorer


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
    # print(lst)
    lst_i = [(lst[i], i) for i in range(len(lst))]
    # print(lst_i)
    
    best = heapq.nlargest(limit, lst_i)
    inds = sorted([best[i][1] for i in range(len(best))])

    # print(best)
    print(inds)

    
    summary = [sentences[i] for i in inds]
    print("summary: ", ' '.join(summary))
    
    # abstract = read_list("trainingAbstracts")
    print("abstract: ", ' '.join(abstract))

    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    print(rouge.score(' '.join(summary), ' '.join(abstract)))


def create_individuals(article):
        words = get_words(article)
        word_weights = get_word_weights(words) 
        sentences, sentence_weights = get_sentences(words, word_weights)
    

def get_word_weights(words):
    weights_to_return = []

    for word in words:
        try:
            weight = vocab[word]
            weights_to_return.append(weight)
        except KeyError:
            weights_to_return.append(0)

    return weights_to_return

def get_sentences(words, word_weights):
    md = MosesDetokenizer()

    sentences = []
    sentence_weights = []

    start_i = 0

    while True:
        try:
            end_i = words.index('.', start_i)
            
            sentence = md.detokenize(words[start_i:end_i+1])
            sentences.append(sentence)

            sentence_weight = stats.mean(word_weights[start_i:end_i+1])
            sentence_weights.append(sentence_weight)

            start_i = end_i+1
            
        except ValueError:
            break

    return sentences, sentence_weights

def get_words(article):
    mt = MosesTokenizer()
    ar = str(article['article'].numpy()).lower()
    ar = " ".join(ar.split('\\n'))
    without_n = "".join(filter(lambda x: x.isalpha() or x.isspace() or x == ".", ar))
    tokens = mt.tokenize(without_n)
    return tokens


def create_dictionary():
    vocab = read_list("filtered")
    weights = read_list("weights")
    word_weights = dict(zip(vocab, weights))
    return word_weights





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
CXPB, MUTPB, NGEN = 0.5, 0.2, 50


word_weights = create_dictionary()





sentences = read_list("trainingSentences")
abstract = read_list("trainingAbstracts")[0]
sent_count = abstract.count('.')
print(sent_count)

# Evaluate the entire population
evaluate_pop()
run_algorithm()

get_best_inds(sent_count)

