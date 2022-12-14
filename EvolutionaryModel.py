# File to use genetic algorithm to update vocab weights using ROUGE-L.

import random
from deap import creator, base, tools
import pickle
import tensorflow as tf
from nltk.tokenize import sent_tokenize
import re
import keras_nlp
import keras

################################ Helpers ################################

class GAHelpers():
    def __init__(self):
        self.WORD = re.compile(r'\w+')
    
    # Read list to memory
    def read_list(self, file_name = 'vocab_dict'):
        with open(file_name, 'rb') as fp:
            n_list = pickle.load(fp)
            return n_list
        
    # add element to already-stored pickle file list
    def edit_list(self, number, file_name = 'new_vocab'):
        old_list = self.read_list(file_name)
        old_list.append(number)
        self.write_list(old_list)
        
    # write list to binary file
    def write_list(self, a_list, file_name = 'new_vocab'):
        with open(file_name, 'wb') as fp:
            pickle.dump(a_list, fp)
    
    # get articles and abstracts into local memory lists
    def read_articles(self):
        return self.read_list('articles.pkl')[0:1], self.read_list('abstracts.pkl')[0:1]
    
    # use new weights and save them into the vocab
    def update_weights(self, vocab, weights):
        dictionary = {}
        for i in range(len(vocab)):
            dictionary[vocab[i]] = weights[i]
        return dictionary
    
    # regex tokenizer since it is much faster than others
    def regTokenize(self, text):
        words = self.WORD.findall(text)
        return words

    # filter non-words from sentences (other than likely characters)
    def filter_sentence(self, article, type):
        if type == "article":
            article = article.lower()
        article = " ".join(article.split('\\n'))
        article = "".join(filter(lambda x: x.isalpha() or x.isspace() or x == ".", article))
        return article

    # return a deque of deques of sentence weights
    def get_summary_weights(self, sentences, vocab, threshold):
        summary = []
        for sentence in sentences:
            words = self.regTokenize(sentence)
            sentence_weight = self.get_word_weights(words, vocab)
            try:
                if sentence_weight/len(words) > threshold:# add summary if greater than threshold
                    summary.append(sentence)
            except ZeroDivisionError:
                pass
        return summary

    # retrieve word weights from vocab and return deque of sentence weights
    def get_word_weights(self, words, vocab):
        weights = []
        for word in words:
            try:
                weights.append(vocab[word])
            except KeyError:
                weights.append(0)
        return sum(weights)
    
    # filter and tokenize article and return summary weights
    def summarize(self, dictionary, article, threshold):
        article = self.filter_sentence(article, "article")
        sentences = sent_tokenize(article)
        return self.get_summary_weights(sentences, dictionary, threshold)
    
    # Use Rouge-L to score created summary against abstract
    def score_summary(self, summary, abstract, model):
        avg = 0
        if len(summary) <= 0:
            return 0
        summary = " ".join(summary) # make sentence list into string summary
        x = tf.constant([abstract])
        y = tf.constant([summary])
        metric_dict = model.evaluate(x, y, return_dict=True, verbose=False)
        avg = metric_dict["f1_score"] + metric_dict['precision'] + metric_dict['recall']
        return (avg/3)*100 # get f1-score, precision, and recall together as percent of 100
        
    # score invalid fitnesses and update vocab
    def evaluate(self, vocab, ind, articles, abstracts, threshold, model):
        dictionary = self.update_weights(vocab, ind)
        score = 0
        length = len(articles)
        for i in range(length): # go through all articles and summarize/score
            summary = self.summarize(dictionary, articles[i], threshold)
            score = score + self.score_summary(summary, abstracts[i], model)
        return {score/length}
    
    
################################ Algorithm ################################

if __name__ == "__main__":
    # setup ROUGE model
    inputs = keras.Input(shape=(), dtype='string')
    outputs = tf.strings.lower(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(metrics=[keras_nlp.metrics.RougeL()])
        
    helpers = GAHelpers()
    helpers.write_list([], 'new_vocab')
    
    # setup creator with individuals
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness = creator.FitnessMax)
    
    # read vocab, articles, and abstract files into lists
    vocab = helpers.read_list()[0:10]
    print(len(vocab))
    articles, abstracts = helpers.read_articles()
    
    # limit size of individual
    IND_SIZE = len(vocab)
    print(IND_SIZE)
    
    # setup individuals to be lists of floats
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     toolbox.attr_float, n = IND_SIZE)
    
    # limit size of population and create population
    POP_SIZE = 75
    pop = list()
    for i in range(POP_SIZE):
        pop.append(toolbox.individual())
    
    # set up mating and mutation
    CXPB = 0.8 # probability of crossing 
    MUTPB = 0.05 # probability of mutating
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize = 5) # select best individuals out of 5
    
    # setup variables for generation
    num_generations = 10
    threshold= 0.6
    max_score = 0
    max_ind = None
    avg_fitness = list()
    best_fitness = list()
    
    # iterate through each generation
    for i in range(num_generations):
        print("Generation " + str(i))
        
        # create offspring from population
        offspring = toolbox.select(pop, POP_SIZE)
        offspring = list(map(toolbox.clone, offspring))

        # apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        print("Mating Done")
        
        # apply mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                mutant[int(random.random() * IND_SIZE)] = 0
                del mutant.fitness.values
        print("Mutations Done")
        
        # evaluate invalid fitness scores using ROUGE-L
        print("Evaluate Invalid Fitness")
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            fitness = helpers.evaluate(vocab, ind, articles, abstracts, threshold, model)
            # print(fitness)
            ind.fitness.values = fitness
        
        # create new population and variables
        pop[:] = offspring
        fitness = 0
        best_gen = 0
        max_ind = []
        
        # get max and average scores for this generation
        print("Get Max Scores")
        for ind in pop:
            fitness += ind.fitness.values[0]
            if max_score < ind.fitness.values[0]:
                max_score = ind.fitness.values[0]
                max_ind = ind
            if best_gen < ind.fitness.values[0]:
                best_gen = ind.fitness.values[0]
        avg_fitness.append(fitness/POP_SIZE)
        best_fitness.append(best_gen)
    print(avg_fitness)
    print(best_fitness)
        
    # update vocab file to reflect new weights
    print("adding weights to file")
    weights = []
    for num in max_ind:
        weights.append(num)
    helpers.write_list(weights)