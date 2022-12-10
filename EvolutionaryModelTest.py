# File to use genetic algorithm to update vocab weights using ROUGE-L.

import random
from deap import creator, base, tools
import pickle
import tensorflow as tf
from nltk.tokenize import sent_tokenize
import re
import keras_nlp
import keras
import winsound

################################ Helpers ################################

class GAHelpers():
    def __init__(self):
        self.WORD = re.compile(r'\w+')
    
    # Read list to memory
    def read_list(self, file_name = 'vocab_dict'):
        with open(file_name, 'rb') as fp:
            n_list = pickle.load(fp)
            return n_list
        
    def edit_list(self, number, file_name = 'new_vocab'):
        old_list = self.read_list(file_name)
        old_list.append(number)
        self.write_list(old_list)
        
    # write list to binary file
    def write_list(self, a_list, file_name = 'new_vocab'):
        with open(file_name, 'wb') as fp:
            pickle.dump(a_list, fp)
            print('Done writing list into a binary file')
    
    # given a sample size, get articles and abstracts from appropriate file into local memory lists 
    def read_articles(self, sample_size):
        articles = self.read_list('articles_test_' + str(sample_size) + '.pkl')
        abstracts = self.read_list('abstracts_test_' + str(sample_size) + '.pkl')
        return articles, abstracts

    
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
            if sentence_weight/len(words) > threshold:
                summary.append(sentence)
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
    def score_summary(self, summary, abstract):
        avg = 0
        if len(summary) <= 0:
            return 0
        summary = " ".join(summary) # make sentence list into string summary
        inputs = keras.Input(shape=(), dtype='string')
        outputs = tf.strings.lower(inputs)
        model = keras.Model(inputs, outputs)
        model.compile(metrics=[keras_nlp.metrics.RougeL()])
        x = tf.constant([abstract])
        y = tf.constant([summary])
        metric_dict = model.evaluate(x, y, return_dict=True, verbose=False)
        avg = metric_dict["f1_score"] + metric_dict['precision'] + metric_dict['recall']
        return (avg/3)*100 # get f1-score as percent of 100
        
    # score invalid fitnesses and update vocab
    def evaluate(self, vocab, ind, articles, abstracts, threshold):
        dictionary = self.update_weights(vocab, ind)
        score = 0
        length = len(articles)
        for i in range(length): # go through all articles and summarize/score
            summary = self.summarize(dictionary, articles[i], threshold)
            score = score + self.score_summary(summary, abstracts[i])
        return {score/length}

    # TODO
    def init_individual(self, icls, sample_size, vocab_size):
        contents = self.read_list("D:/course-project-ayca-a-di-shosuke/vocab_files/new_vocab_g10_p50_a" + sample_size + "_v" + vocab_size)
        return icls(contents)    

    def get_sample_size_str(self, sample_size):
        if sample_size == 10:
            return '0010'
        elif sample_size == 50:
            return '0050'
        elif sample_size == 100:
            return '0100'

    def get_vocab_size_str(self, vocab_size):
        if vocab_size == 1000:
            return '01000'
        elif vocab_size == 10000:
            return '10000'
        elif vocab_size == 50000:
            return '50000'
    
################################ Algorithm ################################

if __name__ == "__main__":
    
    helpers = GAHelpers()
    
    NUM_GENERATIONS = 10
    POP_SIZE = 50

    # for samples in [10, 50, 100]:
    #     for vocabs in [1000, 10000, 50000]:

    for sample, vocab in [(10, 1000), (10, 10000), (10, 50000), (50, 1000), (50, 10000), (100, 10000), (100, 50000)]:

        SAMPLE_SIZE = sample
        VOCAB_SIZE = vocab   

        # setup creator with individuals
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness = creator.FitnessMax)
        
        # read vocab, articles, and abstract files into lists
        vocab = helpers.read_list()[:VOCAB_SIZE]
        print(len(vocab))
        articles, abstracts = helpers.read_articles(SAMPLE_SIZE)
        print(len(articles), len(abstracts))
        
        # limit size of individual
        IND_SIZE = len(vocab)
        print(IND_SIZE)
        
        # setup individuals to be lists of floats
        toolbox = base.Toolbox()
        # toolbox.register("attr_float", random.random)
        # toolbox.register("individual", tools.initRepeat, creator.Individual, 
        #                  toolbox.attr_float, n = IND_SIZE)

        toolbox.register("individual", helpers.init_individual, creator.Individual, helpers.get_sample_size_str(SAMPLE_SIZE), helpers.get_vocab_size_str(VOCAB_SIZE))
        
        # create population
        pop = list()
        for i in range(POP_SIZE):
            pop.append(toolbox.individual())
        
        # set up mating and mutation
        CXPB = 0.8 # probability of crossing 
        MUTPB = 0.05 # probability of mutating
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("select", tools.selTournament, tournsize = 5) # select best individuals out of 5
        
        # setup variables for generation
        threshold= 0.6
        max_score = 0
        max_ind = None
        avg_fitness = list()
        best_fitness = list()
        
        # iterate through each generation
        for i in range(NUM_GENERATIONS):
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
                fitness = helpers.evaluate(vocab, ind, articles, abstracts, threshold)
                # print(fitness)
                ind.fitness.values = fitness
            
            # create new population
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

        helpers.write_list(avg_fitness, 'avg_fitness_g10_p50_a' + helpers.get_sample_size_str(SAMPLE_SIZE) + '_v' + helpers.get_vocab_size_str(VOCAB_SIZE))
        helpers.write_list(best_fitness, 'best_fitness_g10_p50_a' + helpers.get_sample_size_str(SAMPLE_SIZE) + '_v' + helpers.get_vocab_size_str(VOCAB_SIZE))

        duration = 1000  # milliseconds
        freq = 440  # Hz
        winsound.Beep(freq, duration)
        