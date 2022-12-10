# This file builds a vocab dictionary into a pickle file. The vocab consists
# of alphabetic words. This file also creates list-readable versions of abstracts
# and articles. 

import random
import tensorflow_datasets as tfds
import pickle
import re
import nltk.data
import time
import deap


class Setup():
    def __init__(self):
        self.WORD = re.compile(r'\w+')
        self.tokenizer = nltk.tokenize

    ################################ Helpers ################################
    # write list to binary file
    def write_list(self, a_list, file_name = 'vocab_dict'):
        with open(file_name, 'wb') as fp:
            pickle.dump(a_list, fp)
            print('Done writing list into a binary file')

    # Read list to memory
    def read_list(self, file_name = 'vocab_dict'):
        with open(file_name, 'rb') as fp:
            n_list = pickle.load(fp)
            return n_list
        
    # return a printable version of the created vocab file
    def read_vocab_dict(self):
        vocab = self.read_list("vocab_dict")
        return vocab
    
    # read training set from tfds 
    def read_papers_file(self, data_name = 'scientific_papers'):
        print("Reading Data")
        self.train_set = tfds.load(data_name, split='train')
        
    ################################ Create Vocabulary ################################
        
    # setup files and training data for building vocabulary
    def setup_files(self):
        self.read_papers_file()
        self.write_list({}, "vocab_dict")
        self.list_set = list(self.train_set)
        
    # creates a vocabulary from the scientific papers dataset           
    def buildVocabulary(self):
        self.setup_files()
        words = {}
        i = 0
        for article in self.list_set:
            article = str(article['article']).lower() # convert to lower
            tokens = self.regTokenize(article) # tokenize
            tokens = self.filter_all_words(tokens) # filter out non-words
            words.update(dict(zip(tokens, list([0]*len(tokens)))))
            print(i)
            i = i + 1
        self.write_list(list(words.keys()), 'vocab_dict')
        
    # regex tokenizer since it is much faster than others
    def regTokenize(self, text):
        words = self.WORD.findall(text)
        return words
        
    # filter out words with numbers and unlikely characters
    def filter_words(self, word):
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'-")
        validation = set((word))
        if validation.issubset(allowed_chars):
            return word
    
    # loop through words and filter
    def filter_all_words(self, words):
        new_words = list(filter(self.filter_words, words))
        return new_words
    
    ################################ Create Article Lists ################################
    
    def read_articles_train(self):
        train_set = list(tfds.load('scientific_papers', split='train'))
        articles = []
        abstracts = []
        i = 0
        for article in train_set:
            articles.append(str(article['article']))
            abstracts.append(str(article['abstract']))
            print(i)
            i = i + 1
        self.write_list(articles, 'articles.pkl')
        self.write_list(abstracts, 'abstracts.pkl')

    def read_articles_test(self):

        train_set = list(tfds.load('scientific_papers', split='test', data_dir='D:'))
        articles = []
        abstracts = []
        i = 0
        for article in train_set:
            articles.append(str(article['article']))
            abstracts.append(str(article['abstract']))
            print(i)
            i = i + 1
        self.write_list(articles, 'articles_test.pkl')
        self.write_list(abstracts, 'abstracts_test.pkl')

    def write_test_samples(self, sample_size):
        articles = self.read_list('articles_test.pkl')
        abstracts = self.read_list('abstracts_test.pkl')
        article_samples, abstract_samples = zip(*random.sample(list(zip(articles, abstracts)), sample_size))
        self.write_list(article_samples, 'articles_test_' + str(sample_size) + '.pkl')
        self.write_list(abstract_samples, 'abstracts_test_' + str(sample_size) + '.pkl')
        
if __name__ == "__main__":
    tic = time.perf_counter()
    setup = Setup()
    
    # setup.buildVocabulary() # takes ~40 minutes to run on laptop
    # print(setup.read_vocab_dict())
    
    # setup.read_articles_train() # takes < 5-10 minutes to run on laptop
    # print(setup.read_list('articles.pkl')[0])
    # print(setup.read_list('abstracts.pkl')[0])

    # setup.read_articles_test()
    # print(setup.read_list('articles_test.pkl')[0])
    # print(setup.read_list('abstracts_test.pkl')[0])

    # for num in [10, 50, 100]:
    #     setup.write_test_samples(num)

    # print(len(setup.read_list('new_vocab')))
    print((time.perf_counter() - tic)/60) # time process


