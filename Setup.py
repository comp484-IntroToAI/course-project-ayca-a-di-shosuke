# This file builds a vocab dictionary into a pickle file. The vocab consists
# of alphabetic words and corresponding random weights between 0 and 1. 

import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
import random
from collections import deque
import re


class Setup():
    def __init__(self):
        self.WORD = re.compile(r'\w+')

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
    
    # read training set from tfds 
    def read_papers_file(self, data_name = 'scientific_papers'):
        print("Reading Data")
        self.train_set = tfds.load(data_name, split='train')
    
    # setup files and training data for building vocabulary
    def setup_files(self):
        self.read_papers_file()
        self.write_list({}, "vocab_dict")
        self.list_set = list(self.train_set)
        
    # creates a vocabulary from the scientific papers dataset           
    def buildVocabulary(self):
        self.setup_files()
        words = {}
        for article in self.list_set:
            article = str(article['article']).lower() # convert to lower
            tokens = self.regTokenize(article) # tokenize
            tokens = self.filter_all_words(tokens) # filter out non-words
            words.update(dict(zip(tokens, self.create_random_weights(len(tokens))))) # create weights
        self.write_list(words, 'vocab_dict')
        
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
    
    # create weights between 0-1
    def create_random_weights(self, number):
        weights = deque()
        for num in range(number):
            weights.append(random.random())
        return weights
    
    # return a printable version of the created vocab file
    def read_vocab_dict(self):
        vocab = self.read_list("vocab_dict")
        return len(vocab)

        
if __name__ == "__main__":
    setup = Setup()
    # setup.buildVocabulary() # takes 40 minutes to run 
    print(setup.read_vocab_dict())


