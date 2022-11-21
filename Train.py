from sacremoses import MosesTokenizer, MosesDetokenizer
from statistics import mean
import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
import random
import regex as re


class Train():
    def __init__(self):
        self.mt = MosesTokenizer()
        self.md = MosesDetokenizer()

    def tokenize_sample(self):
        for article in list(self.train_set)[0:1]:
            self.create_individuals(article)
            
    def create_individuals(self, article):
        words = self.mt.tokenize(str(article['article']).lower())
        word_weights = self.get_word_weights(words) 

        sentences, sentence_weights = self.get_sentences(words, word_weights)

        for i in range(len(sentences)):
            print(sentences[i], sentence_weights[i])

        # TODO: for each sentence, create an individual with weight_attr sentence_weights[i]


    def get_word_weights(self, words):
        vocab = self.read_list("filtered")
        weights = self.read_list("weights")

        weights_to_return = []

        for word in words:
            try:
                i = vocab.index(word)
                weight = weights[i]
                weights_to_return.append(weight)
            except ValueError:
                weights_to_return.append(0)

        return weights_to_return

    def get_sentences(self, words, word_weights):
        sentences = []
        sentence_weights = []

        start_i = 0

        while True:
            try:
                end_i = words.index('.', start_i)
                
                sentence = self.md.detokenize(words[start_i:end_i+1])
                sentences.append(sentence)

                sentence_weight = mean(word_weights[start_i:end_i+1])
                sentence_weights.append(sentence_weight)

                start_i = end_i+1
                
            except ValueError:
                break

        return sentences, sentence_weights

    # write list to binary file
    def write_list(self, a_list, file_name):
        with open(file_name, 'wb') as fp:
            pickle.dump(a_list, fp)
            print('Done writing list into a binary file')

    # Read list to memory
    def read_list(self, file_name):
        with open(file_name, 'rb') as fp:
            n_list = pickle.load(fp)[0:2000]
            return n_list
    
    def readFile(self, data_name = 'scientific_papers'):
        print("Reading Data")
        self.train_set = tfds.load(data_name, split='train') # delete take for whole training set

    def get_filtered_vocabulary(self, start_index=0, end_index=-1):
        filtered_list = self.read_list("filtered")

        if end_index == -1:
            end_index = len(filtered_list)

        return filtered_list[start_index : end_index]

    def get_weights(self, start_index=0, end_index=-1):
        weights = self.read_list("weights")

        if end_index == -1:
            end_index = len(weights)

        return weights[start_index : end_index]

        
if __name__ == "__main__":
    train = Train()

    ## setup.filter_words_characters()
    # print(setup.get_filtered_vocabulary(0, 10))

    ## setup.create_parallel_weight_list()
    # print(setup.get_weights(0, 10))
    
    train.readFile()
    train.tokenize_sample()



