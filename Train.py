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
        i = 0
        for article in list(self.train_set):
            print(i)
            self.create_individuals(article)
            i += 1
            
    def create_individuals(self, article):
        words = self.get_words(article)
        word_weights = self.get_word_weights(words) 

        sentences, sentence_weights = self.get_sentences(words, word_weights)
        self.write_list(sentences, 'sentences')
        self.write_list(sentence_weights, 'SentenceWeights')
        # for i in range(len(sentences)):
        #     print(sentences[i], sentence_weights[i])

        # TODO: for each sentence, create an individual with weight_attr sentence_weights[i]

    def get_words(self, article):
        ar = article['article'].lower()
        ar_no_new_lines = ar.replace("\ n", "")
        tokens = self.mt.tokenize(ar_no_new_lines)
        return tokens
        

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
            n_list = pickle.load(fp)
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
    
    # train.readFile()
    # train.tokenize_sample()

    sentence = train.read_list('sentences')[5]

    tokens = train.mt.tokenize(sentence)
    print(tokens)

    # print(train.read_list('sentences'))
    # print(train.read_list('SentenceWeights'))



