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
        self.vocab = self.read_list("filtered")
        self.weights = self.read_list("weights")
        self.all_vocab = dict(zip(self.vocab, self.weights))

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

    def get_words(self, article):
        ar = str(article['article'].numpy()).lower()
        ar = " ".join(ar.split('\\n'))
        without_n = "".join(filter(lambda x: x.isalpha() or x.isspace() or x == ".", ar))
        tokens = self.mt.tokenize(without_n)
        return tokens
        

    def get_word_weights(self, words):
        weights_to_return = []

        for word in words:
            try:
                weight = self.all_vocab[word]
                weights_to_return.append(weight)
            except KeyError:
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
        with open(file_name, 'ab') as fp:
            pickle.dump(a_list, fp)
            # print('Done writing list into a binary file')

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
    train.readFile()
    train.tokenize_sample()
    # print(train.read_list('sentences'))
    # print(train.read_list('SentenceWeights'))



