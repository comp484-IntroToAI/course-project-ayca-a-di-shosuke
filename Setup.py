from sacremoses import MosesTokenizer
import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
import regex as re


class Setup():
    def __init__(self):
        self.tokenize = MosesTokenizer()
        
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
            
    def buildVocabulary(self):
        print("building vocabulary")
        self.write_list([], "vocab")
        self.vocab = []
        article_num = 0
        for article in self.train_set:
            vocabu = str(article['article'])
            vocabu = vocabu.lower()
            vocabu = list(set(self.tokenize.tokenize(vocabu)))
            self.vocab.extend(vocabu)
            print("article tokenized: " + str(article_num))
            article_num = article_num + 1
            if article_num % 1000 == 0:
                print("saving article " + str(article_num))
                self.saveWords()
                self.vocab = []
        self.saveWords()
        
    def saveWords(self):
        current_list = self.read_list("vocab")
        current_list.extend(self.vocab)
        current_list = list(set(current_list))
        self.write_list(current_list, "vocab")
        
    def printVocabulary(self):
        print("fetching vocabulary")
        # print(self.read_list("vocab")[2000:2100]/)
        print("filtered")
        print(len(self.read_list("filtered")))
        print(self.read_list("filtered"))
    
    def filter_words_characters(self):
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'-")
        words = self.read_list("filtered")
        new_words = []
        x = 0 
        for word in words:
            validation = set((word))
            if validation.issubset(allowed_chars):
                new_words.append(word)
            x = x + 1
            print(str(x))
        self.write_list(new_words, "filtered")

        
if __name__ == "__main__":
    setup = Setup()
    # setup.filter_words_characters()
    setup.printVocabulary()
