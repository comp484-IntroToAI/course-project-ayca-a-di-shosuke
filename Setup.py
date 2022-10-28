from sacremoses import MosesTokenizer
import tensorflow as tf
import tensorflow_datasets as tfds
import pickle


class Setup():
    def __init__(self):
        self.tokenize = MosesTokenizer()
        
    # write list to binary file
    def write_list(self, a_list):
        with open('vocab', 'wb') as fp:
            pickle.dump(a_list, fp)
            print('Done writing list into a binary file')

    # Read list to memory
    def read_list(self):
        with open('vocab', 'rb') as fp:
            n_list = pickle.load(fp)
            return n_list
    
    def readFile(self, data_name = 'scientific_papers'):
        print("Reading Data")
        self.train_set = tfds.load(data_name, split='train') # delete take for whole training set
            
    def buildVocabulary(self):
        print("building vocabulary")
        self.vocab = []
        article_num = 0
        for article in self.train_set:
            article_num = article_num + 1
            vocabu = str(article['article'])
            vocabu = vocabu.lower()
            vocabu = list(set(self.tokenize.tokenize(vocabu)))
            self.vocab.extend(vocabu)
            self.vocab = list(set(self.vocab))
            print("article tokenized: " + str(article_num))
        self.write_list(self.vocab)
        
    def printVocabulary(self):
        print("fetching vocabulary")
        print(self.read_list())

        
if __name__ == "__main__":
    setup = Setup()
    setup.readFile()
    setup.buildVocabulary()
    # setup.printVocabulary()
