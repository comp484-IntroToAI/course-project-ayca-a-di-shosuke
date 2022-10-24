from ast import main
from lib2to3.pgen2.tokenize import tokenize
from mosestokenizer import *
import tensorflow as tf
import tensorflow_datasets as tfds

class Setup():
    def __init__(self):
        self.tokenize = MosesTokenizer('en')
    
    def readFile(self, data_name = 'scientific_papers'):
        self.train_set = tfds.load(data_name, split='train')
            
    
    def buildVocabulary(self):
        self.vocab = self.tokenize(self.train_set[0].abstract)
        self.tokenize.close()
        
    def printVocabulary(self):
        print(self.vocab)
        

       
if __name__ == "__main__":
    setup = Setup()
    setup.readFile()
    setup.buildVocabulary()
    setup.printVocabulary()
