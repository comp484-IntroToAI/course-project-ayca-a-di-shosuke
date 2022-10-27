from sacremoses import MosesDetokenizer, MosesTokenizer
import tensorflow as tf
import tensorflow_datasets as tfds

class Setup():
    def __init__(self):
        self.tokenize = MosesTokenizer()
    
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
