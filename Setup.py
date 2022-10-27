from sacremoses import MosesDetokenizer, MosesTokenizer
import tensorflow as tf
import tensorflow_datasets as tfds

class Setup():
    def __init__(self):
        self.tokenize = MosesTokenizer()
    
    def readFile(self, data_name = 'scientific_papers'):
        self.train_set = tfds.load(data_name, split='train')
        print('#########################')
        print(list(self.train_set)[0])
            
    
    def buildVocabulary(self):
        self.vocab = self.tokenize.tokenize(self.train_set)
        
    def printVocabulary(self):
        print(self.vocab)
        # pass

        

       
if __name__ == "__main__":
    setup = Setup()
    setup.readFile()
    setup.buildVocabulary()
    setup.printVocabulary()
