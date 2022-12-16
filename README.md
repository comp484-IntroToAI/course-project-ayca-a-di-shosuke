# REvolutionary NLP: Extractive Text Summary using Genetic Algorithms

Ayça Arbay, A’di Dust, Shosuke Noma

## Introduction

Automatic text summarization (ATS) is an increasingly in-demand Natural Language Processing (NLP) task with a wide range of useful applications. Previous ATS research has relied most heavily on deep learning and although significant progress has been made, the computationally expensive nature of deep learning models remains a limiting factor (Chen et al., 2021). Genetic algorithms (GA), on the other hand, require considerably less computational power and training data than deep learning models but are seldom used for NLP tasks. This project contributes to the relatively unexplored field of automatic text summarization using genetic algorithms. Since GAs are most commonly used to solve optimization problems, they are better suited for text analysis than text generation. For this reason, we chose to focus on extractive rather than abstractive summarization in our research. We explored whether our genetic algorithm can produce accurate and cohesive summaries of scientific articles and how its performance correlates with the algorithm’s parameters. To assess the algorithm’s success, we will be using ROUGE-L scores, which measure how well a generated summary matches the actual abstract of an article, as well as subjective evaluations, such as binary feedback from the users.

In this project only focus on summarizing scientific articles, using their abstracts as a reference to judge the quality of our generated summaries. We utilized TensorFlow’s scientific_papers dataset (obtained from ArXiv and PubMed OpenAccess repositories) to both train and test our algorithm, which we will create using Python’s DEAP (Distributed Evolutionary Algorithms in Python) framework. We experimented with various population and vocabulary sizes depending on the limit of our computational resources. Our final product consists of a GUI with a text field and a submission button that then outputs the extracted summary. 

This is an important problem because it is a computationally inexpensive way to summarize text. This method has also been rarely studied, and our project is the first we could find that implements ROUGE-L and has a GUI for public use. The use of AI for summarizing text is especially relevant in academic texts, do to the complexity of academic papers. Using our algorithm and graphical user interface, individuals can quickly understand the gist of a paper before deciding to read the paper or to get the context briefly. 

## Algorithmic Overview

### Genetic Algorithm

A genetic algorithm is a solution to an optimization problem that mimics evolution. The algorithm involves creating a population of individuals, and then mating those individuals with the best match so that future generations have what's positive about both individuals. There are also random mutations involved, which keeps the optimization from hovering at local maximums, so that the solution can go towards the global maximum. The following pseudocode describes how our algorithm approached the genetic algorithm process. 

```
create a population
assign individuals list of weights
for each generation:
  mate individuals
    cross over weights to make children
    mutate weights in some individuals
  evaluate children
  make children the new population
```

### Evaluation

To evaluate the children we used a metric called ROUGE-L. This stands for Recall-Oriented Understudy for Gisting Evaluation Longest common substring. Essentially, ROUGE is used for evaluating summary by analyzing the similarity of content between two texts. We used the Longest Common Substring (LCS) as our evaluation metric, which measures a summary by the longest string of words in common between the two texts. We used this because we wanted not just to capture unigrams or bigrams, but also the structure of the reference summary. 

Our evaluation pseudocode is shown below:

```
Given a vocab and list of weights:
  create a sentence
  keep sentences with weights above a threshold
  combine to a summary
  get ROUGE-L score
  redefine weights basesd on score
```

## Installation

- figure this out when TKinter part done

## File Descriptions

### FileCreator.py

This file is used to setup the training information needed to run the genetic algorithm. The runnable processes are:

- Creating Vocabulary:

**buildVocabulary()** - takes about 40 minutes to run on a laptop. Uses scientific articles dataset and parses unique words into a list, saved as a pickle file. 

- Creating Dataset:

**read_articles()** - takes about 10 minutes to run on a laptop. Creates a list version of articles and abstracts and saves them into 2 pickle files. 

### EvolutionaryModel.py

This file handles the genetic algorithm. Necessary steps before running:

- create vocabulary dictionary
- create article dataset
- create abstract dataset
- change number of generations, population size, and number of training articles as necessary. 

### TextSummaryGUI.py

A file containing a simple GUI for inputting articles, which implements the genetic algorithm output as the summary function. 

### Pickled Files

- **vocab_dict.pkl**: Contains a list of unique words parsed from training set.
- **abstracts.pkl**: Contains a list of all training abstracts.
- **articles.pkl**: Contains a list of all training articles. 
- **new_vocab.pkl**: Contains list of best individual weights from genetic algorithm. 

## Technology Used

### Languages

- Python

### Libraries

- Random
- DEAP
- Pickle
- NLTK
- RE
- Keras
- Keras_NLP
- Time
- TKinter


## Credits

### Databases

- [TensorFlow Datasets Scientific Papers](https://www.tensorflow.org/datasets/catalog/scientific_papers)

### Paper Inpspiration

Chen, William, Kensal Ramos, and Kalyan Naidu Mullaguri. "Genetic Algorithms For Extractive Summarization." *arXiv preprint arXiv:2105.02365 (2021).*

### Code Inspriation

- [GA Text Summarization](https://github.com/wanchichen/GA-Text-Summarization) by William Chen
