# REvolutionary NLP: Extractive Text Summary using Genetic Algorithms

Ayça Arbay, A’di Dust, Shosuke Noma

## Introduction

Automatic text summarization (ATS) is an increasingly in-demand Natural Language Processing (NLP) task with a wide range of useful applications. Previous ATS research has relied most heavily on deep learning and although significant progress has been made, the computationally expensive nature of deep learning models remains a limiting factor (Chen et al., 2021). Genetic algorithms (GA), on the other hand, require considerably less computational power and training data than deep learning models but are seldom used for NLP tasks. This project contributes to the relatively unexplored field of automatic text summarization using genetic algorithms. Since GAs are most commonly used to solve optimization problems, they are better suited for text analysis than text generation. For this reason, we chose to focus on extractive rather than abstractive summarization in our research. We explored whether our genetic algorithm can produce accurate and cohesive summaries of scientific articles and how its performance correlates with the algorithm’s parameters. To assess the algorithm’s success, we will be using ROUGE-L scores, which measure how well a generated summary matches the actual abstract of an article, as well as subjective evaluations, such as binary feedback from the users.

In this project only focus on summarizing scientific articles, using their abstracts as a reference to judge the quality of our generated summaries. We utilized TensorFlow’s scientific_papers dataset (obtained from ArXiv and PubMed OpenAccess repositories) to both train and test our algorithm, which we will create using Python’s DEAP (Distributed Evolutionary Algorithms in Python) framework. We experimented with various population and vocabulary sizes depending on the limit of our computational resources. Our final product consists of a GUI with a text field and a submission button that then outputs the extracted summary. 

This is an important problem because it is a computationally inexpensive way to summarize text. This method has also been rarely studied, and our project is the first we could find that implements ROUGE-L and has a GUI for public use. The use of AI for summarizing text is especially relevant in academic texts, do to the complexity of academic papers. Using our algorithm and graphical user interface, individuals can quickly understand the gist of a paper before deciding to read the paper or to get the context briefly. 

## Algorithmic Overview

### Genetic Algorithm

- description of what GA is

```
add pseudocode
```

### Evaluation

- description of ROUGE

## Installation

- figure this out when TKinter part done

## File Descriptions

### Setup.py

### EvolutionaryModel.py

### TextSummaryGUI.py

### Pickled Files

## Technology Used

### Languages

- Python

### Libraries

- Random
- DEAP
- Pickle
- NLTK.Tokenize
- RE
- Keras
- Keras_NLP
- Time

## Credits

### Databases

- TensorFlow Datasets Scientific Papers

### Sources



### Code Inspriation

- GA-Text-Summarization
