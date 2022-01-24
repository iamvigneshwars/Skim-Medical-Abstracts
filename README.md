# Annotate Sections in Unstructured Medical Abstracts

# Introduction

Close to two million works of medicals literatures are published each year.
Due to the increasing number of publications, the researchers need special
tools to skim through the literature. It is difficult to extract relevant
information quickly from an unstructured abstract. Accessing information
from the literature will be easier if the abstract of the literature is
structured. This project aims to annotate each sentence in
randomized controlled trials medical abstracts into their appropriate
sections thereby helping the researchers to collect the required
information expeditiously. 

## Data

The model is trained using the PubMed RCT 200K dataset which is described in  *Franck Dernoncourt, Ji Young Lee. [PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts](https://arxiv.org/abs/1710.06071). International Joint Conference on Natural Language Processing (IJCNLP). 2017.* The dataset consists of approximately 200,000 medical abstracts of randomized controlled trails (RTC). 

The proprocessed data consists of text, target and position of the sentences in an abstract. The position of the sentence in an abstract if scalled between 1 to 5.

## Model Architecture

The model Contains an attention layer, followed by two Bi-directional LSTM layers. The output for Bi-LSTM layer is concatinated with sentence position feature (one hot vector). The output of concatinated layers if fed into a fully connected layer with 128 neurons followed by an output layer that contains 5 neurons (5 output classes).

<p float="middle">
  <img src="static/model.png" width="450" />
</p>


## Run my project

You can make predictions on CLI or using an web interface. To use the flask web interface run program as follows

```bash
export FLASK_APP=main.py
flask run --host=0.0.0.0 # this will serve at port 5000 (default)
```

or to make predictions on CLI, run the predict.py as follows


```bash
python predict.py
```


## Requirements

- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [AllenNLP](https://github.com/allenai/allennlp) >= 0.6.1
- [spacy](https://github.com/explosion/spaCy)
- [fastText](https://github.com/facebookresearch/fastText)
- [Pubmed RCT](https://github.com/Franck-Dernoncourt/pubmed-rct) - dataset


