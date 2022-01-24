from nltk import sent_tokenize
import pandas as pd
import numpy as np
import joblib

def createData(abstract):

    """
    This function extracts the text from all the sentences and
    extracts the positional information of the sentences in an unstructract.

    Args:

        abstract - raw text of unstructured abstract.

    Returns: 

        A tuple containing list of sentences from the abstract and,
        a list of one hot encoded positional vector for all sentences. 

        Example :

        (
        ["Although immune-mediated ther..... promising treatment options.",
         "In renal cell carcino.....  with metastatic disease",
         "In urothelial carcinoma, cp..... for other indications.], 
          
        [[0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [1, 0, 0, 0, 0]]]
        )

    """


    data = sent_tokenize(abstract) # Tokenize each sentences
    abstracts = []
    
    # Divide abstract into rough sections. 
    position = ['#', 'FIRST', 'SECOND', 'THIRD', 'FOURTH', 'FIFTH']
    for line_no, abst_lines in enumerate(data):
        each_line = {}
        each_line["text"] = abst_lines 
        # Categorizes the position of sentence equally (1 to 5)
        scale_line = round(( (line_no + 1 - 1) / (len(data) - 1) ) * (5 - 1) + 1) 
        each_line['position'] = position[scale_line]
        abstracts.append(each_line)

    abstract = pd.DataFrame(abstracts)
    abs_sent = abstract.text
    one_hot = joblib.load('Model/one_hot.joblib')
    abs_pos = one_hot.transform(np.expand_dims(abstract.position, axis = 1)).toarray()

    return (abs_sent, abs_pos)


